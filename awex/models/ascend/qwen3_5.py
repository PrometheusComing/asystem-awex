# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import torch
from transformers import PretrainedConfig

from awex.converter.mcore_converter import (
    McoreToHFWeightConverter,
    _process_mcore_pp_name,
)
from awex.converter.sglang_converter import SGlangToHFWeightConverter
from awex.converter.weights_converter import append_scale_inv, normalize_scale_inv_name
from awex.sharding.param_sharding import (
    ShardingStrategy,
    ShardingType,
    get_default_sharding_dim,
)


class Qwen3VLShardingStrategy(ShardingStrategy):
    _visual_sharding_dims = {
        "merger.linear_fc1.weight": 0,
        "merger.linear_fc1.bias": 0,
        "merger.linear_fc2.weight": 1,
        "attn.qkv.weight": 0,
        "attn.qkv.bias": 0,
        "attn.proj.weight": 1,
        "mlp.linear_fc1.weight": 0,
        "mlp.linear_fc1.bias": 0,
        "mlp.linear_fc2.weight": 1,
    }

    _gdn_sharding_dims = {}

    def get_shared_expert_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for shared expert parameters.
        Returns (ShardingType, num_shards).
        """
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        if self.tp_size > 1:
            return ShardingType.TP_SHARDING, sharding_dim, self.tp_size
        else:
            return ShardingType.NO_SHARDING, sharding_dim, 1

    def _get_gdn_sharding_strategy(self, parameter_name):
        """Determine sharding strategy for GDN (Gated DeltaNet) linear-attn params.

        Ported from awex Qwen3_5ShardingMixin.get_attention_sharding_strategy
        (awex/sharding/param_sharding.py:356). GDN state params (in_proj_qkvz,
        in_proj_ba, conv1d, A_log, dt_bias) are per-head SSM state and must be
        TP-sharded along dim 0, not split across the data-parallel axis.
        out_proj is row-parallel (dim 1, like attention.dense). norm.weight is
        replicated (NO_SHARDING). input_layernorm.weight does not contain
        "linear_attn" so it never reaches this method (handled by the base
        class as NO_SHARDING).
        """
        # Norm weights are replicated (not TP-sharded).
        if "norm" in parameter_name:
            return ShardingType.NO_SHARDING, 0, 1
        # out_proj is row-parallel (dim 1).
        if "out_proj" in parameter_name:
            sharding_dim = 1
        else:
            # Column-parallel GDN params: in_proj_qkvz, in_proj_ba, conv1d,
            # A_log, dt_bias.
            sharding_dim = 0
        tp_size = self.rank_info.tp_size
        if tp_size > 1:
            return ShardingType.TP_SHARDING, sharding_dim, tp_size
        return ShardingType.NO_SHARDING, sharding_dim, 1

    def _get_visual_sharding_strategy(self, parameter_name, **kwargs):
        tp_size = self.rank_info.tp_size
        if tp_size == 1:
            return ShardingType.NO_SHARDING, 0, 1

        visual_prefix = "model.visual."
        if parameter_name.startswith(visual_prefix):
            suffix = parameter_name[len(visual_prefix) :]

            if suffix.startswith("blocks."):
                # Skip "blocks.{i}."
                parts = suffix.split(".", 2)
                if len(parts) >= 3:
                    suffix = parts[
                        2
                    ]  # e.g. "attn.qkv.weight" or "mlp.linear_fc1.weight"

            sharding_dim = self._visual_sharding_dims.get(suffix)
            if sharding_dim is not None:
                return ShardingType.TP_SHARDING, sharding_dim, tp_size

        return ShardingType.NO_SHARDING, 0, 1

    def get_sharding_strategy(self, parameter_name, **kwargs):
        if "visual." in parameter_name:
            return self._get_visual_sharding_strategy(parameter_name, **kwargs)
        if "linear_attn" in parameter_name:
            return self._get_gdn_sharding_strategy(parameter_name)
        if (
            "shared_expert_gate" in parameter_name
            or "shared_experts.gate_weight" in parameter_name
        ):
            return ShardingType.NO_SHARDING, 0, 1
        return super().get_sharding_strategy(parameter_name, **kwargs)


def _reshard_merged_column_parallel_for_infer(
    full_tensor: torch.Tensor,
    per_rank_component_sizes: list[int],
    dim: int,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> torch.Tensor:
    """Reshard a MergedColumnParallel tensor from train TP to infer TP.

    Ported from awex Qwen3_5McoreConverterMixin._reshard_merged_column_parallel_for_infer
    (awex/converter/mcore_converter.py:1062). See GDN_TP4_TP8_reshard_port.md for the
    verified bug-fix background.

    A MergedColumnParallel weight concatenates several logical components
    (e.g. ``[Q, K, V]`` for conv1d, ``[Q, K, V, Z]`` for in_proj_qkvz,
    ``[B, A]`` for in_proj_ba) along ``dim``. Each component is independently
    sharded across TP ranks by head.

    ``full_tensor`` is the all-gather result, laid out interleaved by train
    rank: ``[r0_comp0|...|r0_compN, r1_comp0|...|r1_compN, ...]``.
    ``per_rank_component_sizes`` gives the size of each component within one
    train-rank block (= global component size / train_tp_size).

    For r = infer_tp / train_tp_size:

    - r > 1: this train rank serves r infer ranks. Within this rank's block,
      each component is split into r equal pieces (one per infer rank), and the
      pieces are re-assembled in infer-rank order so that a downstream naive
      dim-0 split dispatches each piece to the correct infer rank.
    - r < 1 (s = train_tp_size / infer_tp train ranks per infer rank): the
      corresponding components from s consecutive train ranks are merged
      (concatenated per component, e.g. [Q_r0, Q_r1, K_r0, K_r1, ...]) into
      one infer rank's weight. This train rank contributes to infer rank
      ``train_tp_rank // s`` and returns its contiguous chunk of the merged
      infer-rank weight so that concatenating the s chunks in train-rank order
      reproduces the full infer-rank weight.
    """
    per_rank_total = sum(per_rank_component_sizes)
    rank_start = train_tp_rank * per_rank_total
    rank_block = full_tensor.narrow(dim, rank_start, per_rank_total)

    if infer_atten_tp_size == train_tp_size:
        # r == 1: this rank's block maps 1:1 to one infer rank.
        return rank_block.contiguous()
    if infer_atten_tp_size > train_tp_size:
        if infer_atten_tp_size % train_tp_size != 0:
            raise ValueError(
                f"infer_atten_tp_size ({infer_atten_tp_size}) must be a multiple "
                f"of train_tp_size ({train_tp_size}) for merged-column-parallel "
                f"resharding"
            )
        r = infer_atten_tp_size // train_tp_size
        components = list(torch.split(rank_block, per_rank_component_sizes, dim=dim))
        per_component_shards = [torch.chunk(c, r, dim=dim) for c in components]
        merged_infer_shards = [
            torch.cat([comp_shards[i] for comp_shards in per_component_shards], dim=dim)
            for i in range(r)
        ]
        return torch.cat(merged_infer_shards, dim=dim).contiguous()

    # infer_atten_tp_size < train_tp_size: each infer rank merges the
    # corresponding components from s = train_tp_size / infer_atten_tp_size
    # train ranks (e.g. infer_tp=2, train_tp=4 -> s=2, train ranks 0,1 feed
    # infer rank 0; train ranks 2,3 feed infer rank 1). For each component,
    # concatenate the s per-rank pieces so that component boundaries are
    # respected (NOT a flat concat of two ranks' entire blocks).
    if train_tp_size % infer_atten_tp_size != 0:
        raise ValueError(
            f"train_tp_size ({train_tp_size}) must be a multiple of "
            f"infer_atten_tp_size ({infer_atten_tp_size}) for "
            f"merged-column-parallel resharding when infer_tp < train_tp"
        )
    s = train_tp_size // infer_atten_tp_size
    infer_rank = train_tp_rank // s
    pos_in_group = train_tp_rank % s
    train_start = infer_rank * s
    merged_components = []
    for comp_idx, comp_size in enumerate(per_rank_component_sizes):
        comp_offset = sum(per_rank_component_sizes[:comp_idx])
        parts = [
            full_tensor.narrow(dim, tr * per_rank_total + comp_offset, comp_size)
            for tr in range(train_start, train_start + s)
        ]
        merged_components.append(torch.cat(parts, dim=dim))
    infer_weight = torch.cat(merged_components, dim=dim)
    chunk_start = pos_in_group * per_rank_total
    return infer_weight.narrow(dim, chunk_start, per_rank_total).contiguous()


def _q_interleave_index(q_heads_per_rank: int) -> list[int]:
    """Build Q interleave index: [u0..u_{h-1}, c0..c_{h-1}] -> [u0,c0,u1,c1,...].

    Ported from awex Qwen3_5McoreConverterMixin._q_interleave_index
    (awex/converter/mcore_converter.py:1366).

    Q is duplicated: first half unique, second half copies (c_i = u_i).
    HF stores them interleaved [unique, copy, unique, copy, ...].
    """
    half = q_heads_per_rank // 2
    idx = []
    for i in range(half):
        idx.append(i)  # unique head i
        idx.append(half + i)  # its copy
    return idx


def _split_mcore_gated_attn_qkv(
    linear_qkv: torch.Tensor,
    hf_config,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    """Convert Qwen3.5 full-attention QKV: mcore GQA-interleaved -> infer layout.

    u->q_heads=24,using 8 as example.
    g->gate_heads=24,using 8 as example just like q.
    k,v->kv_heads=4

    mcore 4 group,q_heads // kv_heads

    g0:[u0 u1 g0 g1 k0 v0]
    g1:[u2 u3 g2 g3 k1 v1]
    g2:[u4 u5 g4 g5 k2 v2]
    g3:[u6 u7 g6 g7 k3 v3]
    when tp=2, rank_0 has g0、g1 and rank_1 has g2、g3。every rank has full group
    when tp=4, rank_x has gx,x in [0,1,2,3]。every rank has full group
    when tp=8，rank_0 has [u0 u1 g0] and rank_1 has [g1 k0 v0].they split up g0 in half.

    vllm

    [u0 g0 u1 g1 u2 g2 u3 g3 u4 g4 u5 g5 u6 g6 u7 g7]
    [k0 k1 k2 k3]
    [v0 v1 v2 v3]
    when tp=2,rank_0 has [u0 g0 u1 g1 u2 g2 u3 g3 k0 k1 v0 v1]
    when tp=4,rank_0 has [u0 g0 u1 g1 k0 v0]
    when tp=8,repeat kv:
    [u0 g0 u1 g1 u2 g2 u3 g3 u4 g4 u5 g5 u6 g6 u7 g7]
    [k0 k0 k1 k1 k2 k2 k3 k3]
    [v0 v0 v1 v1 v2 v2 v3 v3]
    rank_0 has [u0 g0 k0 v0]
    rank_1 has [u1 g1 k0 v0]

    No mater what the train tp is,just allgather to get all groups.
    then traverse each group to get all q,z,k,v.like:
    g0 [u0 u1 g0 g1 k0 v0] → q_u=[u0,u1], z=[g0,g1], k=[k0], v=[v0]
    g1 [u2 u3 g2 g3 k1 v1] → q_u=[u2,u3], z=[g2,g3], k=[k1], v=[v1]
    g2 [u4 u5 g4 g5 k2 v2] → q_u=[u4,u5], z=[g4,g5], k=[k2], v=[v2]
    g3 [u6 u7 g6 g7 k3 v3] → q_u=[u6,u7], z=[g6,g7], k=[k3], v=[v3]
    q_unique = [u0 u1 u2 u3 u4 u5 u6 u7]
    z_full   = [g0 g1 g2 g3 g4 g5 g6 g7]
    key      = [k0 k1 k2 k3]
    value    = [v0 v1 v2 v3]
    interleave_idx = _q_interleave_index(16)--> [0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15]
    query = [u0 g0 u1 g1 u2 g2 u3 g3 u4 g4 u5 g5 u6 g6 u7 g7]
    key   =  [k0 k1 k2 k3]
    value =  [v0 v1 v2 v3]
    """
    from awex.converter.mcore_converter import get_full_tensor

    text_config = getattr(hf_config, "text_config", None) or hf_config
    total_num_heads = int(getattr(text_config, "num_attention_heads", 0))
    total_num_kv_heads = int(getattr(text_config, "num_key_value_heads", 0))
    head_size = int(getattr(text_config, "head_dim", 0))
    if not head_size:
        head_size = int(getattr(text_config, "hidden_size", 0)) // total_num_heads
    if not (total_num_heads and total_num_kv_heads and head_size):
        raise ValueError(
            f"num_attention_heads/num_key_value_heads/head_dim are required in "
            f"hf_config.text_config to convert the gated-attn QKV, but some "
            f"were missing: heads={total_num_heads}, kv={total_num_kv_heads}, "
            f"head_size={head_size}"
        )
    heads_per_group = total_num_heads // total_num_kv_heads

    per_group_slots = 2 * heads_per_group + 2
    full_rows = per_group_slots * total_num_kv_heads * head_size
    # Row sizes of each component within one kv-group.
    q_u_rows = heads_per_group * head_size
    z_rows = heads_per_group * head_size
    kv_rows = head_size

    train_tp_size_eff = train_tp_size if (train_tp_size and train_tp_size > 0) else 1
    local_size = linear_qkv.shape[0]

    if train_tp_size_eff > 1:
        if local_size == full_rows:
            need_gather = False
        elif local_size * train_tp_size_eff == full_rows:
            need_gather = True
        else:
            raise ValueError(
                f"QKV weight size mismatch: local_size={local_size}, "
                f"train_tp_size={train_tp_size_eff}. Expected a per-rank shard "
                f"({full_rows // train_tp_size_eff}) or full ({full_rows}) for "
                f"the gated GQA layout. num_heads={total_num_heads}, "
                f"num_kv_heads={total_num_kv_heads}, head_size={head_size}"
            )
    else:
        need_gather = False
        if local_size != full_rows:
            raise ValueError(
                f"QKV weight size mismatch: local_size={local_size}, expected "
                f"full={full_rows} for the gated GQA layout. num_heads="
                f"{total_num_heads}, num_kv_heads={total_num_kv_heads}, "
                f"head_size={head_size}"
            )

    weight = get_full_tensor(linear_qkv, dim=0) if need_gather else linear_qkv
    if weight.shape[0] != full_rows or weight.ndim != 2:
        raise ValueError(
            f"QKV full weight mismatch after gather: shape="
            f"{tuple(weight.shape)}, ndim={weight.ndim}, expected "
            f"[{full_rows}, *]. Qwen3.5 uses attention_bias=false so a 2D "
            f"weight is expected on this path."
        )
    feature_dim = weight.shape[-1]
    q_u_parts, z_parts, key_parts, value_parts = [], [], [], []
    for group in torch.chunk(weight, total_num_kv_heads, dim=0):
        q_u, z, k, v = group.split([q_u_rows, z_rows, kv_rows, kv_rows], dim=0)
        q_u_parts.append(q_u)
        z_parts.append(z)
        key_parts.append(k)
        value_parts.append(v)
    q_unique = torch.cat(q_u_parts, dim=0)  # [num_heads * head_size, feature]
    z_full = torch.cat(z_parts, dim=0)  # [num_heads * head_size, feature]
    key = torch.cat(key_parts, dim=0)  # [num_kv * head_size, feature]
    value = torch.cat(value_parts, dim=0)  # [num_kv * head_size, feature]

    # Q is duplicated (unique + gate). mcore stores them per group as
    # [Q_u, Z]; HF wants them interleaved per head [u0, c0, u1, c1, ...].
    # Gather into [all-unique, all-gate] then interleave per head to HF order.
    effective_q = 2 * total_num_heads
    combined = torch.cat([q_unique, z_full], dim=0)
    interleave_idx = _q_interleave_index(effective_q)
    query = combined.view(effective_q, head_size, feature_dim)[interleave_idx].reshape(
        -1, feature_dim
    )
    # Free the gate intermediates (only `query` is needed downstream).
    del q_unique, z_full, combined
    if infer_atten_tp_size >= total_num_kv_heads:
        num_kv_head_replicas = infer_atten_tp_size // total_num_kv_heads
        key_shards = [
            k
            for k in key.chunk(total_num_kv_heads, dim=0)
            for _ in range(num_kv_head_replicas)
        ]
        value_shards = [
            v
            for v in value.chunk(total_num_kv_heads, dim=0)
            for _ in range(num_kv_head_replicas)
        ]
    else:
        key_shards = key.chunk(infer_atten_tp_size, dim=0)
        value_shards = value.chunk(infer_atten_tp_size, dim=0)
    query_shards = query.chunk(infer_atten_tp_size, dim=0)

    qkv_tp_groups = []
    for query_shard, key_shard, value_shard in zip(
        query_shards, key_shards, value_shards
    ):
        qkv_tp_groups.append(query_shard)
        qkv_tp_groups.append(key_shard)
        qkv_tp_groups.append(value_shard)
    merged = torch.cat(qkv_tp_groups, dim=0)

    if train_tp_size_eff > 1:
        if train_tp_rank is None:
            raise ValueError("train_tp_rank is required when train_tp_size > 1")
        shards = torch.chunk(merged, train_tp_size_eff, dim=0)
        if train_tp_rank >= len(shards):
            raise ValueError(
                f"train_tp_rank {train_tp_rank} out of range for "
                f"tp_size {train_tp_size_eff}"
            )
        merged = shards[train_tp_rank]
    return [("self_attn.qkv_proj.weight", merged)]


def _reshard_mcore_gdn_conv1d(
    conv1d: torch.Tensor,
    hf_config,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    """Reshard GDN conv1d.weight from train TP to infer TP.

    Ported from awex Qwen3_5McoreConverterMixin._convert_linear_attention_param
    (conv1d.weight branch, awex/converter/mcore_converter.py:1226).

    conv1d out_channels are [Q, K, V] concatenated (Z does NOT enter conv1d).
    Each component is independently head-sharded, so use the merged-aware
    reshard to respect component boundaries when r > 1. Some mcore GDN impls
    replicate conv1d across TP ranks (small param); detect that case and use
    a single copy so resharding doesn't produce duplicated shards.
    """
    from awex.converter.mcore_converter import get_full_tensor

    text_config = getattr(hf_config, "text_config", None) or hf_config
    linear_num_key_heads = int(getattr(text_config, "linear_num_key_heads", 0))
    linear_key_head_dim = int(getattr(text_config, "linear_key_head_dim", 0))
    linear_num_value_heads = int(getattr(text_config, "linear_num_value_heads", 0))
    linear_value_head_dim = int(getattr(text_config, "linear_value_head_dim", 0))
    if not (
        linear_num_key_heads
        and linear_key_head_dim
        and linear_num_value_heads
        and linear_value_head_dim
    ):
        raise ValueError(
            "linear_num_key_heads / linear_key_head_dim / "
            "linear_num_value_heads / linear_value_head_dim are "
            "required in hf_config.text_config to reshard the GDN "
            "conv1d.weight, but some were missing."
        )
    qk_dim = linear_num_key_heads * linear_key_head_dim
    v_dim = linear_num_value_heads * linear_value_head_dim

    local_shape = tuple(conv1d.shape)
    full = get_full_tensor(conv1d, dim=0)
    train_tp_size_eff = train_tp_size if train_tp_size > 0 else 1

    # Detect replicated conv1d (some mcore GDN impls replicate this small param)
    is_replicated = False
    if (
        train_tp_size_eff > 1
        and full.shape[0] == local_shape[0] * train_tp_size_eff
        and local_shape[0] > 0
    ):
        first = full.narrow(0, 0, local_shape[0])
        second = full.narrow(0, local_shape[0], local_shape[0])
        if torch.equal(first[0], second[0]) and torch.equal(first[-1], second[-1]):
            full = first.contiguous()
            is_replicated = True

    if is_replicated:
        # conv1d is replicated: every train rank holds the same full [Q,K,V]
        # (global, not per-rank). Shard each component by infer_tp and pick
        # this rank's r slices.
        infer_tp = infer_atten_tp_size
        r = infer_tp // train_tp_size_eff if infer_tp > train_tp_size_eff else 1
        start = train_tp_rank * r
        end = start + r
        components = list(torch.split(full, [qk_dim, qk_dim, v_dim], dim=0))
        per_component_shards = [torch.chunk(c, infer_tp, dim=0) for c in components]
        merged = [
            torch.cat([comp_shards[i] for comp_shards in per_component_shards], dim=0)
            for i in range(start, end)
        ]
        out = torch.cat(merged, dim=0).contiguous()
    else:
        # Standard column-parallel: full is interleaved by train rank,
        # each rank block = [Q_local, K_local, V_local].
        if qk_dim % train_tp_size_eff != 0 or v_dim % train_tp_size_eff != 0:
            raise ValueError(
                f"qk_dim ({qk_dim}) and v_dim ({v_dim}) must be "
                f"divisible by train_tp_size ({train_tp_size_eff}) for "
                f"GDN conv1d reshard"
            )
        per_rank_qk = qk_dim // train_tp_size_eff
        per_rank_v = v_dim // train_tp_size_eff
        out = _reshard_merged_column_parallel_for_infer(
            full,
            [per_rank_qk, per_rank_qk, per_rank_v],
            0,
            infer_atten_tp_size,
            train_tp_rank,
            train_tp_size_eff,
        )
    return [("linear_attn.conv1d.weight", out)]


def _split_mcore_gdn_in_proj(
    in_proj: torch.Tensor,
    hf_config,
    infer_atten_tp_size: int,
    train_tp_rank: int,
    train_tp_size: int,
) -> list[tuple[str, torch.Tensor]]:
    """Split GDN in_proj.weight (fused [qkvz; ba]) and reshard train TP -> infer TP.

    Ported from awex Qwen3_5McoreConverterMixin._convert_linear_attention_param
    (in_proj.weight branch, awex/converter/mcore_converter.py:1137).

    GDN in_proj is column-parallel (fused [qkvz; ba] along dim 0). Each train
    rank stores [local_qkvz; local_ba], so after get_full_tensor (all_gather)
    the layout is interleaved by rank: [r0_qkvz, r0_ba, r1_qkvz, r1_ba, ...]
    NOT [global_qkvz; global_ba]. We must de-interleave rank chunks first,
    then reshard each part (qkvz / ba) from train TP to infer TP.

    global_ba = 2 * linear_num_value_heads (the ``b`` and ``a`` projections
    each have one entry per value head); per-rank ba = global_ba / train_tp_size.
    qkvz components = [Q, K, V, Z] (Q=K=qk_dim, V=Z=v_dim).
    ba components = [B, A] (each = num_value_heads per rank).
    """
    from awex.converter.mcore_converter import get_full_tensor

    text_config = getattr(hf_config, "text_config", None) or hf_config
    linear_num_value_heads = int(getattr(text_config, "linear_num_value_heads", 0))
    linear_num_key_heads = int(getattr(text_config, "linear_num_key_heads", 0))
    linear_key_head_dim = int(getattr(text_config, "linear_key_head_dim", 0))
    linear_value_head_dim = int(getattr(text_config, "linear_value_head_dim", 0))
    if not linear_num_value_heads:
        raise ValueError(
            "linear_num_value_heads is required in hf_config.text_config "
            "to split the GDN in_proj.weight, but it was not found."
        )
    if not (linear_num_key_heads and linear_key_head_dim and linear_value_head_dim):
        raise ValueError(
            "linear_num_key_heads / linear_key_head_dim / "
            "linear_value_head_dim are required in hf_config.text_config "
            "to reshard the GDN in_proj.weight, but some were missing."
        )
    qk_dim = linear_num_key_heads * linear_key_head_dim
    v_dim = linear_num_value_heads * linear_value_head_dim
    global_ba = 2 * linear_num_value_heads
    train_tp_size_eff = train_tp_size if train_tp_size > 0 else 1

    full = get_full_tensor(in_proj, dim=0)

    if train_tp_size_eff > 1:
        if global_ba % train_tp_size_eff != 0:
            raise ValueError(
                f"Global GDN ba dim ({global_ba}) is not divisible by "
                f"train_tp_size ({train_tp_size_eff})"
            )
        local_ba = global_ba // train_tp_size_eff
        # De-interleave: [r0_qkvz, r0_ba, r1_qkvz, r1_ba, ...]
        # -> [global_qkvz; global_ba]
        rank_chunks = torch.chunk(full, train_tp_size_eff, dim=0)
        qkvz_parts = []
        ba_parts = []
        for chunk in rank_chunks:
            local_qkvz = chunk.shape[0] - local_ba
            qkvz_parts.append(chunk.narrow(0, 0, local_qkvz))
            ba_parts.append(chunk.narrow(0, local_qkvz, local_ba))
        qkvz_full = torch.cat(qkvz_parts, dim=0)
        ba_full = torch.cat(ba_parts, dim=0)
    else:
        # train_tp_size == 1: local tensor is already the full tensor
        # with layout [global_qkvz; global_ba].
        local_ba = global_ba
        local_qkvz = full.shape[0] - local_ba
        qkvz_full = full.narrow(0, 0, local_qkvz)
        ba_full = full.narrow(0, local_qkvz, local_ba)

    # qkvz and ba are MergedColumnParallel: each component ([Q,K,V,Z] and
    # [B,A]) is independently head-sharded. Use the merged-aware reshard so
    # component boundaries are respected when r > 1. Per-rank component
    # sizes = global component size / train_tp_size.
    if qk_dim % train_tp_size_eff != 0 or v_dim % train_tp_size_eff != 0:
        raise ValueError(
            f"qk_dim ({qk_dim}) and v_dim ({v_dim}) must be divisible "
            f"by train_tp_size ({train_tp_size_eff}) for GDN in_proj reshard"
        )
    per_rank_qk = qk_dim // train_tp_size_eff
    per_rank_v = v_dim // train_tp_size_eff
    per_rank_v_heads = linear_num_value_heads // train_tp_size_eff
    qkvz = _reshard_merged_column_parallel_for_infer(
        qkvz_full,
        [per_rank_qk, per_rank_qk, per_rank_v, per_rank_v],
        0,
        infer_atten_tp_size,
        train_tp_rank,
        train_tp_size_eff,
    )
    ba = _reshard_merged_column_parallel_for_infer(
        ba_full,
        [per_rank_v_heads, per_rank_v_heads],
        0,
        infer_atten_tp_size,
        train_tp_rank,
        train_tp_size_eff,
    )
    return [
        ("linear_attn.in_proj_qkvz.weight", qkvz),
        ("linear_attn.in_proj_ba.weight", ba),
    ]


def reshard_visual_attn_qkv(
    parameter: torch.Tensor,
    infer_atten_tp_size: int,
    vision_config: PretrainedConfig,
    train_tp_rank: int,
    train_tp_size: int,
):
    from awex.converter.mcore_converter import get_full_tensor

    weight = get_full_tensor(parameter, dim=0)
    num_heads = vision_config.num_heads
    head_dim = vision_config.hidden_size // num_heads
    query_list = []
    key_list = []
    value_list = []
    for qkv in torch.chunk(weight, num_heads, dim=0):
        q, k, v = qkv.split([head_dim, head_dim, head_dim], dim=0)
        query_list.append(q)
        key_list.append(k)
        value_list.append(v)
    # concat the query, key, value
    all_query = torch.cat(query_list, dim=0)
    all_key = torch.cat(key_list, dim=0)
    all_value = torch.cat(value_list, dim=0)

    query_shards = all_query.chunk(infer_atten_tp_size, dim=0)
    key_shards = all_key.chunk(infer_atten_tp_size, dim=0)
    value_shards = all_value.chunk(infer_atten_tp_size, dim=0)
    qkv_tp_groups = []
    for query_shard, key_shard, value_shard in zip(
        query_shards, key_shards, value_shards
    ):
        qkv_tp_groups.append(query_shard)
        qkv_tp_groups.append(key_shard)
        qkv_tp_groups.append(value_shard)
    merged = torch.cat(qkv_tp_groups, dim=0)
    if train_tp_size and train_tp_size > 1:
        if train_tp_rank is None:
            raise ValueError("train_tp_rank is required when train_tp_size > 1")
        shards = torch.chunk(merged, train_tp_size, dim=0)
        if train_tp_rank >= len(shards):
            raise ValueError(
                f"train_tp_rank {train_tp_rank} out of range for tp_size {train_tp_size}"
            )
        return shards[train_tp_rank]
    return merged


class McoreToHFWeightConverterQwen3VL(McoreToHFWeightConverter):
    def __init__(self, hf_config, rank_info, infer_conf, tf_config):
        super().__init__(hf_config.text_config, rank_info, infer_conf, tf_config)
        self.vision_config = hf_config.vision_config

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_vision_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert vision encoder (vision_model) parameters from mcore to HF format.

        Name mapping:
          mcore: module.vision_model.decoder.layers.{i}.self_attention.linear_qkv.weight
          HF:    model.visual.blocks.{i}.attn.qkv.weight

          mcore: module.vision_model.decoder.layers.{i}.self_attention.linear_proj.weight
          HF:    model.visual.blocks.{i}.attn.proj.weight

          mcore: module.vision_model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight
          HF:    model.visual.blocks.{i}.norm1.weight

          mcore: module.vision_model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight
          HF:    model.visual.blocks.{i}.norm2.weight

          mcore: module.vision_model.merger.patch_norm.weight
          HF:    model.visual.merger.norm.weight

          mcore: module.vision_model.decoder.layers.{i}  →  model.visual.blocks.{i}
        """
        # Strip "module.vision_model." prefix (already stripped "module.module.")
        # After stripping, remaining starts with "vision_model."
        assert name.startswith("vision_model."), f"Expected vision_model prefix: {name}"
        remaining = name[len("vision_model.") :]

        # --- Top-level vision params (patch_embed, pos_embed, merger) ---
        if remaining.startswith("patch_embed."):
            return [(f"model.visual.{remaining}", parameter)]
        if remaining.startswith("pos_embed."):
            return [(f"model.visual.{remaining}", parameter)]
        if remaining.startswith("merger."):
            # merger.patch_norm → merger.norm
            remaining = remaining.replace("merger.patch_norm.", "merger.norm.", 1)
            return [(f"model.visual.{remaining}", parameter)]

        # --- Block-level vision params ---
        # mcore: decoder.layers.{i}.self_attention.* or decoder.layers.{i}.mlp.*
        if remaining.startswith("decoder.layers."):
            # Extract layer index and sub-name
            rest = remaining[len("decoder.layers.") :]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse vision block name: {name}")
            block_idx = parts[0]
            sub_name = parts[1]

            # self_attention → attn
            if sub_name.startswith("self_attention."):
                attn_sub = sub_name[len("self_attention.") :]
                # linear_qkv.layer_norm_weight → norm1.weight
                if attn_sub == "linear_qkv.layer_norm_weight":
                    return [
                        (f"model.visual.blocks.{block_idx}.norm1.weight", parameter)
                    ]
                if attn_sub == "linear_qkv.layer_norm_bias":
                    return [(f"model.visual.blocks.{block_idx}.norm1.bias", parameter)]
                # linear_qkv.weight/bias → attn.qkv.weight/bias
                if attn_sub == "linear_qkv.weight":
                    reshard_param = reshard_visual_attn_qkv(
                        parameter,
                        self.infer_atten_tp_size,
                        self.vision_config,
                        self.rank_info.attn_tp_rank,
                        self.rank_info.attn_tp_size,
                    )
                    return [
                        (
                            f"model.visual.blocks.{block_idx}.attn.qkv.weight",
                            reshard_param,
                        )
                    ]
                if attn_sub == "linear_qkv.bias":
                    reshard_param = reshard_visual_attn_qkv(
                        parameter,
                        self.infer_atten_tp_size,
                        self.vision_config,
                        self.rank_info.attn_tp_rank,
                        self.rank_info.attn_tp_size,
                    )
                    return [
                        (
                            f"model.visual.blocks.{block_idx}.attn.qkv.bias",
                            reshard_param,
                        )
                    ]
                # linear_proj.weight/bias → attn.proj.weight/bias
                if attn_sub == "linear_proj.weight":
                    return [
                        (f"model.visual.blocks.{block_idx}.attn.proj.weight", parameter)
                    ]
                if attn_sub == "linear_proj.bias":
                    return [
                        (f"model.visual.blocks.{block_idx}.attn.proj.bias", parameter)
                    ]
                raise NotImplementedError(f"Unsupported vision attn param: {name}")

            # mlp
            if sub_name.startswith("mlp."):
                mlp_sub = sub_name[len("mlp.") :]
                # linear_fc1.layer_norm_weight → norm2.weight
                if mlp_sub == "linear_fc1.layer_norm_weight":
                    return [
                        (f"model.visual.blocks.{block_idx}.norm2.weight", parameter)
                    ]
                if mlp_sub == "linear_fc1.layer_norm_bias":
                    return [(f"model.visual.blocks.{block_idx}.norm2.bias", parameter)]
                # linear_fc1/fc2 weight/bias → mlp.linear_fc1/fc2 weight/bias
                if mlp_sub in (
                    "linear_fc1.weight",
                    "linear_fc1.bias",
                    "linear_fc2.weight",
                    "linear_fc2.bias",
                ):
                    return [
                        (f"model.visual.blocks.{block_idx}.mlp.{mlp_sub}", parameter)
                    ]
                raise NotImplementedError(f"Unsupported vision mlp param: {name}")

            raise NotImplementedError(f"Unsupported vision block param: {name}")

        raise NotImplementedError(f"Unsupported vision param: {name}")

    @staticmethod
    def _resolve_dtype_from_config(dtype_value):
        """Resolve a dtype value from hf_config into a torch.dtype.

        ``hf_config`` may store dtypes either as strings (e.g. ``"float32"``,
        ``"bfloat16"``) or already as ``torch.dtype``. This helper normalizes
        both forms so callers can ``.to(target_dtype)`` directly.
        """
        if isinstance(dtype_value, torch.dtype):
            return dtype_value
        if isinstance(dtype_value, str):
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
            }
            key = dtype_value.strip().lower()
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported dtype value from config: {dtype_value!r}")

    def _gdn_state_param_dtype(self) -> torch.dtype:
        """Dtype used by the inference side for GDN/SSM state params (A_log).

        Read from ``hf_config.mamba_ssm_dtype`` so the conversion tracks the
        actual model config rather than a hardcoded target.
        """
        return self._resolve_dtype_from_config(
            getattr(self.hf_config, "mamba_ssm_dtype", "float32")
        )

    def _gdn_compute_dtype(self) -> torch.dtype:
        """Dtype used by the inference side for regular GDN params (dt_bias, norm).

        Read from ``hf_config.dtype`` (the model compute dtype) so the
        conversion tracks the actual model config rather than a hardcoded target.
        """
        return self._resolve_dtype_from_config(
            getattr(self.hf_config, "dtype", "bfloat16")
        )

    def _convert_gdn_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> list[tuple[str, torch.Tensor]]:
        if "in_proj.weight" in name:
            return _split_mcore_gdn_in_proj(
                parameter,
                self.hf_config,
                self.infer_atten_tp_size,
                self.rank_info.attn_tp_rank,
                self.rank_info.attn_tp_size,
            )
        elif "in_proj.layer_norm_weight" in name:
            return [("input_layernorm.weight", parameter)]
        elif "conv1d.weight" in name:
            return _reshard_mcore_gdn_conv1d(
                parameter,
                self.hf_config,
                self.infer_atten_tp_size,
                self.rank_info.attn_tp_rank,
                self.rank_info.attn_tp_size,
            )
        elif "dt_bias" in name:
            # dt_bias is a regular bias param on the inference side and is
            # stored in the model compute dtype (hf_config.dtype, e.g.
            # bfloat16), NOT in mamba_ssm_dtype. Convert accordingly so the
            # meta check passes.
            target_dtype = self._gdn_compute_dtype()
            if parameter.dtype != target_dtype:
                parameter = parameter.to(target_dtype)
            return [("linear_attn.dt_bias", parameter)]
        elif "A_log" in name:
            # A_log is an SSM state param on the inference side and is stored
            # in mamba_ssm_dtype (e.g. float32) for numerical stability,
            # NOT in the model compute dtype. Convert accordingly so the
            # meta check passes.
            target_dtype = self._gdn_state_param_dtype()
            if parameter.dtype != target_dtype:
                parameter = parameter.to(target_dtype)
            return [("linear_attn.A_log", parameter)]
        elif "out_norm.weight" in name:
            # ``parameter + 1.0`` promotes to float32; cast back to the model
            # compute dtype so the norm weight matches the inference side.
            target_dtype = self._gdn_compute_dtype()
            return [("linear_attn.norm.weight", (parameter + 1.0).to(target_dtype))]
        elif "out_proj.weight" in name:
            return [("linear_attn.out_proj.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported GDN parameter name: {name}")

    def _is_linear_attn_layer(self, layer_number: int, name: str) -> bool:
        # first time here,self._pp_stage_layer_id_map is {}
        # when pp > 1,layer_number is local rank。it does not support the num_hidden_layers % (pp * full_attention_interval) != 0
        if not self._pp_stage_layer_id_map:
            gdn_keys = ["dt_bias", "A_log", "in_proj", "conv1d", "out_norm", "out_proj"]
            for key in gdn_keys:
                if key in name:
                    return True
            return False

        text_config = self.hf_config
        layer_types = getattr(text_config, "layer_types", [])
        if layer_types:
            return layer_types[layer_number] == "linear_attention"
        interval = getattr(text_config, "full_attention_interval", 4)
        return (layer_number + 1) % interval != 0

    def _convert_attn_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> list[tuple[str, torch.Tensor]]:
        name = _process_mcore_pp_name(
            name,
            self.rank_info,
            self.hf_config,
            self.tf_config,
            vp_stage=vp_stage,
            pp_stage_layer_id_map=self._pp_stage_layer_id_map,
        )
        rest = name.split("decoder.layers.", 1)[1]
        layer_str = rest.split(".", 1)[0]
        layer_number = int(layer_str)

        if self._is_linear_attn_layer(layer_number, name):
            # GDN linear attention
            converted = []
            for sub_name, param in self._convert_gdn_param(
                name, parameter, str(layer_number)
            ):
                converted.append((f"model.layers.{layer_number}.{sub_name}", param))
            return converted
        else:
            # Gated full attention with attention_output_gate
            if "linear_qkv.weight" in name or name.endswith("linear_qkv"):
                converted = _split_mcore_gated_attn_qkv(
                    parameter,
                    self.hf_config,
                    self.infer_atten_tp_size,
                    self.rank_info.attn_tp_rank,
                    self.rank_info.attn_tp_size,
                )
                result = []
                for sub_name, param in converted:
                    sub_name = self._normalize_attn_name(sub_name)
                    result.append((f"model.layers.{layer_number}.{sub_name}", param))
                return result
            else:
                converted = []
                for attn_name, param in self._convert_attention_param(
                    name, parameter, layer_str
                ):
                    attn_name = self._normalize_attn_name(attn_name)
                    converted.append(
                        (f"model.layers.{layer_number}.{attn_name}", param)
                    )
                return converted

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> list[tuple[str, torch.Tensor]]:
        name = name.replace("module.", "")

        # ---- Vision encoder parameters ----
        if name.startswith("vision_model."):
            return self._convert_vision_param(name, parameter)

        language_prefix = "language_model."
        name = name.replace(language_prefix, "")

        if "self_attention" in name:
            converted_params = self._convert_attn_param(
                name, parameter, vp_stage=vp_stage
            )
        else:
            converted_params = super().convert_param(name, parameter, vp_stage=vp_stage)

        if len(converted_params) == 1 and converted_params[0][0] == "lm_head.weight":
            return converted_params
        return [
            (s.replace("model.", "model.language_model.", 1), t)
            for s, t in converted_params
        ]


class VLLMToHFWeightConverterQwen3VL(
    SGlangToHFWeightConverter,
):
    """vLLM-side converter for Qwen3-VL multimodal models.

    Handles the ``model.language_model.*`` prefix, GDN linear_attn
    parameters, and standard MHA self_attn with separate Q/K/V projections.
    vLLM uses the same parameter names as the HF checkpoint for Qwen3-VL,
    so minimal normalization is needed.
    """

    def __init__(
        self,
        model_config,
        infer_engine_config,
        rank_info,
    ):
        super().__init__(model_config.text_config, infer_engine_config, rank_info)

    def _fuse_qkv(self, name: str) -> bool:
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_visual_param(self, name: str, parameter):
        return [(f"model.{name}", parameter)]

    def _convert_gdn_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        rest = name.split("layers.", 1)[1]
        layer_str = rest.split(".", 1)[0]
        prefix = f"model.layers.{layer_str}."

        if "in_proj_qkvz" in name:
            return [(f"{prefix}linear_attn.in_proj_qkvz.weight", parameter)]
        elif "in_proj_ba" in name:
            return [(f"{prefix}linear_attn.in_proj_ba.weight", parameter)]
        elif "conv1d.weight" in name:
            return [(f"{prefix}linear_attn.conv1d.weight", parameter)]
        elif "dt_bias" in name:
            return [(f"{prefix}linear_attn.dt_bias", parameter)]
        elif "A_log" in name:
            return [(f"{prefix}linear_attn.A_log", parameter)]
        elif "norm.weight" in name:
            return [(f"{prefix}linear_attn.norm.weight", parameter)]
        elif "out_proj.weight" in name:
            return [(f"{prefix}linear_attn.out_proj.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported vLLM GDN parameter name: {name}")

    def _normalize_name(self, name: str) -> str:
        name, has_scale_inv = normalize_scale_inv_name(name)
        replacements = [
            (".self_attn.attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.qkv", ".attention.query_key_value_proj"),
            (".self_attn.qkv_proj", ".attention.query_key_value_proj"),
            (".self_attn.attn.o_proj", ".attention.dense"),
            (".self_attn.o_proj", ".attention.dense"),
            (".self_attn.proj", ".attention.dense"),
            (".self_attn.q_norm", ".attention.query_layernorm"),
            (".self_attn.k_norm", ".attention.key_layernorm"),
        ]
        for old, new in replacements:
            if old in name:
                name = name.replace(old, new)
        # Guard against double normalization.
        name = name.replace("query_key_value_proj_proj", "query_key_value_proj")
        return append_scale_inv(name, has_scale_inv)

    def convert_param(self, name, parameter):
        if name.startswith("visual."):
            converted_params = self._convert_visual_param(name, parameter)
            return converted_params

        language_prefix = "language_model."
        name = name.replace(language_prefix, "").replace(
            "shared_expert.", "shared_experts."
        )

        if "linear_attn" in name:
            converted_params = self._convert_gdn_param(name, parameter)
        else:
            converted_params = super().convert_param(
                self._normalize_name(name), parameter
            )

        if len(converted_params) == 1 and converted_params[0][0] == "lm_head.weight":
            return converted_params
        return [
            (s.replace("model.", "model.language_model.", 1), t)
            for s, t in converted_params
        ]


CONFIG = [
    {
        "model_name": "Qwen3_5ForCausalLM",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
    {
        "model_name": "Qwen3_5MoeForCausalLM",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
    {
        "model_name": "Qwen3_5ForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
    {
        "model_name": "Qwen3_5MoeForConditionalGeneration",
        "sharding_strategy": Qwen3VLShardingStrategy,
        "mcore_converter": McoreToHFWeightConverterQwen3VL,
        "vllm_converter": VLLMToHFWeightConverterQwen3VL,
    },
]
