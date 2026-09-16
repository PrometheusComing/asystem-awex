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

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)


def _ensure_awex_importable():
    try:
        import awex  # noqa: F401

        return
    except Exception:
        pass
    awex = types.ModuleType("awex")
    awex.__path__ = []
    ac = types.ModuleType("awex.converter")
    ac.__path__ = []
    awex.converter = ac
    am = types.ModuleType("awex.converter.mcore_converter")
    ac.mcore_converter = am
    am.get_full_tensor = lambda w, dim=0: w
    am.McoreToHFWeightConverter = type("X", (), {})
    am._process_mcore_pp_name = lambda n, *a, **k: n
    sg = types.ModuleType("awex.converter.sglang_converter")
    ac.sglang_converter = sg
    sg.SGlangToHFWeightConverter = type("X", (), {})
    wc = types.ModuleType("awex.converter.weights_converter")
    ac.weights_converter = wc
    wc.append_scale_inv = lambda n, b: n
    wc.normalize_scale_inv_name = lambda n: (n, False)
    ashp = types.ModuleType("awex.sharding")
    ashp.__path__ = []
    awex.sharding = ashp
    shp = types.ModuleType("awex.sharding.param_sharding")
    ashp.param_sharding = shp
    shp.ShardingStrategy = type("X", (), {})
    shp.ShardingType = type("X", (), {})
    shp.get_default_sharding_dim = lambda n: 0
    sys.modules.update(
        {
            "awex": awex,
            "awex.converter": ac,
            "awex.converter.mcore_converter": am,
            "awex.converter.sglang_converter": sg,
            "awex.converter.weights_converter": wc,
            "awex.sharding": ashp,
            "awex.sharding.param_sharding": shp,
        }
    )


def _load_converter():
    from awex.models.ascend.qwen3_5 import _split_mcore_gated_attn_qkv as _fn

    return _fn


_ensure_awex_importable()
_split_mcore_gated_attn_qkv = _load_converter()


def _make_hf_qkv(num_heads: int, num_kv: int, head_dim: int, hidden: int):
    """HF q_proj (per-head [query, gate] -> HF order [u0, c0, u1, c1, ...]),
    plus k_proj / v_proj (one head per kv group)."""
    q_proj = torch.zeros(num_heads * head_dim * 2, hidden)
    for h in range(num_heads):
        q_proj[h * 2 * head_dim : h * 2 * head_dim + head_dim] = 1000 + h
        q_proj[h * 2 * head_dim + head_dim : h * 2 * head_dim + 2 * head_dim] = 2000 + h
    k_proj = torch.zeros(num_kv * head_dim, hidden)
    v_proj = torch.zeros(num_kv * head_dim, hidden)
    for g in range(num_kv):
        k_proj[g * head_dim : (g + 1) * head_dim] = 3000 + g
        v_proj[g * head_dim : (g + 1) * head_dim] = 4000 + g
    return q_proj, k_proj, v_proj


def _make_mcore_weight(q_proj, k_proj, v_proj, num_heads, num_kv, head_dim):
    """Build the mcore GQA-interleaved `linear_qkv` (mirrors Megatron-Bridge
    `merge_qkv_weights` with `attention_output_gate=True`):
    per kv-group = `[Q_unique(hpg), Z/gate(hpg), K(1), V(1)]`."""
    hpg = num_heads // num_kv
    q_r = q_proj.view(num_heads, head_dim * 2, -1)
    q_r, z_r = torch.chunk(q_r, 2, dim=1)  # [num_heads, head_dim, hidden] each
    k_r = k_proj.view(num_kv, head_dim, -1)
    v_r = v_proj.view(num_kv, head_dim, -1)
    parts = []
    for g in range(num_kv):
        parts.extend(
            [
                q_r[g * hpg : (g + 1) * hpg],
                z_r[g * hpg : (g + 1) * hpg],
                k_r[g : g + 1],
                v_r[g : g + 1],
            ]
        )
    return torch.cat(parts, dim=0).reshape(-1, q_proj.shape[-1])


def _kv_heads_for_infer_rank(i: int, infer_tp: int, num_kv: int) -> list[int]:
    """kv-head indices vLLM assigns to infer rank i (replicated when
    infer_tp > num_kv, chunked when infer_tp < num_kv)."""
    if infer_tp >= num_kv:
        replicas = infer_tp // num_kv
        return [i // replicas]
    per = num_kv // infer_tp
    return list(range(i * per, (i + 1) * per))


def _vllm_expected(q_proj, k_proj, v_proj, num_heads, num_kv, head_dim, infer_tp):
    """Fused qkv_proj layout vLLM loads: per rank `[Q_shard, K*, V*]` (all K
    heads then all V heads), Q in HF order. Derived from vLLM semantics only."""
    q_total = 2 * num_heads  # query + gate
    q_per = q_total // infer_tp
    parts = []
    for i in range(infer_tp):
        parts.append(q_proj[i * q_per * head_dim : (i + 1) * q_per * head_dim])
        heads = _kv_heads_for_infer_rank(i, infer_tp, num_kv)
        for h in heads:  # all K heads for this rank
            parts.append(k_proj[h * head_dim : (h + 1) * head_dim])
        for h in heads:  # all V heads for this rank
            parts.append(v_proj[h * head_dim : (h + 1) * head_dim])
    return torch.cat(parts, dim=0)


def _cfg(num_heads, num_kv, head_dim, hidden):
    return SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        attn_output_gate=True,
    )


def _run_converter(mcore_w, cfg, train_tp, infer_tp, monkeypatch):
    """Run the converter for every train rank (mocking the TP all-gather) and
    concatenate per-rank outputs into the full infer-layout."""

    def _fake_get_full_tensor(weight, dim=0):  # noqa: ARG001
        return mcore_w

    monkeypatch.setattr(
        "awex.converter.mcore_converter.get_full_tensor", _fake_get_full_tensor
    )
    per = mcore_w.shape[0] // train_tp if train_tp > 1 else mcore_w.shape[0]
    outs = []
    for r in range(train_tp):
        shard = mcore_w[r * per : (r + 1) * per] if train_tp > 1 else mcore_w
        outs.append(
            _split_mcore_gated_attn_qkv(
                shard,
                cfg,
                infer_tp,
                train_tp_rank=(r if train_tp > 1 else None),
                train_tp_size=train_tp,
            )[0][1]
        )
    return torch.cat(outs, dim=0)


# Two head structures: hpg=2 (small) and hpg=6 (real Qwen3.5: 24 q / 4 kv).

CONFIGS = [
    pytest.param(8, 4, 4, 4, id="hpg2"),
    pytest.param(24, 4, 4, 4, id="hpg6"),
]

CASES = [
    pytest.param(2, 2, id="train2_infer2_bug"),
    pytest.param(2, 4, id="train2_infer4_prod"),
    pytest.param(4, 4, id="train4_infer4_nocp"),
    pytest.param(1, 4, id="train1_infer4"),
    pytest.param(2, 8, id="train2_infer8_repl"),
    pytest.param(4, 2, id="train4_infer2"),
    pytest.param(1, 2, id="train1_infer2"),
    pytest.param(1, 1, id="train1_infer1"),
    pytest.param(8, 4, id="train8_infer4_tp_gt_kv"),  # train_tp>num_kv (contiguous)
    pytest.param(8, 8, id="train8_infer8_tp_gt_kv"),
    pytest.param(8, 2, id="train8_infer2_tp_gt_kv"),
]


@pytest.mark.parametrize("train_tp,infer_tp", CASES)
@pytest.mark.parametrize("num_heads,num_kv,head_dim,hidden", CONFIGS)
def test_split_qkv_matches_vllm_layout_for_tp_combo(
    num_heads, num_kv, head_dim, hidden, train_tp, infer_tp, monkeypatch
):
    """Converter output equals vLLM's fused qkv_proj layout for any TP combo."""
    # Arrange
    qp, kp, vp = _make_hf_qkv(num_heads, num_kv, head_dim, hidden)
    mcore_w = _make_mcore_weight(qp, kp, vp, num_heads, num_kv, head_dim)
    expected = _vllm_expected(qp, kp, vp, num_heads, num_kv, head_dim, infer_tp)
    cfg = _cfg(num_heads, num_kv, head_dim, hidden)

    got = _run_converter(mcore_w, cfg, train_tp, infer_tp, monkeypatch)

    # Assert
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_heads,num_kv,head_dim,hidden", CONFIGS)
def test_split_qkv_all_k_then_all_v_when_infer_tp_below_num_kv(
    num_heads, num_kv, head_dim, hidden, monkeypatch
):
    """Regression: with infer_tp=2 < num_kv=4, each rank's qkv_proj is
    `[Q, K*, V*]` -- the K block holds only k-head rows (3xxx) and the V
    block only v-head rows (4xxx), NOT interleaved per kv-head."""
    # Arrange
    train_tp, infer_tp = 2, 2
    qp, kp, vp = _make_hf_qkv(num_heads, num_kv, head_dim, hidden)
    mcore_w = _make_mcore_weight(qp, kp, vp, num_heads, num_kv, head_dim)
    cfg = _cfg(num_heads, num_kv, head_dim, hidden)

    # Act
    got = _run_converter(mcore_w, cfg, train_tp, infer_tp, monkeypatch)

    # Assert: split each rank's qkv into [Q, K*, V*] and check id ranges.
    q_per = (2 * num_heads) // infer_tp
    kv_per = num_kv // infer_tp  # >1 here (infer_tp < num_kv)
    per_rank = (q_per + 2 * kv_per) * head_dim
    for r in range(train_tp):
        block = got[r * per_rank : (r + 1) * per_rank]
        q_ids = {int(x) for x in block[: q_per * head_dim][:, 0].tolist()}
        k_ids = {
            int(x)
            for x in block[q_per * head_dim : (q_per + kv_per) * head_dim][
                :, 0
            ].tolist()
        }
        v_ids = {int(x) for x in block[(q_per + kv_per) * head_dim :][:, 0].tolist()}
        assert all(1000 <= i < 3000 for i in q_ids), f"rank {r} Q block leaked: {q_ids}"
        assert all(3000 <= i < 4000 for i in k_ids), f"rank {r} K block leaked: {k_ids}"
        assert all(4000 <= i < 5000 for i in v_ids), f"rank {r} V block leaked: {v_ids}"


def test_split_qkv_raises_on_unexpected_weight_size(monkeypatch):
    """A local shard whose size matches neither the per-rank shard nor the full
    weight is rejected with a clear error (guards against silent corruption)."""
    # Arrange
    num_heads, num_kv, head_dim, hidden = 8, 4, 4, 4
    qp, kp, vp = _make_hf_qkv(num_heads, num_kv, head_dim, hidden)
    mcore_w = _make_mcore_weight(qp, kp, vp, num_heads, num_kv, head_dim)
    cfg = _cfg(num_heads, num_kv, head_dim, hidden)
    wrong = mcore_w[:3]
    with pytest.raises(ValueError, match="QKV weight size mismatch"):
        _split_mcore_gated_attn_qkv(
            wrong, cfg, infer_atten_tp_size=4, train_tp_rank=0, train_tp_size=2
        )
