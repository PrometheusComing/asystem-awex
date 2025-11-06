from awex.sharding.rank_info import RankInfo


def get_sglang_sharding_strategy(
    model_name: str, server_args, rank_info: RankInfo, **kwargs
):
    """
    Get the sharding strategy class for a given model architecture name.
    """
    from awex.models import get_sharding_strategy
    cls = get_sharding_strategy(model_name)
    return cls(
        engine_type="sglang",
        enable_dp_attention=server_args.enable_dp_attention,
        enable_dp_lm_head=server_args.enable_dp_lm_head,
        moe_dense_tp_size=server_args.moe_dense_tp_size,
        tp_size=rank_info.tp_size,
        ep_size=server_args.ep_size,
        ep_tp_size=1,
        rank_info=rank_info,
        **kwargs,
    )



def get_sglang_rank_info(model_context, engine_rank) -> RankInfo:
    scheduler = model_context["scheduler"]
    server_args = scheduler.server_args
    if server_args.dp_size != 1 and not server_args.enable_dp_attention:
        raise ValueError(
            f"DP size is not 1, but {server_args.dp_size}. This is not supported yet."
        )
    tp_size = model_context["tp_size"]
    tp_rank = model_context["tp_rank"]
    ep_size = server_args.ep_size
    if (
        server_args.enable_ep_moe
        or server_args.enable_deepep_moe
        or (hasattr(server_args, "enable_pplx_moe") and server_args.enable_pplx_moe)
    ):
        assert ep_size == tp_size, "ep_size must be equal to tp_size"
        ep_rank = tp_rank
    else:
        assert ep_size == 1, "ep_size must be 1"
        ep_rank = 0
    ep_tp_size = 1
    ep_tp_rank = 0
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=model_context["pp_rank"],
        pp_size=model_context["pp_size"],
        dp_size=model_context["dp_size"],
        dp_rank=0,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=model_context["attn_tp_rank"],
        attn_tp_size=model_context["attn_tp_size"],
        attn_dp_rank=model_context["attn_dp_rank"],
        world_size=model_context["world_size"],
        global_rank=model_context["global_rank"],
        engine_rank=engine_rank,
        local_rank=model_context["local_rank"],
        is_infer=True,
    )

