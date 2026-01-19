import torch
import pytest
from src.stage1.cosmos.modules.transformer import AttentionBlock
from megatron.core.process_groups_config import ProcessGroupCollection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoEFFN forward requires CUDA")
def test_attention_block_moe_instantiation():
    """Test AttentionBlock instantiation with MoEFFN"""

    # For single-process unit tests, do not initialize torch.distributed.
    # Megatron uses process groups only when TP/EP sizes are > 1.
    pg_collection = ProcessGroupCollection(
        tp=None,
        ep=None,
        cp=None,
        tp_cp=None,
        tp_dp_cp=None,
        expt_tp=None,
        tp_ep=None,
        expt_dp=None,
    )

    moe_kwargs = {
        "num_experts": 4,
        "topk": 1,
        "pg_collection": pg_collection,
        "moe_token_dispatcher_type": "allgather",  # For local test
        "moe_router_load_balancing_type": "none",
        "moe_router_pre_softmax": True,
    }

    block = AttentionBlock(
        dim=256,
        n_heads=8,
        ffn_type="moe",
        moe_kwargs=moe_kwargs,
        norm_layer="layernorm",
    ).cuda()

    # Check if FFN is MoEFFN
    from src.stage1.cosmos.modules.moe import MoEFFN

    assert isinstance(block.ffn, MoEFFN)

    # Basic forward pass
    x = torch.randn(2, 64, 256, device="cuda")  # [B, S, H]
    out = block(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_attention_block_moe_instantiation()
