import torch
import pytest
from src.stage1.cosmos.modules.moe import SwiGLUExpert, MyExperts, MoEFFN
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


def test_swiglu_expert_initialization():
    """Test SwiGLUExpert initialization and inheritance from SwiGLU"""
    expert = SwiGLUExpert(hidden_size=256, ffn_hidden_size=1024, ffn_drop=0.1, fused_type=None)

    # Check that the expert is properly initialized
    assert hasattr(expert, "w1")
    assert hasattr(expert, "w2")
    assert hasattr(expert, "w3")
    assert expert.w1.in_features == 256
    assert expert.w1.out_features == 1024


def test_swiglu_expert_forward():
    """Test SwiGLUExpert forward pass"""
    expert = SwiGLUExpert(hidden_size=256, ffn_hidden_size=1024, ffn_drop=0.0, fused_type=None)

    # Test with a batch of inputs
    x = torch.randn(8, 256)  # [batch, dim]
    output = expert(x)

    # Check output shape
    assert output.shape == (8, 256), f"Expected shape (8, 256), got {output.shape}"

    # Check that output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))


def test_my_experts_initialization():
    """Test MyExperts initialization with multiple experts"""
    config = TransformerConfig(
        num_layers=12,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        num_moe_experts=4,
    )

    experts = MyExperts(num_local_experts=4, config=config, ffn_drop=0.1, fused_type=None)

    # Check that the correct number of experts are created
    assert len(experts.experts) == 4

    # Check that each expert is a SwiGLUExpert
    for expert in experts.experts:
        assert isinstance(expert, SwiGLUExpert)


def test_my_experts_forward():
    """Test MyExperts forward pass with token routing"""
    config = TransformerConfig(
        num_layers=12,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        num_moe_experts=4,
    )

    experts = MyExperts(num_local_experts=4, config=config, ffn_drop=0.0, fused_type=None)

    # Create dummy inputs simulating routed tokens
    # Assume 16 tokens total, distributed across 4 experts
    num_tokens = 16
    hidden_states = torch.randn(num_tokens, 256)  # [num_tokens, hidden]
    tokens_per_expert = torch.tensor([4, 4, 4, 4])  # Equal distribution
    permuted_probs = torch.ones(num_tokens)  # Router weights for each routed token (flattened)

    output_ones, bias = experts(hidden_states, tokens_per_expert, permuted_probs)

    # If we scale probs, output should scale accordingly (since probs are applied on output)
    scaled_probs = torch.full((num_tokens,), 0.5)
    output_scaled, _ = experts(hidden_states, tokens_per_expert, scaled_probs)

    # Check output shape
    assert output_ones.shape == hidden_states.shape, f"Expected shape {hidden_states.shape}, got {output_ones.shape}"

    # Check that bias is None (as per implementation)
    assert bias is None

    assert torch.allclose(output_scaled, output_ones * 0.5, rtol=1e-4, atol=1e-5)


def test_my_experts_forward_uneven_distribution():
    """Test MyExperts forward with uneven token distribution"""
    config = TransformerConfig(
        num_layers=12,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        num_moe_experts=4,
    )

    experts = MyExperts(num_local_experts=4, config=config, ffn_drop=0.0, fused_type=None)

    # Uneven distribution with some experts getting 0 tokens
    num_tokens = 10
    hidden_states = torch.randn(num_tokens, 256)
    tokens_per_expert = torch.tensor([5, 3, 2, 0])  # Last expert gets no tokens
    permuted_probs = torch.rand(num_tokens)

    output, bias = experts(hidden_states, tokens_per_expert, permuted_probs)

    # Check output shape
    assert output.shape == hidden_states.shape

    # Check that the function handles zero-token experts gracefully
    assert not torch.isnan(output).any()


def test_moe_ffn_initialization():
    """Test MoEFFN initialization"""
    # Provide an explicit pg_collection so this unit test does not depend on Megatron global parallel_state.
    pg_collection = ProcessGroupCollection(
        tp=None,
        cp=None,
        tp_cp=None,
        tp_dp_cp=None,
        ep=None,
        expt_tp=None,
        tp_ep=None,
        expt_dp=None,
    )
    moe_ffn = MoEFFN(
        dim=256,
        n_heads=8,
        mlp_ratio=4.0,
        num_layers=12,
        layer_idx=0,
        num_experts=8,
        topk=2,
        moe_token_dispatcher_type="allgather",
        ffn_drop=0.1,
        fused_type=None,
        pg_collection=pg_collection,
        assume_bsh=True,
    )

    # Check that the MoE layer was created
    assert hasattr(moe_ffn, "moe")
    assert hasattr(moe_ffn.moe, "experts")

    # Check that experts are MyExperts instance
    assert isinstance(moe_ffn.moe.experts, MyExperts)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MoEFFN forward requires CUDA because Megatron router moves weights to GPU."
)
def test_moe_ffn_forward_bsh():
    """Test MoEFFN forward pass with [B, S, H] format"""
    pg_collection = ProcessGroupCollection(
        tp=None,
        cp=None,
        tp_cp=None,
        tp_dp_cp=None,
        ep=None,
        expt_tp=None,
        tp_ep=None,
        expt_dp=None,
    )
    moe_ffn = MoEFFN(
        dim=256,
        n_heads=8,
        mlp_ratio=4.0,
        num_layers=12,
        layer_idx=0,
        num_experts=4,
        topk=2,
        moe_token_dispatcher_type="allgather",
        moe_router_load_balancing_type="none",
        ffn_drop=0.0,
        fused_type=None,
        pg_collection=pg_collection,
        assume_bsh=True,
    ).cuda()

    # Input in [B, S, H] format
    x = torch.randn(2, 16, 256, device="cuda")  # [batch, seq_len, hidden]
    output = moe_ffn(x)

    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MoEFFN forward requires CUDA because Megatron router moves weights to GPU."
)
def test_moe_ffn_forward_sbh():
    """Test MoEFFN forward pass with [S, B, H] format"""
    pg_collection = ProcessGroupCollection(
        tp=None,
        cp=None,
        tp_cp=None,
        tp_dp_cp=None,
        ep=None,
        expt_tp=None,
        tp_ep=None,
        expt_dp=None,
    )
    moe_ffn = MoEFFN(
        dim=256,
        n_heads=8,
        mlp_ratio=4.0,
        num_layers=12,
        layer_idx=0,
        num_experts=4,
        topk=2,
        moe_token_dispatcher_type="allgather",
        moe_router_load_balancing_type="none",
        ffn_drop=0.0,
        fused_type=None,
        pg_collection=pg_collection,
        assume_bsh=False,  # Expects [S, B, H]
    ).cuda()

    # Input in [S, B, H] format
    x = torch.randn(16, 2, 256, device="cuda")  # [seq_len, batch, hidden]
    output = moe_ffn(x)

    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"


if __name__ == "__main__":
    # Run tests
    test_swiglu_expert_initialization()
    test_swiglu_expert_forward()
    test_my_experts_initialization()
    test_my_experts_forward()
    test_my_experts_forward_uneven_distribution()

    print("\n✓ All unit tests passed!")
    print("\nNote: MoEFFN forward tests are skipped when CUDA is unavailable.")
