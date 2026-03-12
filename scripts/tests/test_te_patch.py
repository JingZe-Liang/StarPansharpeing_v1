import os

import pytest
import torch
import torch.nn as nn
from timm.layers.attention import Attention as TimmAttention

te = pytest.importorskip("transformer_engine.pytorch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for TransformerEngine tests.", allow_module_level=True)

os.environ["MODEL_COMPILED"] = "0"

from src.stage1.cosmos.modules.naflex import GatedAttentionTimmWrapped
from src.stage1.cosmos.modules.te_patch import apply_te_patches, patch_te_linears
from src.stage1.cosmos.modules.transformer import Attention, GatedAttention
from src.stage1.cosmos.modules.variants.mlp import SwiGLU

DEVICE = "cuda"
DTYPE = torch.float16


def _make_rope(seq_len: int, head_dim: int, device: str = DEVICE, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.randn(1, 1, seq_len, head_dim * 2, device=device, dtype=dtype)


def test_attention_forward_with_and_without_te_dpa() -> None:
    x = torch.randn(2, 8, 32, device=DEVICE, dtype=DTYPE)

    native_attn = Attention(dim=32, num_heads=4, use_te_dpa=False).to(device=DEVICE, dtype=DTYPE)
    te_attn = Attention(dim=32, num_heads=4, use_te_dpa=True).to(device=DEVICE, dtype=DTYPE)

    native_out = native_attn(x)
    te_out = te_attn(x)

    assert native_out.shape == x.shape
    assert te_out.shape == x.shape
    assert te_attn.te_dpa is not None


def test_gated_attention_forward_with_te_dpa_and_rope() -> None:
    x = torch.randn(2, 8, 32, device=DEVICE, dtype=DTYPE)
    rope = _make_rope(seq_len=8, head_dim=8)
    attn = GatedAttention(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        headwise_attn_output_gate=True,
        use_te_dpa=True,
        fused_rope=False,
    ).to(device=DEVICE, dtype=DTYPE)

    output = attn(x, rope=rope)

    assert output.shape == x.shape
    assert attn.te_dpa is not None


def test_gated_attention_timm_wrapper_uses_te_path() -> None:
    x = torch.randn(2, 8, 32, device=DEVICE, dtype=DTYPE)
    rope = _make_rope(seq_len=7, head_dim=8)
    wrapped = GatedAttentionTimmWrapped(
        dim=32,
        num_heads=4,
        num_prefix_tokens=1,
        use_te_dpa=True,
        fused_rope=False,
    ).to(device=DEVICE, dtype=DTYPE)

    output = wrapped(x, rope=rope)

    assert output.shape == x.shape
    assert wrapped.te_dpa is not None


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention(dim=32, num_heads=4, use_te_dpa=False)
        self.ffn = SwiGLU(
            in_features=32,
            hidden_features=64,
            out_features=32,
            norm_layer=None,
            bias=True,
            drop=0.0,
            is_fused=None,
        )
        self.external = TimmAttention(dim=32, num_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))


def test_patch_te_linears_skips_external_timm_modules() -> None:
    model = _ToyModel().to(device=DEVICE, dtype=DTYPE)

    patch_te_linears(model)

    assert isinstance(model.attn.qkv, te.Linear)
    assert isinstance(model.attn.proj, te.Linear)
    assert isinstance(model.ffn.w1, te.Linear)
    assert isinstance(model.ffn.w2, te.Linear)
    assert isinstance(model.ffn.w3, te.Linear)
    assert isinstance(model.external.qkv, nn.Linear)
    assert isinstance(model.external.proj, nn.Linear)


def test_apply_te_patches_preserves_forward() -> None:
    model = _ToyModel().to(device=DEVICE, dtype=DTYPE)
    x = torch.randn(2, 8, 32, device=DEVICE, dtype=DTYPE)

    apply_te_patches(model)
    output = model(x)

    assert output.shape == x.shape
    assert model.attn.te_dpa is not None
    assert isinstance(model.attn.qkv, te.Linear)


def test_set_te_parallel_groups_handles_none_and_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    attn = Attention(dim=32, num_heads=4, use_te_dpa=True).to(device=DEVICE, dtype=DTYPE)
    assert attn.te_dpa is not None

    calls: dict[str, object] = {}

    monkeypatch.setattr(attn.te_dpa, "set_tensor_parallel_group", lambda group: calls.setdefault("tp", group))
    monkeypatch.setattr(
        attn.te_dpa,
        "set_context_parallel_group",
        lambda group, ranks, stream, comm: calls.setdefault("cp", (group, ranks, stream, comm)),
    )

    attn.set_te_parallel_groups(tp_group=None, cp_group=None)
    assert "tp" in calls
    assert calls["tp"] is None

    calls.clear()
    mock_tp_group = object()
    mock_cp_group = object()
    attn.set_te_parallel_groups(
        tp_group=mock_tp_group,
        cp_group=mock_cp_group,
        cp_global_ranks=[0, 1],
        cp_stream=None,
        cp_comm_type="p2p",
    )

    assert calls["tp"] is mock_tp_group
    cp_group, cp_ranks, cp_stream, cp_comm = calls["cp"]
    assert cp_group is mock_cp_group
    assert cp_ranks == [0, 1]
    assert cp_stream is not None
    assert cp_comm == "p2p"
