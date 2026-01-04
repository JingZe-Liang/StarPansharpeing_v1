from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


def _load_mhc_py_module() -> Any:
    mhc_py = Path(__file__).resolve().parents[1] / "mHC.py"
    spec = importlib.util.spec_from_file_location("stage2_layers_mhc", mhc_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{mhc_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim 必须能被 num_heads 整除，但得到 dim={dim}, num_heads={num_heads}")

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_expansion: int,
        layer_idx: int,
        init_hc: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim=dim, num_heads=num_heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, expansion=mlp_expansion)

        # 注意：此处的 `init_hc` 就是 `get_init_and_expand_reduce_stream_functions_cuda(...)` 返回的 init_fn
        self.hc_attn = init_hc(
            dim=dim,
            branch=nn.Sequential(self.ln_1, self.attn),
            layer_index=layer_idx * 2,
            sinkhorn_iters=5,
        )
        self.hc_mlp = init_hc(
            dim=dim,
            branch=nn.Sequential(self.ln_2, self.mlp),
            layer_index=layer_idx * 2 + 1,
            sinkhorn_iters=5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hc_attn(x)
        x = self.hc_mlp(x)
        return x


class TinyGPTLikeModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        init_hc: Callable[..., nn.Module],
        expand_stream: nn.Module,
        reduce_stream: nn.Module,
        mlp_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        self.h = nn.ModuleList(
            [
                Block(
                    dim=n_embd,
                    num_heads=n_head,
                    mlp_expansion=mlp_expansion,
                    layer_idx=i,
                    init_hc=init_hc,
                )
                for i in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch, seq = idx.shape
        if seq > self.block_size:
            raise ValueError(f"seq_len 超过 block_size：{seq} > {self.block_size}")

        pos = torch.arange(seq, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)

        # 全程多流：从这里开始 x 变成 [(B*s), T, C]
        x = self.expand_stream(x)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        x = self.reduce_stream(x)

        return self.lm_head(x)


def main() -> int:
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() 为 False：当前进程拿不到 CUDA，无法运行该示例。")
        return 0

    try:
        from mhc import ops as _  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"无法导入 `mhc` CUDA 扩展：{exc!r}")
        print("请先在 `src/stage2/layers/mHC.cu` 下执行：`make install` 或 `pip install -e .`")
        return 1

    mhc_mod = _load_mhc_py_module()

    device = torch.device("cuda")
    dtype = torch.float16

    streams = 4
    vocab_size = 256
    block_size = 64
    batch = 2
    n_layer = 2
    n_head = 4
    n_embd = 64

    init_hc, expand_stream, reduce_stream = mhc_mod.get_init_and_expand_reduce_stream_functions_cuda(streams)

    model = TinyGPTLikeModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        init_hc=init_hc,
        expand_stream=expand_stream,
        reduce_stream=reduce_stream,
    ).to(device=device, dtype=dtype)

    idx = torch.randint(0, vocab_size, (batch, block_size), device=device, dtype=torch.long)
    logits = model(idx)
    loss = logits.float().mean()
    loss.backward()

    print("TinyGPTLikeModel + HyperConnectionsCUDA 跑通")
    print(f"logits: {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
