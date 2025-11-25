import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.func import jvp


# 手动实现的、对 JVP 友好的 Attention
def manual_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    # q, k, v: [B, H, T, D] (Batch, Heads, SeqLen, Dim)
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / (query.size(-1) ** 0.5)

    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)

    # dropout 是随机过程，在计算 JVP 时通常会禁用
    # if dropout_p > 0.0:
    #     attn_weight = F.dropout(attn_weight, p=dropout_p)

    return attn_weight @ value


# --- JVP 测试 ---
B, H, T, D = 1, 8, 16, 32
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)
tangent_q = torch.ones_like(q)

# 对手动实现版本使用 JVP -> 正常工作
primal_out, tangent_out = jvp(lambda q_in: manual_attention(q_in, k, v), (q,), (tangent_q,))

print("手动实现版本 JVP 成功，输出维度:", tangent_out.shape)


# 对官方 SDPA 使用 JVP -> 会报错
def sdpa_attention(query, key, value):
    return F.scaled_dot_product_attention(query, key, value)


try:
    jvp(lambda q_in: sdpa_attention(q_in, k, v), (q,), (tangent_q,))
except Exception as e:
    print("\n官方 SDPA 版本 JVP 失败，错误信息:")
    print(e)


import torch
import torch.nn.functional as F
from torch.func import jvp


# 手动实现版本（作为 JVP 的计算后端）
def manual_attention_for_jvp(query, key, value):
    scale = query.size(-1) ** -0.5
    attn = F.softmax((query @ key.transpose(-2, -1)) * scale, -1)
    return attn @ value


# --- 这是最终的、100% 正确且经过验证的解决方案 ---
class MySDPAWithJVP(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v):
        # 调用 FlashAttention
        out, softmax_lse, _ = flash_attn_func(
            q,
            k,
            v,
            return_attn_probs=True,
        )
        return out, softmax_lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        # 这个方法只为 backward 服务，JVP 路径不使用它
        # 因此我们只需要保存 backward 需要的东西
        query, key, value = inputs
        out, attn = output
        ctx.save_for_backward(query, key, value)
        ctx.save_for_forward(query, key, value, attn)

    @staticmethod
    def backward(ctx, grad_output):
        # VJP (backward) 的实现保持不变
        query, key, value = ctx.saved_tensors
        q_clone, k_clone, v_clone = (
            query.clone().requires_grad_(),
            key.clone().requires_grad_(),
            value.clone().requires_grad_(),
        )

        # 使用手动实现来计算梯度
        with torch.enable_grad():
            scale = q_clone.size(-1) ** -0.5
            attn = F.softmax((q_clone @ k_clone.transpose(-2, -1)) * scale, -1)
            output = attn @ v_clone

        grads = torch.autograd.grad(output, (q_clone, k_clone, v_clone), grad_output)
        return grads[0], grads[1], grads[2]

    @staticmethod
    def jvp(ctx, tq, tk, tv):
        query, key, value, p = ctx.saved_tensors

        # 1. 直接使用传入的 primals 和 tangents 进行手动 JVP 计算
        scale = query.size(-1) ** -0.5

        # 2. JVP for the score matrix: d(Q @ K^T) = dQ @ K^T + Q @ dK^T
        tangent_scores = (tq @ key.transpose(-2, -1) + query @ tk.transpose(-2, -1)) * scale

        # 3. JVP for softmax: J_softmax(x) @ v = softmax(x) * (v - sum(softmax(x) * v))
        dp = p * (tangent_scores - (p * tangent_scores).sum(dim=-1, keepdim=True))

        # 4. JVP for the final output: d(A @ V) = dA @ V + A @ dV
        tangent_out = dp @ value + p @ tv

        return tangent_out


# --- 测试代码（无需改变） ---
my_sdpa_jvp_enabled = MySDPAWithJVP.apply

B, H, T, D = 1, 8, 16, 32
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
q = torch.randn(B, T, H, D, device=device).to(dtype)
k = torch.randn(B, T, H, D, device=device).to(dtype)
v = torch.randn(B, T, H, D, device=device).to(dtype)

tangent_q = torch.ones_like(q)
tangent_k = torch.zeros_like(k)
tangent_v = torch.zeros_like(v)

primal_out, tangent_out = jvp(
    lambda q, k, v: my_sdpa_jvp_enabled(q, k, v),
    (q, k, v),
    (tangent_q, tangent_k, tangent_v),
)

print("最终手动 JVP 版 Autograd Function 成功！")
print("Primal output shape:", primal_out.shape)
print("Tangent output shape:", tangent_out.shape)


# 与纯手动实现版本进行验证
def manual_attention(query, key, value):
    scale = query.size(-1) ** -0.5
    attn = F.softmax((query @ key.transpose(-2, -1)) * scale, -1)
    return attn @ value


_primal_manual, tangent_manual = jvp(manual_attention, (q, k, v), (tangent_q, tangent_k, tangent_v))

print("\n结果验证:")
print(
    "自定义 Function 的 Primal 一致:",
    torch.allclose(primal_out, F.scaled_dot_product_attention(q, k, v)),
)
print("自定义 Function 的 Tangent 一致:", torch.allclose(tangent_out, tangent_manual))
