import torch
from sageattention import sageattn


def test_sage_attention(attn_type="sageattn"):
    b, nh, l, nd = 1, 8, 256, 64
    q, k, v = (
        torch.randn(b, nh, l, nd),
        torch.randn(b, nh, l, nd),
        torch.randn(b, nh, l, nd),
    )

    torch.cuda.set_device("cuda:2")
    to_fn = lambda x: x.to(torch.bfloat16).to("cuda")
    q, k, v = to_fn(q), to_fn(k), to_fn(v)

    with torch.no_grad():
        att_out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        att_out = sageattn(q, k, v, attn_mask=None, dropout_p=0.0)
        # att_out_fp8 = sageattn_qk_int8_pv_fp8_cuda(
        #     q, k, v, attn_mask=None, dropout_p=0.0
        # )
        print(att_out.shape)

        # diff
        diff = torch.mean((att_out - att_out_ref) ** 2)
        max_diff = torch.max(torch.abs(att_out - att_out_ref))
        print(f"Max diff: {max_diff}")
        print(f"Diff bf16: {diff}")

        # diff_fp8 = torch.mean((att_out_fp8-att_out_ref)**2)
        # print(f'Diff fp8: {diff_fp8}')


if __name__ == "__main__":
    test_sage_attention(attn_type="sageattn")
    print("Sage Attention test passed.")
