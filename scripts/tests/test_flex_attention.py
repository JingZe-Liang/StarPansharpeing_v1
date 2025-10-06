import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

q_heads = 8
txt_len = 32
img_len = 128
txt_tokens = torch.randn(2, 8, txt_len, 64).cuda()
img_tokens = torch.randn(2, 8, img_len, 64).cuda()

x = torch.cat([txt_tokens, img_tokens], dim=-2)  # (2, 8, 520, 64)

# img attend to txt and itself, but txt can only attend to itself and is causal

TXT_LEN = 32


def mask_mod(b, h, q_idx, kv_idx):
    is_txt_causal = (q_idx >= kv_idx) & (q_idx < TXT_LEN)
    is_img_full = q_idx >= TXT_LEN
    return is_txt_causal | is_img_full


# mask_mod = get_mask_mod(txt_len, img_len)
l = x.shape[-2]
mask = create_block_mask(mask_mod, None, None, l, l, BLOCK_SIZE=1)
# mask = create_block_mask(sliding_window_causal, None, None, 64, 64, BLOCK_SIZE=1)

# bh_mask = mask[0, 0]

# mask = mask.to_dense()
# print(mask.to_string((64, 64), limit=10))

o = flex_attention(x, x, x, block_mask=mask)
print(o.shape)
