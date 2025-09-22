import timm
import torch
from timm.layers.pos_embed_sincos import apply_rot_embed, apply_rot_embed_cat
from timm.models.vision_transformer import VisionTransformer
from timm.models.vitamin import GeGluMlp


def test_create():
    model = timm.create_model(
        "vitamin_large_224",
        patch_size=1,
        fc_norm=False,
        drop_rate=0.0,
        num_classes=0,
        global_pool="",
        pos_embed="none",
        class_token=False,
        mlp_layer=GeGluMlp,
        reg_tokens=512,
        img_size=256,
        drop_path_rate=0.1,
    )

    print(model)


def test_create_vision_transformer():
    model = VisionTransformer(
        num_heads=8,
        patch_size=16,
        embed_dim=512,
        global_pool="",
        reg_tokens=8,
        pos_embed="learn",
        class_token=False,
        num_classes=0,
    ).cuda()
    # print(model)
    x = torch.randn(1, 3, 224, 224).cuda()
    print(model(x).shape)


def test_rope_mixed():
    from timm.layers import RotaryEmbeddingMixed

    rope_mixed = RotaryEmbeddingMixed(
        dim=512,
        depth=8,
        num_heads=8,
        feat_shape=None,
    )
    l = 14 * 14
    dim_head = 512 // 8
    head = 8
    depthed_rope_embd = rope_mixed.get_embed(shape=[14, 14])
    print("mixed rope embedding shaped as ", depthed_rope_embd.shape)
    # Apply rope
    l0_rope = depthed_rope_embd[:, 0]
    q = torch.randn(1, head, l, dim_head)
    q = apply_rot_embed_cat(q, l0_rope)
    print(q.shape)


def test_create_naflex_vit():
    from timm.models.naflexvit import NaFlexVit, NaFlexVitCfg

    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=384,
        num_heads=8,
        depth=8,
        qk_norm=True,
        reg_tokens=8,
        pos_embed_grid_size=(224 // 16, 224 // 16),
        rope_type="axial",
        global_pool="",  # return the full sequence
    )
    model = NaFlexVit(
        cfg=cfg,
        in_chans=3,
        num_classes=0,
        img_size=224,
    ).cuda()
    print(model)

    img_size = 256
    x = torch.randn(1, 3, img_size, img_size).cuda()
    out = model(x)
    print(out.shape)  # [1, num_tokens, embed_dim]


if __name__ == "__main__":
    # test_create()
    # test_create_vision_transformer()
    # test_rope_mixed()
    test_create_naflex_vit()
