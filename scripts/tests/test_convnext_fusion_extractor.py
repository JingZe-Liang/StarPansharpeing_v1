import torch

from src.stage2.layers.dinov3_adapter import ConvnextExtractor


def test_convnext_extractor_fuses_feat_into_query() -> None:
    torch.manual_seed(0)

    b, c = 2, 64
    htoks, wtoks = 8, 8
    h_c, w_c = 8, 8

    # query packs (2H,2W), (H,W), (H//2,W//2)
    l2 = (h_c * 2) * (w_c * 2)
    l3 = h_c * w_c
    l4 = (h_c // 2) * (w_c // 2)
    query_len = l2 + l3 + l4

    query = torch.randn(b, query_len, c, requires_grad=True)
    feat = torch.randn(b, htoks * wtoks, c, requires_grad=True)
    spatial_shapes = torch.tensor([[htoks, wtoks]], dtype=torch.long)

    extractor = ConvnextExtractor(dim=c, with_cffn=False)
    out = extractor(
        query=query,
        reference_points=None,
        feat=feat,
        spatial_shapes=spatial_shapes,
        level_start_index=None,
        H=h_c,
        W=w_c,
    )

    assert out.shape == query.shape

    out.sum().backward()
    assert query.grad is not None
    assert feat.grad is not None
