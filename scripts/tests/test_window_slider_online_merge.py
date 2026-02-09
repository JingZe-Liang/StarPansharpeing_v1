import torch

from src.data.window_slider import model_predict_patcher


def test_model_predict_patcher_online_merge_average_matches_offline() -> None:
    torch.manual_seed(0)
    img = torch.randn(1, 3, 150, 127, dtype=torch.float32)
    gt = torch.zeros(1, 150, 127, dtype=torch.long)

    def _model(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_logits = batch["img"] * 1.5 + 0.25
        return {"pred_logits": pred_logits}

    outputs_offline = model_predict_patcher(
        _model,
        {"img": img, "gt": gt},
        patch_keys=["img", "gt"],
        merge_keys=["pred_logits"],
        patch_size=64,
        stride=32,
        merge_method="average",
        online_merge=False,
    )
    outputs_online = model_predict_patcher(
        _model,
        {"img": img, "gt": gt},
        patch_keys=["img", "gt"],
        merge_keys=["pred_logits"],
        patch_size=64,
        stride=32,
        merge_method="average",
        online_merge=True,
    )

    assert torch.allclose(outputs_online["pred_logits"], outputs_offline["pred_logits"], atol=1e-5, rtol=1e-5)


def test_model_predict_patcher_online_merge_last_matches_offline() -> None:
    torch.manual_seed(1)
    img = torch.randn(1, 3, 133, 141, dtype=torch.float32)
    gt = torch.zeros(1, 133, 141, dtype=torch.long)

    def _model(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_logits = batch["img"] - 0.75
        return {"pred_logits": pred_logits}

    outputs_offline = model_predict_patcher(
        _model,
        {"img": img, "gt": gt},
        patch_keys=["img", "gt"],
        merge_keys=["pred_logits"],
        patch_size=64,
        stride=40,
        merge_method="last",
        online_merge=False,
    )
    outputs_online = model_predict_patcher(
        _model,
        {"img": img, "gt": gt},
        patch_keys=["img", "gt"],
        merge_keys=["pred_logits"],
        patch_size=64,
        stride=40,
        merge_method="last",
        online_merge=True,
    )

    assert torch.equal(outputs_online["pred_logits"], outputs_offline["pred_logits"])
