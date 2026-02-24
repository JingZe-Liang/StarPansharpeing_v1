from types import MethodType

import torch
from torch import nn

from scripts.trainer.hyper_latent_classification_trainer import HyperClassificationTrainer
from src.stage2.classification.data.TreeSatAI_timeseries import (
    TREE_SAT_GENUS_CLASSES,
    TreeSatAITimeSeriesDataset,
    _replace_nans_with_spatial_mean,
    _select_list_by_indices,
    _select_temporal_indices,
    _select_tensor_by_indices,
    treesatai_timeseries_collate_fn,
)
from src.stage2.classification.model.cosmos_tokenizer_classifier_ts import CosmosTokenizerClassifierTS


def test_label_threshold_multihot_uses_area_threshold() -> None:
    dataset = TreeSatAITimeSeriesDataset.__new__(TreeSatAITimeSeriesDataset)
    dataset.classes = TREE_SAT_GENUS_CLASSES
    dataset.labels = {
        "sample.tif": [
            ["Abies", 0.0],
            ["Acer", 0.05],
            ["Alnus", 0.07],
            ["Betula", 0.2],
        ]
    }
    dataset.label_mode = "multi_hot"
    dataset.label_area_threshold = 0.07

    output = dataset._build_label_tensors("sample.tif")
    label = output["label"]
    assert int(label[0].item()) == 0
    assert int(label[1].item()) == 0
    assert int(label[2].item()) == 0
    assert int(label[3].item()) == 1


def test_temporal_sampling_random_then_max_t_syncs_modalities() -> None:
    torch.manual_seed(2026)
    length = 80
    indices = _select_temporal_indices(
        length=length,
        sample_cap=50,
        max_t=7,
        mode="official_random_cap_then_max_t",
    )
    assert int(indices.numel()) == 7
    assert bool(((indices >= 0) & (indices < length)).all())

    tensor = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    values = [f"p{i}" for i in range(length)]
    selected_tensor = _select_tensor_by_indices(tensor, indices)
    selected_values = _select_list_by_indices(values, indices)
    assert selected_tensor.shape[0] == len(selected_values)
    for i, idx in enumerate(indices.tolist()):
        assert float(selected_tensor[i, 0].item()) == float(idx)
        assert selected_values[i] == f"p{idx}"

    indices_truncate = _select_temporal_indices(
        length=10,
        sample_cap=50,
        max_t=3,
        mode="truncate_only",
    )
    assert indices_truncate.tolist() == [0, 1, 2]


def test_s1_nan_fill_spatial_mean() -> None:
    tensor = torch.tensor(
        [
            [
                [[float("nan"), 1.0], [3.0, float("nan")]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            [
                [[float("nan"), float("nan")], [float("nan"), float("nan")]],
                [[2.0, 4.0], [6.0, 8.0]],
            ],
        ],
        dtype=torch.float32,
    )
    filled = _replace_nans_with_spatial_mean(tensor)
    assert not torch.isnan(filled).any()
    assert torch.isclose(filled[0, 0, 0, 0], torch.tensor(2.0))
    assert torch.isclose(filled[0, 0, 1, 1], torch.tensor(2.0))
    assert torch.isclose(filled[1, 0, 0, 0], torch.tensor(0.0))


def test_collate_emit_valid_mask_switch() -> None:
    batch = [
        {
            "label": torch.tensor([1.0, 0.0]),
            "image_s2": torch.ones((2, 2, 2, 2), dtype=torch.float32),
            "doy_s2": torch.tensor([1, 2], dtype=torch.long),
            "sample_id": "a",
        },
        {
            "label": torch.tensor([0.0, 1.0]),
            "image_s2": torch.ones((3, 2, 2, 2), dtype=torch.float32),
            "doy_s2": torch.tensor([3, 4, 5], dtype=torch.long),
            "sample_id": "b",
        },
    ]
    collated_no_valid = treesatai_timeseries_collate_fn(batch, emit_valid_mask=False)
    assert "image_s2_valid" not in collated_no_valid
    assert "doy_s2_valid" not in collated_no_valid
    assert collated_no_valid["image_s2"].shape == (2, 3, 2, 2, 2)
    assert collated_no_valid["doy_s2"].shape == (2, 3)

    collated_valid = treesatai_timeseries_collate_fn(batch, emit_valid_mask=True)
    assert "image_s2_valid" in collated_valid
    assert "doy_s2_valid" in collated_valid
    assert collated_valid["image_s2_valid"].shape == (2, 3)
    assert collated_valid["doy_s2_valid"].shape == (2, 3)


def test_batch_compat_for_trainer_parse_and_model_forward_without_valid_mask() -> None:
    trainer = HyperClassificationTrainer.__new__(HyperClassificationTrainer)
    trainer.label_key = "label"
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.metric_task = "multilabel"

    batch = {
        "label": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "image_aerial": torch.randn((2, 4, 8, 8), dtype=torch.float32),
        "image_s2": torch.randn((2, 3, 10, 8, 8), dtype=torch.float32),
        "doy_s2": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
    }
    image, label = HyperClassificationTrainer._parse_batch(trainer, batch)
    assert isinstance(image, dict)
    assert label.dtype == torch.float32

    model = CosmosTokenizerClassifierTS.__new__(CosmosTokenizerClassifierTS)
    nn.Module.__init__(model)
    model.modality_order = ("aerial", "s2")
    model.modality_fusion = "mean"
    model.frame_feature_dim = 4
    model.classifier = nn.Linear(4, 3)

    def _fake_encode_modality(
        self: CosmosTokenizerClassifierTS, x: torch.Tensor, valid: object, modality: str
    ) -> torch.Tensor:
        del valid
        seq = self._prepare_temporal_tensor(x)
        b = int(seq.shape[0])
        base = 1.0 if modality == "aerial" else 2.0
        return torch.full((b, self.frame_feature_dim), base, dtype=torch.float32, device=seq.device)

    model._encode_modality = MethodType(_fake_encode_modality, model)
    output = CosmosTokenizerClassifierTS.forward(model, image)
    assert "logits" in output
    assert output["logits"].shape == (2, 3)
