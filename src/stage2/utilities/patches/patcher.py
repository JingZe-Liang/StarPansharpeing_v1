from typing import Any, Callable

import torch
import torch.nn as nn
from kornia.contrib import combine_tensor_patches, extract_tensor_patches
from tqdm import tqdm

from src.utilities.logging import log


class ModelForwardPatcher(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        patch_size: dict[str, int],
        stride: dict[str, int],
        out_patch_size: int | None = None,
        out_stride: int | None = None,
        model_out_processer: Callable[[Any], torch.Tensor] = lambda x: x,
    ) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.out_patch_size = out_patch_size or patch_size[list(patch_size.keys())[0]]
        self.out_stride = out_stride or stride[list(stride.keys())[0]]
        self._patching_input_names = list(patch_size.keys())
        self.model_out_processer = model_out_processer
        assert self._patching_input_names == list(stride.keys()), (
            "patch_size and stride should have the same keys"
        )
        log(
            f"[ModelForwardPatcher] Patching model {model.__class__.__name__} with patch_size={patch_size} and stride={stride}"
        )

    def forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        original_size: tuple | None = None,
    ):
        # 1. extract patches for inputs that need patching
        patched_inputs = {}
        for k, v in model_inputs.items():
            if k in self._patching_input_names:
                patched_inputs[k] = extract_tensor_patches(
                    v,
                    window_size=self.patch_size[k],
                    stride=self.stride[k],
                )

        # 2. forward the model with patched inputs
        _n_patches = patched_inputs[self._patching_input_names[0]].shape[
            1
        ]  # [B, N, C, H, W]
        model_outs = []
        for i in tqdm(range(_n_patches), desc="Patching model forward ..."):
            model_in = {}
            for k, v in patched_inputs.items():
                model_in[k] = v[:, i]  # [B, C, H, W]
            patched_out = self.model(**model_in)  # dict of [B, C, H, W]
            patched_out = self.model_out_processer(patched_out)
            assert isinstance(patched_out, torch.Tensor), (
                "model output should be a tensor"
            )
            model_outs.append(patched_out)

        # 3. combine patches for outputs that need patching
        if original_size is None:
            original_size = tuple(
                model_inputs[self._patching_input_names[0]].shape[-2:]
            )

        # 4. combine patches
        model_outs = torch.stack(model_outs, dim=1)  # [B, N, C, H, W]
        combined = combine_tensor_patches(
            model_outs,
            original_size=original_size,
            window_size=self.out_patch_size,
            stride=self.out_stride,
        )
        return combined


def test_patching_model_forward():
    inputs = {
        "x": torch.randn(2, 3, 64, 64),
        "y": torch.randn(2, 3, 256, 256),
        "z": torch.randn(2, 1, 256, 256),
    }
    dummy_model = lambda x, y, z: y
    model = ModelForwardPatcher(
        model=dummy_model,
        patch_size={"x": 16, "y": 64, "z": 64},
        stride={"x": 8, "y": 32, "z": 32},
        out_patch_size=64,
        out_stride=32,
    )
    out = model(inputs, original_size=(256, 256))
    assert out.shape == (2, 3, 256, 256)
    assert torch.isclose(out, inputs["y"]).all()


if __name__ == "__main__":
    test_patching_model_forward()
