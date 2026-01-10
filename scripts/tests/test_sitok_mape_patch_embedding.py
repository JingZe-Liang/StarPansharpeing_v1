import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.modules.patching import SiTokMAPEPatchEmbedding


class TestSiTokMAPEPatchEmbedding(unittest.TestCase):
    def test_forward_shape_and_kernel_resample(self) -> None:
        embed_dim = 4
        patch_size = 3
        module = SiTokMAPEPatchEmbedding(
            img_size=None,
            patch_size=patch_size,
            embed_dim=embed_dim,
            base_patch_sizes=(2, 4),
            strict_img_size=False,
            dynamic_img_pad=False,
            norm_layer=torch.nn.LayerNorm,
        )

        x = torch.randn(2, 5, 12, 12)
        y = module(x)
        self.assertEqual(tuple(y.shape), (2, 5 * (12 // patch_size) * (12 // patch_size), embed_dim))

        w, b = module.get_kernel(patch_size, device=x.device, dtype=x.dtype)
        self.assertEqual(tuple(w.shape), (embed_dim, 1, patch_size, patch_size))
        self.assertIsNotNone(b)


if __name__ == "__main__":
    unittest.main()
