import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, cast

import torch

_LossFn = Callable[..., torch.Tensor]


def _load_multiview_info_nce_loss() -> _LossFn:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "stage1" / "self_supervised" / "info_nce.py"
    spec = spec_from_file_location("_info_nce_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_LossFn, module.multiview_info_nce_loss)


class TestMultiViewInfoNCELoss(unittest.TestCase):
    def test_v2_objectives_match(self):
        multiview_info_nce_loss = _load_multiview_info_nce_loss()
        torch.manual_seed(0)
        embeddings = torch.randn(2, 4, 8, requires_grad=True)

        loss_supcon = multiview_info_nce_loss(embeddings, temperature=0.2, objective="supcon")
        loss_mp = multiview_info_nce_loss(embeddings, temperature=0.2, objective="multi_positive_infonce")

        self.assertTrue(torch.isfinite(loss_supcon).item())
        self.assertTrue(torch.isfinite(loss_mp).item())
        self.assertTrue(torch.allclose(loss_supcon, loss_mp, rtol=1e-6, atol=1e-6))

        (loss_supcon + loss_mp).backward()

    def test_v3_objectives_differ(self):
        multiview_info_nce_loss = _load_multiview_info_nce_loss()
        torch.manual_seed(0)
        embeddings = torch.randn(3, 4, 8)

        loss_supcon = multiview_info_nce_loss(embeddings, temperature=0.2, objective="supcon")
        loss_mp = multiview_info_nce_loss(embeddings, temperature=0.2, objective="multi_positive_infonce")

        self.assertTrue(torch.isfinite(loss_supcon).item())
        self.assertTrue(torch.isfinite(loss_mp).item())
        self.assertGreater((loss_supcon - loss_mp).abs().item(), 1e-6)

    def test_anchor_views_meaningful(self):
        multiview_info_nce_loss = _load_multiview_info_nce_loss()
        torch.manual_seed(0)
        embeddings = torch.randn(3, 4, 8)

        for objective in ("supcon", "multi_positive_infonce"):
            loss_all = multiview_info_nce_loss(embeddings, temperature=0.2, objective=objective)
            loss_v0 = multiview_info_nce_loss(embeddings, temperature=0.2, objective=objective, anchor_views=[0])
            loss_v1 = multiview_info_nce_loss(embeddings, temperature=0.2, objective=objective, anchor_views=[1])
            loss_v2 = multiview_info_nce_loss(embeddings, temperature=0.2, objective=objective, anchor_views=[2])

            self.assertTrue(torch.isfinite(loss_all).item())
            self.assertTrue(torch.isfinite(loss_v0).item())
            self.assertTrue(torch.isfinite(loss_v1).item())
            self.assertTrue(torch.isfinite(loss_v2).item())

            # 因为每个 view 的 B 相同，all-anchor 的均值应等于各 view-anchor 均值的平均
            self.assertTrue(torch.allclose(loss_all, (loss_v0 + loss_v1 + loss_v2) / 3, rtol=1e-6, atol=1e-6))

    def test_input_validation(self):
        multiview_info_nce_loss = _load_multiview_info_nce_loss()
        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(1, 4, 8))

        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(2, 0, 8))

        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(2, 4, 0))

        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(2, 4), temperature=0.2)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(2, 4, 8), temperature=0.0)

        with self.assertRaises(TypeError):
            multiview_info_nce_loss(torch.randint(0, 10, (2, 4, 8)))

        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(2, 4, 8), anchor_views=[])

        with self.assertRaises(ValueError):
            multiview_info_nce_loss(torch.randn(2, 4, 8), anchor_views=[2])


if __name__ == "__main__":
    unittest.main()
