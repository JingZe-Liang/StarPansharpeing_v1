import torch

from src.utilities.train_utils.visualization import visualize_segmentation_map


def test_visualize_segmentation_map_coco_palette_with_small_n_class() -> None:
    gt = torch.randint(0, 2, (1, 16, 16), dtype=torch.long)
    vis = visualize_segmentation_map(gt, n_class=2, use_coco_colors=True, to_pil=False)
    assert vis.shape == (1, 16, 16, 3)
