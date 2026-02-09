from __future__ import annotations

import argparse

from scripts.dataset.cross_city_segmentation.make_train_patches_litdata import _resolve_output_dir


def test_resolve_output_dir_default_no_resize() -> None:
    args = argparse.Namespace(
        output_dir="",
        data_root="data/Downstreams/CrossCitySegmentation",
        dataset_name="beijing",
        patch_size=256,
        patch_resize_to=0,
        stride=128,
    )
    out = _resolve_output_dir(args)
    assert out.endswith("litdata_train/beijing_ps256_rsnone_s128_split_modal/train")


def test_resolve_output_dir_custom() -> None:
    args = argparse.Namespace(
        output_dir="/tmp/custom_litdata",
        data_root="data/Downstreams/CrossCitySegmentation",
        dataset_name="augsburg",
        patch_size=256,
        patch_resize_to=0,
        stride=128,
    )
    out = _resolve_output_dir(args)
    assert out == "/tmp/custom_litdata"
