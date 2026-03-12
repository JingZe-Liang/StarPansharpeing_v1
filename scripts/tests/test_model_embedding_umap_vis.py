from pathlib import Path

from omegaconf import OmegaConf

from scripts.infer.model_embedding_umap_vis import (
    _build_dataset_sources,
    _choose_sample_indices,
    _dataset_cache_path,
    _parse_cache_fields,
    _resolve_cache_dir,
    _resolve_target_sample_count,
    DatasetSource,
)


def test_build_dataset_sources_splits_each_path_when_group_by_path() -> None:
    dataset_cfg = OmegaConf.create(
        {
            "train_loader": {
                "paths": {
                    "rgb_path": {
                        "3bands_512": [
                            [
                                "data/AerialVG/LitData_hyper_images",
                                "data/RS5M/LitData_images_train",
                            ],
                            {"resize_before_transform": 512, "force_to_rgb": True},
                        ]
                    }
                }
            }
        }
    )

    sources = _build_dataset_sources(dataset_cfg=dataset_cfg, split="train", group_by="path")

    assert [source.label for source in sources] == [
        "AerialVG/LitData_hyper_images",
        "RS5M/LitData_images_train",
    ]
    assert all(source.stream_kwargs["is_cycled"] is False for source in sources)
    assert all(source.stream_kwargs["shuffle"] is False for source in sources)


def test_resolve_target_sample_count_prefers_smaller_limit() -> None:
    sample_count = _resolve_target_sample_count(
        dataset_len=1000,
        samples_per_dataset=120,
        sample_ratio=0.05,
    )

    assert sample_count == 50


def test_choose_sample_indices_is_sorted_and_deterministic() -> None:
    indices_a = _choose_sample_indices(dataset_len=20, sample_count=5, seed=7)
    indices_b = _choose_sample_indices(dataset_len=20, sample_count=5, seed=7)

    assert indices_a.tolist() == indices_b.tolist()
    assert indices_a.tolist() == sorted(indices_a.tolist())
    assert len(set(indices_a.tolist())) == 5


def test_parse_cache_fields_requires_embedding() -> None:
    fields = _parse_cache_fields("embedding,dataset_label,sample_key")

    assert fields == ["embedding", "dataset_label", "sample_key"]


def test_resolve_cache_dir_turns_npz_path_into_directory() -> None:
    cache_dir = _resolve_cache_dir(Path("tmp/cache/all_embeddings.npz"))

    assert cache_dir == Path("tmp/cache/all_embeddings")


def test_dataset_cache_path_uses_per_dataset_file() -> None:
    source = DatasetSource(
        label="AerialVG/LitData_hyper_images",
        input_paths=["data/AerialVG/LitData_hyper_images"],
        stream_kwargs={},
        group_name="3bands_512",
    )

    cache_path = _dataset_cache_path(Path("tmp/cache"), source, 3)

    assert cache_path == Path("tmp/cache/003_AerialVG_LitData_hyper_images.npz")
