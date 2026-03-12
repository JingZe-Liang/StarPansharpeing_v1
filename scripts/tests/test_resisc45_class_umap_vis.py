from pathlib import Path

from scripts.infer.resisc45_class_umap_vis import _collect_class_image_paths, _sample_images_per_class


def test_collect_class_image_paths_filters_extensions(tmp_path: Path) -> None:
    airplane_dir = tmp_path / "airplane"
    airport_dir = tmp_path / "airport"
    airplane_dir.mkdir()
    airport_dir.mkdir()

    (airplane_dir / "a.jpg").write_bytes(b"x")
    (airplane_dir / "ignore.txt").write_text("x", encoding="utf-8")
    (airport_dir / "b.png").write_bytes(b"x")

    class_to_paths = _collect_class_image_paths(tmp_path, extensions={".jpg", ".png"})

    assert sorted(class_to_paths) == ["airplane", "airport"]
    assert [path.name for path in class_to_paths["airplane"]] == ["a.jpg"]
    assert [path.name for path in class_to_paths["airport"]] == ["b.png"]


def test_sample_images_per_class_returns_fixed_count_per_class(tmp_path: Path) -> None:
    class_to_paths = {
        "airplane": [tmp_path / f"airplane_{index}.jpg" for index in range(5)],
        "airport": [tmp_path / f"airport_{index}.jpg" for index in range(5)],
    }

    samples = _sample_images_per_class(class_to_paths, samples_per_class=3, seed=7)

    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.class_name] = counts.get(sample.class_name, 0) + 1

    assert counts == {"airplane": 3, "airport": 3}
