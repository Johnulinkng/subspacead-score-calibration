from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from industrial_anomaly_lab.data_protocols import (
    DraemSyntheticTrainDataset,
    DraemTestDataset,
    build_test_manifest,
)
from industrial_anomaly_lab.draem_runtime import configure_reproducibility


def _write_rgb(path: Path, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 48, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((40, 48), dtype=np.uint8)
    mask[5:15, 9:20] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_mvtec_manifest_resolves_mask_suffix(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "bottle/test/good/000.png")
    _write_rgb(tmp_path / "bottle/test/crack/001.png")
    _write_mask(tmp_path / "bottle/ground_truth/crack/001_mask.png")

    manifest = build_test_manifest("mvtec_ad", tmp_path, "bottle")

    assert [sample.label for sample in manifest] == [1, 0]
    assert manifest[0].mask_path.name == "001_mask.png"
    assert manifest[1].mask_path is None


def test_visa_manifest_accepts_uppercase_jpeg_and_stem_mask(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "candle/test/good/000.JPG")
    _write_rgb(tmp_path / "candle/test/bad/001.JPG")
    _write_mask(tmp_path / "candle/ground_truth/bad/001.png")

    manifest = build_test_manifest("visa", tmp_path, "candle")
    dataset = DraemTestDataset("visa", tmp_path, "candle", image_size=32)
    bad = dataset[0]
    good = dataset[1]

    assert [sample.label for sample in manifest] == [1, 0]
    assert manifest[0].mask_path.name == "001.png"
    assert tuple(bad["image"].shape) == (3, 32, 32)
    assert tuple(bad["mask"].shape) == (1, 32, 32)
    assert set(np.unique(bad["mask"].numpy())).issubset({0.0, 1.0})
    assert float(good["mask"].sum()) == 0.0


def test_training_adapter_is_deterministic_and_accepts_jpeg(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "visa/candle/train/good/000.JPG", 120)
    _write_rgb(tmp_path / "dtd/woven/source.jpg", 180)

    configure_reproducibility(17)
    first_dataset = DraemSyntheticTrainDataset(
        tmp_path / "visa", "candle", tmp_path / "dtd", image_size=32
    )
    first = first_dataset[0]
    configure_reproducibility(17)
    second_dataset = DraemSyntheticTrainDataset(
        tmp_path / "visa", "candle", tmp_path / "dtd", image_size=32
    )
    second = second_dataset[0]

    assert len(first_dataset) == 1
    assert first_dataset.image_paths[0].suffix == ".JPG"
    assert first["image"].shape == (3, 32, 32)
    assert np.array_equal(first["image"].numpy(), second["image"].numpy())
    assert np.array_equal(
        first["augmented_image"].numpy(), second["augmented_image"].numpy()
    )
    assert np.array_equal(
        first["anomaly_mask"].numpy(), second["anomaly_mask"].numpy()
    )
