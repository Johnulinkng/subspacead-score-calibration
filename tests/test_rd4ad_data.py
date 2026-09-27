from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from industrial_anomaly_lab.rd4ad_data import (
    Rd4adNormalTrainDataset,
    Rd4adTestDataset,
)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((27, 39, 3), value, dtype=np.uint8), mode="RGB").save(
        path
    )


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((27, 39), dtype=np.uint8)
    mask[4:15, 10:24] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_rd4ad_mvtec_adapter_uses_suffix_mask_and_normalized_rgb(
    tmp_path: Path,
) -> None:
    _write_rgb(tmp_path / "bottle/train/good/train.png", 100)
    _write_rgb(tmp_path / "bottle/test/good/000.png", 80)
    _write_rgb(tmp_path / "bottle/test/crack/001.png", 180)
    _write_mask(tmp_path / "bottle/ground_truth/crack/001_mask.png")

    train = Rd4adNormalTrainDataset(tmp_path, "bottle", image_size=32)
    test = Rd4adTestDataset("mvtec_ad", tmp_path, "bottle", image_size=32)

    assert len(train) == 1
    assert tuple(train[0]["image"].shape) == (3, 32, 32)
    assert train[0]["image"].dtype.is_floating_point
    assert [sample.label for sample in test.samples] == [1, 0]
    assert test.samples[0].mask_path.name == "001_mask.png"
    assert tuple(test[0]["mask"].shape) == (1, 32, 32)
    assert set(test[0]["mask"].unique().tolist()) == {0.0, 1.0}
    assert float(test[1]["mask"].sum()) == 0.0


def test_rd4ad_visa_adapter_accepts_uppercase_jpeg_and_stem_mask(
    tmp_path: Path,
) -> None:
    _write_rgb(tmp_path / "candle/train/good/train.JPG", 100)
    _write_rgb(tmp_path / "candle/test/good/000.JPG", 80)
    _write_rgb(tmp_path / "candle/test/bad/001.JPG", 180)
    _write_mask(tmp_path / "candle/ground_truth/bad/001.png")

    train = Rd4adNormalTrainDataset(tmp_path, "candle", image_size=32)
    test = Rd4adTestDataset("visa", tmp_path, "candle", image_size=32)

    assert train.image_paths[0].suffix == ".JPG"
    assert [sample.label for sample in test.samples] == [1, 0]
    assert test.samples[0].mask_path.name == "001.png"
    assert test[0]["sample_id"] == "candle/bad/001"
    assert tuple(test[0]["image"].shape) == (3, 32, 32)
    assert np.isfinite(test[0]["image"].numpy()).all()
