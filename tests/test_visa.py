from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.visa import (
    OFFICIAL_1CLS_COUNTS,
    audit_visa_prepared_layout,
    audit_visa_split,
    read_visa_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SPLIT = PROJECT_ROOT / "third_party" / "spot-diff" / "split_csv" / "1cls.csv"


def _write_split(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows(rows)


def test_official_visa_one_class_split_is_frozen() -> None:
    result = audit_visa_split(OFFICIAL_SPLIT)

    assert result["row_count"] == 10_821
    assert result["normal_count"] == 9_621
    assert result["anomaly_count"] == 1_200
    assert result["partition_counts"] == OFFICIAL_1CLS_COUNTS


def test_split_rejects_anomalous_training_sample(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    _write_split(
        split,
        [["candle", "train", "anomaly", "candle/image.JPG", "candle/mask.png"]],
    )

    with pytest.raises(ValueError, match="only normal"):
        read_visa_split(split, enforce_official_counts=False)


def test_prepared_layout_matches_split_and_binary_masks(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    _write_split(
        split,
        [
            ["candle", "train", "normal", "candle/Data/Images/Normal/001.JPG", ""],
            ["candle", "test", "normal", "candle/Data/Images/Normal/002.JPG", ""],
            [
                "candle",
                "test",
                "anomaly",
                "candle/Data/Images/Anomaly/003.JPG",
                "candle/Data/Masks/Anomaly/003.png",
            ],
        ],
    )
    prepared = tmp_path / "prepared"
    for relative in (
        "candle/train/good/001.JPG",
        "candle/test/good/002.JPG",
        "candle/test/bad/003.JPG",
    ):
        output = prepared / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"image")
    mask_path = prepared / "candle/ground_truth/bad/003.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    Image.fromarray(mask).save(mask_path)

    result = audit_visa_prepared_layout(
        prepared,
        split,
        enforce_official_counts=False,
    )

    assert result["image_count"] == 3
    assert result["mask_count"] == 1
    assert result["exact_split_match"] is True


def test_prepared_layout_rejects_nonbinary_mask(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    _write_split(
        split,
        [["candle", "test", "anomaly", "candle/a.JPG", "candle/a.png"]],
    )
    prepared = tmp_path / "prepared"
    image = prepared / "candle/test/bad/a.JPG"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    mask_path = prepared / "candle/ground_truth/bad/a.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(np.array([[0, 127], [0, 255]], dtype=np.uint8)).save(mask_path)

    with pytest.raises(ValueError, match="non-binary"):
        audit_visa_prepared_layout(
            prepared,
            split,
            enforce_official_counts=False,
        )
