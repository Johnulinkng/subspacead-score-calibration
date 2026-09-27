"""Auditable adapters for the official VisA one-class split.

The Amazon ``spot-diff`` repository remains an untouched third-party source.
This module validates its split CSV and the MVTec-style tree produced by the
official preparation script.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from PIL import Image


VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

OFFICIAL_1CLS_COUNTS = {
    "candle": {"train_normal": 900, "test_normal": 100, "test_anomaly": 100},
    "capsules": {"train_normal": 542, "test_normal": 60, "test_anomaly": 100},
    "cashew": {"train_normal": 450, "test_normal": 50, "test_anomaly": 100},
    "chewinggum": {"train_normal": 453, "test_normal": 50, "test_anomaly": 100},
    "fryum": {"train_normal": 450, "test_normal": 50, "test_anomaly": 100},
    "macaroni1": {"train_normal": 900, "test_normal": 100, "test_anomaly": 100},
    "macaroni2": {"train_normal": 900, "test_normal": 100, "test_anomaly": 100},
    "pcb1": {"train_normal": 904, "test_normal": 100, "test_anomaly": 100},
    "pcb2": {"train_normal": 901, "test_normal": 100, "test_anomaly": 100},
    "pcb3": {"train_normal": 905, "test_normal": 101, "test_anomaly": 100},
    "pcb4": {"train_normal": 904, "test_normal": 101, "test_anomaly": 100},
    "pipe_fryum": {"train_normal": 450, "test_normal": 50, "test_anomaly": 100},
}

_SPLIT_HEADER = ("object", "split", "label", "image", "mask")


@dataclass(frozen=True)
class VisaSplitRow:
    category: str
    split: str
    label: str
    image: str
    mask: str

    @property
    def prepared_label(self) -> str:
        return "good" if self.label == "normal" else "bad"

    @property
    def prepared_image(self) -> Path:
        return Path(self.category, self.split, self.prepared_label, PurePosixPath(self.image).name)

    @property
    def prepared_mask(self) -> Path | None:
        if self.label == "normal":
            return None
        return Path(
            self.category,
            "ground_truth",
            "bad",
            PurePosixPath(self.mask).name,
        )


def _safe_posix_relative(value: str, *, field: str, row_number: int) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"row {row_number}: unsafe {field} path {value!r}")


def read_visa_split(
    split_file: str | Path,
    *,
    enforce_official_counts: bool = True,
) -> tuple[VisaSplitRow, ...]:
    """Read and validate a VisA 1-class split CSV.

    With ``enforce_official_counts=True``, the exact category and partition
    counts published in the official ``split_csv/1cls.csv`` are required.
    """

    split_path = Path(split_file)
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _SPLIT_HEADER:
            raise ValueError(
                f"unexpected split header {reader.fieldnames!r}; expected {_SPLIT_HEADER!r}"
            )

        rows: list[VisaSplitRow] = []
        for row_number, record in enumerate(reader, start=2):
            category = record["object"].strip()
            split = record["split"].strip()
            label = record["label"].strip()
            image = record["image"].strip()
            mask = record["mask"].strip()

            if category not in VISA_CATEGORIES:
                raise ValueError(f"row {row_number}: unknown category {category!r}")
            if split not in {"train", "test"}:
                raise ValueError(f"row {row_number}: unknown split {split!r}")
            if label not in {"normal", "anomaly"}:
                raise ValueError(f"row {row_number}: unknown label {label!r}")
            if split == "train" and label != "normal":
                raise ValueError(f"row {row_number}: 1-class training must contain only normal images")

            _safe_posix_relative(image, field="image", row_number=row_number)
            image_parts = PurePosixPath(image).parts
            if not image_parts or image_parts[0] != category:
                raise ValueError(f"row {row_number}: image path is outside category {category!r}")

            if label == "anomaly":
                if split != "test":
                    raise ValueError(f"row {row_number}: anomaly row must belong to test split")
                _safe_posix_relative(mask, field="mask", row_number=row_number)
                mask_parts = PurePosixPath(mask).parts
                if not mask_parts or mask_parts[0] != category:
                    raise ValueError(f"row {row_number}: mask path is outside category {category!r}")
            elif mask:
                raise ValueError(f"row {row_number}: normal sample must not declare a mask")

            rows.append(VisaSplitRow(category, split, label, image, mask))

    if not rows:
        raise ValueError("split CSV contains no samples")

    image_paths = [row.image for row in rows]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("split CSV contains duplicate source image paths")

    prepared_images = [row.prepared_image.as_posix() for row in rows]
    if len(prepared_images) != len(set(prepared_images)):
        raise ValueError("split CSV produces colliding prepared image paths")

    prepared_masks = [
        row.prepared_mask.as_posix() for row in rows if row.prepared_mask is not None
    ]
    if len(prepared_masks) != len(set(prepared_masks)):
        raise ValueError("split CSV produces colliding prepared mask paths")

    if enforce_official_counts:
        actual_categories = {row.category for row in rows}
        if actual_categories != set(VISA_CATEGORIES):
            raise ValueError("split CSV does not contain the exact 12 official categories")
        actual = _partition_counts(rows)
        if actual != OFFICIAL_1CLS_COUNTS:
            raise ValueError(f"official partition counts do not match: {actual!r}")

    return tuple(rows)


def _partition_counts(rows: Iterable[VisaSplitRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row.category][f"{row.split}_{row.label}"] += 1
    return {
        category: {
            "train_normal": counts[category]["train_normal"],
            "test_normal": counts[category]["test_normal"],
            "test_anomaly": counts[category]["test_anomaly"],
        }
        for category in sorted(counts)
    }


def audit_visa_split(
    split_file: str | Path,
    *,
    enforce_official_counts: bool = True,
) -> dict[str, object]:
    rows = read_visa_split(split_file, enforce_official_counts=enforce_official_counts)
    counts = _partition_counts(rows)
    return {
        "row_count": len(rows),
        "category_count": len(counts),
        "normal_count": sum(row.label == "normal" for row in rows),
        "anomaly_count": sum(row.label == "anomaly" for row in rows),
        "partition_counts": counts,
    }


def audit_visa_raw_layout(
    raw_root: str | Path,
    split_file: str | Path,
    *,
    enforce_official_counts: bool = True,
) -> dict[str, object]:
    """Verify every source image/mask named by the official split exists."""

    root = Path(raw_root)
    rows = read_visa_split(split_file, enforce_official_counts=enforce_official_counts)
    expected = [root.joinpath(*PurePosixPath(row.image).parts) for row in rows]
    expected.extend(
        root.joinpath(*PurePosixPath(row.mask).parts)
        for row in rows
        if row.mask
    )
    missing = [str(path) for path in expected if not path.is_file()]
    empty = [str(path) for path in expected if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise ValueError(
            f"raw VisA tree failed validation: {len(missing)} missing, {len(empty)} empty; "
            f"examples={missing[:3] + empty[:3]}"
        )
    return {
        "root": str(root.resolve()),
        "source_image_count": len(rows),
        "source_mask_count": sum(bool(row.mask) for row in rows),
        "all_files_present_and_nonempty": True,
    }


def audit_visa_prepared_layout(
    prepared_root: str | Path,
    split_file: str | Path,
    *,
    enforce_official_counts: bool = True,
    check_mask_values: bool = True,
) -> dict[str, object]:
    """Verify the prepared MVTec-style VisA tree against the split CSV."""

    root = Path(prepared_root)
    rows = read_visa_split(split_file, enforce_official_counts=enforce_official_counts)
    expected_images = {row.prepared_image.as_posix() for row in rows}
    expected_masks = {
        row.prepared_mask.as_posix() for row in rows if row.prepared_mask is not None
    }
    expected_files = expected_images | expected_masks

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } if root.is_dir() else set()
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    empty = sorted(
        relative
        for relative in expected_files & actual_files
        if (root / Path(relative)).stat().st_size == 0
    )
    if missing or unexpected or empty:
        raise ValueError(
            "prepared VisA tree failed validation: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, {len(empty)} empty; "
            f"examples={missing[:3] + unexpected[:3] + empty[:3]}"
        )

    if check_mask_values:
        invalid_masks: list[str] = []
        empty_masks: list[str] = []
        for relative in sorted(expected_masks):
            mask = np.asarray(Image.open(root / Path(relative)))
            values = set(np.unique(mask).tolist())
            if not values.issubset({0, 255}):
                invalid_masks.append(relative)
            if 255 not in values:
                empty_masks.append(relative)
        if invalid_masks or empty_masks:
            raise ValueError(
                "prepared VisA masks failed binary/non-empty validation: "
                f"{len(invalid_masks)} non-binary, {len(empty_masks)} empty; "
                f"examples={invalid_masks[:3] + empty_masks[:3]}"
            )

    return {
        "root": str(root.resolve()),
        "image_count": len(expected_images),
        "mask_count": len(expected_masks),
        "category_count": len({row.category for row in rows}),
        "exact_split_match": True,
        "binary_nonempty_masks": check_mask_values,
    }


def split_rows_as_dicts(split_file: str | Path) -> list[dict[str, str]]:
    """Return frozen split records for callers that need path-level metadata."""

    return [asdict(row) for row in read_visa_split(split_file)]
