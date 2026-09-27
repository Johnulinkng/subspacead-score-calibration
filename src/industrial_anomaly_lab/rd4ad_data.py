"""Project-owned data adapters for the unmodified RD4AD implementation.

The author loader only discovers lower-case PNG files.  These adapters use the
shared, validated MVTec-style manifest so the same code reads MVTec AD and the
official VisA one-class preparation without copying or renaming data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data_protocols import (
    TestSample,
    _validate_explicit_image_paths,
    _validate_explicit_test_samples,
    build_test_manifest,
    validate_train_layout,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_rd4ad_image_transform(image_size: int) -> transforms.Compose:
    """Freeze the author's RGB/ImageNet transform under torchvision 0.22."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_binary_mask(path: Path | None, image_size: int) -> torch.Tensor:
    if path is None:
        return torch.zeros((1, image_size, image_size), dtype=torch.float32)
    with Image.open(path) as image:
        mask = transforms.functional.resize(
            image.convert("L"),
            [image_size, image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        mask = transforms.functional.center_crop(mask, [image_size, image_size])
        tensor = transforms.functional.pil_to_tensor(mask)
    return (tensor > 0).to(dtype=torch.float32)


class Rd4adNormalTrainDataset(Dataset[dict[str, object]]):
    """Normal-only training set for one category, independent of file suffix."""

    def __init__(
        self,
        dataset_root: str | Path,
        category: str,
        *,
        image_size: int = 256,
        image_paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.image_paths = (
            validate_train_layout(dataset_root, category)
            if image_paths is None
            else _validate_explicit_image_paths(image_paths, "RD4AD normal training")
        )
        self.transform = build_rd4ad_image_transform(image_size)
        self.category = category

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "sample_id": f"{self.category}/train/good/{path.stem}",
            "image_path": str(path),
        }


class Rd4adTestDataset(Dataset[dict[str, object]]):
    """MVTec AD/VisA test data with a nearest-neighbor evaluation mask."""

    def __init__(
        self,
        dataset: str,
        dataset_root: str | Path,
        category: str,
        *,
        image_size: int = 256,
        samples: Sequence[TestSample] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.samples: list[TestSample] = (
            build_test_manifest(dataset, dataset_root, category)
            if samples is None
            else _validate_explicit_test_samples(
                samples,
                dataset=dataset,
                dataset_root=dataset_root,
                category=category,
            )
        )
        self.transform = build_rd4ad_image_transform(image_size)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
            original_width, original_height = rgb.size
            tensor = self.transform(rgb)
        mask = _load_binary_mask(sample.mask_path, self.image_size)
        return {
            "image": tensor,
            "mask": mask,
            "label": torch.tensor(sample.label, dtype=torch.int64),
            "sample_id": sample.sample_id,
            "defect_type": sample.defect_type,
            "image_path": str(sample.image_path),
            "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
            "original_height": original_height,
            "original_width": original_width,
        }
