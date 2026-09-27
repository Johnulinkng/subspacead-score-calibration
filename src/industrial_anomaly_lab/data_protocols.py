"""Dataset adapters for the MVTec AD and VisA one-class layouts.

Both datasets are represented as ``category/{train,test,ground_truth}`` after
the official VisA preparation step.  The adapter keeps two format differences
explicit: VisA images commonly use upper-case JPEG suffixes and its masks keep
the image stem, while MVTec masks conventionally add ``_mask``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from .draem_runtime import DEFAULT_VENDOR_ROOT, _load_vendor_file


DatasetName = Literal["mvtec_ad", "visa"]
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class TestSample:
    dataset: DatasetName
    category: str
    defect_type: str
    sample_id: str
    image_path: Path
    mask_path: Path | None
    label: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["image_path"] = str(self.image_path)
        payload["mask_path"] = None if self.mask_path is None else str(self.mask_path)
        return payload


def _validate_dataset_name(dataset: str) -> DatasetName:
    if dataset not in {"mvtec_ad", "visa"}:
        raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
    return dataset  # type: ignore[return-value]


def discover_images(directory: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return supported image files in deterministic, case-insensitive order."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory is missing: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    images = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda path: path.as_posix().casefold())


def _resolve_mask(mask_directory: Path, image_stem: str) -> Path:
    if not mask_directory.is_dir():
        raise FileNotFoundError(f"ground-truth directory is missing: {mask_directory}")

    candidates: list[Path] = []
    accepted_stems = {image_stem.casefold(), f"{image_stem}_mask".casefold()}
    for path in mask_directory.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and path.stem.casefold() in accepted_stems
        ):
            candidates.append(path.resolve())
    candidates.sort(key=lambda path: path.as_posix().casefold())
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one mask for stem '{image_stem}' in {mask_directory}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def build_test_manifest(
    dataset: str,
    dataset_root: str | Path,
    category: str,
) -> list[TestSample]:
    """Validate and enumerate one complete test category."""

    resolved_dataset = _validate_dataset_name(dataset)
    root = Path(dataset_root).resolve()
    category_root = root / category
    test_root = category_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"test directory is missing: {test_root}")

    defect_directories = sorted(
        (path for path in test_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not defect_directories:
        raise ValueError(f"no test defect directories found below {test_root}")

    samples: list[TestSample] = []
    seen_ids: set[str] = set()
    for defect_directory in defect_directories:
        defect_type = defect_directory.name
        is_normal = defect_type.casefold() == "good"
        images = discover_images(defect_directory)
        if not images:
            raise ValueError(f"no test images found in {defect_directory}")
        for image_path in images:
            mask_path = None
            if not is_normal:
                mask_path = _resolve_mask(
                    category_root / "ground_truth" / defect_type,
                    image_path.stem,
                )
            sample_id = f"{category}/{defect_type}/{image_path.stem}"
            if sample_id.casefold() in seen_ids:
                raise ValueError(f"duplicate sample id after suffix removal: {sample_id}")
            seen_ids.add(sample_id.casefold())
            samples.append(
                TestSample(
                    dataset=resolved_dataset,
                    category=category,
                    defect_type=defect_type,
                    sample_id=sample_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    label=0 if is_normal else 1,
                )
            )
    return samples


def validate_train_layout(dataset_root: str | Path, category: str) -> list[Path]:
    train_directory = Path(dataset_root).resolve() / category / "train" / "good"
    images = discover_images(train_directory)
    if not images:
        raise ValueError(f"no normal training images found in {train_directory}")
    return images


def _validate_explicit_image_paths(
    paths: Sequence[str | Path], label: str
) -> list[Path]:
    """Validate a previously authenticated path selection without rediscovery."""

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} image is missing or linked: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"{label} image suffix is unsupported: {path}")
        key = str(path).casefold()
        if key in seen:
            raise ValueError(f"duplicate {label} image path: {path}")
        seen.add(key)
        resolved.append(path)
    if not resolved:
        raise ValueError(f"explicit {label} image list is empty")
    return resolved


def _validate_explicit_test_samples(
    samples: Sequence[TestSample],
    *,
    dataset: str,
    dataset_root: str | Path,
    category: str,
) -> list[TestSample]:
    """Validate an authenticated test selection without directory discovery."""

    resolved_dataset = _validate_dataset_name(dataset)
    root = Path(dataset_root).resolve()
    if not samples:
        raise ValueError("explicit test sample list is empty")
    checked: list[TestSample] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for position, sample in enumerate(samples):
        if not isinstance(sample, TestSample):
            raise TypeError(f"explicit test sample {position} is not a TestSample")
        if sample.dataset != resolved_dataset or sample.category != category:
            raise ValueError(f"explicit test sample {position} dataset/category mismatch")
        if isinstance(sample.label, bool) or sample.label not in (0, 1):
            raise ValueError(f"explicit test sample {position} has invalid label")
        if (sample.label == 0) != (sample.defect_type.casefold() == "good"):
            raise ValueError(f"explicit test sample {position} label/defect mismatch")
        image_path = Path(sample.image_path)
        if image_path.is_symlink() or not image_path.is_file():
            raise FileNotFoundError(
                f"explicit test sample {position} image is missing or linked: {image_path}"
            )
        image_path = image_path.resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"explicit test sample {position} image escapes dataset root"
            ) from error
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"explicit test sample {position} image suffix is unsupported")
        mask_path: Path | None = None
        if sample.label == 0:
            if sample.mask_path is not None:
                raise ValueError(f"normal explicit test sample {position} has a mask")
        else:
            if sample.mask_path is None:
                raise ValueError(f"anomalous explicit test sample {position} has no mask")
            candidate_mask = Path(sample.mask_path)
            if candidate_mask.is_symlink() or not candidate_mask.is_file():
                raise FileNotFoundError(
                    f"explicit test sample {position} mask is missing or linked: "
                    f"{candidate_mask}"
                )
            mask_path = candidate_mask.resolve()
            try:
                mask_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"explicit test sample {position} mask escapes dataset root"
                ) from error
            if mask_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise ValueError(f"explicit test sample {position} mask suffix is unsupported")
        sample_key = sample.sample_id.casefold()
        image_key = str(image_path).casefold()
        if not sample.sample_id or sample_key in seen_ids or image_key in seen_images:
            raise ValueError("explicit test samples contain empty/duplicate IDs or images")
        seen_ids.add(sample_key)
        seen_images.add(image_key)
        checked.append(
            TestSample(
                dataset=resolved_dataset,
                category=category,
                defect_type=sample.defect_type,
                sample_id=sample.sample_id,
                image_path=image_path,
                mask_path=mask_path,
                label=sample.label,
            )
        )
    return checked


class DraemTestDataset(Dataset[dict[str, object]]):
    """DRAEM-compatible test loader with nearest-neighbor mask resizing."""

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
        self.dataset_root = Path(dataset_root).resolve()
        self.samples = (
            build_test_manifest(dataset, self.dataset_root, category)
            if samples is None
            else _validate_explicit_test_samples(
                samples,
                dataset=dataset,
                dataset_root=self.dataset_root,
                category=category,
            )
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {sample.image_path}")
        original_height, original_width = image.shape[:2]
        image = cv2.resize(
            image,
            dsize=(self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.from_numpy(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        )

        if sample.mask_path is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"OpenCV could not read mask: {sample.mask_path}")
            mask = cv2.resize(
                mask,
                dsize=(self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0).astype(np.uint8)

        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask[None, :, :].astype(np.float32)),
            "label": torch.tensor(sample.label, dtype=torch.int64),
            "sample_id": sample.sample_id,
            "defect_type": sample.defect_type,
            "image_path": str(sample.image_path),
            "mask_path": "" if sample.mask_path is None else str(sample.mask_path),
            "original_height": original_height,
            "original_width": original_width,
        }


_PERLIN_FUNCTION: Callable[..., np.ndarray] | None = None


def _get_perlin_function(vendor_root: Path) -> Callable[..., np.ndarray]:
    global _PERLIN_FUNCTION
    if _PERLIN_FUNCTION is None:
        module = _load_vendor_file("perlin", "perlin.py", vendor_root)
        _PERLIN_FUNCTION = module.rand_perlin_2d_np
    return _PERLIN_FUNCTION


class DraemSyntheticTrainDataset(Dataset[dict[str, object]]):
    """Author-compatible synthesis with file discovery extended for VisA JPEGs.

    The sampling and augmentation sequence mirrors the MIT-licensed DRAEM
    ``MVTecDRAEMTrainDataset``.  In particular, ``__getitem__`` intentionally
    resamples a random normal image, preserving the author's baseline behavior.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        category: str,
        anomaly_source_root: str | Path,
        *,
        image_size: int = 256,
        vendor_root: Path = DEFAULT_VENDOR_ROOT,
        normal_image_paths: Sequence[str | Path] | None = None,
        anomaly_source_paths: Sequence[str | Path] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if (normal_image_paths is None) != (anomaly_source_paths is None):
            raise ValueError(
                "normal_image_paths and anomaly_source_paths must be provided together"
            )
        self.image_paths = (
            validate_train_layout(dataset_root, category)
            if normal_image_paths is None
            else _validate_explicit_image_paths(normal_image_paths, "normal training")
        )
        self.anomaly_source_paths = (
            discover_images(anomaly_source_root, recursive=True)
            if anomaly_source_paths is None
            else _validate_explicit_image_paths(anomaly_source_paths, "anomaly source")
        )
        if not self.anomaly_source_paths:
            raise ValueError(f"no anomaly-source images found in {anomaly_source_root}")
        self.image_size = image_size
        self.vendor_root = Path(vendor_root)
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _random_augmenter(self) -> iaa.Sequential:
        indices = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        return iaa.Sequential([self.augmenters[index] for index in indices])

    def _augment_image(
        self, image: np.ndarray, anomaly_source_path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = cv2.imread(str(anomaly_source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"OpenCV could not read anomaly source: {anomaly_source_path}")
        source = cv2.resize(source, (self.image_size, self.image_size))
        source = self._random_augmenter()(image=source)

        perlin_scale_x = 2 ** int(torch.randint(0, 6, (1,)).item())
        perlin_scale_y = 2 ** int(torch.randint(0, 6, (1,)).item())
        perlin = _get_perlin_function(self.vendor_root)(
            (self.image_size, self.image_size),
            (perlin_scale_x, perlin_scale_y),
        )
        perlin = self.rot(image=perlin)
        mask = (perlin > 0.5).astype(np.float32)[:, :, None]
        source_region = source.astype(np.float32) * mask / 255.0
        beta = float(torch.rand(1).item()) * 0.8
        augmented = (
            image * (1.0 - mask)
            + (1.0 - beta) * source_region
            + beta * image * mask
        )

        if float(torch.rand(1).item()) > 0.5:
            return (
                image.astype(np.float32),
                np.zeros_like(mask, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        augmented = mask * augmented + (1.0 - mask) * image
        has_anomaly = 1.0 if float(mask.sum()) > 0.0 else 0.0
        return (
            augmented.astype(np.float32),
            mask.astype(np.float32),
            np.array([has_anomaly], dtype=np.float32),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        image_index = int(torch.randint(0, len(self.image_paths), (1,)).item())
        source_index = int(torch.randint(0, len(self.anomaly_source_paths), (1,)).item())
        image_path = self.image_paths[image_index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read training image: {image_path}")
        image = cv2.resize(image, (self.image_size, self.image_size))
        if float(torch.rand(1).item()) > 0.7:
            image = self.rot(image=image)
        image = image.astype(np.float32) / 255.0
        augmented, mask, has_anomaly = self._augment_image(
            image, self.anomaly_source_paths[source_index]
        )
        return {
            "image": torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()),
            "augmented_image": torch.from_numpy(
                np.transpose(augmented, (2, 0, 1)).copy()
            ),
            "anomaly_mask": torch.from_numpy(np.transpose(mask, (2, 0, 1)).copy()),
            "has_anomaly": torch.from_numpy(has_anomaly.copy()),
            "idx": image_index,
            "image_path": str(image_path),
        }
