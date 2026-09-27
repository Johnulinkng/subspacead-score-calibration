"""Project-layer adapter for the frozen PatchCore author implementation.

This module intentionally does not import ``third_party/PatchCore`` at import
time.  The author runtime has optional dependencies that are not part of the
verified DRAEM environment.  Dataset geometry and prediction persistence can
therefore be tested independently before a PatchCore runtime is installed.

"Raw map" below means the author's final floating-point segmentation output
(including its fixed Gaussian smoothing), with no per-image normalization.
The only project-side operation is one bilinear resize from the model plane to
the frozen 256 x 256 evaluation plane.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data_protocols import TestSample, build_test_manifest, validate_train_layout
from .draem_runtime import (
    configure_reproducibility,
    make_generator,
    seed_dataloader_worker,
)
from .metrics import (
    EVALUATION_PROTOCOL,
    METRICS_SCHEMA_VERSION,
    evaluate_category,
    image_auroc_from_scores,
    image_score_contract,
    raw_map_max_image_scores,
)
from .formal_provenance import RunPurpose, classify_run_purpose
from .formal_test_inventory import validate_formal_prediction_rows
from .run_artifacts import (
    prepare_output_directory,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from .wrn50_weights import build_torchvision_wrn50_from_local_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "third_party" / "PatchCore"
PATCHCORE_AUTHOR_COMMIT = "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
PATCHCORE_EVALUATION_PROTOCOL = EVALUATION_PROTOCOL
DatasetName = Literal["mvtec_ad", "visa"]
DatasetSplit = Literal["train", "test"]
ExecutionScope = Literal["full", "debug_limited"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PatchCoreGeometry:
    """Frozen author input geometry and common evaluation geometry."""

    resize: int = 256
    image_size: int = 224
    evaluation_size: int = 256

    def validate(self) -> None:
        if self.resize <= 0 or self.image_size <= 0 or self.evaluation_size <= 0:
            raise ValueError("all PatchCore geometry sizes must be positive")
        if self.resize < self.image_size:
            raise ValueError("resize must be greater than or equal to image_size")


@dataclass(frozen=True)
class PatchCoreExportSettings:
    """Metadata required to persist one complete category prediction set."""

    dataset: DatasetName
    category: str
    dataset_root: Path
    output_dir: Path
    seed: int
    execution_scope: ExecutionScope
    model_provenance: dict[str, Any]
    run_purpose: RunPurpose | None = None
    formal_provenance: dict[str, Any] | None = None
    geometry: PatchCoreGeometry = PatchCoreGeometry()
    vendor_commit: str = PATCHCORE_AUTHOR_COMMIT
    sample_indices: tuple[int, ...] | None = None

    def resolved_run_purpose(self) -> RunPurpose:
        if self.run_purpose is not None:
            return self.run_purpose
        return (
            "debug_limited"
            if self.execution_scope == "debug_limited"
            else "author_reproduction"
        )

    def validate(self) -> None:
        if self.dataset not in {"mvtec_ad", "visa"}:
            raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
        if not self.category:
            raise ValueError("category must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.execution_scope not in {"full", "debug_limited"}:
            raise ValueError("execution_scope must be 'full' or 'debug_limited'")
        purpose = classify_run_purpose(
            self.resolved_run_purpose(),
            max_steps=1 if self.execution_scope == "debug_limited" else None,
        )
        if purpose["execution_scope"] != self.execution_scope:
            raise ValueError("run_purpose and execution_scope disagree")
        if self.resolved_run_purpose() == "development_full":
            raise ValueError(
                "development_full must use a frozen development evaluator, not "
                "the formal test export entry point"
            )
        if self.resolved_run_purpose() == "formal_full":
            if not isinstance(self.formal_provenance, dict) or not self.formal_provenance:
                raise ValueError("formal_full PatchCore export requires formal provenance")
        elif self.formal_provenance is not None:
            raise ValueError("formal provenance is forbidden outside formal_full")
        if self.execution_scope == "full" and self.geometry != PatchCoreGeometry():
            raise ValueError(
                "full PatchCore exports require frozen geometry "
                "resize=256, image_size=224, evaluation_size=256"
            )
        if self.execution_scope == "full" and self.sample_indices is not None:
            raise ValueError("sample_indices are forbidden for a full PatchCore export")
        if self.sample_indices is not None:
            if (
                not self.sample_indices
                or any(index < 0 for index in self.sample_indices)
                or tuple(sorted(set(self.sample_indices))) != self.sample_indices
            ):
                raise ValueError(
                    "sample_indices must be a non-empty sorted tuple of unique "
                    "non-negative indices"
                )
        if self.vendor_commit != PATCHCORE_AUTHOR_COMMIT:
            raise ValueError("PatchCore vendor commit differs from the frozen source")
        if not isinstance(self.model_provenance, dict) or not self.model_provenance:
            raise ValueError("model_provenance must be a non-empty dictionary")
        self.geometry.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(Path(self.dataset_root).resolve())
        payload["output_dir"] = str(Path(self.output_dir).resolve())
        payload["run_purpose"] = self.resolved_run_purpose()
        return payload


class PatchCoreDataset(Dataset[dict[str, object]]):
    """One-class MVTec/VisA adapter with PatchCore's author preprocessing.

    Images use the author's ``Resize(int) -> CenterCrop -> ImageNet`` pipeline.
    Test masks follow the identical spatial transform with nearest-neighbour
    interpolation, then one nearest-neighbour resize to the common evaluation
    plane.  No anomaly label or mask is used for the training split.
    """

    def __init__(
        self,
        dataset: DatasetName,
        dataset_root: str | Path,
        category: str,
        *,
        split: DatasetSplit,
        geometry: PatchCoreGeometry = PatchCoreGeometry(),
        training_image_paths: Sequence[str | Path] | None = None,
        samples: Sequence[TestSample] | None = None,
    ) -> None:
        if dataset not in {"mvtec_ad", "visa"}:
            raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not category:
            raise ValueError("category must not be empty")
        geometry.validate()

        self.dataset_name = dataset
        self.dataset_root = Path(dataset_root).resolve()
        self.category = category
        self.split = split
        self.geometry = geometry
        self.imagesize = (3, geometry.image_size, geometry.image_size)

        self._image_transform = transforms.Compose(
            [
                transforms.Resize(geometry.resize),
                transforms.CenterCrop(geometry.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    geometry.resize, interpolation=InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(geometry.image_size),
                transforms.Resize(
                    (geometry.evaluation_size, geometry.evaluation_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.PILToTensor(),
            ]
        )

        self.test_samples: list[TestSample] | None
        if training_image_paths is not None and split != "train":
            raise ValueError("training_image_paths are valid only for the training split")
        if samples is not None and split != "test":
            raise ValueError("samples are valid only for the test split")
        if split == "train":
            if training_image_paths is None:
                self.image_paths = validate_train_layout(self.dataset_root, category)
            else:
                resolved_paths = [Path(path).resolve() for path in training_image_paths]
                if not resolved_paths or len(resolved_paths) != len(set(resolved_paths)):
                    raise ValueError(
                        "training_image_paths must be non-empty and contain no duplicates"
                    )
                for path in resolved_paths:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                self.image_paths = resolved_paths
            self.test_samples = None
        else:
            if samples is None:
                self.test_samples = build_test_manifest(
                    dataset, self.dataset_root, category
                )
            else:
                self.test_samples = list(samples)
                if not self.test_samples:
                    raise ValueError("explicit PatchCore test samples must not be empty")
                if any(
                    sample.dataset != dataset or sample.category != category
                    for sample in self.test_samples
                ):
                    raise ValueError("explicit PatchCore test sample identity mismatch")
                sample_ids = [sample.sample_id for sample in self.test_samples]
                if len(sample_ids) != len(set(sample_ids)):
                    raise ValueError("explicit PatchCore test sample IDs must be unique")
            self.image_paths = [sample.image_path for sample in self.test_samples]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as source:
            original_width, original_height = source.size
            image = self._image_transform(source.convert("RGB"))

        if self.test_samples is None:
            sample_id = f"{self.category}/good/{image_path.stem}"
            defect_type = "good"
            label = 0
            mask_path: Path | None = None
            mask = torch.zeros(
                (1, self.geometry.evaluation_size, self.geometry.evaluation_size),
                dtype=torch.float32,
            )
        else:
            sample = self.test_samples[index]
            sample_id = sample.sample_id
            defect_type = sample.defect_type
            label = sample.label
            mask_path = sample.mask_path
            if mask_path is None:
                mask = torch.zeros(
                    (1, self.geometry.evaluation_size, self.geometry.evaluation_size),
                    dtype=torch.float32,
                )
            else:
                with Image.open(mask_path) as source_mask:
                    mask = (self._mask_transform(source_mask.convert("L")) > 0).to(
                        torch.float32
                    )

        return {
            "image": image,
            "mask": mask,
            "classname": self.category,
            "anomaly": defect_type,
            "is_anomaly": torch.tensor(label, dtype=torch.int64),
            "label": torch.tensor(label, dtype=torch.int64),
            "sample_id": sample_id,
            "image_path": str(image_path),
            "mask_path": "" if mask_path is None else str(mask_path),
            "original_height": original_height,
            "original_width": original_width,
        }


def build_patchcore_loader(
    dataset: PatchCoreDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    pin_memory: bool = False,
) -> DataLoader[dict[str, object]]:
    """Build the non-shuffled loader used by the author implementation."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_dataloader_worker,
        generator=make_generator(seed),
    )


def audit_patchcore_runtime(
    vendor_root: str | Path = DEFAULT_VENDOR_ROOT,
) -> dict[str, Any]:
    """Return a read-only dependency/source audit without importing PatchCore."""

    root = Path(vendor_root).resolve()
    expected_files = (
        root / "src" / "patchcore" / "patchcore.py",
        root / "src" / "patchcore" / "common.py",
        root / "src" / "patchcore" / "backbones.py",
        root / "src" / "patchcore" / "sampler.py",
        root / "LICENSE",
    )
    required_modules = ("faiss", "timm")
    modules = {
        name: importlib.util.find_spec(name) is not None for name in required_modules
    }
    git_commit: str | None = None
    git_clean: bool | None = None
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_commit = commit_result.stdout.strip()
        git_clean = not bool(status_result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    source_matches = git_commit == PATCHCORE_AUTHOR_COMMIT and git_clean is True
    return {
        "schema_version": 2,
        "vendor_root": str(root),
        "frozen_commit": PATCHCORE_AUTHOR_COMMIT,
        "actual_commit": git_commit,
        "commit_matches": git_commit == PATCHCORE_AUTHOR_COMMIT,
        "vendor_worktree_clean": git_clean,
        "expected_files": {
            path.relative_to(root).as_posix(): path.is_file() for path in expected_files
        },
        "required_runtime_modules": modules,
        "runtime_import_ready": all(modules.values())
        and all(path.is_file() for path in expected_files)
        and source_matches,
        "import_attempted": False,
        "training_or_test_evaluation_run": False,
        "checked_at_utc": utc_now(),
    }


def import_patchcore_vendor(
    vendor_root: str | Path = DEFAULT_VENDOR_ROOT,
) -> tuple[Any, Any, Any, Any]:
    """Import the frozen author modules only after dependencies are available."""

    audit = audit_patchcore_runtime(vendor_root)
    missing_files = [
        name for name, exists in audit["expected_files"].items() if not exists
    ]
    missing_modules = [
        name
        for name, available in audit["required_runtime_modules"].items()
        if not available
    ]
    source_problem = not audit["commit_matches"] or not audit["vendor_worktree_clean"]
    if missing_files or missing_modules or source_problem:
        details = []
        if missing_files:
            details.append(f"missing files: {', '.join(missing_files)}")
        if missing_modules:
            details.append(f"missing modules: {', '.join(missing_modules)}")
        if source_problem:
            details.append(
                "vendor source is not a clean checkout of " + PATCHCORE_AUTHOR_COMMIT
            )
        raise RuntimeError("PatchCore runtime is not ready; " + "; ".join(details))

    source_root = Path(vendor_root).resolve() / "src"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import patchcore.backbones as backbones
    import patchcore.common as common
    import patchcore.patchcore as patchcore_module
    import patchcore.sampler as sampler

    return patchcore_module, backbones, common, sampler


def build_patchcore_backbone_from_local_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device,
    model_factory: Callable[[], torch.nn.Module] | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Build PatchCore's WRN50 backbone without the vendor download path."""

    source_audit = audit_patchcore_runtime(DEFAULT_VENDOR_ROOT)
    if not source_audit["commit_matches"] or not source_audit["vendor_worktree_clean"]:
        raise RuntimeError("PatchCore vendor source is not the frozen clean checkout")
    backbone, report = build_torchvision_wrn50_from_local_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        device=device,
        model_factory=model_factory,
    )
    backbone.name = "wideresnet50"  # type: ignore[attr-defined]
    backbone.seed = None  # type: ignore[attr-defined]
    return backbone, {
        **report,
        "consumer": "PatchCore",
        "vendor_backbones_load_called": False,
        "vendor_pretrained_true_called": False,
        "backbone_name": "wideresnet50",
        "patchcore_vendor_commit": PATCHCORE_AUTHOR_COMMIT,
    }


def build_patchcore_model_from_local_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device,
    coreset_device: str | torch.device,
    faiss_workers: int = 8,
    model_factory: Callable[[], torch.nn.Module] | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Assemble the frozen PatchCore configuration around a local backbone.

    This constructs the model and sampler only.  It neither fits a memory bank
    nor touches any train/test image.  The caller must run the project resource
    gate before choosing the model and coreset devices for a smoke or formal run.
    """

    if faiss_workers <= 0:
        raise ValueError("faiss_workers must be positive")
    model_device = torch.device(device)
    resolved_coreset_device = torch.device(coreset_device)
    patchcore_module, _backbones, common, sampler = import_patchcore_vendor()
    backbone, checkpoint_report = build_patchcore_backbone_from_local_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        device=model_device,
        model_factory=model_factory,
    )
    feature_sampler = sampler.ApproximateGreedyCoresetSampler(
        0.10,
        resolved_coreset_device,
    )
    nn_method = common.FaissNN(on_gpu=False, num_workers=faiss_workers)
    model = patchcore_module.PatchCore(model_device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=model_device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=feature_sampler,
        anomaly_score_num_nn=1,
        nn_method=nn_method,
    )
    return model, {
        "schema_version": 1,
        "method": "PatchCore",
        "checkpoint": checkpoint_report,
        "configuration": {
            "backbone": "wideresnet50",
            "layers": ["layer2", "layer3"],
            "input_shape": [3, 224, 224],
            "pretrain_embed_dimension": 1024,
            "target_embed_dimension": 1024,
            "patchsize": 3,
            "anomaly_score_num_nn": 1,
            "coreset": "approx_greedy",
            "coreset_percentage": 0.10,
            "model_device": str(model_device),
            "coreset_device": str(resolved_coreset_device),
            "faiss_on_gpu": False,
            "faiss_workers": faiss_workers,
        },
        "model_constructed": True,
        "memory_bank_fitted": False,
        "dataset_content_accessed": False,
        "implicit_download_used": False,
    }


def run_patchcore_synthetic_smoke(seed: int = 20260927) -> dict[str, Any]:
    """Exercise the author CPU/FAISS path using random tensors only.

    The smoke deliberately supplies a tiny local backbone rather than calling
    the author's ImageNet backbone loader.  It therefore validates imports,
    feature hooks, coreset sampling, FAISS indexing and prediction without a
    weight download or any dataset access.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    started = time.perf_counter()
    source_audit = audit_patchcore_runtime(DEFAULT_VENDOR_ROOT)
    patchcore_module, _backbones, common, sampler = import_patchcore_vendor(
        DEFAULT_VENDOR_ROOT
    )
    configure_reproducibility(seed, deterministic=True)
    device = torch.device("cpu")

    class TinySyntheticBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            )

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.layer3(self.layer2(image))

    feature_sampler = sampler.ApproximateGreedyCoresetSampler(
        0.25,
        device,
        number_of_starting_points=3,
        dimension_to_project_features_to=4,
    )
    nn_method = common.FaissNN(on_gpu=False, num_workers=1)
    model = patchcore_module.PatchCore(device)
    model.load(
        backbone=TinySyntheticBackbone(),
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 8, 8),
        pretrain_embed_dimension=16,
        target_embed_dimension=8,
        patchsize=3,
        featuresampler=feature_sampler,
        anomaly_score_num_nn=1,
        nn_method=nn_method,
    )
    synthetic_support = torch.rand(2, 3, 8, 8)
    synthetic_query = torch.rand(1, 3, 8, 8)
    model.fit([{"image": synthetic_support}])
    scores, maps = model.predict(synthetic_query)

    score_array = np.asarray(scores, dtype=np.float64)
    map_array = np.asarray(maps, dtype=np.float64)
    feature_bank = np.asarray(model.anomaly_scorer.detection_features)
    index_size = int(model.anomaly_scorer.nn_method.search_index.ntotal)
    if score_array.shape != (1,) or map_array.shape != (1, 8, 8):
        raise RuntimeError(
            f"unexpected synthetic PatchCore outputs: {score_array.shape}, {map_array.shape}"
        )
    if not np.isfinite(score_array).all() or not np.isfinite(map_array).all():
        raise RuntimeError("synthetic PatchCore output is non-finite")
    if feature_bank.shape != (32, 8) or index_size != 32:
        raise RuntimeError(
            f"unexpected PatchCore memory bank: {feature_bank.shape}, index={index_size}"
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "kind": "patchcore_author_runtime_synthetic_cpu_smoke",
        "claim_eligible": False,
        "formal_training_run": False,
        "formal_test_content_accessed": False,
        "synthetic_memory_bank_fit": True,
        "pretrained_weights_requested": False,
        "author_backbone_loader_called": False,
        "seed": seed,
        "device": "cpu",
        "support_tensor_shape": [2, 3, 8, 8],
        "query_tensor_shape": [1, 3, 8, 8],
        "score_shape": list(score_array.shape),
        "map_shape": list(map_array.shape),
        "outputs_finite": True,
        "feature_bank_shape": list(feature_bank.shape),
        "faiss_index_size": index_size,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "faiss": getattr(common.faiss, "__version__", "unknown"),
            "timm": getattr(_backbones.timm, "__version__", "unknown"),
        },
        "source": {
            "commit": source_audit["actual_commit"],
            "commit_matches": source_audit["commit_matches"],
            "vendor_worktree_clean": source_audit["vendor_worktree_clean"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": utc_now(),
    }


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative_to_root(value: Path | None, root: Path) -> str | None:
    if value is None:
        return None
    return value.resolve().relative_to(root).as_posix()


def _resize_author_map(
    anomaly_map: np.ndarray,
    geometry: PatchCoreGeometry,
) -> np.ndarray:
    array = np.asarray(anomaly_map)
    expected = (geometry.image_size, geometry.image_size)
    if array.shape != expected:
        raise ValueError(
            f"PatchCore author map must have shape {expected}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("PatchCore author map contains NaN or infinite values")
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))[None, None]
    resized = torch_functional.interpolate(
        tensor,
        size=(geometry.evaluation_size, geometry.evaluation_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy().astype(np.float32, copy=False)


def export_patchcore_predictions(
    settings: PatchCoreExportSettings,
    *,
    author_maps: Sequence[np.ndarray],
    author_image_scores: Sequence[float],
    formal_test_selection: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Persist and evaluate one complete category from author outputs.

    The function deliberately accepts already-produced outputs.  Model fitting
    and test inference stay outside this dependency-light layer.  Metrics are
    computed only after reloading the exact saved arrays.
    """

    settings.validate()
    source_audit = audit_patchcore_runtime(DEFAULT_VENDOR_ROOT)
    if not source_audit["commit_matches"] or not source_audit["vendor_worktree_clean"]:
        raise RuntimeError(
            "refusing to export PatchCore predictions from an unfrozen vendor source"
        )
    dataset_root = Path(settings.dataset_root).resolve()
    run_purpose = settings.resolved_run_purpose()
    purpose = classify_run_purpose(
        run_purpose,
        max_steps=1 if settings.execution_scope == "debug_limited" else None,
    )
    formal_test_evidence: Mapping[str, object] | None = None
    formal_test_entries: list[Mapping[str, object]] | None = None
    formal_test_samples: list[TestSample] | None = None
    if run_purpose == "formal_full":
        if not isinstance(formal_test_selection, Mapping):
            raise ValueError(
                "formal_full PatchCore export requires formal test inventory selection"
            )
        evidence = formal_test_selection.get("evidence")
        entries = formal_test_selection.get("entries")
        samples = formal_test_selection.get("samples")
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(entries, list)
            or not isinstance(samples, list)
            or not all(isinstance(entry, Mapping) for entry in entries)
            or not all(isinstance(sample, TestSample) for sample in samples)
        ):
            raise ValueError("formal test inventory category selection is invalid")
        formal_test_evidence = evidence
        formal_test_entries = entries
        formal_test_samples = samples
    elif formal_test_selection is not None:
        raise ValueError("formal test inventory selection is forbidden outside formal_full")
    evaluation_protocol = (
        PATCHCORE_EVALUATION_PROTOCOL
        if settings.geometry == PatchCoreGeometry()
        else (
            "synthetic_nonstandard_patchcore_"
            f"r{settings.geometry.resize}_i{settings.geometry.image_size}_"
            f"e{settings.geometry.evaluation_size}"
        )
    )
    output_dir = prepare_output_directory(settings.output_dir)
    test_dataset = PatchCoreDataset(
        settings.dataset,
        dataset_root,
        settings.category,
        split="test",
        geometry=settings.geometry,
        samples=formal_test_samples,
    )
    selected_indices = (
        list(range(len(test_dataset)))
        if settings.sample_indices is None
        else list(settings.sample_indices)
    )
    if selected_indices and selected_indices[-1] >= len(test_dataset):
        raise ValueError("sample_indices contain an index outside the test dataset")
    if len(author_maps) != len(selected_indices):
        raise ValueError(
            f"received {len(author_maps)} maps for {len(selected_indices)} selected test samples"
        )
    author_scores = np.asarray(author_image_scores, dtype=np.float64)
    if author_scores.shape != (len(selected_indices),):
        raise ValueError(
            "author_image_scores must have one value for every selected test sample"
        )
    if not np.isfinite(author_scores).all():
        raise ValueError("author_image_scores contains NaN or infinite values")

    raw_dir = output_dir / "predictions" / "raw_maps"
    mask_dir = output_dir / "predictions" / "evaluation_masks"
    run_config = {
        "schema_version": 2,
        "method": "PatchCore",
        "evaluation_protocol": evaluation_protocol,
        "execution_scope": settings.execution_scope,
        "run_purpose": run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        "formal_provenance": settings.formal_provenance,
        "standalone_performance_claim_eligible": False,
        "settings": settings.to_dict(),
        "source_provenance": {
            "repository": "https://github.com/amazon-science/patchcore-inspection.git",
            "vendor_commit": settings.vendor_commit,
            "vendor_directory": str(DEFAULT_VENDOR_ROOT.resolve()),
            "actual_commit": source_audit["actual_commit"],
            "vendor_worktree_clean": source_audit["vendor_worktree_clean"],
        },
        "prediction_contract": {
            "raw_map_dtype": "float32",
            "raw_map_normalization": "none",
            "author_map_semantics": "author_output_after_gaussian_sigma_4",
            "author_map_shape": [
                settings.geometry.image_size,
                settings.geometry.image_size,
            ],
            "evaluation_map_shape": [
                settings.geometry.evaluation_size,
                settings.geometry.evaluation_size,
            ],
            "prediction_resize": "single_bilinear_align_corners_false_equivalent",
            "mask_geometry": "author_resize_and_center_crop_then_nearest_to_evaluation_plane",
            "image_scores": image_score_contract(
                "patchcore_author_patch_score_aggregation"
            ),
            "metrics_reload_saved_arrays": True,
            "complete_category": settings.sample_indices is None,
        },
        "created_at_utc": utc_now(),
    }
    if formal_test_evidence is not None:
        run_config["formal_test_inventory"] = dict(formal_test_evidence)
    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "exporting",
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        },
    )

    rows: list[dict[str, Any]] = []
    try:
        for index, (dataset_index, author_map, author_score) in enumerate(
            zip(selected_indices, author_maps, author_scores, strict=True)
        ):
            sample = test_dataset.test_samples[dataset_index]  # type: ignore[index]
            dataset_item = test_dataset[dataset_index]
            evaluation_map = _resize_author_map(author_map, settings.geometry)
            evaluation_mask = (
                dataset_item["mask"].numpy()[0].astype(bool, copy=False)
            )
            raw_path = raw_dir / f"{index:06d}.npy"
            mask_path = mask_dir / f"{index:06d}.npy"
            _atomic_save_npy(raw_path, evaluation_map)
            _atomic_save_npy(mask_path, evaluation_mask)
            row: dict[str, Any] = {
                    "index": index,
                    "dataset_index": dataset_index,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "defect_type": sample.defect_type,
                    "input_image": _relative_to_root(sample.image_path, dataset_root),
                    "source_ground_truth": _relative_to_root(
                        sample.mask_path, dataset_root
                    ),
                    "raw_map": raw_path.relative_to(output_dir).as_posix(),
                    "raw_map_sha256": sha256_file(raw_path),
                    "evaluation_mask": mask_path.relative_to(output_dir).as_posix(),
                    "evaluation_mask_sha256": sha256_file(mask_path),
                    "method_native_image_score": float(author_score),
                    "original_height": int(dataset_item["original_height"]),
                    "original_width": int(dataset_item["original_width"]),
                }
            if formal_test_evidence is not None:
                assert formal_test_entries is not None
                entry = formal_test_entries[index]
                row.update(
                    {
                        "source_image_sha256": entry["source_image_sha256"],
                        "source_ground_truth_sha256": entry["source_mask_sha256"],
                        "formal_test_inventory_id": formal_test_evidence[
                            "inventory_id"
                        ],
                        "formal_test_inventory_sha256": formal_test_evidence[
                            "inventory_sha256"
                        ],
                        "formal_test_inventory_receipt_sha256": formal_test_evidence[
                            "receipt_sha256"
                        ],
                    }
                )
            rows.append(row)

        write_jsonl(output_dir / "predictions" / "index.jsonl", rows)
        saved_maps = np.stack(
            [np.load(output_dir / row["raw_map"], allow_pickle=False) for row in rows]
        )
        saved_masks = np.stack(
            [
                np.load(output_dir / row["evaluation_mask"], allow_pickle=False)
                for row in rows
            ]
        )
        saved_labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        saved_author_scores = np.asarray(
            [row["method_native_image_score"] for row in rows], dtype=np.float64
        )
        raw_map_max_scores = raw_map_max_image_scores(saved_maps)
        for row, common_score in zip(rows, raw_map_max_scores, strict=True):
            row["raw_map_max_score"] = float(common_score)
        if formal_test_selection is not None:
            validate_formal_prediction_rows(rows, formal_test_selection)
        write_jsonl(output_dir / "predictions" / "index.jsonl", rows)

        metrics_evaluable = (
            len(np.unique(saved_labels)) == 2
            and bool(saved_masks.any())
            and bool((~saved_masks).any())
        )
        if metrics_evaluable:
            metrics = evaluate_category(
                saved_maps, saved_masks, saved_labels
            ).to_dict()
            metrics["image_auroc_method_native"] = image_auroc_from_scores(
                saved_labels, saved_author_scores
            )
            metrics_status = "computed"
        else:
            if settings.sample_indices is None:
                raise ValueError(
                    "complete-category metrics require both image classes and "
                    "both pixel classes"
                )
            metrics = {}
            metrics_status = "unavailable_for_partial_debug_selection"
        metrics_payload: dict[str, Any] = {
                "schema_version": METRICS_SCHEMA_VERSION,
                "dataset": settings.dataset,
                "category": settings.category,
                "num_images": len(rows),
                "protocol": evaluation_protocol,
                "image_score_contract": image_score_contract(
                    "patchcore_author_patch_score_aggregation"
                ),
                "metrics": metrics,
                "metrics_status": metrics_status,
                "complete_category": settings.sample_indices is None,
                "computed_from": "predictions/index.jsonl raw_map and evaluation_mask files",
                "computed_at_utc": utc_now(),
            }
        if formal_test_evidence is not None:
            metrics_payload["formal_test_inventory"] = dict(formal_test_evidence)
        write_json(output_dir / "metrics.json", metrics_payload)
        final_state = {
            "status": "completed",
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "formal_matrix_eligible": purpose["formal_matrix_eligible"],
            "num_images": len(rows),
            "run_config_sha256": sha256_file(output_dir / "run_config.json"),
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
            "predictions_index_sha256": sha256_file(
                output_dir / "predictions" / "index.jsonl"
            ),
            "updated_at_utc": utc_now(),
        }
        if formal_test_evidence is not None:
            final_state["formal_test_inventory"] = dict(formal_test_evidence)
        write_json(output_dir / "run_state.json", final_state)
        return {**final_state, "metrics": metrics}
    except BaseException as error:
        write_json(
            output_dir / "run_state.json",
            {
                "status": "failed",
                "execution_scope": settings.execution_scope,
                "run_purpose": run_purpose,
                "claim_scope": purpose["claim_scope"],
                "formal_matrix_eligible": False,
                "num_images_written": len(rows),
                "updated_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
