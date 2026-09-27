"""Auditable fit/load/inference runner for the frozen PatchCore comparator.

The vendor serializer is intentionally not used.  It stores Python pickle and
reconstructs the backbone through a loader that can download weights.  This
module stores the fitted state as a plain float32 NumPy memory bank, a FAISS
index, and a hash-gated JSON manifest.  The network is always reconstructed
from the project's explicit local WideResNet50-2 checkpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .draem_runtime import configure_reproducibility
from .formal_provenance import (
    RunPurpose,
    classify_run_purpose,
    validate_development_authority,
    validate_formal_authority,
    validate_formal_checkpoint_manifest,
)
from .formal_source_provenance import (
    DEFAULT_SCOPE as DEFAULT_FORMAL_SOURCE_SCOPE,
    capture_formal_source_provenance,
    validate_formal_source_provenance,
)
from .formal_test_inventory import resolve_formal_test_inventory_category
from .formal_training_inputs import (
    DEFAULT_FORMAL_INPUT_MANIFEST,
    resolve_frozen_formal_training_inputs,
)
from .patchcore_adapter import (
    PATCHCORE_AUTHOR_COMMIT,
    PatchCoreDataset,
    PatchCoreExportSettings,
    PatchCoreGeometry,
    audit_patchcore_runtime,
    build_patchcore_loader,
    build_patchcore_model_from_local_checkpoint,
    export_patchcore_predictions,
)
from .run_artifacts import (
    append_jsonl,
    prepare_output_directory,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from .wrn50_weights import inspect_wrn50_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRATION_PATH = PROJECT_ROOT / "configs" / "preregistered_experiments.json"
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "configs" / "project.json"
DEFAULT_DEVELOPMENT_PROTOCOL = PROJECT_ROOT / "configs" / "development_protocol.json"
DEFAULT_BACKBONE_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "weights" / "wide_resnet50_2-95faca4d.pth"
)
WRN50_IMAGENET1K_V1_SHA256 = (
    "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
)
ExecutionScope = Literal["full", "debug_limited"]

FROZEN_CONFIGURATION: dict[str, Any] = {
    "backbone": "wideresnet50",
    "layers": ["layer2", "layer3"],
    "input_shape": [3, 224, 224],
    "pretrain_embed_dimension": 1024,
    "target_embed_dimension": 1024,
    "patchsize": 3,
    "anomaly_score_num_nn": 1,
    "coreset": "approx_greedy",
    "coreset_percentage": 0.10,
    "faiss_metric": "squared_l2",
    "faiss_on_gpu": False,
}
_REQUIRED_PROJECT_ENVIRONMENT = (
    "TEMP",
    "TMP",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "PYTHONPYCACHEPREFIX",
)


ModelBuilder = Callable[["PatchCoreFitSettings"], tuple[Any, dict[str, Any]]]
CheckpointInspector = Callable[[Path, str], dict[str, Any]]


def _safe_category(category: str) -> bool:
    return bool(category) and Path(category).name == category and not any(
        separator in category for separator in ("/", "\\")
    )


def _validate_sha256(value: str, field_name: str) -> str:
    normalized = value.lower().strip()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a full 64-character SHA-256")
    return normalized


def registered_model_seeds(
    registration_path: str | Path = DEFAULT_REGISTRATION_PATH,
) -> tuple[int, ...]:
    """Read the already-frozen model seeds without consulting any test result."""

    path = Path(registration_path).resolve()
    with path.open(encoding="utf-8") as handle:
        registration = json.load(handle)
    values = registration.get("model_seeds")
    if (
        not isinstance(values, list)
        or len(values) != 3
        or not all(isinstance(seed, int) and seed >= 0 for seed in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("registration must contain three distinct non-negative model_seeds")
    return tuple(values)


def audit_project_local_environment(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Fail closed unless caches and temporary files resolve under the project."""

    root = Path(project_root).resolve()
    resolved: dict[str, str] = {}
    invalid: list[str] = []
    for name in _REQUIRED_PROJECT_ENVIRONMENT:
        value = os.environ.get(name)
        if not value:
            invalid.append(f"{name}=<missing>")
            continue
        path = Path(value).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            invalid.append(f"{name}={path}")
        resolved[name] = str(path)
    if invalid:
        raise RuntimeError(
            "PatchCore requires project-local cache/temp paths; run "
            "scripts/Set-ProjectEnvironment.ps1 first: " + ", ".join(invalid)
        )
    return resolved


@dataclass(frozen=True)
class PatchCoreFitSettings:
    dataset: Literal["mvtec_ad", "visa"]
    category: str
    dataset_root: Path
    output_dir: Path
    backbone_checkpoint: Path = DEFAULT_BACKBONE_CHECKPOINT
    backbone_sha256: str = WRN50_IMAGENET1K_V1_SHA256
    seed: int = 1887537248
    execution_scope: ExecutionScope = "full"
    batch_size: int = 8
    workers: int = 0
    device: str = "cpu"
    coreset_device: str = "cpu"
    faiss_workers: int = 8
    max_steps: int | None = None
    registration_path: Path = DEFAULT_REGISTRATION_PATH
    project_config_path: Path = DEFAULT_PROJECT_CONFIG
    development_protocol_path: Path = DEFAULT_DEVELOPMENT_PROTOCOL
    formal_input_manifest: Path = DEFAULT_FORMAL_INPUT_MANIFEST
    formal_source_scope: Path = DEFAULT_FORMAL_SOURCE_SCOPE
    run_purpose: RunPurpose | None = None
    geometry: PatchCoreGeometry = PatchCoreGeometry()

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
        if not _safe_category(self.category):
            raise ValueError("category must be one safe directory name")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.execution_scope not in {"full", "debug_limited"}:
            raise ValueError("execution_scope must be 'full' or 'debug_limited'")
        purpose = classify_run_purpose(
            self.resolved_run_purpose(), max_steps=self.max_steps
        )
        if purpose["execution_scope"] != self.execution_scope:
            raise ValueError("run_purpose and execution_scope disagree")
        if self.batch_size <= 0 or self.faiss_workers <= 0:
            raise ValueError("batch_size and faiss_workers must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if not Path(self.backbone_checkpoint).is_file():
            raise FileNotFoundError(self.backbone_checkpoint)
        _validate_sha256(self.backbone_sha256, "backbone_sha256")
        self.geometry.validate()
        if self.execution_scope == "full":
            if self.max_steps is not None:
                raise ValueError("max_steps is forbidden for a full PatchCore fit")
            if self.geometry != PatchCoreGeometry():
                raise ValueError("full PatchCore fit requires frozen 256->224 geometry")
            if self.seed not in registered_model_seeds(self.registration_path):
                raise ValueError(f"seed {self.seed} is not pre-registered")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "dataset_root",
            "output_dir",
            "backbone_checkpoint",
            "registration_path",
            "project_config_path",
            "development_protocol_path",
            "formal_input_manifest",
            "formal_source_scope",
        ):
            payload[name] = str(Path(payload[name]).resolve())
        payload["run_purpose"] = self.resolved_run_purpose()
        return payload


@dataclass(frozen=True)
class PatchCoreInferenceSettings:
    dataset: Literal["mvtec_ad", "visa"]
    category: str
    dataset_root: Path
    fitted_model_dir: Path
    output_dir: Path
    backbone_checkpoint: Path = DEFAULT_BACKBONE_CHECKPOINT
    backbone_sha256: str = WRN50_IMAGENET1K_V1_SHA256
    seed: int = 1887537248
    execution_scope: ExecutionScope = "full"
    batch_size: int = 8
    workers: int = 0
    device: str = "cpu"
    coreset_device: str = "cpu"
    faiss_workers: int = 8
    max_steps: int | None = None
    registration_path: Path = DEFAULT_REGISTRATION_PATH
    project_config_path: Path = DEFAULT_PROJECT_CONFIG
    formal_input_manifest: Path = DEFAULT_FORMAL_INPUT_MANIFEST
    run_purpose: RunPurpose | None = None
    checkpoint_manifest: Path | None = None
    formal_test_inventory_dir: Path | None = None
    geometry: PatchCoreGeometry = PatchCoreGeometry()

    def resolved_run_purpose(self) -> RunPurpose:
        if self.run_purpose is not None:
            return self.run_purpose
        return (
            "debug_limited"
            if self.execution_scope == "debug_limited"
            else "author_reproduction"
        )

    def validate(self) -> None:
        fit = PatchCoreFitSettings(
            dataset=self.dataset,
            category=self.category,
            dataset_root=self.dataset_root,
            output_dir=self.output_dir,
            backbone_checkpoint=self.backbone_checkpoint,
            backbone_sha256=self.backbone_sha256,
            seed=self.seed,
            execution_scope=self.execution_scope,
            batch_size=self.batch_size,
            workers=self.workers,
            device=self.device,
            coreset_device=self.coreset_device,
            faiss_workers=self.faiss_workers,
            max_steps=self.max_steps,
            registration_path=self.registration_path,
            project_config_path=self.project_config_path,
            formal_input_manifest=self.formal_input_manifest,
            run_purpose=self.resolved_run_purpose(),
            geometry=self.geometry,
        )
        fit.validate()
        if self.resolved_run_purpose() == "development_full":
            raise ValueError(
                "development_full must use a frozen development evaluator, not "
                "the formal test inference entry point"
            )
        if (
            self.resolved_run_purpose() != "formal_full"
            and self.formal_test_inventory_dir is not None
        ):
            raise ValueError(
                "formal_test_inventory_dir is only valid for formal_full inference"
            )
        model_dir = Path(self.fitted_model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "dataset_root",
            "fitted_model_dir",
            "output_dir",
            "backbone_checkpoint",
            "registration_path",
            "project_config_path",
            "formal_input_manifest",
        ):
            payload[name] = str(Path(payload[name]).resolve())
        payload["checkpoint_manifest"] = (
            None
            if self.checkpoint_manifest is None
            else str(Path(self.checkpoint_manifest).resolve())
        )
        payload["formal_test_inventory_dir"] = (
            None
            if self.formal_test_inventory_dir is None
            else str(Path(self.formal_test_inventory_dir).resolve())
        )
        payload["run_purpose"] = self.resolved_run_purpose()
        return payload


def patchcore_training_recipe(settings: PatchCoreFitSettings) -> dict[str, object]:
    """Canonical PatchCore fit recipe used by the frozen formal registry."""

    return {
        "schema_version": 2,
        "method": "PatchCore",
        "implementation": "project_safe_runner_over_frozen_author_source",
        "vendor_commit": PATCHCORE_AUTHOR_COMMIT,
        "backbone": {
            "name": "wide_resnet50_2_imagenet1k_v1",
            "sha256": settings.backbone_sha256.lower(),
        },
        "geometry": asdict(settings.geometry),
        "batch_size": settings.batch_size,
        "loader_shuffle": False,
        "loader_workers": settings.workers,
        "model_device_type": torch.device(settings.device).type,
        "coreset_device_type": torch.device(settings.coreset_device).type,
        "faiss_workers": settings.faiss_workers,
        "configuration": FROZEN_CONFIGURATION,
        "serialization": "float32_npy_plus_faiss_plus_hash_gated_json_no_pickle",
        "checkpoint_rule": "complete_train_good_memory_bank_no_test_selection",
    }


def patchcore_inference_recipe(
    settings: PatchCoreInferenceSettings,
) -> dict[str, object]:
    """Canonical prediction/export recipe used by the frozen formal registry."""

    return {
        "schema_version": 2,
        "method": "PatchCore",
        "vendor_commit": PATCHCORE_AUTHOR_COMMIT,
        "geometry": asdict(settings.geometry),
        "batch_size": settings.batch_size,
        "loader_workers": settings.workers,
        "model_device_type": torch.device(settings.device).type,
        "faiss_workers": settings.faiss_workers,
        "configuration": FROZEN_CONFIGURATION,
        "author_map": "author_output_after_gaussian_sigma_4",
        "evaluation_resize": "single_bilinear_align_corners_false",
        "common_image_score": "max_of_persisted_raw_anomaly_map",
        "method_native_image_score": "patchcore_author_patch_score_aggregation",
        "raw_map_normalization": "none",
    }


def _default_checkpoint_inspector(path: Path, expected_sha256: str) -> dict[str, Any]:
    return inspect_wrn50_checkpoint(path, expected_sha256=expected_sha256)


def _train_inventory(dataset: PatchCoreDataset, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_root = (root / dataset.category / "train" / "good").resolve()
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(dataset.image_paths):
        resolved = path.resolve()
        try:
            relative_to_allowed = resolved.relative_to(allowed_root).as_posix()
        except ValueError as error:
            raise RuntimeError(f"training path escaped train/good: {resolved}") from error
        size = resolved.stat().st_size
        if size <= 0:
            raise ValueError(f"normal training image is empty: {resolved}")
        rows.append(
            {
                "index": index,
                "relative_path": resolved.relative_to(root).as_posix(),
                "relative_to_train_good": relative_to_allowed,
                "size_bytes": size,
                "sha256": sha256_file(resolved),
            }
        )
    digest_input = "".join(
        f"{row['relative_path']}\0{row['size_bytes']}\0{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    import hashlib

    return rows, {
        "normal_training_images": len(rows),
        "normal_training_file_bytes": int(sum(row["size_bytes"] for row in rows)),
        "relative_path_size_and_content_sha256": hashlib.sha256(digest_input).hexdigest(),
        "allowed_split": "train/good only",
        "test_or_ground_truth_enumerated": False,
    }


def _project_root_from_config(path: Path) -> Path:
    resolved = path.resolve()
    return (
        resolved.parent.parent
        if resolved.parent.name == "configs"
        else PROJECT_ROOT
    )


def _resolve_training_dataset(
    settings: PatchCoreFitSettings,
) -> tuple[PatchCoreDataset, dict[str, Any]]:
    root = Path(settings.dataset_root).resolve()
    if settings.resolved_run_purpose() == "formal_full":
        project_root = _project_root_from_config(settings.project_config_path)
        inputs = resolve_frozen_formal_training_inputs(
            method="PatchCore",
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=root,
            bundle_manifest_path=settings.formal_input_manifest,
            project_config_path=settings.project_config_path,
            project_root=project_root,
        )
        return (
            PatchCoreDataset(
                settings.dataset,
                root,
                settings.category,
                split="train",
                geometry=settings.geometry,
                training_image_paths=inputs.normal_image_paths,
            ),
            inputs.evidence,
        )
    dataset = PatchCoreDataset(
        settings.dataset,
        root,
        settings.category,
        split="train",
        geometry=settings.geometry,
    )
    return dataset, {
        "schema_version": 1,
        "contract": "directory_discovery_nonformal_only",
        "method": "PatchCore",
        "dataset": settings.dataset,
        "category": settings.category,
        "manifest_backed": False,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "test_or_ground_truth_enumerated": False,
    }


def inspect_fit_inputs(
    settings: PatchCoreFitSettings,
    *,
    checkpoint_inspector: CheckpointInspector = _default_checkpoint_inspector,
    require_project_environment: bool = True,
) -> dict[str, Any]:
    """Validate only train/good, source, seed and local checkpoint; never fit."""

    settings.validate()
    environment = audit_project_local_environment() if require_project_environment else {}
    root = Path(settings.dataset_root).resolve()
    dataset, input_evidence = _resolve_training_dataset(settings)
    rows, inventory = _train_inventory(dataset, root)
    source = audit_patchcore_runtime()
    if not source["runtime_import_ready"]:
        raise RuntimeError("frozen PatchCore vendor runtime is not ready")
    checkpoint = checkpoint_inspector(
        Path(settings.backbone_checkpoint).resolve(), settings.backbone_sha256
    )
    run_purpose = settings.resolved_run_purpose()
    purpose = classify_run_purpose(run_purpose, max_steps=settings.max_steps)
    authority: dict[str, object] | None
    source_provenance: dict[str, Any] | None = None
    if run_purpose == "development_full":
        authority = validate_development_authority(
            method="PatchCore",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
            development_protocol_path=settings.development_protocol_path,
        )
    elif run_purpose == "formal_full":
        authority = validate_formal_authority(
            method="PatchCore",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            phase="training",
            recipe=patchcore_training_recipe(settings),
            training_input_evidence=input_evidence,
            formal_input_project_root=_project_root_from_config(
                settings.project_config_path
            ),
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
        )
        source_provenance = capture_formal_source_provenance(
            project_root=_project_root_from_config(settings.project_config_path),
            scope_path=settings.formal_source_scope,
        )
    else:
        authority = None
    return {
        "schema_version": 2,
        "method": "PatchCore",
        "dataset": settings.dataset,
        "category": settings.category,
        "execution_scope": settings.execution_scope,
        "run_purpose": run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        "formal_provenance": authority,
        "formal_source_provenance": source_provenance,
        "settings": settings.to_dict(),
        "frozen_configuration": FROZEN_CONFIGURATION,
        "registered_seeds": list(registered_model_seeds(settings.registration_path)),
        "input_inventory": inventory,
        "input_evidence": input_evidence,
        "train_manifest_rows": rows,
        "backbone_checkpoint": checkpoint,
        "vendor_source": source,
        "project_environment": environment,
        "fit_started": False,
        "test_split_accessed": False,
        "validated_at_utc": utc_now(),
    }


class _LimitedLoader(Iterable[dict[str, Any]]):
    def __init__(self, loader: DataLoader[dict[str, Any]], max_steps: int | None) -> None:
        self.loader = loader
        self.max_steps = max_steps

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for step, batch in enumerate(self.loader, start=1):
            if self.max_steps is not None and step > self.max_steps:
                break
            yield batch

    def __len__(self) -> int:
        return min(len(self.loader), self.max_steps) if self.max_steps is not None else len(self.loader)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _feature_bank_from_model(model: Any) -> np.ndarray:
    try:
        value = model.anomaly_scorer.detection_features
    except AttributeError as error:
        raise ValueError("PatchCore model has no fitted detection_features") from error
    features = np.asarray(value)
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError(f"PatchCore detection_features must be non-empty 2-D, got {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"PatchCore detection_features must be float32, got {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("PatchCore detection_features contain NaN or infinity")
    return np.ascontiguousarray(features)


def _index_from_model(model: Any) -> Any:
    try:
        index = model.anomaly_scorer.nn_method.search_index
    except AttributeError as error:
        raise ValueError("PatchCore model has no FAISS nearest-neighbour scorer") from error
    if index is None:
        raise ValueError("PatchCore FAISS index is not fitted")
    return index


def _verify_index_matches_features(index: Any, features: np.ndarray) -> dict[str, Any]:
    dimension = int(index.d)
    count = int(index.ntotal)
    if (count, dimension) != features.shape:
        raise ValueError(
            "FAISS index and memory-bank shape differ: "
            f"index={(count, dimension)}, features={features.shape}"
        )
    try:
        reconstructed = np.asarray(index.reconstruct_n(0, count), dtype=np.float32)
    except BaseException as error:
        raise ValueError("FAISS index does not support auditable vector reconstruction") from error
    if reconstructed.shape != features.shape or not np.array_equal(reconstructed, features):
        raise ValueError("FAISS index vectors differ from the persisted memory bank")
    return {
        "index_type": type(index).__name__,
        "dimension": dimension,
        "ntotal": count,
        "reconstruction_verified": True,
    }


def _remove_fitted_artifacts(model_dir: str | Path) -> list[str]:
    """Remove only the three final fitted-state files and their known temporaries."""

    supplied = Path(model_dir)
    if supplied.is_symlink():
        raise ValueError("PatchCore model directory cannot be a symbolic link")
    directory = supplied.resolve()
    removed: list[str] = []
    for name in (
        "manifest.json",
        "memory_bank.npy",
        "nnscorer_search_index.faiss",
        ".manifest.json.tmp",
        ".memory_bank.npy.tmp",
        ".nnscorer_search_index.faiss.tmp",
    ):
        path = directory / name
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(name)
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
    return removed


def _safe_fitted_file(directory: Path, value: object, *, role: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"PatchCore fitted artifact path is unsafe: {role}")
    candidate = directory / value
    if candidate.is_symlink():
        raise ValueError(f"PatchCore fitted artifact cannot be a symlink: {role}")
    resolved = candidate.resolve()
    if resolved.parent != directory:
        raise ValueError(f"PatchCore fitted artifact escaped model directory: {role}")
    return resolved


def save_fitted_patchcore_state(
    model: Any,
    model_dir: str | Path,
    *,
    context: Mapping[str, Any],
    backbone_checkpoint: str | Path,
) -> dict[str, Any]:
    """Persist fitted PatchCore state without pickle or implicit code loading."""

    supplied_directory = Path(model_dir)
    if supplied_directory.is_symlink():
        raise ValueError("PatchCore model directory cannot be a symbolic link")
    directory = supplied_directory.resolve()
    required = {
        "dataset",
        "category",
        "seed",
        "execution_scope",
        "run_purpose",
        "claim_scope",
        "formal_matrix_eligible",
        "backbone_sha256",
        "train_inventory_sha256",
        "train_image_count_total",
        "train_image_count_selected",
        "max_steps",
        "training_recipe",
        "formal_provenance",
        "formal_training_inputs",
        "formal_source_provenance",
        "vendor_source_audit",
    }
    if set(context) != required:
        raise ValueError(
            "PatchCore fitted-state context schema mismatch: "
            f"missing={sorted(required - set(context))}, "
            f"extra={sorted(set(context) - required)}"
        )
    purpose = classify_run_purpose(
        str(context["run_purpose"]),  # type: ignore[arg-type]
        max_steps=context["max_steps"],  # type: ignore[arg-type]
    )
    for key in ("execution_scope", "claim_scope", "formal_matrix_eligible"):
        if context[key] != purpose[key]:
            raise ValueError(f"PatchCore fitted-state purpose mismatch: {key}")
    formal = context["run_purpose"] == "formal_full"
    if formal:
        for key in (
            "formal_provenance",
            "formal_training_inputs",
            "formal_source_provenance",
        ):
            if not isinstance(context[key], Mapping):
                raise ValueError(f"formal PatchCore fitted state is missing {key}")
    elif context["formal_source_provenance"] is not None:
        raise ValueError("formal source provenance is forbidden outside formal_full")
    vendor = context["vendor_source_audit"]
    if (
        not isinstance(vendor, Mapping)
        or vendor.get("actual_commit") != PATCHCORE_AUTHOR_COMMIT
        or vendor.get("vendor_worktree_clean") is not True
    ):
        raise ValueError("PatchCore fitted state requires the frozen clean vendor source")
    checkpoint_path = Path(backbone_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != context["backbone_sha256"]:
        raise ValueError("PatchCore fitted-state backbone SHA-256 mismatch")

    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"model directory is not empty: {directory}")
    features = _feature_bank_from_model(model)
    index = _index_from_model(model)
    index_report = _verify_index_matches_features(index, features)
    memory_path = directory / "memory_bank.npy"
    index_path = directory / "nnscorer_search_index.faiss"
    manifest_path = directory / "manifest.json"
    temporary_index = index_path.with_name(f".{index_path.name}.tmp")
    try:
        _atomic_save_npy(memory_path, features)
        model.anomaly_scorer.nn_method.save(str(temporary_index))
        # Windows rejects FlushFileBuffers on a read-only descriptor.  Reopen
        # read/write after FAISS closes its writer, then durably flush it.
        with temporary_index.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_index, index_path)
        # Verify the exact on-disk index, not merely the in-memory source object.
        import faiss

        disk_index = faiss.read_index(str(index_path))
        disk_index_report = _verify_index_matches_features(disk_index, features)
        memory_entry = {
            "path": memory_path.name,
            "sha256": sha256_file(memory_path),
            "size_bytes": memory_path.stat().st_size,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "finite": True,
        }
        index_entry = {
            "path": index_path.name,
            "sha256": sha256_file(index_path),
            "size_bytes": index_path.stat().st_size,
            **index_report,
            "disk_readback": disk_index_report,
        }
        backbone_entry = {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "origin": "torchvision_imagenet1k_v1_local_hash_verified",
        }
        manifest = {
            "schema_version": 2,
            "artifact_type": "patchcore_safe_fitted_state",
            "method": "PatchCore",
            "dataset": context["dataset"],
            "category": context["category"],
            "seed": context["seed"],
            "run_purpose": context["run_purpose"],
            "execution_scope": context["execution_scope"],
            "claim_scope": context["claim_scope"],
            "formal_matrix_eligible": context["formal_matrix_eligible"],
            "checkpoint_origin": "project_training_run",
            "final_checkpoint_only": True,
            "test_selected": False,
            "vendor_commit": PATCHCORE_AUTHOR_COMMIT,
            "vendor_source_audit": dict(vendor),
            "configuration": FROZEN_CONFIGURATION,
            "training_recipe": context["training_recipe"],
            "formal_provenance": context["formal_provenance"],
            "formal_training_inputs": context["formal_training_inputs"],
            "formal_source_provenance": context["formal_source_provenance"],
            "serialization": {
                "pickle_used": False,
                "vendor_save_to_path_used": False,
                "vendor_load_from_path_required": False,
                "implicit_backbone_download_allowed": False,
            },
            "context": dict(context),
            "memory_bank": memory_entry,
            "faiss_index": index_entry,
            "checkpoints": {
                "memory_bank": memory_entry,
                "faiss_index": index_entry,
                "backbone": backbone_entry,
            },
            "external_hashes": {"backbone": checkpoint_sha},
            "created_at_utc": utc_now(),
        }
        write_json(manifest_path, manifest)
        return {**manifest, "manifest_sha256": sha256_file(manifest_path)}
    except BaseException:
        _remove_fitted_artifacts(directory)
        raise
    finally:
        if temporary_index.exists():
            temporary_index.unlink()


def load_fitted_patchcore_state(
    model: Any,
    model_dir: str | Path,
    *,
    backbone_checkpoint: str | Path,
    expected_context: Mapping[str, Any] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Hash-check and inject the project-owned memory bank and FAISS index."""

    supplied_directory = Path(model_dir)
    if supplied_directory.is_symlink():
        raise ValueError("PatchCore model directory cannot be a symbolic link")
    directory = supplied_directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 2 or manifest.get("artifact_type") != "patchcore_safe_fitted_state":
        raise ValueError("unsupported PatchCore fitted-state manifest")
    if manifest.get("vendor_commit") != PATCHCORE_AUTHOR_COMMIT:
        raise ValueError("fitted state uses a different PatchCore vendor commit")
    if manifest.get("configuration") != FROZEN_CONFIGURATION:
        raise ValueError("fitted state differs from the frozen PatchCore configuration")
    serialization = manifest.get("serialization")
    if not isinstance(serialization, dict) or serialization.get("pickle_used") is not False:
        raise ValueError("fitted state does not prove pickle-free serialization")
    context = manifest.get("context")
    if not isinstance(context, dict):
        raise ValueError("fitted state has no valid context")
    if expected_context is not None:
        mismatches = {
            key: {"expected": value, "actual": context.get(key)}
            for key, value in expected_context.items()
            if context.get(key) != value
        }
        if mismatches:
            raise ValueError(f"fitted-state context mismatch: {mismatches}")
    run_purpose = context.get("run_purpose")
    max_steps = context.get("max_steps")
    if not isinstance(run_purpose, str):
        raise ValueError("fitted state has no valid run_purpose")
    purpose = classify_run_purpose(run_purpose, max_steps=max_steps)  # type: ignore[arg-type]
    for key in ("execution_scope", "claim_scope", "formal_matrix_eligible"):
        if context.get(key) != purpose[key] or manifest.get(key) != purpose[key]:
            raise ValueError(f"fitted-state purpose mismatch: {key}")
    for key in ("dataset", "category", "seed", "run_purpose"):
        if manifest.get(key) != context.get(key):
            raise ValueError(f"fitted-state identity mismatch: {key}")

    current_vendor = audit_patchcore_runtime()
    recorded_vendor = manifest.get("vendor_source_audit")
    if (
        not isinstance(recorded_vendor, Mapping)
        or recorded_vendor.get("actual_commit") != PATCHCORE_AUTHOR_COMMIT
        or recorded_vendor.get("vendor_worktree_clean") is not True
        or current_vendor.get("actual_commit") != PATCHCORE_AUTHOR_COMMIT
        or current_vendor.get("vendor_worktree_clean") is not True
    ):
        raise ValueError("PatchCore vendor source provenance is no longer valid")
    source_validation: dict[str, Any] | None = None
    if run_purpose == "formal_full":
        source = manifest.get("formal_source_provenance")
        if not isinstance(source, Mapping):
            raise ValueError("formal PatchCore fitted state lacks source provenance")
        source_validation = validate_formal_source_provenance(
            source, project_root=project_root
        )

    memory = manifest.get("memory_bank")
    faiss_info = manifest.get("faiss_index")
    if not isinstance(memory, dict) or not isinstance(faiss_info, dict):
        raise ValueError("fitted state is missing memory-bank or FAISS metadata")
    memory_path = _safe_fitted_file(directory, memory.get("path"), role="memory_bank")
    index_path = _safe_fitted_file(directory, faiss_info.get("path"), role="faiss_index")
    for path, info in ((memory_path, memory), (index_path, faiss_info)):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != info.get("sha256"):
            raise ValueError(f"PatchCore fitted artifact SHA-256 mismatch: {path.name}")
        if info.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"PatchCore fitted artifact size mismatch: {path.name}")
    checkpoint_entries = manifest.get("checkpoints")
    if not isinstance(checkpoint_entries, Mapping) or set(checkpoint_entries) != {
        "memory_bank",
        "faiss_index",
        "backbone",
    }:
        raise ValueError("PatchCore fitted-state checkpoint role set mismatch")
    if checkpoint_entries.get("memory_bank") != memory or checkpoint_entries.get("faiss_index") != faiss_info:
        raise ValueError("PatchCore fitted-state checkpoint metadata mismatch")
    backbone_entry = checkpoint_entries.get("backbone")
    if not isinstance(backbone_entry, Mapping):
        raise ValueError("PatchCore fitted state is missing backbone metadata")
    checkpoint_path = Path(backbone_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if (
        backbone_entry.get("path") != checkpoint_path.name
        or backbone_entry.get("sha256") != sha256_file(checkpoint_path)
        or backbone_entry.get("size_bytes") != checkpoint_path.stat().st_size
        or manifest.get("external_hashes")
        != {"backbone": backbone_entry.get("sha256")}
    ):
        raise ValueError("PatchCore fitted-state backbone provenance mismatch")
    features = np.load(memory_path, allow_pickle=False)
    if features.dtype != np.float32 or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("persisted PatchCore memory bank must be finite float32 2-D")
    if list(features.shape) != memory.get("shape") or memory.get("dtype") != "float32":
        raise ValueError("memory-bank array metadata mismatch")
    model.anomaly_scorer.nn_method.load(str(index_path))
    model.anomaly_scorer.detection_features = features
    index_report = _verify_index_matches_features(_index_from_model(model), features)
    return {
        "schema_version": 2,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "memory_bank_sha256": memory["sha256"],
        "faiss_index_sha256": faiss_info["sha256"],
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "faiss": index_report,
        "pickle_loaded": False,
        "vendor_load_from_path_called": False,
        "context": context,
        "formal_source_validation": source_validation,
    }


def _default_model_builder(settings: PatchCoreFitSettings) -> tuple[Any, dict[str, Any]]:
    return build_patchcore_model_from_local_checkpoint(
        settings.backbone_checkpoint,
        expected_sha256=settings.backbone_sha256,
        device=settings.device,
        coreset_device=settings.coreset_device,
        faiss_workers=settings.faiss_workers,
    )


def run_patchcore_fit(
    settings: PatchCoreFitSettings,
    *,
    model_builder: ModelBuilder = _default_model_builder,
    checkpoint_inspector: CheckpointInspector = _default_checkpoint_inspector,
    require_project_environment: bool = True,
) -> dict[str, Any]:
    """Fit one category from train/good and persist a safe memory-bank artifact."""

    inspection = inspect_fit_inputs(
        settings,
        checkpoint_inspector=checkpoint_inspector,
        require_project_environment=require_project_environment,
    )
    run_purpose = settings.resolved_run_purpose()
    purpose = classify_run_purpose(run_purpose, max_steps=settings.max_steps)
    configure_reproducibility(settings.seed, deterministic=True)
    output = prepare_output_directory(settings.output_dir)
    events_path = output / "events.jsonl"
    all_rows = inspection.pop("train_manifest_rows")
    selected_count = len(all_rows)
    if settings.max_steps is not None:
        selected_count = min(selected_count, settings.max_steps * settings.batch_size)
    manifest_rows = [
        {**row, "selected_for_fit": row["index"] < selected_count} for row in all_rows
    ]
    write_jsonl(output / "inputs" / "train_manifest.jsonl", manifest_rows)
    run_config = {
        "schema_version": 2,
        "method": "PatchCore",
        "implementation": "project_safe_runner_over_frozen_author_source",
        "vendor_commit": PATCHCORE_AUTHOR_COMMIT,
        "execution_scope": settings.execution_scope,
        "run_purpose": run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        "formal_provenance": inspection["formal_provenance"],
        "formal_source_provenance": inspection["formal_source_provenance"],
        "formal_matrix_complete": False,
        "settings": settings.to_dict(),
        "frozen_configuration": FROZEN_CONFIGURATION,
        "input_inventory": inspection["input_inventory"],
        "input_evidence": inspection["input_evidence"],
        "selected_for_fit": selected_count,
        "test_split_accessed": False,
        "serialization_contract": "float32_npy_plus_faiss_plus_hash_gated_json_no_pickle",
        "created_at_utc": utc_now(),
    }
    write_json(output / "run_config.json", run_config)
    write_json(
        output / "run_state.json",
        {
            "status": "running",
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "formal_matrix_eligible": purpose["formal_matrix_eligible"],
            "test_split_accessed": False,
            "updated_at_utc": utc_now(),
        },
    )
    append_jsonl(events_path, {"event": "fit_started", "at_utc": utc_now()})
    try:
        root = Path(settings.dataset_root).resolve()
        selected_training_paths = [
            root / Path(row["relative_path"]) for row in all_rows
        ]
        train_dataset = PatchCoreDataset(
            settings.dataset,
            root,
            settings.category,
            split="train",
            geometry=settings.geometry,
            training_image_paths=selected_training_paths,
        )
        loader = build_patchcore_loader(
            train_dataset,
            batch_size=settings.batch_size,
            workers=settings.workers,
            seed=settings.seed,
            pin_memory=torch.device(settings.device).type == "cuda",
        )
        limited_loader = _LimitedLoader(loader, settings.max_steps)
        if len(limited_loader) == 0:
            raise RuntimeError("PatchCore fit has no train/good batches")
        model, build_report = model_builder(settings)
        append_jsonl(
            events_path,
            {
                "event": "memory_bank_fit_started",
                "planned_batches": len(limited_loader),
                "at_utc": utc_now(),
            },
        )
        model.fit(limited_loader)
        context = {
            "dataset": settings.dataset,
            "category": settings.category,
            "seed": settings.seed,
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "formal_matrix_eligible": purpose["formal_matrix_eligible"],
            "backbone_sha256": settings.backbone_sha256.lower(),
            "train_inventory_sha256": inspection["input_inventory"][
                "relative_path_size_and_content_sha256"
            ],
            "train_image_count_total": len(manifest_rows),
            "train_image_count_selected": selected_count,
            "max_steps": settings.max_steps,
            "training_recipe": patchcore_training_recipe(settings),
            "formal_provenance": inspection["formal_provenance"],
            "formal_training_inputs": (
                inspection["input_evidence"] if run_purpose == "formal_full" else None
            ),
            "formal_source_provenance": inspection["formal_source_provenance"],
            "vendor_source_audit": inspection["vendor_source"],
        }
        model_manifest = save_fitted_patchcore_state(
            model,
            output / "model",
            context=context,
            backbone_checkpoint=settings.backbone_checkpoint,
        )
        run_config["model_build"] = build_report
        run_config["backbone_checkpoint"] = inspection["backbone_checkpoint"]
        run_config["vendor_source"] = inspection["vendor_source"]
        run_config["model_artifact"] = {
            "manifest": "model/manifest.json",
            "manifest_sha256": model_manifest["manifest_sha256"],
        }
        write_json(output / "run_config.json", run_config)
        final_state = {
            "status": "completed" if settings.execution_scope == "full" else "debug_completed",
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "formal_matrix_eligible": purpose["formal_matrix_eligible"],
            "memory_bank_fitted": True,
            "fit_batches": len(limited_loader),
            "train_images_selected": selected_count,
            "test_split_accessed": False,
            "model_manifest_sha256": model_manifest["manifest_sha256"],
            "run_config_sha256": sha256_file(output / "run_config.json"),
            "train_manifest_sha256": sha256_file(
                output / "inputs" / "train_manifest.jsonl"
            ),
            "standalone_performance_claim_eligible": False,
            "updated_at_utc": utc_now(),
        }
        append_jsonl(events_path, {"event": "fit_completed", **final_state})
        write_json(output / "run_state.json", final_state)
        return final_state
    except BaseException as error:
        removed_artifacts = _remove_fitted_artifacts(output / "model")
        write_json(
            output / "run_state.json",
            {
                "status": "failed",
                "execution_scope": settings.execution_scope,
                "run_purpose": run_purpose,
                "claim_scope": purpose["claim_scope"],
                "formal_matrix_eligible": False,
                "test_split_accessed": False,
                "removed_final_fitted_artifacts": removed_artifacts,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at_utc": utc_now(),
            },
        )
        append_jsonl(
            events_path,
            {
                "event": "fit_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "at_utc": utc_now(),
            },
        )
        raise


def _fit_settings_from_inference(settings: PatchCoreInferenceSettings) -> PatchCoreFitSettings:
    return PatchCoreFitSettings(
        dataset=settings.dataset,
        category=settings.category,
        dataset_root=settings.dataset_root,
        output_dir=settings.output_dir,
        backbone_checkpoint=settings.backbone_checkpoint,
        backbone_sha256=settings.backbone_sha256,
        seed=settings.seed,
        execution_scope=settings.execution_scope,
        batch_size=settings.batch_size,
        workers=settings.workers,
        device=settings.device,
        coreset_device=settings.coreset_device,
        faiss_workers=settings.faiss_workers,
        max_steps=settings.max_steps,
        registration_path=settings.registration_path,
        project_config_path=settings.project_config_path,
        formal_input_manifest=settings.formal_input_manifest,
        run_purpose=settings.resolved_run_purpose(),
        geometry=settings.geometry,
    )


def inspect_inference_inputs(
    settings: PatchCoreInferenceSettings,
    *,
    model_builder: ModelBuilder = _default_model_builder,
    require_project_environment: bool = True,
    return_formal_test_selection: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Verify fitted inputs; formal_full also revalidates its frozen test inventory."""

    settings.validate()
    environment = audit_project_local_environment() if require_project_environment else {}
    run_purpose = settings.resolved_run_purpose()
    purpose = classify_run_purpose(run_purpose, max_steps=settings.max_steps)
    formal_provenance: dict[str, object] | None = None
    checkpoint_validation: dict[str, object] | None = None
    formal_test_selection: dict[str, Any] | None = None
    formal_test_evidence: dict[str, Any] | None = None
    if run_purpose == "formal_full":
        inference_recipe = patchcore_inference_recipe(settings)
        inference_authority = validate_formal_authority(
            method="PatchCore",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            phase="inference",
            recipe=inference_recipe,
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
        )
        manifest_path = settings.checkpoint_manifest or (
            Path(settings.fitted_model_dir).resolve() / "manifest.json"
        )
        checkpoint_validation = validate_formal_checkpoint_manifest(
            manifest_path,
            method="PatchCore",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            checkpoint_files={
                "memory_bank": Path(settings.fitted_model_dir).resolve()
                / "memory_bank.npy",
                "faiss_index": Path(settings.fitted_model_dir).resolve()
                / "nnscorer_search_index.faiss",
                "backbone": settings.backbone_checkpoint,
            },
            external_hashes={"backbone": settings.backbone_sha256.lower()},
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
        )
        if settings.formal_test_inventory_dir is None:
            raise ValueError(
                "formal_full PatchCore inference requires formal_test_inventory_dir"
            )
        formal_test_selection = resolve_formal_test_inventory_category(
            output_dir=settings.formal_test_inventory_dir,
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=settings.dataset_root,
            project_root=_project_root_from_config(settings.project_config_path),
        )
        formal_test_evidence = formal_test_selection["evidence"]
        formal_provenance = {
            "claim_scope": "formal_matrix_candidate",
            "inference_recipe": inference_recipe,
            "inference_authority": inference_authority,
            "checkpoint_validation": checkpoint_validation,
            "formal_test_inventory": formal_test_evidence,
        }
    fit_settings = _fit_settings_from_inference(settings)
    model, build_report = model_builder(fit_settings)
    load_report = load_fitted_patchcore_state(
        model,
        settings.fitted_model_dir,
        backbone_checkpoint=settings.backbone_checkpoint,
        expected_context={
            "dataset": settings.dataset,
            "category": settings.category,
            "seed": settings.seed,
            "backbone_sha256": settings.backbone_sha256.lower(),
        },
        project_root=_project_root_from_config(settings.project_config_path),
    )
    fitted_scope = load_report["context"].get("execution_scope")
    if fitted_scope not in {"full", "debug_limited"}:
        raise ValueError("fitted state has no valid execution_scope")
    if settings.execution_scope == "full" and fitted_scope != "full":
        raise ValueError(
            "a debug_limited fitted state cannot be used for full inference"
        )
    fitted_purpose = load_report["context"].get("run_purpose")
    if run_purpose == "formal_full" and fitted_purpose != "formal_full":
        raise ValueError("formal_full inference requires a formal_full fitted state")
    if (
        settings.execution_scope == "full"
        and run_purpose != "formal_full"
        and fitted_purpose != run_purpose
    ):
        raise ValueError("full PatchCore inference and fitted-state run_purpose disagree")
    if checkpoint_validation is not None and (
        checkpoint_validation.get("manifest_sha256")
        != load_report["manifest_sha256"]
    ):
        raise ValueError("formal checkpoint validation differs from loaded fitted state")
    report: dict[str, Any] = {
        "schema_version": 2,
        "method": "PatchCore",
        "dataset": settings.dataset,
        "category": settings.category,
        "execution_scope": settings.execution_scope,
        "run_purpose": run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        "formal_provenance": formal_provenance,
        "formal_test_inventory": formal_test_evidence,
        "settings": settings.to_dict(),
        "frozen_configuration": FROZEN_CONFIGURATION,
        "model_build": build_report,
        "fitted_state": load_report,
        "vendor_source": audit_patchcore_runtime(),
        "project_environment": environment,
        "test_manifest_enumerated": formal_test_selection is not None,
        "formal_test_inventory_validated": formal_test_selection is not None,
        "test_dataset_directory_discovery_used": False
        if formal_test_selection is not None
        else None,
        "inference_started": False,
        "validated_at_utc": utc_now(),
    }
    if return_formal_test_selection:
        report["_formal_test_selection"] = formal_test_selection
    return model, report


def run_patchcore_inference(
    settings: PatchCoreInferenceSettings,
    *,
    model_builder: ModelBuilder = _default_model_builder,
    require_project_environment: bool = True,
) -> dict[str, Any]:
    """Load the safe fitted state, run author prediction, then use the export contract."""

    model, inspection = inspect_inference_inputs(
        settings,
        model_builder=model_builder,
        require_project_environment=require_project_environment,
        return_formal_test_selection=True,
    )
    formal_test_selection = inspection.pop("_formal_test_selection")
    configure_reproducibility(settings.seed, deterministic=True)
    dataset = PatchCoreDataset(
        settings.dataset,
        settings.dataset_root,
        settings.category,
        split="test",
        geometry=settings.geometry,
        samples=(
            None
            if formal_test_selection is None
            else formal_test_selection["samples"]
        ),
    )
    loader = build_patchcore_loader(
        dataset,
        batch_size=settings.batch_size,
        workers=settings.workers,
        seed=settings.seed,
        pin_memory=torch.device(settings.device).type == "cuda",
    )
    limited_loader = _LimitedLoader(loader, settings.max_steps)
    author_scores: list[float] = []
    author_maps: list[np.ndarray] = []
    for batch in limited_loader:
        scores, maps = model.predict(batch["image"])
        author_scores.extend(float(value) for value in scores)
        author_maps.extend(np.asarray(value) for value in maps)
    if not author_maps:
        raise RuntimeError("PatchCore inference produced no predictions")
    sample_indices: tuple[int, ...] | None = None
    if len(author_maps) != len(dataset):
        if settings.execution_scope != "debug_limited":
            raise RuntimeError("full PatchCore inference did not cover the complete category")
        sample_indices = tuple(range(len(author_maps)))
    provenance = {
        "kind": "patchcore_safe_fitted_state",
        "fit_manifest": inspection["fitted_state"]["manifest_path"],
        "fit_manifest_sha256": inspection["fitted_state"]["manifest_sha256"],
        "memory_bank_sha256": inspection["fitted_state"]["memory_bank_sha256"],
        "faiss_index_sha256": inspection["fitted_state"]["faiss_index_sha256"],
        "checkpoint_sha256": settings.backbone_sha256.lower(),
        "pickle_loaded": False,
        "implicit_download_used": False,
        "run_purpose": settings.resolved_run_purpose(),
    }
    return export_patchcore_predictions(
        PatchCoreExportSettings(
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=Path(settings.dataset_root),
            output_dir=Path(settings.output_dir),
            seed=settings.seed,
            execution_scope=settings.execution_scope,
            model_provenance=provenance,
            run_purpose=settings.resolved_run_purpose(),
            formal_provenance=inspection["formal_provenance"],
            geometry=settings.geometry,
            sample_indices=sample_indices,
        ),
        author_maps=author_maps,
        author_image_scores=author_scores,
        formal_test_selection=formal_test_selection,
    )
