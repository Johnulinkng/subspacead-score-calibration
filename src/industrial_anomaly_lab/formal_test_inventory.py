"""F0-only, hash-bound inventories for formal MVTec AD and VisA test sets.

The builder is deliberately unavailable during model development.  It first
checks that the formal protocol and method registry have been frozen, then
enumerates the complete official test split and decodes masks only to derive
the pre-registered anomaly-size strata used by the visualization protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .data_protocols import TestSample, build_test_manifest
from .run_artifacts import sha256_file, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "configs" / "project.json"
DEFAULT_REGISTRATION = PROJECT_ROOT / "configs" / "preregistered_experiments.json"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "formal_test_inventory_protocol.json"
INVENTORY_FILENAME = "inventory.json"
RECEIPT_FILENAME = "receipt.json"
SUPPORTED_DATASETS = frozenset({"mvtec_ad", "visa"})
FORMAL_TEST_SELECTION_CONTRACT = "formal_test_inventory_category_selection_v1"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise FileNotFoundError(f"{label} is missing or linked: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing or linked: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return resolved, payload


def _relative_to_root(path: str | Path, root: str | Path, label: str) -> str:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must stay below project root: {resolved}") from error
    if not relative.parts:
        raise ValueError(f"{label} cannot be the project root")
    return PurePosixPath(*relative.parts).as_posix()


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{label} is not a safe relative path: {value!r}")
    return candidate.as_posix()


def _resolve_project_file(root: Path, relative: object, label: str) -> Path:
    safe = _safe_relative_path(relative, label)
    candidate = root / Path(*PurePosixPath(safe).parts)
    if candidate.is_symlink():
        raise FileNotFoundError(f"{label} is missing or linked: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes project root") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing or linked: {resolved}")
    return resolved


def _resolve_dataset_file(root: Path, path: Path, label: str) -> tuple[Path, str]:
    if path.is_symlink():
        raise FileNotFoundError(f"{label} is missing or linked: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes dataset root: {resolved}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing or linked: {resolved}")
    return resolved, PurePosixPath(*relative.parts).as_posix()


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _load_protocol(path: str | Path) -> tuple[Path, dict[str, Any], float, float]:
    resolved, protocol = _load_json_object(path, "formal test inventory protocol")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported formal test inventory protocol schema")
    if protocol.get("kind") != "formal_test_inventory_protocol":
        raise ValueError("unexpected formal test inventory protocol kind")
    if protocol.get("status") != "frozen_before_formal_test_enumeration":
        raise ValueError("formal test inventory protocol is not frozen")
    if protocol.get("datasets") != ["mvtec_ad", "visa"]:
        raise ValueError("formal test inventory protocol dataset scope changed")
    mask = protocol.get("mask_interpretation")
    if not isinstance(mask, Mapping) or mask.get("positive_rule") != "pixel_value_gt_0":
        raise ValueError("formal mask interpretation is missing or unsupported")
    if mask.get("area_denominator") != "all_native_mask_pixels":
        raise ValueError("formal anomaly-area denominator is missing or unsupported")
    thresholds = protocol.get("anomaly_scale_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("formal anomaly-scale thresholds are missing")
    small = thresholds.get("small_max_fraction")
    medium = thresholds.get("medium_max_fraction")
    if (
        isinstance(small, bool)
        or isinstance(medium, bool)
        or not isinstance(small, (int, float))
        or not isinstance(medium, (int, float))
        or not math.isfinite(float(small))
        or not math.isfinite(float(medium))
        or not (0.0 < float(small) < float(medium) < 1.0)
    ):
        raise ValueError("formal anomaly-scale thresholds must satisfy 0 < small < medium < 1")
    return resolved, protocol, float(small), float(medium)


def _validate_registry(registration: Mapping[str, Any]) -> None:
    registry = registration.get("formal_method_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("formal method registry is missing")
    if registry.get("status") != "frozen_before_formal_evaluation":
        raise ValueError("formal method registry is not frozen before F0")
    methods = registry.get("methods")
    baseline = registry.get("baseline")
    candidate = registry.get("candidate")
    recipes = registry.get("recipes")
    if (
        not isinstance(methods, list)
        or len(methods) < 2
        or any(not isinstance(method, str) or not method for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("formal method list must contain at least two unique methods")
    if not isinstance(baseline, str) or baseline not in methods:
        raise ValueError("formal baseline is not frozen in the method list")
    if not isinstance(candidate, str) or candidate not in methods or candidate == baseline:
        raise ValueError("formal candidate is not frozen distinctly from the baseline")
    if not isinstance(recipes, Mapping) or set(recipes) != set(methods):
        raise ValueError("formal recipes must exactly cover the frozen method list")
    for method in methods:
        recipe = recipes.get(method)
        if (
            not isinstance(recipe, Mapping)
            or not isinstance(recipe.get("recipe_id"), str)
            or not recipe.get("recipe_id")
            or not isinstance(recipe.get("training"), Mapping)
            or not isinstance(recipe.get("inference"), Mapping)
        ):
            raise ValueError(f"formal recipe is incomplete for method: {method}")


def _load_authority(
    *,
    dataset: str,
    project_root: str | Path,
    project_config_path: str | Path,
    registration_path: str | Path,
    protocol_path: str | Path,
    prepared_manifest_path: str | Path | None,
) -> dict[str, Any]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    project_path, project = _load_json_object(project_config_path, "project config")
    registration_resolved, registration = _load_json_object(
        registration_path, "experiment registration"
    )
    protocol_resolved, protocol, small, medium = _load_protocol(protocol_path)
    for path, label in (
        (project_path, "project config"),
        (registration_resolved, "experiment registration"),
        (protocol_resolved, "formal test inventory protocol"),
    ):
        _relative_to_root(path, root, label)

    evaluation = project.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("formal_protocol_frozen") is not True:
        raise ValueError("formal evaluation protocol is not frozen")
    datasets = project.get("datasets")
    dataset_config = datasets.get(dataset) if isinstance(datasets, Mapping) else None
    if not isinstance(dataset_config, Mapping):
        raise ValueError(f"dataset is not registered for formal evaluation: {dataset}")
    categories = dataset_config.get("categories")
    if (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(category, str) or not category for category in categories)
        or len(set(categories)) != len(categories)
    ):
        raise ValueError(f"registered categories are invalid for {dataset}")
    configured_manifest = dataset_config.get("prepared_manifest")
    if not isinstance(configured_manifest, str) or not configured_manifest:
        raise ValueError(f"prepared_manifest is not registered for {dataset}")
    expected_manifest = _resolve_project_file(root, configured_manifest, "prepared manifest")
    observed_manifest = (
        expected_manifest
        if prepared_manifest_path is None
        else Path(prepared_manifest_path).resolve()
    )
    if observed_manifest != expected_manifest:
        raise ValueError("prepared manifest path differs from project registration")
    prepared_path, prepared = _load_json_object(observed_manifest, "prepared manifest")
    _relative_to_root(prepared_path, root, "prepared manifest")
    _validate_registry(registration)
    return {
        "project_root": root,
        "project_path": project_path,
        "project": project,
        "registration_path": registration_resolved,
        "registration": registration,
        "protocol_path": protocol_resolved,
        "protocol": protocol,
        "prepared_path": prepared_path,
        "prepared": prepared,
        "categories": list(categories),
        "small_max": small,
        "medium_max": medium,
    }


def _validate_prepared_manifest(
    *,
    dataset: str,
    dataset_root: Path,
    project_root: Path,
    categories: list[str],
    prepared: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    expected_name = "MVTec AD" if dataset == "mvtec_ad" else "VisA"
    if prepared.get("dataset") != expected_name:
        raise ValueError("prepared manifest dataset identity mismatch")
    expected: dict[str, dict[str, int]] = {}
    if dataset == "mvtec_ad":
        layout = prepared.get("layout")
        category_rows = layout.get("categories") if isinstance(layout, Mapping) else None
        if not isinstance(category_rows, Mapping) or set(category_rows) != set(categories):
            raise ValueError("MVTec prepared-manifest categories differ from registration")
        manifest_root = layout.get("root")
        if not isinstance(manifest_root, str):
            raise ValueError("MVTec prepared-manifest root is missing")
        recorded_root = (project_root / manifest_root).resolve()
        if recorded_root != dataset_root.resolve():
            raise ValueError("MVTec prepared-manifest root differs from dataset root")
        for category in categories:
            row = category_rows.get(category)
            if not isinstance(row, Mapping):
                raise ValueError(f"MVTec prepared row is missing for {category}")
            expected[category] = {
                "normal": int(row.get("test_good_images", -1)),
                "anomaly": int(row.get("test_anomaly_images", -1)),
                "masks": int(row.get("ground_truth_masks", -1)),
            }
    else:
        prepared_section = prepared.get("prepared")
        recorded = prepared_section.get("root") if isinstance(prepared_section, Mapping) else None
        if not isinstance(recorded, str):
            raise ValueError("VisA prepared-manifest root is missing")
        recorded_path = Path(recorded)
        if not recorded_path.is_absolute():
            recorded_path = project_root / recorded_path
        if recorded_path.resolve() != dataset_root.resolve():
            raise ValueError("VisA prepared-manifest root differs from dataset root")
        split = prepared.get("split")
        partitions = split.get("partition_counts") if isinstance(split, Mapping) else None
        if not isinstance(partitions, Mapping) or set(partitions) != set(categories):
            raise ValueError("VisA prepared-manifest categories differ from registration")
        for category in categories:
            row = partitions.get(category)
            if not isinstance(row, Mapping):
                raise ValueError(f"VisA prepared row is missing for {category}")
            anomaly = int(row.get("test_anomaly", -1))
            expected[category] = {
                "normal": int(row.get("test_normal", -1)),
                "anomaly": anomaly,
                "masks": anomaly,
            }
    if any(value < 0 for row in expected.values() for value in row.values()):
        raise ValueError("prepared manifest contains invalid formal-test counts")
    return expected


def _scale_for_area(area: float, *, small_max: float, medium_max: float) -> str:
    if area <= small_max:
        return "small"
    if area <= medium_max:
        return "medium"
    return "large"


def _construct_inventory(
    *,
    dataset: str,
    dataset_root: Path,
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    root = dataset_root.resolve()
    project_root = Path(authority["project_root"])
    try:
        root.relative_to(project_root)
    except ValueError as error:
        raise ValueError("dataset root must stay below project root") from error
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"dataset root is missing or linked: {root}")
    categories = list(authority["categories"])
    actual_category_dirs = {
        path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if actual_category_dirs != set(categories):
        missing = sorted(set(categories) - actual_category_dirs)
        extra = sorted(actual_category_dirs - set(categories))
        raise ValueError(f"dataset category directories differ: missing={missing}, extra={extra}")

    expected_counts = _validate_prepared_manifest(
        dataset=dataset,
        dataset_root=root,
        project_root=project_root,
        categories=categories,
        prepared=authority["prepared"],
    )
    entries: list[dict[str, Any]] = []
    observed: dict[str, dict[str, int]] = {}
    for category in categories:
        samples = build_test_manifest(dataset, root, category)
        counts: Counter[str] = Counter()
        for sample in samples:
            image_path, image_relative = _resolve_dataset_file(
                root, sample.image_path, f"{sample.sample_id} source image"
            )
            if sample.label == 0:
                area = 0.0
                scale = "normal"
                mask_relative: str | None = None
                mask_sha: str | None = None
                counts["normal"] += 1
            else:
                if sample.mask_path is None:
                    raise ValueError(f"anomalous sample has no source mask: {sample.sample_id}")
                mask_path, mask_relative = _resolve_dataset_file(
                    root, sample.mask_path, f"{sample.sample_id} source mask"
                )
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.ndim != 2 or mask.size == 0:
                    raise ValueError(f"could not decode source mask: {mask_path}")
                unique = set(int(value) for value in np.unique(mask))
                if not unique.issubset({0, 255}) or 255 not in unique:
                    raise ValueError(f"source mask must be non-empty binary 0/255: {mask_path}")
                area = float((mask > 0).sum() / mask.size)
                if not (0.0 < area <= 1.0):
                    raise ValueError(f"invalid anomaly area for {sample.sample_id}")
                scale = _scale_for_area(
                    area,
                    small_max=float(authority["small_max"]),
                    medium_max=float(authority["medium_max"]),
                )
                mask_sha = sha256_file(mask_path)
                counts["anomaly"] += 1
                counts["masks"] += 1
            entries.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "defect_type": sample.defect_type,
                    "anomaly_area_fraction": area,
                    "anomaly_scale": scale,
                    "source_image": image_relative,
                    "source_image_sha256": sha256_file(image_path),
                    "source_mask": mask_relative,
                    "source_mask_sha256": mask_sha,
                }
            )
        observed[category] = {
            "normal": counts["normal"],
            "anomaly": counts["anomaly"],
            "masks": counts["masks"],
        }
        if observed[category] != expected_counts[category]:
            raise ValueError(
                f"formal-test counts differ for {category}: "
                f"expected={expected_counts[category]}, observed={observed[category]}"
            )

    sample_ids = [entry["sample_id"] for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("formal test inventory contains duplicate sample IDs")
    provenance = {
        "dataset": dataset,
        "project_config_sha256": sha256_file(authority["project_path"]),
        "registration_sha256": sha256_file(authority["registration_path"]),
        "protocol_sha256": sha256_file(authority["protocol_path"]),
        "prepared_manifest_sha256": sha256_file(authority["prepared_path"]),
        "entries": entries,
    }
    inventory_id = f"{dataset}-test-{_canonical_sha256(provenance)[:20]}"
    return (
        {
            "schema_version": 1,
            "kind": "formal_test_inventory",
            "inventory_id": inventory_id,
            "dataset": dataset,
            "split": "test",
            "entries": entries,
        },
        observed,
    )


def build_formal_test_inventory_bundle(
    *,
    dataset: str,
    dataset_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    registration_path: str | Path = DEFAULT_REGISTRATION,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    prepared_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the complete F0 inventory after all formal freeze gates pass."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    _relative_to_root(output, root, "formal test inventory output")
    if Path(output_dir).is_symlink():
        raise ValueError("formal test inventory output cannot be a symlink")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output}")
    authority = _load_authority(
        dataset=dataset,
        project_root=root,
        project_config_path=project_config_path,
        registration_path=registration_path,
        protocol_path=protocol_path,
        prepared_manifest_path=prepared_manifest_path,
    )
    inventory, counts = _construct_inventory(
        dataset=dataset,
        dataset_root=Path(dataset_root),
        authority=authority,
    )
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / INVENTORY_FILENAME
    write_json(inventory_path, inventory)
    receipt = {
        "schema_version": 1,
        "kind": "formal_test_inventory_receipt",
        "contract": "formal_test_inventory_v1",
        "dataset": dataset,
        "split": "test",
        "inventory_id": inventory["inventory_id"],
        "inventory": INVENTORY_FILENAME,
        "inventory_sha256": sha256_file(inventory_path),
        "dataset_root": _relative_to_root(dataset_root, root, "dataset root"),
        "project_config": _relative_to_root(authority["project_path"], root, "project config"),
        "project_config_sha256": sha256_file(authority["project_path"]),
        "registration": _relative_to_root(
            authority["registration_path"], root, "experiment registration"
        ),
        "registration_sha256": sha256_file(authority["registration_path"]),
        "protocol": _relative_to_root(authority["protocol_path"], root, "inventory protocol"),
        "protocol_sha256": sha256_file(authority["protocol_path"]),
        "prepared_manifest": _relative_to_root(
            authority["prepared_path"], root, "prepared manifest"
        ),
        "prepared_manifest_sha256": sha256_file(authority["prepared_path"]),
        "entry_count": len(inventory["entries"]),
        "category_counts": counts,
        "formal_test_content_accessed": True,
        "ground_truth_masks_decoded_for_preregistered_scale_strata": True,
        "model_predictions_accessed": False,
        "result_driven_selection_performed": False,
    }
    write_json(output / RECEIPT_FILENAME, receipt)
    return receipt


def validate_formal_test_inventory_bundle(
    *, output_dir: str | Path, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Rebuild the inventory in memory and reject any authority or data drift."""

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    _relative_to_root(output, root, "formal test inventory output")
    receipt_path, receipt = _load_json_object(output / RECEIPT_FILENAME, "inventory receipt")
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "formal_test_inventory_receipt":
        raise ValueError("unsupported formal test inventory receipt schema")
    if receipt.get("contract") != "formal_test_inventory_v1":
        raise ValueError("unexpected formal test inventory contract")
    if receipt.get("split") != "test" or receipt.get("inventory") != INVENTORY_FILENAME:
        raise ValueError("formal test inventory receipt identity is invalid")
    expected_access_flags = {
        "formal_test_content_accessed": True,
        "ground_truth_masks_decoded_for_preregistered_scale_strata": True,
        "model_predictions_accessed": False,
        "result_driven_selection_performed": False,
    }
    for field, expected in expected_access_flags.items():
        if receipt.get(field) is not expected:
            raise ValueError(f"formal test inventory receipt flag is invalid: {field}")
    dataset = receipt.get("dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError("inventory receipt dataset is invalid")
    paths: dict[str, Path] = {}
    for field, label in (
        ("project_config", "project config"),
        ("registration", "experiment registration"),
        ("protocol", "inventory protocol"),
        ("prepared_manifest", "prepared manifest"),
    ):
        path = _resolve_project_file(root, receipt.get(field), label)
        expected_hash = _require_hash(receipt.get(f"{field}_sha256"), f"{label} SHA-256")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{label} SHA-256 drifted after inventory creation")
        paths[field] = path
    dataset_root_relative = _safe_relative_path(receipt.get("dataset_root"), "dataset root")
    dataset_root = (root / Path(*PurePosixPath(dataset_root_relative).parts)).resolve()
    inventory_relative = _safe_relative_path(receipt.get("inventory"), "inventory path")
    inventory_path = (output / Path(*PurePosixPath(inventory_relative).parts)).resolve()
    try:
        inventory_path.relative_to(output)
    except ValueError as error:
        raise ValueError("inventory path escapes its bundle") from error
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    expected_inventory_hash = _require_hash(
        receipt.get("inventory_sha256"), "inventory SHA-256"
    )
    if sha256_file(inventory_path) != expected_inventory_hash:
        raise ValueError("formal test inventory SHA-256 mismatch")
    _, stored_inventory = _load_json_object(inventory_path, "formal test inventory")
    authority = _load_authority(
        dataset=str(dataset),
        project_root=root,
        project_config_path=paths["project_config"],
        registration_path=paths["registration"],
        protocol_path=paths["protocol"],
        prepared_manifest_path=paths["prepared_manifest"],
    )
    rebuilt, counts = _construct_inventory(
        dataset=str(dataset), dataset_root=dataset_root, authority=authority
    )
    if stored_inventory != rebuilt:
        raise ValueError("formal test inventory differs from current complete test split")
    if receipt.get("inventory_id") != rebuilt["inventory_id"]:
        raise ValueError("formal test inventory ID mismatch")
    if receipt.get("entry_count") != len(rebuilt["entries"]):
        raise ValueError("formal test inventory entry count mismatch")
    if receipt.get("category_counts") != counts:
        raise ValueError("formal test inventory category counts mismatch")
    if receipt_path.parent != output:
        raise ValueError("inventory receipt is not stored at the bundle root")
    return receipt


def _project_root_from_config(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent


def resolve_formal_test_inventory_category(
    *,
    output_dir: str | Path,
    dataset: str,
    category: str,
    dataset_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate a complete bundle and select one category without rediscovery.

    ``validate_formal_test_inventory_bundle`` deliberately re-enumerates the
    complete official split to detect source drift.  After that one validation,
    this resolver constructs the category sample list only from the authenticated
    inventory rows.  Inference adapters must pass these explicit samples to their
    datasets instead of calling ``build_test_manifest`` again.
    """

    root = Path(project_root).resolve()
    raw_output = Path(output_dir)
    if raw_output.is_symlink():
        raise ValueError("formal test inventory bundle cannot be a symlink")
    output = raw_output.resolve()
    receipt = validate_formal_test_inventory_bundle(
        output_dir=output,
        project_root=root,
    )
    if receipt.get("dataset") != dataset:
        raise ValueError(
            "formal test inventory dataset mismatch: "
            f"expected={dataset!r}, observed={receipt.get('dataset')!r}"
        )
    if not isinstance(category, str) or not category:
        raise ValueError("formal test inventory category must be non-empty")

    dataset_root_relative = _safe_relative_path(
        receipt.get("dataset_root"), "formal test inventory dataset root"
    )
    recorded_dataset_candidate = root / Path(*PurePosixPath(dataset_root_relative).parts)
    if recorded_dataset_candidate.is_symlink():
        raise ValueError("formal test inventory dataset root cannot be a symlink")
    recorded_dataset_root = recorded_dataset_candidate.resolve()
    try:
        recorded_dataset_root.relative_to(root)
    except ValueError as error:
        raise ValueError("formal test inventory dataset root escapes project root") from error
    if recorded_dataset_root.is_symlink() or not recorded_dataset_root.is_dir():
        raise FileNotFoundError(
            f"formal test inventory dataset root is missing or linked: "
            f"{recorded_dataset_root}"
        )
    if dataset_root is not None and Path(dataset_root).resolve() != recorded_dataset_root:
        raise ValueError("supplied dataset root differs from formal test inventory")

    inventory_path, inventory = _load_json_object(
        output / INVENTORY_FILENAME, "formal test inventory"
    )
    if inventory_path.parent != output:
        raise ValueError("formal test inventory is not stored at the bundle root")
    if inventory.get("schema_version") != 1 or inventory.get("kind") != "formal_test_inventory":
        raise ValueError("unsupported formal test inventory schema")
    if inventory.get("dataset") != dataset or inventory.get("split") != "test":
        raise ValueError("formal test inventory identity mismatch")
    if inventory.get("inventory_id") != receipt.get("inventory_id"):
        raise ValueError("formal test inventory ID differs from its receipt")
    raw_entries = inventory.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("formal test inventory entries are missing")
    if len(raw_entries) != receipt.get("entry_count"):
        raise ValueError("formal test inventory entry count differs from its receipt")

    category_counts = receipt.get("category_counts")
    counts = category_counts.get(category) if isinstance(category_counts, Mapping) else None
    if not isinstance(counts, Mapping):
        raise ValueError(
            f"category is absent from the complete formal test inventory: {category}"
        )
    normal_count = counts.get("normal")
    anomaly_count = counts.get("anomaly")
    if (
        isinstance(normal_count, bool)
        or isinstance(anomaly_count, bool)
        or not isinstance(normal_count, int)
        or not isinstance(anomaly_count, int)
        or normal_count < 0
        or anomaly_count < 0
        or normal_count + anomaly_count <= 0
    ):
        raise ValueError(f"invalid formal test category counts: {category}")
    expected_count = normal_count + anomaly_count

    selected_entries: list[dict[str, Any]] = []
    samples: list[TestSample] = []
    seen_sample_ids: set[str] = set()
    seen_images: set[str] = set()
    for position, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"formal test inventory entry {position} must be an object")
        if raw.get("dataset") != dataset:
            raise ValueError(f"formal test inventory entry {position} dataset mismatch")
        if raw.get("category") != category:
            continue
        sample_id = raw.get("sample_id")
        defect_type = raw.get("defect_type")
        label = raw.get("label")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"formal test inventory entry {position} has no sample_id")
        if not isinstance(defect_type, str) or not defect_type:
            raise ValueError(f"formal test inventory entry {position} has no defect_type")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"formal test inventory entry {position} has invalid label")
        if (label == 0) != (defect_type.casefold() == "good"):
            raise ValueError(
                f"formal test inventory entry {position} label/defect_type mismatch"
            )
        image_relative = _safe_relative_path(
            raw.get("source_image"), f"formal test inventory entry {position} image"
        )
        image_path, canonical_image_relative = _resolve_dataset_file(
            recorded_dataset_root,
            recorded_dataset_root / Path(*PurePosixPath(image_relative).parts),
            f"formal test inventory entry {position} image",
        )
        if canonical_image_relative != image_relative:
            raise ValueError(f"formal test inventory entry {position} image path drifted")
        image_sha = _require_hash(
            raw.get("source_image_sha256"),
            f"formal test inventory entry {position} image SHA-256",
        )
        if sha256_file(image_path) != image_sha:
            raise ValueError(f"formal test inventory entry {position} source image drifted")

        raw_mask = raw.get("source_mask")
        raw_mask_sha = raw.get("source_mask_sha256")
        mask_path: Path | None
        if label == 0:
            if raw_mask is not None or raw_mask_sha is not None:
                raise ValueError(
                    f"normal formal test entry {position} unexpectedly has a mask"
                )
            mask_path = None
        else:
            mask_relative = _safe_relative_path(
                raw_mask, f"formal test inventory entry {position} mask"
            )
            mask_path, canonical_mask_relative = _resolve_dataset_file(
                recorded_dataset_root,
                recorded_dataset_root / Path(*PurePosixPath(mask_relative).parts),
                f"formal test inventory entry {position} mask",
            )
            if canonical_mask_relative != mask_relative:
                raise ValueError(f"formal test inventory entry {position} mask path drifted")
            mask_sha = _require_hash(
                raw_mask_sha,
                f"formal test inventory entry {position} mask SHA-256",
            )
            if sha256_file(mask_path) != mask_sha:
                raise ValueError(f"formal test inventory entry {position} source mask drifted")

        sample_key = sample_id.casefold()
        image_key = image_relative.casefold()
        if sample_key in seen_sample_ids or image_key in seen_images:
            raise ValueError("formal test category contains duplicate sample IDs or images")
        seen_sample_ids.add(sample_key)
        seen_images.add(image_key)
        selected_entries.append(dict(raw))
        samples.append(
            TestSample(
                dataset=dataset,  # type: ignore[arg-type]
                category=category,
                defect_type=defect_type,
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
                label=label,
            )
        )

    if len(samples) != expected_count:
        raise ValueError(
            f"formal test inventory category is incomplete: {category}; "
            f"expected={expected_count}, observed={len(samples)}"
        )
    observed_normal = sum(sample.label == 0 for sample in samples)
    observed_anomaly = sum(sample.label == 1 for sample in samples)
    if observed_normal != normal_count or observed_anomaly != anomaly_count:
        raise ValueError(f"formal test inventory category label counts drifted: {category}")

    evidence = {
        "schema_version": 1,
        "contract": FORMAL_TEST_SELECTION_CONTRACT,
        "bundle": _relative_to_root(output, root, "formal test inventory bundle"),
        "inventory_id": inventory["inventory_id"],
        "dataset": dataset,
        "split": "test",
        "category": category,
        "inventory_sha256": sha256_file(inventory_path),
        "receipt_sha256": sha256_file(output / RECEIPT_FILENAME),
        "entry_count": len(raw_entries),
        "category_entry_count": len(samples),
        "dataset_root": dataset_root_relative,
    }
    return {
        "evidence": evidence,
        "entries": selected_entries,
        "samples": samples,
        "dataset_root": recorded_dataset_root,
    }


def validate_formal_test_inventory_evidence(
    evidence: Mapping[str, object],
    *,
    dataset: str,
    category: str,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
) -> dict[str, Any]:
    """Re-resolve persisted category evidence and reject bundle/data drift."""

    if not isinstance(evidence, Mapping):
        raise ValueError("formal result is missing formal test inventory evidence")
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported formal test inventory evidence schema")
    if evidence.get("contract") != FORMAL_TEST_SELECTION_CONTRACT:
        raise ValueError("unexpected formal test inventory evidence contract")
    root = _project_root_from_config(project_config_path)
    bundle_relative = _safe_relative_path(
        evidence.get("bundle"), "formal test inventory evidence bundle"
    )
    selection = resolve_formal_test_inventory_category(
        output_dir=root / Path(*PurePosixPath(bundle_relative).parts),
        dataset=dataset,
        category=category,
        project_root=root,
    )
    if dict(evidence) != selection["evidence"]:
        raise ValueError("formal test inventory evidence differs from current bundle")
    return selection


def validate_formal_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
) -> None:
    """Prove prediction rows are a one-to-one export of a category inventory."""

    entries = selection.get("entries")
    evidence = selection.get("evidence")
    if not isinstance(entries, list) or not isinstance(evidence, Mapping):
        raise ValueError("invalid formal test inventory category selection")
    if len(rows) != len(entries):
        raise ValueError("prediction index does not cover the formal test category")
    bindings = {
        "formal_test_inventory_id": evidence.get("inventory_id"),
        "formal_test_inventory_sha256": evidence.get("inventory_sha256"),
        "formal_test_inventory_receipt_sha256": evidence.get("receipt_sha256"),
    }
    for position, (row, entry) in enumerate(zip(rows, entries, strict=True)):
        if not isinstance(row, Mapping) or not isinstance(entry, Mapping):
            raise ValueError(f"invalid formal prediction row at position {position}")
        expected = {
            "index": position,
            "sample_id": entry.get("sample_id"),
            "label": entry.get("label"),
            "defect_type": entry.get("defect_type"),
            "input_image": entry.get("source_image"),
            "source_ground_truth": entry.get("source_mask"),
            "source_image_sha256": entry.get("source_image_sha256"),
            "source_ground_truth_sha256": entry.get("source_mask_sha256"),
            **bindings,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"formal prediction row {position} differs from inventory field {field}"
                )
