"""Fail-closed construction of the final, portable experiment delivery.

The builder deliberately accepts only files named by a reviewed delivery
specification.  Dataset images, checkpoints, caches and run directories are
never copied into the archive.  Formal run directories are read only to
revalidate the evidence behind the final tables and figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import stat
import subprocess
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .metrics import EVALUATION_PROTOCOL, METRICS_SCHEMA_VERSION
from .result_aggregation import METRIC_NAMES, ResultRow, macro_by_seed, method_summary
from .run_artifacts import sha256_file
from .visualization_delivery import (
    _load_frozen_method_registry as _load_visualization_method_registry,
    _load_selection_config as _load_visualization_selection_config,
)


DELIVERY_SPEC_SCHEMA_VERSION = 1
DELIVERY_MANIFEST_SCHEMA_VERSION = 1
DELIVERY_MANIFEST_NAME = "DELIVERY-MANIFEST.json"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SHA256_LENGTH = 64
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FORBIDDEN_TOP_LEVEL = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".runtime",
    ".tox",
    ".venv",
    "cache",
    "caches",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "model",
    "models",
    "runs",
    "tmp",
    "weights",
}
_FORBIDDEN_ANYWHERE = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "caches",
    "checkpoints",
    "dataset",
    "datasets",
    "model",
    "models",
    "weights",
}
_FORBIDDEN_ARTIFACT_SUBTREES = {"weights", "preflight", "validation"}
_FORBIDDEN_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".bin",
    ".data",
    ".db",
    ".engine",
    ".faiss",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".lmdb",
    ".model",
    ".npy",
    ".npz",
    ".onnx",
    ".pb",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".tflite",
    ".weights",
    ".xz",
    ".zip",
}
_EXPECTED_FORMAL_FIELDS: dict[str, object] = {
    "execution_scope": "full",
    "run_purpose": "formal_full",
    "claim_scope": "formal_matrix_candidate",
    "formal_matrix_eligible": True,
}
_EXPECTED_VISUALIZATION_COHORTS = {
    "result_independent_preregistered",
    "outcome_dependent_machine_selected",
}


@dataclass(frozen=True)
class PackagedFile:
    path: str
    size_bytes: int
    sha256: str
    deliverable: str


@dataclass(frozen=True)
class FormalRunEvidence:
    run_directory: str
    method: str
    dataset: str
    category: str
    seed: int
    run_config_sha256: str
    run_state_sha256: str
    metrics_sha256: str
    prediction_index_sha256: str
    source_provenance_sha256: str
    source_project_commit: str


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {context}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object: {path}")
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _validate_relative_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError(f"{context} must not be absolute or escape the project: {value!r}")
    canonical = relative.as_posix()
    if canonical != value:
        raise ValueError(f"{context} is not a canonical POSIX path: {value!r}")
    return canonical


def _reject_forbidden_payload_path(relative: str, *, allow_png: bool = False) -> None:
    parts = tuple(part.lower() for part in PurePosixPath(relative).parts)
    if not parts:
        raise ValueError("empty payload path")
    if parts[0] in _FORBIDDEN_TOP_LEVEL or any(
        part in _FORBIDDEN_ANYWHERE for part in parts
    ):
        raise ValueError(f"forbidden dataset/cache/runtime path in delivery: {relative}")
    if (
        len(parts) >= 2
        and parts[0] == "artifacts"
        and parts[1] in _FORBIDDEN_ARTIFACT_SUBTREES
    ):
        raise ValueError(f"forbidden weight/temporary evidence path in delivery: {relative}")
    suffix = PurePosixPath(relative).suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden model/data/archive file in delivery: {relative}")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"} and not allow_png:
        raise ValueError(f"image files are allowed only as declared visualization PNGs: {relative}")


def _resolve_project_file(
    project_root: Path,
    value: object,
    context: str,
    *,
    allow_png: bool = False,
) -> tuple[str, Path]:
    relative = _validate_relative_text(value, context)
    _reject_forbidden_payload_path(relative, allow_png=allow_png)
    root = project_root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if _is_reparse_or_symlink(current):
            raise ValueError(f"links/reparse points are forbidden in delivery inputs: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"delivery path escapes the project: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing delivery file ({context}): {resolved}")
    return relative, resolved


def _resolve_project_directory(project_root: Path, value: object, context: str) -> tuple[str, Path]:
    relative = _validate_relative_text(value, context)
    root = project_root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if _is_reparse_or_symlink(current):
            raise ValueError(f"links/reparse points are forbidden in {context}: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes the project: {relative}") from error
    if not resolved.is_dir():
        raise FileNotFoundError(f"missing {context}: {resolved}")
    return relative, resolved


def _require_exact_keys(
    payload: Mapping[str, Any], required: Iterable[str], context: str
) -> None:
    expected = set(required)
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{context} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_string_list(value: object, context: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{context} must be {'a non-empty' if nonempty else 'an'} list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{context} contains a non-string or empty value")
    resolved = list(value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{context} contains duplicates")
    return resolved


def _git(project_root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _verify_project_git(
    project_root: Path, tracked_paths: Sequence[str]
) -> dict[str, object]:
    commit = _git(project_root, "rev-parse", "HEAD")
    # Git object IDs in this repository are SHA-1. File identities elsewhere
    # in the delivery use SHA-256.
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("project HEAD is not a full 40-character Git commit")
    for relative in tracked_paths:
        result = subprocess.run(
            ["git", "-C", str(project_root), "ls-files", "--error-unmatch", "--", relative],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise ValueError(f"project delivery source is not tracked by Git: {relative}")
    status = _git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *tracked_paths,
    )
    if status:
        raise ValueError(
            "project delivery sources differ from HEAD; commit them before packaging: "
            + status.splitlines()[0]
        )
    return {
        "commit": commit,
        "included_project_files_match_head": True,
        "tracked_file_count": len(tracked_paths),
    }


def _contains_value(value: object, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


def _verify_third_party(
    project_root: Path,
    sources: object,
    included_paths: set[str],
) -> list[dict[str, object]]:
    if not isinstance(sources, list) or not sources:
        raise ValueError("third_party_sources must be a non-empty list")
    output: list[dict[str, object]] = []
    names: set[str] = set()
    for position, item in enumerate(sources):
        if not isinstance(item, Mapping):
            raise ValueError(f"third_party_sources[{position}] must be an object")
        _require_exact_keys(item, ("name", "path", "commit", "source_manifest"), f"third_party_sources[{position}]")
        name = item["name"]
        expected_commit = item["commit"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("third-party names must be non-empty and unique")
        names.add(name)
        if (
            not isinstance(expected_commit, str)
            or len(expected_commit) != 40
            or any(character not in "0123456789abcdef" for character in expected_commit)
        ):
            raise ValueError(f"invalid third-party commit for {name}")
        path_text = _validate_relative_text(item["path"], f"third-party path for {name}")
        if PurePosixPath(path_text).parts[0].lower() != "third_party":
            raise ValueError(f"third-party source must stay under third_party/: {name}")
        _, path = _resolve_project_directory(project_root, path_text, f"third-party source {name}")
        observed_commit = _git(path, "rev-parse", "HEAD")
        if observed_commit != expected_commit:
            raise ValueError(
                f"third-party commit mismatch for {name}: expected={expected_commit}, observed={observed_commit}"
            )
        if _git(path, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError(f"third-party worktree is not clean: {name}")
        manifest_text, manifest_path = _resolve_project_file(
            project_root,
            item["source_manifest"],
            f"third-party source manifest for {name}",
        )
        if manifest_text not in included_paths:
            raise ValueError(f"third-party source manifest is absent from package allowlist: {manifest_text}")
        manifest = _load_json(manifest_path, f"third-party source manifest for {name}")
        if not _contains_value(manifest, expected_commit):
            raise ValueError(f"third-party source manifest does not bind commit for {name}")
        output.append(
            {
                "name": name,
                "path": path_text,
                "commit": observed_commit,
                "worktree_clean": True,
                "source_manifest": manifest_text,
                "source_manifest_sha256": sha256_file(manifest_path),
            }
        )
    required = {"DRAEM", "PatchCore", "RD4AD"}
    if not required.issubset(names):
        raise ValueError(f"third-party provenance is missing: {sorted(required - names)}")
    return sorted(output, key=lambda row: str(row["name"]))


def _verify_formal_test_inventories(
    project_root: Path,
    inventory_specs: object,
    *,
    required_datasets: Sequence[str],
    categories: Mapping[str, Sequence[str]],
    project_config_path: Path,
    registration_path: Path,
    included_paths: set[str],
) -> tuple[
    list[tuple[str, Path]],
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    """Verify frozen F0 bundle metadata without opening test images or masks."""

    if not isinstance(inventory_specs, list) or not inventory_specs:
        raise ValueError("formal_test_inventories must be a non-empty list")
    packaged: list[tuple[str, Path]] = []
    evidence: list[dict[str, object]] = []
    entries_by_dataset: dict[str, list[dict[str, object]]] = {}
    seen_datasets: set[str] = set()
    for position, item in enumerate(inventory_specs):
        if not isinstance(item, Mapping):
            raise ValueError(f"formal_test_inventories[{position}] must be an object")
        _require_exact_keys(
            item,
            ("dataset", "inventory", "receipt"),
            f"formal_test_inventories[{position}]",
        )
        dataset = item["dataset"]
        if not isinstance(dataset, str) or dataset not in required_datasets or dataset in seen_datasets:
            raise ValueError(f"invalid or duplicate formal test inventory dataset: {dataset!r}")
        seen_datasets.add(dataset)
        inventory_text, inventory_path = _resolve_project_file(
            project_root, item["inventory"], f"formal test inventory for {dataset}"
        )
        receipt_text, receipt_path = _resolve_project_file(
            project_root, item["receipt"], f"formal test inventory receipt for {dataset}"
        )
        if inventory_path.parent != receipt_path.parent:
            raise ValueError(f"formal test inventory and receipt are not one bundle: {dataset}")
        inventory = _load_json(inventory_path, f"formal test inventory for {dataset}")
        receipt = _load_json(receipt_path, f"formal test inventory receipt for {dataset}")
        if (
            inventory.get("schema_version") != 1
            or inventory.get("kind") != "formal_test_inventory"
            or inventory.get("dataset") != dataset
            or inventory.get("split") != "test"
        ):
            raise ValueError(f"unsupported or non-formal test inventory: {dataset}")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("kind") != "formal_test_inventory_receipt"
            or receipt.get("contract") != "formal_test_inventory_v1"
            or receipt.get("dataset") != dataset
            or receipt.get("split") != "test"
        ):
            raise ValueError(f"unsupported or non-formal test inventory receipt: {dataset}")
        if receipt.get("inventory") != inventory_path.name:
            raise ValueError(f"formal test receipt points to a different inventory: {dataset}")
        if receipt.get("inventory_sha256") != sha256_file(inventory_path):
            raise ValueError(f"formal test inventory SHA-256 mismatch: {dataset}")
        if receipt.get("inventory_id") != inventory.get("inventory_id"):
            raise ValueError(f"formal test inventory ID mismatch: {dataset}")
        if not isinstance(inventory.get("inventory_id"), str) or not inventory["inventory_id"]:
            raise ValueError(f"formal test inventory ID is invalid: {dataset}")
        _validate_relative_text(
            receipt.get("dataset_root"), f"formal test dataset root for {dataset}"
        )
        for field, path in (
            ("project_config", project_config_path),
            ("registration", registration_path),
        ):
            relative = path.resolve().relative_to(project_root.resolve()).as_posix()
            if receipt.get(field) != relative or receipt.get(f"{field}_sha256") != sha256_file(path):
                raise ValueError(f"formal test receipt {field} authority mismatch: {dataset}")
        for field in ("protocol", "prepared_manifest"):
            authority_text = _validate_relative_text(
                receipt.get(field), f"formal test receipt {field} for {dataset}"
            )
            if authority_text not in included_paths:
                raise ValueError(
                    f"formal test receipt authority is absent from package allowlist: {authority_text}"
                )
            _, authority_path = _resolve_project_file(
                project_root, authority_text, f"formal test receipt {field} for {dataset}"
            )
            if receipt.get(f"{field}_sha256") != sha256_file(authority_path):
                raise ValueError(f"formal test receipt {field} SHA-256 mismatch: {dataset}")
        if (
            receipt.get("formal_test_content_accessed") is not True
            or receipt.get("ground_truth_masks_decoded_for_preregistered_scale_strata") is not True
            or receipt.get("model_predictions_accessed") is not False
            or receipt.get("result_driven_selection_performed") is not False
        ):
            raise ValueError(f"formal test receipt has an invalid F0 boundary: {dataset}")
        entries = inventory.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"formal test inventory has no entries: {dataset}")
        sample_ids: set[str] = set()
        validated_entries: list[dict[str, object]] = []
        counts: dict[str, Counter[str]] = {
            category: Counter() for category in categories[dataset]
        }
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"formal test inventory entry is invalid: {dataset}/{entry_index}")
            category = entry.get("category")
            sample_id = entry.get("sample_id")
            if entry.get("dataset") != dataset or category not in counts:
                raise ValueError(f"formal test inventory has an unknown identity: {dataset}/{entry_index}")
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"formal test inventory has an invalid/duplicate sample ID: {dataset}")
            sample_ids.add(sample_id)
            _validate_relative_text(entry.get("source_image"), f"formal source image {sample_id}")
            if not _is_sha256(entry.get("source_image_sha256")):
                raise ValueError(f"formal source image hash is invalid: {sample_id}")
            label = entry.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                raise ValueError(f"formal test label is invalid: {sample_id}")
            defect_type = entry.get("defect_type")
            area = entry.get("anomaly_area_fraction")
            scale = entry.get("anomaly_scale")
            if not isinstance(defect_type, str) or not defect_type:
                raise ValueError(f"formal test defect type is invalid: {sample_id}")
            if label == 0:
                if (
                    defect_type != "good"
                    or isinstance(area, bool)
                    or not isinstance(area, (int, float))
                    or float(area) != 0.0
                    or scale != "normal"
                    or entry.get("source_mask") is not None
                    or entry.get("source_mask_sha256") is not None
                ):
                    raise ValueError(f"normal formal sample unexpectedly has a mask: {sample_id}")
                counts[str(category)]["normal"] += 1
            else:
                if (
                    defect_type == "good"
                    or isinstance(area, bool)
                    or not isinstance(area, (int, float))
                    or not math.isfinite(float(area))
                    or not 0.0 < float(area) <= 1.0
                    or scale not in {"small", "medium", "large"}
                ):
                    raise ValueError(f"anomalous formal sample metadata is invalid: {sample_id}")
                _validate_relative_text(entry.get("source_mask"), f"formal source mask {sample_id}")
                if not _is_sha256(entry.get("source_mask_sha256")):
                    raise ValueError(f"formal source mask hash is invalid: {sample_id}")
                counts[str(category)]["anomaly"] += 1
                counts[str(category)]["masks"] += 1
            validated_entries.append(dict(entry))
        observed_counts = {
            category: {
                "normal": counter["normal"],
                "anomaly": counter["anomaly"],
                "masks": counter["masks"],
            }
            for category, counter in counts.items()
        }
        if any(counter["normal"] + counter["anomaly"] == 0 for counter in counts.values()):
            raise ValueError(f"formal test inventory omits a registered category: {dataset}")
        if receipt.get("entry_count") != len(entries) or receipt.get("category_counts") != observed_counts:
            raise ValueError(f"formal test receipt counts disagree with inventory: {dataset}")
        packaged.extend(((inventory_text, inventory_path), (receipt_text, receipt_path)))
        evidence.append(
            {
                "dataset": dataset,
                "bundle": inventory_path.parent.relative_to(project_root).as_posix(),
                "inventory_id": inventory["inventory_id"],
                "dataset_root": receipt["dataset_root"],
                "inventory": inventory_text,
                "inventory_sha256": sha256_file(inventory_path),
                "receipt": receipt_text,
                "receipt_sha256": sha256_file(receipt_path),
                "entry_count": len(entries),
                "category_counts": observed_counts,
                "content_files_reopened_by_packager": False,
            }
        )
        entries_by_dataset[dataset] = validated_entries
    if seen_datasets != set(required_datasets):
        raise ValueError(
            "formal test inventory delivery does not cover both datasets: "
            f"missing={sorted(set(required_datasets) - seen_datasets)}"
        )
    return (
        packaged,
        sorted(evidence, key=lambda row: str(row["dataset"])),
        entries_by_dataset,
    )


def _validated_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} is not a numeric metric")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{context} is not finite in [0, 1]")
    return resolved


def _validate_formal_source_binding(
    project_root: Path,
    formal_provenance: Mapping[str, object],
) -> tuple[str, str]:
    """Recheck the source identity embedded by the formal checkpoint gate.

    Result artifacts may legitimately be untracked when the delivery is built,
    so this compares the captured clean-launch evidence with the current Git
    objects and authority-file bytes instead of requiring an entirely clean
    post-result worktree.
    """

    checkpoint = formal_provenance.get("checkpoint_validation")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("formal result is missing checkpoint source validation")
    source = checkpoint.get("formal_source_validation")
    if (
        not isinstance(source, Mapping)
        or source.get("schema_version") != 1
        or source.get("contract") != "clean_git_source_provenance_v1"
        or source.get("all_registered_repositories_clean") is not True
        or source.get("project_head_binds_all_tracked_source_files") is not True
        or source.get("untracked_source_files_forbidden") is not True
    ):
        raise ValueError("formal result lacks clean Git source provenance")
    scope_text, scope_path = _resolve_project_file(
        project_root, source.get("scope"), "formal source provenance scope"
    )
    if source.get("scope_sha256") != sha256_file(scope_path):
        raise ValueError("formal source provenance scope SHA-256 drifted")
    authorities = source.get("authority_files")
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("formal source provenance has no authority files")
    seen_authorities: set[str] = set()
    for position, entry in enumerate(authorities):
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid formal source authority file {position}")
        relative, path = _resolve_project_file(
            project_root,
            entry.get("path"),
            f"formal source authority file {position}",
        )
        if relative in seen_authorities:
            raise ValueError("formal source provenance repeats an authority file")
        seen_authorities.add(relative)
        if entry.get("size_bytes") != path.stat().st_size or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"formal source authority file drifted: {relative}")
    if scope_text not in seen_authorities:
        raise ValueError("formal source scope is not self-bound as an authority file")

    repositories = source.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("formal source provenance has no repositories")
    seen_names: set[str] = set()
    project_commit = ""
    for position, entry in enumerate(repositories):
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid formal source repository {position}")
        name = entry.get("name")
        relative = entry.get("path")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("formal source repository names are invalid or duplicated")
        seen_names.add(name)
        if relative == ".":
            if name != "project":
                raise ValueError("only the project repository may use path='.'")
            repository = project_root.resolve()
        else:
            _, repository = _resolve_project_directory(
                project_root,
                relative,
                f"formal source repository {name}",
            )
        head = _git(repository, "rev-parse", "HEAD")
        tree = _git(repository, "rev-parse", "HEAD^{tree}")
        if (
            entry.get("head_commit") != head
            or entry.get("head_tree") != tree
            or entry.get("worktree_clean") is not True
            or entry.get("untracked_files_checked") is not True
            or entry.get("submodule_changes_checked") is not True
        ):
            raise ValueError(f"formal source repository identity drifted: {name}")
        expected_commit = entry.get("expected_commit")
        if relative == ".":
            if expected_commit is not None:
                raise ValueError("project source evidence must not pin a separate expected commit")
            project_commit = head
        else:
            expected_path = {
                "DRAEM": "third_party/DRAEM",
                "PatchCore": "third_party/PatchCore",
                "RD4AD": "third_party/RD4AD",
            }.get(name)
            if expected_path != relative or expected_commit != head:
                raise ValueError(
                    f"formal vendor source path/commit differs from its frozen identity: {name}"
                )
    if seen_names != {"project", "DRAEM", "PatchCore", "RD4AD"}:
        raise ValueError("formal source provenance must cover project, DRAEM, PatchCore and RD4AD")
    source_sha = hashlib.sha256(_canonical_json_bytes(dict(source))).hexdigest()
    return source_sha, project_commit


def _validate_formal_run(
    root: Path,
    results_root: Path,
    inventory_bundles: Mapping[str, Mapping[str, object]],
    inventory_entries: Mapping[str, Sequence[Mapping[str, object]]],
    project_root: Path,
) -> tuple[ResultRow, FormalRunEvidence]:
    try:
        lexical_relative = root.relative_to(results_root)
    except ValueError as error:
        raise ValueError(f"formal run escapes formal_results_root: {root}") from error
    current = results_root
    for part in lexical_relative.parts:
        current = current / part
        if _is_reparse_or_symlink(current):
            raise ValueError(f"links/reparse points are forbidden in formal runs: {root}")
    try:
        root.resolve().relative_to(results_root.resolve())
    except ValueError as error:
        raise ValueError(f"formal run resolves outside formal_results_root: {root}") from error
    for required in ("run_config.json", "run_state.json", "metrics.json", "predictions/index.jsonl"):
        path = root.joinpath(*PurePosixPath(required).parts)
        if _is_reparse_or_symlink(path) or not path.is_file():
            raise ValueError(f"formal run is missing a regular {required}: {root}")
    config_path = root / "run_config.json"
    state_path = root / "run_state.json"
    metrics_path = root / "metrics.json"
    index_path = root / "predictions" / "index.jsonl"
    config = _load_json(config_path, "formal run config")
    state = _load_json(state_path, "formal run state")
    metrics = _load_json(metrics_path, "formal metrics")
    if config.get("schema_version") != 2:
        raise ValueError(f"formal run config must use schema_version 2: {root}")
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"legacy metrics schema is forbidden; expected {METRICS_SCHEMA_VERSION}: {root}"
        )
    if state.get("status") != "completed":
        raise ValueError(f"formal run is not completed: {root}")
    for field, expected in _EXPECTED_FORMAL_FIELDS.items():
        if config.get(field) != expected or state.get(field) != expected:
            raise ValueError(f"non-formal result field {field!r}: {root}")
    if metrics.get("protocol") != EVALUATION_PROTOCOL or config.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ValueError(f"formal result uses an unexpected evaluation protocol: {root}")
    if not isinstance(config.get("formal_provenance"), Mapping):
        raise ValueError(f"formal result is missing formal_provenance: {root}")
    source_sha, source_project_commit = _validate_formal_source_binding(
        project_root, config["formal_provenance"]
    )
    expected_hashes = {
        "run_config_sha256": config_path,
        "metrics_sha256": metrics_path,
        "predictions_index_sha256": index_path,
    }
    for field, path in expected_hashes.items():
        observed = state.get(field)
        if not _is_sha256(observed) or observed != sha256_file(path):
            raise ValueError(f"formal run hash mismatch for {field}: {root}")
    method = config.get("method")
    dataset = metrics.get("dataset")
    category = metrics.get("category")
    settings = config.get("settings")
    if not isinstance(method, str) or not method:
        raise ValueError(f"formal run is missing method: {root}")
    if not isinstance(dataset, str) or not dataset or not isinstance(category, str) or not category:
        raise ValueError(f"formal run is missing dataset/category: {root}")
    if not isinstance(settings, Mapping):
        raise ValueError(f"formal run settings are missing: {root}")
    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"formal run has an invalid seed: {root}")
    if settings.get("dataset") != dataset or settings.get("category") != category:
        raise ValueError(f"formal settings and metrics identity disagree: {root}")
    formal_provenance = config["formal_provenance"]
    inventory_evidence = formal_provenance.get("formal_test_inventory")
    if not isinstance(inventory_evidence, Mapping):
        raise ValueError(f"formal result is missing test inventory evidence: {root}")
    expected_inventory = inventory_bundles.get(dataset)
    if expected_inventory is None:
        raise ValueError(f"formal result uses an unregistered inventory dataset: {root}")
    category_counts = expected_inventory.get("category_counts")
    if not isinstance(category_counts, Mapping) or not isinstance(category_counts.get(category), Mapping):
        raise ValueError(f"formal inventory has no registered category evidence: {root}")
    category_count = category_counts[category]
    expected_category_entries = int(category_count.get("normal", 0)) + int(
        category_count.get("anomaly", 0)
    )
    expected_inventory_evidence = {
        "schema_version": 1,
        "contract": "formal_test_inventory_category_selection_v1",
        "bundle": expected_inventory["bundle"],
        "inventory_id": expected_inventory["inventory_id"],
        "dataset": dataset,
        "split": "test",
        "category": category,
        "inventory_sha256": expected_inventory["inventory_sha256"],
        "receipt_sha256": expected_inventory["receipt_sha256"],
        "entry_count": expected_inventory["entry_count"],
        "category_entry_count": expected_category_entries,
        "dataset_root": expected_inventory["dataset_root"],
    }
    if dict(inventory_evidence) != expected_inventory_evidence:
        raise ValueError(f"formal result test inventory evidence differs from delivery bundle: {root}")
    for payload, label in ((config, "run config"), (state, "run state"), (metrics, "metrics")):
        if payload.get("formal_test_inventory") != inventory_evidence:
            raise ValueError(f"{label} formal test inventory binding mismatch: {root}")
    category_entries = [
        entry
        for entry in inventory_entries.get(dataset, ())
        if entry.get("category") == category
    ]
    if len(category_entries) != expected_category_entries:
        raise ValueError(f"formal inventory category entry set is incomplete: {root}")
    prediction_rows: list[Mapping[str, object]] = []
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"formal prediction index contains a blank line {line_number}: {root}")
            try:
                prediction = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid formal prediction index JSON on line {line_number}: {root}"
                ) from error
            if not isinstance(prediction, Mapping):
                raise ValueError(f"formal prediction row is not an object: {root}")
            prediction_rows.append(prediction)
    if (
        len(prediction_rows) != expected_category_entries
        or metrics.get("num_images") != expected_category_entries
        or state.get("num_images") != expected_category_entries
    ):
        raise ValueError(f"formal result does not cover the complete inventory category: {root}")
    inventory_bindings = {
        "formal_test_inventory_id": expected_inventory["inventory_id"],
        "formal_test_inventory_sha256": expected_inventory["inventory_sha256"],
        "formal_test_inventory_receipt_sha256": expected_inventory["receipt_sha256"],
    }
    seen_prediction_ids: set[str] = set()
    for position, (prediction, entry) in enumerate(
        zip(prediction_rows, category_entries, strict=True)
    ):
        sample_id = entry.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in seen_prediction_ids:
            raise ValueError(f"formal prediction index repeats a sample: {root}")
        seen_prediction_ids.add(sample_id)
        expected_prediction = {
            "index": position,
            "sample_id": sample_id,
            "label": entry.get("label"),
            "defect_type": entry.get("defect_type"),
            "input_image": entry.get("source_image"),
            "source_ground_truth": entry.get("source_mask"),
            "source_image_sha256": entry.get("source_image_sha256"),
            "source_ground_truth_sha256": entry.get("source_mask_sha256"),
            **inventory_bindings,
        }
        for field, expected_value in expected_prediction.items():
            if prediction.get(field) != expected_value:
                raise ValueError(
                    f"formal prediction row {position} differs from inventory field {field}: {root}"
                )
    metric_values = metrics.get("metrics")
    if not isinstance(metric_values, Mapping):
        raise ValueError(f"formal metrics mapping is missing: {root}")
    values = {
        name: _validated_number(metric_values.get(name), f"{root}/{name}")
        for name in METRIC_NAMES
    }
    relative_run = root.resolve().relative_to(results_root.resolve()).as_posix()
    row = ResultRow(
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        run_id=root.resolve().as_posix(),
        **values,
    )
    evidence = FormalRunEvidence(
        run_directory=relative_run,
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        run_config_sha256=sha256_file(config_path),
        run_state_sha256=sha256_file(state_path),
        metrics_sha256=sha256_file(metrics_path),
        prediction_index_sha256=sha256_file(index_path),
        source_provenance_sha256=source_sha,
        source_project_commit=source_project_commit,
    )
    return row, evidence


def _load_formal_runs(
    results_root: Path,
    inventory_evidence: Sequence[Mapping[str, object]],
    inventory_entries: Mapping[str, Sequence[Mapping[str, object]]],
    project_root: Path,
) -> tuple[list[ResultRow], list[FormalRunEvidence]]:
    metric_paths = sorted(results_root.rglob("metrics.json"), key=lambda path: path.as_posix())
    if not metric_paths:
        raise ValueError(f"formal_results_root contains no metrics.json files: {results_root}")
    rows: list[ResultRow] = []
    evidence: list[FormalRunEvidence] = []
    inventory_bundles = {
        str(item["dataset"]): item for item in inventory_evidence
    }
    for metrics_path in metric_paths:
        row, proof = _validate_formal_run(
            metrics_path.parent,
            results_root,
            inventory_bundles,
            inventory_entries,
            project_root,
        )
        rows.append(row)
        evidence.append(proof)
    source_identities = {
        (item.source_provenance_sha256, item.source_project_commit) for item in evidence
    }
    if len(source_identities) != 1:
        raise ValueError("formal result matrix was produced from multiple source identities")
    return rows, evidence


def _assert_complete_matrix(
    rows: Sequence[ResultRow],
    *,
    methods: Sequence[str],
    categories: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
) -> None:
    expected = {
        (method, dataset, category, seed)
        for method in methods
        for dataset, names in categories.items()
        for category in names
        for seed in seeds
    }
    observed_list = [(row.method, row.dataset, row.category, row.seed) for row in rows]
    observed = set(observed_list)
    if len(observed) != len(observed_list):
        raise ValueError("formal result matrix contains duplicate method/dataset/category/seed rows")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"formal result matrix is incomplete or contains extras: "
            f"missing={missing[:5]} ({len(missing)} total), extra={extra[:5]} ({len(extra)} total)"
        )


def _read_csv(path: Path, context: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{context} has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{context} is empty: {path}")
    return rows


def _csv_float(value: object, context: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"missing CSV value: {context}")
    try:
        resolved = float(value)
    except ValueError as error:
        raise ValueError(f"invalid CSV float: {context}") from error
    if not math.isfinite(resolved):
        raise ValueError(f"non-finite CSV float: {context}")
    return resolved


def _assert_float_equal(observed: float, expected: float, context: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"final table disagrees with formal runs for {context}")


def _verify_per_category_table(path: Path, rows: Sequence[ResultRow]) -> None:
    table = _read_csv(path, "per-category formal table")
    expected_fields = {"method", "dataset", "category", "seed", "run_id", *METRIC_NAMES}
    if set(table[0]) != expected_fields:
        raise ValueError("per-category table uses an unsupported or legacy schema")
    lookup = {(row.method, row.dataset, row.category, row.seed): row for row in rows}
    seen: set[tuple[str, str, str, int]] = set()
    for record in table:
        try:
            seed = int(record["seed"])
        except (TypeError, ValueError) as error:
            raise ValueError("per-category table has an invalid seed") from error
        key = (record["method"], record["dataset"], record["category"], seed)
        if key in seen or key not in lookup:
            raise ValueError(f"per-category table has a duplicate or unknown row: {key}")
        seen.add(key)
        expected = lookup[key]
        if Path(record["run_id"]).resolve() != Path(expected.run_id).resolve():
            raise ValueError(f"per-category table run_id disagrees with formal run: {key}")
        for metric in METRIC_NAMES:
            _assert_float_equal(_csv_float(record[metric], f"{key}/{metric}"), expected.metric(metric), f"{key}/{metric}")
    if seen != set(lookup):
        raise ValueError("per-category table omits formal matrix rows")


def _verify_derived_table(
    path: Path,
    expected_rows: Sequence[Mapping[str, object]],
    context: str,
    identity_fields: Sequence[str],
) -> None:
    table = _read_csv(path, context)
    if set(table[0]) != set(expected_rows[0]):
        raise ValueError(f"{context} uses an unsupported schema")
    lookup = {
        tuple(str(row[field]) for field in identity_fields): row for row in expected_rows
    }
    seen: set[tuple[str, ...]] = set()
    for record in table:
        key = tuple(record[field] for field in identity_fields)
        if key in seen or key not in lookup:
            raise ValueError(f"{context} has a duplicate or unknown row: {key}")
        seen.add(key)
        expected = lookup[key]
        for field, expected_value in expected.items():
            if field in identity_fields:
                continue
            if isinstance(expected_value, int) and not isinstance(expected_value, bool):
                if record[field] != str(expected_value):
                    raise ValueError(f"{context} disagrees for {key}/{field}")
            elif isinstance(expected_value, float):
                _assert_float_equal(
                    _csv_float(record[field], f"{key}/{field}"),
                    expected_value,
                    f"{key}/{field}",
                )
            elif record[field] != str(expected_value):
                raise ValueError(f"{context} disagrees for {key}/{field}")
    if seen != set(lookup):
        raise ValueError(f"{context} omits expected rows")


def _verify_aggregation(
    project_root: Path,
    table_spec: Mapping[str, Any],
    *,
    rows: Sequence[ResultRow],
    methods: Sequence[str],
    categories: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
    registration_path: Path,
    required_claim_decision: str,
) -> tuple[list[tuple[str, Path]], dict[str, object]]:
    _require_exact_keys(
        table_spec,
        ("aggregation_manifest", "per_category_seed", "macro_by_seed", "method_summary"),
        "comparison_tables",
    )
    resolved: list[tuple[str, Path]] = []
    for field in ("aggregation_manifest", "per_category_seed", "macro_by_seed", "method_summary"):
        resolved.append(
            _resolve_project_file(project_root, table_spec[field], f"comparison_tables.{field}")
        )
    paths = dict(resolved)
    manifest = _load_json(paths[str(table_spec["aggregation_manifest"])], "aggregation manifest")
    if manifest.get("schema_version") != 1 or manifest.get("matrix_validation") != "passed":
        raise ValueError("aggregation manifest is incomplete or uses an unsupported schema")
    if manifest.get("methods") != list(methods) or manifest.get("seeds") != list(seeds):
        raise ValueError("aggregation manifest differs from the frozen formal registry")
    if manifest.get("dataset_categories") != {key: list(value) for key, value in categories.items()}:
        raise ValueError("aggregation manifest dataset/category matrix differs from project config")
    if manifest.get("row_count") != len(rows):
        raise ValueError("aggregation manifest row_count differs from formal runs")
    registration = manifest.get("registration")
    if not isinstance(registration, Mapping) or registration.get("sha256") != sha256_file(registration_path):
        raise ValueError("aggregation manifest is not bound to the current registration")
    adjudication = manifest.get("claim_adjudication")
    if not isinstance(adjudication, Mapping) or adjudication.get("decision") != required_claim_decision:
        raise ValueError(
            f"formal claim adjudication must be {required_claim_decision!r} for final delivery"
        )
    gates = adjudication.get("gates")
    if required_claim_decision == "pass" and (
        not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values())
    ):
        raise ValueError("claim adjudication says pass without all registered gates passing")
    if not isinstance(manifest.get("comparisons"), list) or not manifest["comparisons"]:
        raise ValueError("aggregation manifest contains no paired comparisons")
    _verify_per_category_table(paths[str(table_spec["per_category_seed"])], rows)
    _verify_derived_table(
        paths[str(table_spec["macro_by_seed"])],
        macro_by_seed(rows),
        "macro-by-seed table",
        ("method", "dataset", "seed"),
    )
    _verify_derived_table(
        paths[str(table_spec["method_summary"])],
        method_summary(rows),
        "method summary table",
        ("method", "dataset"),
    )
    return resolved, {
        "aggregation_manifest_sha256": sha256_file(paths[str(table_spec["aggregation_manifest"])]),
        "claim_decision": required_claim_decision,
        "row_count": len(rows),
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }


def _verify_visualizations(
    project_root: Path,
    visualizations: object,
    *,
    required_datasets: Sequence[str],
    baseline: str,
    candidate: str,
    registered_methods: set[str],
    registered_seeds: set[int],
    formal_evidence: Sequence[FormalRunEvidence],
    formal_results_root: Path,
    registration_path: Path,
    inventory_evidence: Sequence[Mapping[str, object]],
    inventory_entries: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[
    list[tuple[str, Path]],
    list[dict[str, object]],
    list[str],
]:
    if not isinstance(visualizations, list) or not visualizations:
        raise ValueError("visualizations must be a non-empty list")
    run_lookup = {
        (proof.method, proof.dataset, proof.category, proof.seed): proof
        for proof in formal_evidence
    }
    inventory_lookup = {
        str(item["dataset"]): item for item in inventory_evidence
    }
    visualization_registry = _load_visualization_method_registry(registration_path)
    packaged: list[tuple[str, Path]] = []
    summaries: list[dict[str, object]] = []
    pre_f0_tracked_paths: list[str] = []
    seen_datasets: set[str] = set()
    for index, item in enumerate(visualizations):
        if not isinstance(item, Mapping):
            raise ValueError(f"visualizations[{index}] must be an object")
        _require_exact_keys(
            item,
            (
                "dataset",
                "selection_config",
                "selection_manifest",
                "failure_resolution_manifest",
                "figure_manifest",
            ),
            f"visualizations[{index}]",
        )
        dataset = item["dataset"]
        if not isinstance(dataset, str) or dataset not in required_datasets or dataset in seen_datasets:
            raise ValueError(f"invalid or duplicate visualization dataset: {dataset!r}")
        seen_datasets.add(dataset)
        manifest_text, manifest_path = _resolve_project_file(
            project_root, item["figure_manifest"], f"visualization manifest for {dataset}"
        )
        selection_text, selection_path = _resolve_project_file(
            project_root,
            item["selection_manifest"],
            f"visualization selection manifest for {dataset}",
        )
        selection_config_text, selection_config_path = _resolve_project_file(
            project_root,
            item["selection_config"],
            f"visualization selection config for {dataset}",
        )
        failure_text, failure_path = _resolve_project_file(
            project_root,
            item["failure_resolution_manifest"],
            f"visualization failure resolution for {dataset}",
        )
        selection = _load_json(selection_path, f"visualization selection for {dataset}")
        if (
            selection.get("schema_version") != 1
            or selection.get("kind") != "formal_visualization_selection"
            or selection.get("status") != "frozen_before_formal_results"
            or selection.get("freeze_phase") != "F0_post_method_freeze_pre_inference"
            or selection.get("dataset") != dataset
            or selection.get("prediction_artifacts_accessed") is not False
            or selection.get("manual_post_result_selection_allowed") is not False
            or selection.get("payload_sha256") != _canonical_payload_sha256(selection)
            or selection.get("selection_config_sha256") != sha256_file(selection_config_path)
        ):
            raise ValueError(f"visualization selection is not a valid F0 freeze: {dataset}")
        selection_config = _load_visualization_selection_config(
            selection_config_path,
            dataset=dataset,
            registry=visualization_registry,
        )
        if selection.get("selection_config") != selection_config:
            raise ValueError(
                f"visualization selection normalized config payload drifted: {dataset}"
            )
        selection_seed = selection.get("model_seed")
        if (
            isinstance(selection_seed, bool)
            or not isinstance(selection_seed, int)
            or selection_seed not in registered_seeds
        ):
            raise ValueError(f"visualization selection uses an unregistered seed: {dataset}")
        frozen_registry = selection.get("method_registry")
        if not isinstance(frozen_registry, Mapping):
            raise ValueError(f"visualization selection lacks method registry: {dataset}")
        if frozen_registry.get("sha256") != sha256_file(registration_path):
            raise ValueError(f"visualization selection registry hash mismatch: {dataset}")
        frozen_methods = frozen_registry.get("methods")
        if (
            not isinstance(frozen_methods, list)
            or any(not isinstance(method, str) or not method for method in frozen_methods)
            or len(set(frozen_methods)) != len(frozen_methods)
        ):
            raise ValueError(f"visualization selection registry methods are invalid: {dataset}")
        if (
            frozen_registry.get("baseline") != baseline
            or frozen_registry.get("candidate") != candidate
            or set(frozen_methods) != registered_methods
        ):
            raise ValueError(f"visualization selection registry identity mismatch: {dataset}")
        inventory = inventory_lookup[dataset]
        dataset_entries = inventory_entries.get(dataset)
        if not isinstance(dataset_entries, Sequence) or not dataset_entries:
            raise ValueError(f"visualization inventory entries are missing: {dataset}")
        dataset_entry_lookup = {
            str(entry.get("sample_id")): entry
            for entry in dataset_entries
            if isinstance(entry, Mapping)
        }
        if len(dataset_entry_lookup) != len(dataset_entries):
            raise ValueError(f"visualization inventory entry identities are invalid: {dataset}")
        selection_inventory = selection.get("test_inventory")
        if (
            not isinstance(selection_inventory, Mapping)
            or selection_inventory.get("sha256") != inventory["inventory_sha256"]
            or selection_inventory.get("inventory_id") != inventory["inventory_id"]
            or selection_inventory.get("entry_count") != inventory["entry_count"]
        ):
            raise ValueError(f"visualization selection uses a different F0 inventory: {dataset}")
        frozen_samples = selection.get("result_independent_samples")
        if not isinstance(frozen_samples, list) or not frozen_samples:
            raise ValueError(f"visualization selection has no frozen samples: {dataset}")
        frozen_ids: list[str] = []
        for position, entry in enumerate(frozen_samples):
            if not isinstance(entry, Mapping):
                raise ValueError(f"invalid frozen visualization sample {position}: {dataset}")
            sample_id = entry.get("sample_id")
            expected_entry = dataset_entry_lookup.get(str(sample_id))
            if expected_entry is None or any(
                entry.get(field) != value for field, value in expected_entry.items()
            ):
                raise ValueError(f"frozen visualization sample is outside the F0 inventory: {dataset}")
            frozen_ids.append(str(sample_id))
        if len(set(frozen_ids)) != len(frozen_ids):
            raise ValueError(f"frozen visualization samples contain duplicates: {dataset}")
        failure_pool = selection.get("failure_candidate_pool")
        if not isinstance(failure_pool, list) or not failure_pool:
            raise ValueError(f"visualization selection has no frozen failure pool: {dataset}")
        failure_pool_ids: list[str] = []
        for position, entry in enumerate(failure_pool):
            if not isinstance(entry, Mapping):
                raise ValueError(f"invalid frozen failure candidate {position}: {dataset}")
            sample_id = entry.get("sample_id")
            expected_entry = dataset_entry_lookup.get(str(sample_id))
            if (
                expected_entry is None
                or expected_entry.get("label") != 1
                or any(entry.get(field) != value for field, value in expected_entry.items())
            ):
                raise ValueError(f"failure candidate is outside the F0 anomaly inventory: {dataset}")
            failure_pool_ids.append(str(sample_id))
        expected_failure_pool = sorted(
            sample_id
            for sample_id, entry in dataset_entry_lookup.items()
            if entry.get("label") == 1
        )
        if failure_pool_ids != expected_failure_pool:
            raise ValueError(f"failure candidate pool is not the complete frozen anomaly pool: {dataset}")
        failure = _load_json(failure_path, f"visualization failure resolution for {dataset}")
        if (
            failure.get("schema_version") != 1
            or failure.get("kind") != "formal_visualization_failure_resolution"
            or failure.get("status") != "resolved_by_frozen_rule"
            or failure.get("cohort") != "outcome_dependent_machine_selected"
            or failure.get("dataset") != dataset
            or failure.get("model_seed") != selection_seed
            or failure.get("rule") != selection.get("failure_case_rule")
            or failure.get("manual_selection_performed") is not False
            or failure.get("selection_manifest_sha256") != sha256_file(selection_path)
            or failure.get("selection_payload_sha256") != selection.get("payload_sha256")
            or failure.get("payload_sha256") != _canonical_payload_sha256(failure)
            or not isinstance(failure.get("selected_cases"), list)
            or not failure["selected_cases"]
        ):
            raise ValueError(f"visualization failure resolution is not bound to F0: {dataset}")
        failure_case_keys: set[tuple[str, str]] = set()
        for position, entry in enumerate(failure["selected_cases"]):
            if not isinstance(entry, Mapping):
                raise ValueError(f"invalid resolved failure case {position}: {dataset}")
            sample_id = str(entry.get("sample_id"))
            ranking = entry.get("failure_ranking")
            expected_entry = dataset_entry_lookup.get(sample_id)
            if (
                sample_id not in set(failure_pool_ids)
                or expected_entry is None
                or not isinstance(ranking, str)
                or not ranking
                or any(entry.get(field) != value for field, value in expected_entry.items())
            ):
                raise ValueError(f"resolved failure case is outside the frozen pool: {dataset}")
            case_key = (sample_id, ranking)
            if case_key in failure_case_keys:
                raise ValueError(f"resolved failure cases contain duplicates: {dataset}")
            failure_case_keys.add(case_key)
        manifest = _load_json(manifest_path, f"visualization manifest for {dataset}")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "formal_visualization_figure_manifest"
            or manifest.get("status") != "rendered_from_sha_verified_metric_arrays"
            or manifest.get("dataset") != dataset
            or manifest.get("selection_manifest_sha256") != sha256_file(selection_path)
            or manifest.get("selection_payload_sha256") != selection.get("payload_sha256")
            or manifest.get("failure_resolution_sha256") != sha256_file(failure_path)
        ):
            raise ValueError(f"visualization manifest is incomplete or non-formal: {manifest_path}")
        if manifest.get("payload_sha256") != _canonical_payload_sha256(manifest):
            raise ValueError(f"visualization manifest payload SHA-256 mismatch: {manifest_path}")
        if (
            manifest.get("metric_arrays_modified") is not False
            or manifest.get("raw_maps_rewritten") is not False
            or manifest.get("manual_sample_selection_performed") is not False
        ):
            raise ValueError(f"visualization manifest violates raw-map/sample-selection contract: {manifest_path}")
        seed = manifest.get("model_seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in registered_seeds
            or seed != selection_seed
        ):
            raise ValueError(f"visualization uses an unregistered seed: {manifest_path}")
        method_order = manifest.get("method_order")
        if not isinstance(method_order, list) or len(set(method_order)) != len(method_order):
            raise ValueError(f"visualization method order is invalid: {manifest_path}")
        if not {baseline, candidate}.issubset(set(method_order)) or not set(method_order).issubset(registered_methods):
            raise ValueError(f"visualization does not compare the frozen baseline and candidate: {manifest_path}")
        prediction_runs = manifest.get("prediction_runs")
        if not isinstance(prediction_runs, list) or not prediction_runs:
            raise ValueError(f"visualization manifest has no formal prediction provenance: {manifest_path}")
        if failure.get("prediction_runs") != prediction_runs:
            raise ValueError(f"visualization failure/figure run provenance differs: {dataset}")
        prediction_keys: set[tuple[str, str]] = set()
        for record in prediction_runs:
            if not isinstance(record, Mapping):
                raise ValueError(f"invalid visualization prediction provenance: {manifest_path}")
            if record.get("dataset") != dataset or record.get("seed") != selection_seed:
                raise ValueError(f"visualization prediction dataset/seed differs from F0: {dataset}")
            method = record.get("method")
            category = record.get("category")
            if not isinstance(method, str) or not isinstance(category, str):
                raise ValueError(f"visualization prediction identity is invalid: {dataset}")
            prediction_key = (method, category)
            if prediction_key in prediction_keys:
                raise ValueError(f"visualization prediction provenance contains duplicates: {dataset}")
            prediction_keys.add(prediction_key)
            key = (record.get("method"), record.get("dataset"), record.get("category"), record.get("seed"))
            proof = run_lookup.get(key)  # type: ignore[arg-type]
            if proof is None:
                raise ValueError(f"visualization references a run outside the formal matrix: {key}")
            try:
                referenced_root = Path(str(record.get("run_directory"))).resolve()
                referenced_root.relative_to(formal_results_root.resolve())
            except ValueError as error:
                raise ValueError(f"visualization run directory escapes formal_results_root: {key}") from error
            if referenced_root != (formal_results_root / proof.run_directory).resolve():
                raise ValueError(f"visualization run directory disagrees with formal evidence: {key}")
            for field in ("run_config_sha256", "metrics_sha256", "prediction_index_sha256"):
                if record.get(field) != getattr(proof, field):
                    raise ValueError(f"visualization formal provenance mismatch for {key}/{field}")
        formal_categories = {
            proof.category for proof in formal_evidence if proof.dataset == dataset
        }
        expected_prediction_keys = {
            (method, category)
            for method in method_order
            for category in formal_categories
        }
        if prediction_keys != expected_prediction_keys:
            raise ValueError(
                f"visualization prediction runs do not cover method_order x categories: {dataset}"
            )
        figure_records = manifest.get("figures")
        if not isinstance(figure_records, list) or not figure_records:
            raise ValueError(f"visualization manifest has no figures: {manifest_path}")
        cohorts: set[str] = set()
        fixed_figure_ids: list[str] = []
        failure_figure_ids: list[tuple[object, object]] = []
        for position, record in enumerate(figure_records):
            if not isinstance(record, Mapping):
                raise ValueError(f"invalid visualization figure record {position}: {manifest_path}")
            if str(record.get("sample_id")) not in dataset_entry_lookup:
                raise ValueError(f"visualization figure sample is outside the F0 inventory: {dataset}")
            cohorts.add(str(record.get("cohort")))
            if record.get("cohort") == "result_independent_preregistered":
                fixed_figure_ids.append(str(record.get("sample_id")))
            elif record.get("cohort") == "outcome_dependent_machine_selected":
                failure_figure_ids.append(
                    (record.get("sample_id"), record.get("failure_ranking"))
                )
            figure_rel = _validate_relative_text(record.get("figure"), f"figure record {position}")
            package_relative = (PurePosixPath(manifest_text).parent / PurePosixPath(figure_rel)).as_posix()
            path_text, figure_path = _resolve_project_file(
                project_root,
                package_relative,
                f"visualization PNG {position}",
                allow_png=True,
            )
            if figure_path.suffix.lower() != ".png" or figure_path.read_bytes()[:8] != _PNG_SIGNATURE:
                raise ValueError(f"visualization figure is not a PNG: {figure_path}")
            if record.get("figure_sha256") != sha256_file(figure_path):
                raise ValueError(f"visualization PNG SHA-256 mismatch: {figure_path}")
            methods = record.get("methods")
            if not isinstance(methods, list) or [entry.get("method") for entry in methods if isinstance(entry, Mapping)] != method_order:
                raise ValueError(f"visualization figure methods differ from method_order: {figure_path}")
            display = record.get("display")
            if not isinstance(display, Mapping) or display.get("affects_metric_data") is not False:
                raise ValueError(f"visualization display contract is missing: {figure_path}")
            packaged.append((path_text, figure_path))
        if cohorts != _EXPECTED_VISUALIZATION_COHORTS:
            raise ValueError(
                f"visualization must contain preregistered and machine-selected failure cohorts: {dataset}"
            )
        selected_failures = failure.get("selected_cases")
        if (
            not isinstance(frozen_samples, list)
            or [str(entry.get("sample_id")) for entry in frozen_samples if isinstance(entry, Mapping)]
            != fixed_figure_ids
        ):
            raise ValueError(f"visualization figures differ from frozen sample selection: {dataset}")
        if (
            not isinstance(selected_failures, list)
            or [
                (entry.get("sample_id"), entry.get("failure_ranking"))
                for entry in selected_failures
                if isinstance(entry, Mapping)
            ]
            != failure_figure_ids
        ):
            raise ValueError(f"visualization figures differ from machine failure selection: {dataset}")
        packaged.extend(
            (
                (selection_config_text, selection_config_path),
                (selection_text, selection_path),
                (failure_text, failure_path),
                (manifest_text, manifest_path),
            )
        )
        pre_f0_tracked_paths.extend((selection_config_text, selection_text))
        summaries.append(
            {
                "dataset": dataset,
                "figure_manifest": manifest_text,
                "figure_manifest_sha256": sha256_file(manifest_path),
                "selection_manifest": selection_text,
                "selection_manifest_sha256": sha256_file(selection_path),
                "selection_config": selection_config_text,
                "selection_config_sha256": sha256_file(selection_config_path),
                "failure_resolution_manifest": failure_text,
                "failure_resolution_manifest_sha256": sha256_file(failure_path),
                "figure_count": len(figure_records),
                "method_order": method_order,
                "cohorts": sorted(cohorts),
            }
        )
    if seen_datasets != set(required_datasets):
        raise ValueError(
            f"visualization delivery does not cover every formal dataset: "
            f"missing={sorted(set(required_datasets) - seen_datasets)}"
        )
    return (
        packaged,
        sorted(summaries, key=lambda row: str(row["dataset"])),
        sorted(pre_f0_tracked_paths),
    )


def validate_delivery_spec(
    project_root: str | Path,
    spec_path: str | Path,
) -> dict[str, Any]:
    """Validate every final-delivery input without creating an archive."""

    root = Path(project_root).resolve()
    spec_source = Path(spec_path)
    if not spec_source.is_absolute():
        spec_source = root / spec_source
    try:
        spec_relative = spec_source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("delivery spec must stay inside the project") from error
    _, spec_source = _resolve_project_file(root, spec_relative, "delivery spec")
    spec = _load_json(spec_source, "delivery spec")
    required_keys = (
        "schema_version",
        "kind",
        "package_name",
        "project_config",
        "registration",
        "formal_test_inventories",
        "formal_results_root",
        "expected",
        "experiment_package",
        "comparison_tables",
        "visualizations",
        "reproduction_document",
        "third_party_sources",
    )
    _require_exact_keys(spec, required_keys, "delivery spec")
    if spec["schema_version"] != DELIVERY_SPEC_SCHEMA_VERSION or spec["kind"] != "industrial_anomaly_final_delivery":
        raise ValueError("unsupported delivery spec schema or kind")
    package_name = spec["package_name"]
    if not isinstance(package_name, str) or not package_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in package_name):
        raise ValueError("package_name contains unsupported characters")

    expected = spec["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError("expected must be an object")
    _require_exact_keys(
        expected,
        ("datasets", "metrics_schema_version", "evaluation_protocol", "claim_decision"),
        "expected",
    )
    required_datasets = _require_string_list(expected["datasets"], "expected.datasets")
    if set(required_datasets) != {"mvtec_ad", "visa"}:
        raise ValueError("final delivery must cover exactly MVTec AD and VisA")
    if expected["metrics_schema_version"] != METRICS_SCHEMA_VERSION:
        raise ValueError("delivery spec expects an unsupported or legacy metrics schema")
    if expected["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("delivery spec expects an unsupported evaluation protocol")
    if expected["claim_decision"] != "pass":
        raise ValueError("final delivery requires the registered improvement claim to pass")

    project_text, project_path = _resolve_project_file(root, spec["project_config"], "project config")
    registration_text, registration_path = _resolve_project_file(root, spec["registration"], "registration")
    project = _load_json(project_path, "project config")
    registration = _load_json(registration_path, "registration")
    if project.get("evaluation", {}).get("formal_protocol_frozen") is not True:
        raise ValueError("formal evaluation protocol is not frozen")
    if project.get("baseline", {}).get("formal_training_recipe_frozen") is not True:
        raise ValueError("formal baseline training recipe is not frozen")
    registry = registration.get("formal_method_registry")
    if not isinstance(registry, Mapping) or registry.get("status") != "frozen_before_formal_evaluation":
        raise ValueError("formal method registry is not frozen before evaluation")
    methods = _require_string_list(registry.get("methods"), "formal_method_registry.methods")
    baseline = registry.get("baseline")
    candidate = registry.get("candidate")
    if not isinstance(baseline, str) or not isinstance(candidate, str) or baseline == candidate:
        raise ValueError("formal baseline/candidate registry is incomplete")
    if baseline not in methods or candidate not in methods:
        raise ValueError("formal baseline/candidate is absent from registered methods")
    seeds = registration.get("model_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("registration has invalid model seeds")
    project_datasets = project.get("datasets")
    if not isinstance(project_datasets, Mapping) or set(project_datasets) != set(required_datasets):
        raise ValueError("project datasets differ from delivery scope")
    categories: dict[str, list[str]] = {}
    for dataset in required_datasets:
        payload = project_datasets.get(dataset)
        if not isinstance(payload, Mapping):
            raise ValueError(f"missing project dataset config: {dataset}")
        categories[dataset] = _require_string_list(payload.get("categories"), f"project categories for {dataset}")

    experiment = spec["experiment_package"]
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment_package must be an object")
    _require_exact_keys(experiment, ("files", "method_roles", "method_implementation_files"), "experiment_package")
    common_files = _require_string_list(experiment["files"], "experiment_package.files")
    roles = experiment["method_roles"]
    if not isinstance(roles, Mapping):
        raise ValueError("method_roles must be an object")
    _require_exact_keys(roles, ("baseline", "candidate", "ablations", "strong_controls"), "method_roles")
    if roles["baseline"] != baseline or roles["candidate"] != candidate:
        raise ValueError("delivery method roles differ from formal registry")
    ablations = _require_string_list(roles["ablations"], "method_roles.ablations")
    controls = _require_string_list(roles["strong_controls"], "method_roles.strong_controls")
    if not {"PatchCore", "RD4AD"}.issubset(set(controls)):
        raise ValueError("strong controls must include PatchCore and RD4AD")
    role_methods = [baseline, candidate, *ablations, *controls]
    if len(set(role_methods)) != len(role_methods) or set(role_methods) != set(methods):
        raise ValueError("baseline/candidate/ablation/control roles must exactly partition registered methods")
    implementations = experiment["method_implementation_files"]
    if not isinstance(implementations, Mapping) or set(implementations) != set(methods):
        raise ValueError("method_implementation_files must cover exactly the registered methods")

    payloads: dict[str, tuple[Path, str]] = {}

    def add_payload(path_value: object, deliverable: str, context: str, *, allow_png: bool = False) -> str:
        relative, path = _resolve_project_file(root, path_value, context, allow_png=allow_png)
        if relative in payloads and payloads[relative][1] != deliverable:
            # One file may support several methods, but its archive identity is
            # recorded once under the first declared deliverable.
            return relative
        payloads[relative] = (path, deliverable)
        return relative

    for position, value in enumerate(common_files):
        add_payload(value, "experiment_package", f"experiment_package.files[{position}]")
    for method in methods:
        files = _require_string_list(implementations[method], f"implementation files for {method}")
        for position, value in enumerate(files):
            add_payload(value, "experiment_package", f"implementation files for {method}[{position}]")
    if project_text not in payloads or registration_text not in payloads:
        raise ValueError("project config and registration must be in the experiment package allowlist")

    inventory_files, inventory_evidence, inventory_entries = _verify_formal_test_inventories(
        root,
        spec["formal_test_inventories"],
        required_datasets=required_datasets,
        categories=categories,
        project_config_path=project_path,
        registration_path=registration_path,
        included_paths=set(payloads),
    )
    for relative, path in inventory_files:
        if relative in payloads:
            raise ValueError(f"formal test inventory duplicates another delivery path: {relative}")
        payloads[relative] = (path, "experiment_package")

    reproduction_text = add_payload(
        spec["reproduction_document"],
        "reproduction_document",
        "reproduction document",
    )
    if reproduction_text != "docs/reproduction.md":
        raise ValueError("the final reproduction document must be docs/reproduction.md")

    results_text, results_root = _resolve_project_directory(root, spec["formal_results_root"], "formal_results_root")
    results_parts = tuple(part.lower() for part in PurePosixPath(results_text).parts)
    if (
        results_parts[0] in _FORBIDDEN_TOP_LEVEL - {"runs"}
        or (
            len(results_parts) >= 2
            and results_parts[0] == "artifacts"
            and results_parts[1] in _FORBIDDEN_ARTIFACT_SUBTREES
        )
    ):
        raise ValueError("formal_results_root is in a cache/data directory")
    rows, formal_evidence = _load_formal_runs(
        results_root,
        inventory_evidence,
        inventory_entries,
        root,
    )
    _assert_complete_matrix(rows, methods=methods, categories=categories, seeds=seeds)

    if not isinstance(spec["comparison_tables"], Mapping):
        raise ValueError("comparison_tables must be an object")
    table_files, aggregation_evidence = _verify_aggregation(
        root,
        spec["comparison_tables"],
        rows=rows,
        methods=methods,
        categories=categories,
        seeds=seeds,
        registration_path=registration_path,
        required_claim_decision="pass",
    )
    for relative, path in table_files:
        if relative in payloads:
            raise ValueError(f"comparison table path duplicates another deliverable: {relative}")
        payloads[relative] = (path, "comparison_tables")

    visual_files, visualization_evidence, visualization_pre_f0_paths = _verify_visualizations(
        root,
        spec["visualizations"],
        required_datasets=required_datasets,
        baseline=baseline,
        candidate=candidate,
        registered_methods=set(methods),
        registered_seeds=set(seeds),
        formal_evidence=formal_evidence,
        formal_results_root=results_root,
        registration_path=registration_path,
        inventory_evidence=inventory_evidence,
        inventory_entries=inventory_entries,
    )
    for relative, path in visual_files:
        if relative in payloads:
            raise ValueError(f"visualization path duplicates another deliverable: {relative}")
        payloads[relative] = (path, "visualizations")

    add_payload(spec_relative, "delivery_spec", "delivery spec")
    tracked_paths = sorted(
        {spec_relative, reproduction_text, *common_files}
        | {relative for relative, _path in inventory_files}
        | set(visualization_pre_f0_paths)
        | {
            _validate_relative_text(value, f"implementation file for {method}")
            for method in methods
            for value in implementations[method]
        }
    )
    repository = _verify_project_git(root, tracked_paths)
    third_party = _verify_third_party(root, spec["third_party_sources"], set(payloads))

    packaged_files = [
        PackagedFile(
            path=relative,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            deliverable=deliverable,
        )
        for relative, (path, deliverable) in sorted(payloads.items())
    ]
    tree_payload = [
        {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
        for item in packaged_files
    ]
    tree_sha = hashlib.sha256(_canonical_json_bytes(tree_payload)).hexdigest()
    return {
        "schema_version": DELIVERY_MANIFEST_SCHEMA_VERSION,
        "kind": "industrial_anomaly_final_delivery_manifest",
        "package_name": package_name,
        "delivery_status": "complete_formal_claim_passed",
        "spec": {"path": spec_relative, "sha256": sha256_file(spec_source)},
        "scope": {
            "datasets": required_datasets,
            "methods": methods,
            "method_roles": dict(roles),
            "seeds": seeds,
        },
        "repository": repository,
        "third_party_sources": third_party,
        "formal_evidence": {
            "results_root": results_text,
            "run_count": len(formal_evidence),
            "runs": [asdict(item) for item in sorted(formal_evidence, key=lambda item: (item.method, item.dataset, item.category, item.seed))],
            "aggregation": aggregation_evidence,
        },
        "deliverables": {
            "experiment_package": {
                "present": True,
                "method_implementation_files": dict(implementations),
                "formal_test_inventories": inventory_evidence,
            },
            "comparison_tables": {"present": True, **dict(spec["comparison_tables"])},
            "visualizations": {"present": True, "datasets": visualization_evidence},
            "reproduction_document": {"present": True, "path": reproduction_text},
        },
        "files": [asdict(item) for item in packaged_files],
        "payload_tree_sha256": tree_sha,
        "archive_contract": {
            "member_order": "UTF-8 POSIX path ascending; manifest last",
            "timestamp": "1980-01-01T00:00:00",
            "unix_mode": "100644",
            "compression": "deflate level 9",
        },
        "_payload_paths": {item.path: str(payloads[item.path][0]) for item in packaged_files},
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info.flag_bits |= 0x800
    return info


def verify_delivery_archive(archive_path: str | Path) -> dict[str, Any]:
    """Reopen an archive and verify CRC, member set, sizes and SHA-256."""

    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            infos = handle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("delivery ZIP contains duplicate member names")
            if not names or names[-1] != DELIVERY_MANIFEST_NAME:
                raise ValueError("delivery ZIP manifest must be the final member")
            if names[:-1] != sorted(names[:-1]):
                raise ValueError("delivery ZIP payload members are not deterministically ordered")
            if handle.testzip() is not None:
                raise ValueError("delivery ZIP CRC verification failed")
            for info in infos:
                _validate_relative_text(info.filename, "ZIP member")
                if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG:
                    raise ValueError(f"delivery ZIP contains a non-regular member: {info.filename}")
                if info.date_time != _ZIP_TIMESTAMP:
                    raise ValueError(f"delivery ZIP timestamp is not deterministic: {info.filename}")
                if (info.external_attr >> 16) & 0o777 != 0o644:
                    raise ValueError(f"delivery ZIP permission is not deterministic: {info.filename}")
                if info.compress_type != zipfile.ZIP_DEFLATED:
                    raise ValueError(f"delivery ZIP member is not deflate-compressed: {info.filename}")
            if DELIVERY_MANIFEST_NAME not in names:
                raise ValueError("delivery ZIP is missing DELIVERY-MANIFEST.json")
            manifest = json.loads(handle.read(DELIVERY_MANIFEST_NAME).decode("utf-8"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != DELIVERY_MANIFEST_SCHEMA_VERSION
                or manifest.get("kind") != "industrial_anomaly_final_delivery_manifest"
                or manifest.get("delivery_status") != "complete_formal_claim_passed"
            ):
                raise ValueError("delivery manifest uses an unsupported schema")
            repository = manifest.get("repository")
            third_party = manifest.get("third_party_sources")
            if (
                not isinstance(repository, Mapping)
                or not isinstance(repository.get("commit"), str)
                or len(repository["commit"]) != 40
                or not isinstance(third_party, list)
                or not third_party
            ):
                raise ValueError("delivery manifest lacks project/third-party commit provenance")
            records = manifest.get("files")
            if not isinstance(records, list) or not records:
                raise ValueError("delivery manifest has no file records")
            archive_contract = manifest.get("archive_contract")
            if archive_contract != {
                "member_order": "UTF-8 POSIX path ascending; manifest last",
                "timestamp": "1980-01-01T00:00:00",
                "unix_mode": "100644",
                "compression": "deflate level 9",
            }:
                raise ValueError("delivery manifest archive contract is missing or changed")
            deliverables = manifest.get("deliverables")
            required_deliverables = {
                "experiment_package",
                "comparison_tables",
                "visualizations",
                "reproduction_document",
            }
            if (
                not isinstance(deliverables, Mapping)
                or set(deliverables) != required_deliverables
                or any(
                    not isinstance(deliverables[name], Mapping)
                    or deliverables[name].get("present") is not True
                    for name in required_deliverables
                )
            ):
                raise ValueError("delivery manifest does not declare all four deliverables present")
            expected_names = {DELIVERY_MANIFEST_NAME}
            tree_payload: list[dict[str, object]] = []
            recorded_paths: set[str] = set()
            deliverable_counts = Counter()
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("delivery manifest contains an invalid file record")
                relative = _validate_relative_text(record.get("path"), "delivery manifest file")
                if relative in recorded_paths:
                    raise ValueError(f"delivery manifest repeats a file path: {relative}")
                recorded_paths.add(relative)
                deliverable = record.get("deliverable")
                if deliverable not in {
                    "delivery_spec",
                    "experiment_package",
                    "comparison_tables",
                    "visualizations",
                    "reproduction_document",
                }:
                    raise ValueError(f"delivery manifest has an unknown deliverable type: {relative}")
                deliverable_counts[str(deliverable)] += 1
                _reject_forbidden_payload_path(
                    relative, allow_png=deliverable == "visualizations"
                )
                expected_names.add(relative)
                data = handle.read(relative)
                observed_sha = hashlib.sha256(data).hexdigest()
                if record.get("size_bytes") != len(data) or record.get("sha256") != observed_sha:
                    raise ValueError(f"delivery ZIP member integrity mismatch: {relative}")
                tree_payload.append(
                    {"path": relative, "size_bytes": len(data), "sha256": observed_sha}
                )
            if set(names) != expected_names:
                raise ValueError("delivery ZIP members differ from the manifest allowlist")
            if any(deliverable_counts[name] <= 0 for name in required_deliverables):
                raise ValueError("delivery ZIP payload is missing one of the four deliverables")
            if deliverable_counts["delivery_spec"] != 1:
                raise ValueError("delivery ZIP must contain exactly one delivery specification")
            tree_payload.sort(key=lambda item: str(item["path"]))
            tree_sha = hashlib.sha256(_canonical_json_bytes(tree_payload)).hexdigest()
            if manifest.get("payload_tree_sha256") != tree_sha:
                raise ValueError("delivery payload tree SHA-256 mismatch")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or corrupted delivery ZIP: {archive}") from error
    return {
        "status": "verified",
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "payload_tree_sha256": manifest["payload_tree_sha256"],
        "file_count": len(manifest["files"]),
        "project_commit": manifest["repository"]["commit"],
        "third_party_sources": manifest["third_party_sources"],
    }


def build_delivery_archive(
    project_root: str | Path,
    spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate, build deterministically, then independently verify a ZIP."""

    root = Path(project_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    try:
        output_relative = output.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("delivery ZIP output must stay inside the project") from error
    _validate_relative_text(output_relative, "delivery ZIP output")
    if output.suffix.lower() != ".zip":
        raise ValueError("delivery output must have a .zip suffix")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite delivery ZIP: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    current = root
    for part in output.parent.relative_to(root).parts:
        current = current / part
        if _is_reparse_or_symlink(current):
            raise ValueError("delivery output path must not contain a link/reparse point")
    output = output.resolve()

    validation = validate_delivery_spec(root, spec_path)
    payload_paths = validation.pop("_payload_paths")
    manifest_bytes = _canonical_json_bytes(validation)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as handle:
            for relative in sorted(payload_paths):
                data = Path(payload_paths[relative]).read_bytes()
                handle.writestr(_zip_info(relative), data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            handle.writestr(
                _zip_info(DELIVERY_MANIFEST_NAME),
                manifest_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        return verify_delivery_archive(output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
