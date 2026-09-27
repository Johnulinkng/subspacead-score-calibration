"""Fail-closed training-input resolution for frozen development manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from .development_protocol import (
    IMAGE_SUFFIXES,
    collect_renderer_provenance,
    derive_expected_source_split_rows,
    require_exact_canonical_rows,
    validate_registered_model_seeds,
    validate_source_split_config,
    validate_synthetic_development_config,
)
from .run_artifacts import sha256_file


AUTHOR_FULL_TRAIN = "author_full_train"
VISA_DEVELOPMENT_PROTOCOL = "visa_train_dev_dtd_v1"
SUPPORTED_INPUT_PROTOCOLS = frozenset({AUTHOR_FULL_TRAIN, VISA_DEVELOPMENT_PROTOCOL})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ManifestTrainingInputs:
    normal_image_paths: tuple[Path, ...]
    anomaly_source_paths: tuple[Path, ...]
    evidence: dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    payload, _ = _load_json_object_and_sha256(path)
    return payload


def _load_json_object_and_sha256(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_safe_project_path(path: Path, root: Path, label: str) -> Path:
    _require_no_link_components(path, root, label)
    resolved = path.resolve()
    if not _inside(resolved, root):
        raise ValueError(f"{label} escapes the project root: {path}")
    if resolved.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    return resolved


def _require_no_link_components(path: Path, root: Path, label: str) -> None:
    """Reject symlinks and Windows reparse points before path resolution."""

    candidate = Path(os.path.abspath(path))
    anchor = Path(os.path.abspath(root))
    try:
        relative = candidate.relative_to(anchor)
    except ValueError as error:
        raise ValueError(f"{label} escapes its allowed root: {path}") from error
    current = anchor
    for part in relative.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
            raise ValueError(f"{label} contains a linked/reparse component: {current}")


def _parse_jsonl_bytes(payload: bytes, path: Path) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"manifest must be UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line is not allowed: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row: {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL object expected: {path}:{line_number}")
        rows.append(row)
    return rows


def _artifact_index(
    *,
    summary: dict[str, Any],
    manifest_root: Path,
    expected_names: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    entries = summary.get("artifact_files")
    if not isinstance(entries, list):
        raise ValueError("root manifest artifact_files must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("root manifest artifact entry must be an object")
        recorded_path = raw_entry.get("path")
        if not isinstance(recorded_path, str) or "\\" in recorded_path:
            raise ValueError("artifact path must be a POSIX-style string")
        name = PurePosixPath(recorded_path).name
        if name in by_name:
            raise ValueError(f"duplicate artifact name in root manifest: {name}")
        if not isinstance(raw_entry.get("rows"), int) or int(raw_entry["rows"]) < 0:
            raise ValueError(f"invalid row count for artifact: {name}")
        recorded_hash = raw_entry.get("sha256")
        if not isinstance(recorded_hash, str) or not _SHA256_PATTERN.fullmatch(recorded_hash):
            raise ValueError(f"invalid artifact SHA-256 for: {name}")
        by_name[name] = raw_entry
    if set(by_name) != expected_names:
        missing = sorted(expected_names - set(by_name))
        extra = sorted(set(by_name) - expected_names)
        raise ValueError(f"frozen artifact set mismatch; missing={missing}, extra={extra}")

    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(expected_names):
        artifact_path = _require_safe_project_path(
            manifest_root / name, manifest_root, f"artifact {name}"
        )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {artifact_path}")
        payload = artifact_path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != by_name[name]["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")
        rows = _parse_jsonl_bytes(payload, artifact_path)
        if len(rows) != by_name[name]["rows"]:
            raise ValueError(f"artifact row-count mismatch: {name}")
        rows_by_name[name] = rows
    return rows_by_name, by_name


def _validated_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} relative_path must be a non-empty POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} relative_path is unsafe: {value}")
    if relative.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"{label} has unsupported image suffix: {value}")
    return relative


def _validate_common_row(row: dict[str, Any], *, label: str) -> tuple[str, int, str]:
    split = row.get("split")
    if split not in {"train", "dev"}:
        raise ValueError(f"{label} split must be train or dev")
    size_bytes = row.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"{label} size_bytes must be a non-negative integer")
    content_hash = row.get("sha256")
    if not isinstance(content_hash, str) or not _SHA256_PATTERN.fullmatch(content_hash):
        raise ValueError(f"{label} sha256 must be lowercase hexadecimal")
    split_key = row.get("split_key")
    if not isinstance(split_key, str) or not _SHA256_PATTERN.fullmatch(split_key):
        raise ValueError(f"{label} split_key must be lowercase hexadecimal")
    return split, size_bytes, content_hash


def _validate_visa_rows(
    rows: Sequence[dict[str, Any]], category: str
) -> list[dict[str, Any]]:
    seen_paths: set[str] = set()
    hash_splits: dict[str, set[str]] = {}
    for row_number, row in enumerate(rows, start=1):
        label = f"VisA {category} row {row_number}"
        if row.get("dataset") != "visa":
            raise ValueError(f"{label} dataset must be visa")
        if row.get("category") != category:
            raise ValueError(f"{label} category mismatch")
        split, _, content_hash = _validate_common_row(row, label=label)
        relative = _validated_relative_path(row.get("relative_path"), label=label)
        if relative.parts[:3] != (category, "train", "good") or len(relative.parts) < 4:
            raise ValueError(f"{label} path is outside {category}/train/good")
        key = relative.as_posix().casefold()
        if key in seen_paths:
            raise ValueError(f"duplicate VisA relative_path: {relative}")
        seen_paths.add(key)
        hash_splits.setdefault(content_hash, set()).add(split)
    if any(len(splits) != 1 for splits in hash_splits.values()):
        raise ValueError(f"VisA {category} content hash appears in both train and dev")
    selected = [row for row in rows if row["split"] == "train"]
    if not selected:
        raise ValueError(f"VisA {category} selected training set is empty")
    return selected


def _verify_selected_files(
    rows: Sequence[dict[str, Any]],
    *,
    source_root: Path,
    allowed_root: Path,
    label: str,
) -> tuple[tuple[Path, ...], int]:
    paths: list[Path] = []
    total_bytes = 0
    for row_number, row in enumerate(rows, start=1):
        relative = _validated_relative_path(
            row.get("relative_path"), label=f"{label} row {row_number}"
        )
        candidate = source_root / Path(*relative.parts)
        _require_no_link_components(candidate, allowed_root, label)
        path = candidate.resolve()
        if not _inside(path, allowed_root):
            raise ValueError(f"{label} path escapes its allowed training root: {relative}")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} selected file is missing or linked: {path}")
        expected_size = int(row["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"{label} file-size mismatch: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{label} source SHA-256 mismatch: {relative}")
        paths.append(path)
        total_bytes += actual_size
    if not paths:
        raise ValueError(f"{label} selected training set is empty")
    return tuple(paths), total_bytes


def _selection_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_manifest_training_inputs(
    *,
    dataset: str,
    category: str,
    dataset_root: Path,
    anomaly_source_root: Path,
    seed: int,
    protocol_config_path: Path,
) -> ManifestTrainingInputs:
    """Resolve only frozen ``split=train`` paths and verify their file content."""

    if dataset != "visa":
        raise ValueError(f"{VISA_DEVELOPMENT_PROTOCOL} supports only dataset='visa'")
    config_candidate = Path(protocol_config_path).absolute()
    project_root = config_candidate.parent.parent.absolute()
    config_path = _require_safe_project_path(
        config_candidate, project_root, "development protocol config"
    )
    config, config_hash = _load_json_object_and_sha256(config_path)
    if not _inside(config_path, project_root):
        raise ValueError("development protocol config is outside the project root")
    if config.get("protocol_id") != VISA_DEVELOPMENT_PROTOCOL:
        raise ValueError("development protocol_id mismatch")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("development protocol schema_version must be integer 1")
    if config.get("status") != "frozen_before_model_development":
        raise ValueError("development protocol is not frozen_before_model_development")
    registered_seeds = validate_registered_model_seeds(config)
    if seed not in registered_seeds:
        raise ValueError(f"seed {seed} is not one of the three registered model seeds")

    visa_config = config.get("visa")
    dtd_config = config.get("dtd")
    frozen = config.get("frozen_manifest")
    if not isinstance(visa_config, dict) or not isinstance(dtd_config, dict):
        raise ValueError("development protocol visa and dtd entries must be objects")
    synthetic_config = config.get("synthetic_development")
    if not isinstance(synthetic_config, dict):
        raise ValueError("development protocol synthetic_development must be an object")
    validate_source_split_config(visa_config, label="VisA")
    validate_source_split_config(dtd_config, label="DTD")
    validate_synthetic_development_config(synthetic_config)
    if not isinstance(frozen, dict):
        raise ValueError("development protocol frozen_manifest entry is required")
    categories = visa_config.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("development protocol categories must be a non-empty list")
    if any(
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).name != name
        for name in categories
    ):
        raise ValueError("development protocol contains an unsafe category name")
    if len(categories) != len(set(categories)):
        raise ValueError("development protocol categories must be unique")
    if category not in categories:
        raise ValueError(f"category is not registered by the development protocol: {category}")

    expected_visa_root = (project_root / str(visa_config["root"])).resolve()
    expected_dtd_root = (project_root / str(dtd_config["root"])).resolve()
    if dataset_root.resolve() != expected_visa_root:
        raise ValueError("dataset_root does not match the frozen development protocol")
    if anomaly_source_root.resolve() != expected_dtd_root:
        raise ValueError("anomaly_source_root does not match the frozen development protocol")

    manifest_relative = _validated_relative_path_for_json(
        frozen.get("path"), label="frozen root manifest"
    )
    root_manifest_path = _require_safe_project_path(
        project_root / Path(*manifest_relative.parts), project_root, "frozen root manifest"
    )
    if root_manifest_path.suffix.lower() != ".json" or not root_manifest_path.is_file():
        raise FileNotFoundError(f"frozen root manifest is missing: {root_manifest_path}")
    pinned_root_hash = frozen.get("sha256")
    if not isinstance(pinned_root_hash, str) or not _SHA256_PATTERN.fullmatch(
        pinned_root_hash
    ):
        raise ValueError("frozen root manifest SHA-256 is invalid")
    summary, actual_root_hash = _load_json_object_and_sha256(root_manifest_path)
    if actual_root_hash != pinned_root_hash:
        raise ValueError("frozen root manifest SHA-256 mismatch")
    if summary.get("protocol_id") != VISA_DEVELOPMENT_PROTOCOL:
        raise ValueError("root manifest protocol_id mismatch")
    if summary.get("formal_test_content_accessed") is not False:
        raise ValueError("root manifest does not assert formal-test isolation")
    if summary.get("training_started") is not False:
        raise ValueError("root manifest was not produced by the no-training builder")
    if summary.get("model_seeds") != registered_seeds:
        raise ValueError("root manifest registered seeds mismatch")
    current_renderer_provenance = collect_renderer_provenance()
    if summary.get("renderer_provenance") != current_renderer_provenance:
        raise ValueError("root manifest renderer provenance does not match this runtime")

    expected_names = {f"visa_{name}_split.jsonl" for name in categories}
    expected_names.update({"dtd_source_split.jsonl", "synthetic_dev_recipes.jsonl"})
    manifest_root = root_manifest_path.parent
    rows_by_name, artifact_entries = _artifact_index(
        summary=summary,
        manifest_root=manifest_root,
        expected_names=expected_names,
    )
    manifest_prefix = manifest_root.relative_to(project_root).as_posix()
    for name, entry in artifact_entries.items():
        if entry.get("path") != f"{manifest_prefix}/{name}":
            raise ValueError(f"root manifest artifact path mismatch: {name}")
    if summary.get("path_base") != "project_root" or summary.get("project_root") != ".":
        raise ValueError("root manifest portable path contract mismatch")
    if summary.get("config_path") != config_path.relative_to(project_root).as_posix():
        raise ValueError("root manifest config_path mismatch")

    visa_summary = summary.get("visa")
    if not isinstance(visa_summary, dict) or not isinstance(
        visa_summary.get("categories"), dict
    ):
        raise ValueError("root manifest VisA summary is missing")
    if set(visa_summary["categories"]) != set(categories):
        raise ValueError("root manifest VisA category set mismatch")
    selected_visa_by_category: dict[str, list[dict[str, Any]]] = {}
    total_visa_rows = 0
    total_visa_train = 0
    global_visa_hash_splits: dict[str, set[str]] = {}
    for registered_category in categories:
        registered_rows = rows_by_name[f"visa_{registered_category}_split.jsonl"]
        registered_train = _validate_visa_rows(registered_rows, registered_category)
        expected_rows = derive_expected_source_split_rows(
            registered_rows,
            dataset="visa",
            dev_fraction=float(visa_config["dev_fraction"]),
            split_salt=str(visa_config["split_salt"]),
        )
        require_exact_canonical_rows(
            registered_rows,
            expected_rows,
            label=f"VisA {registered_category} source split",
        )
        expected_category_counts = visa_summary["categories"].get(registered_category)
        actual_category_counts = {
            "total": len(registered_rows),
            "train": len(registered_train),
            "dev": len(registered_rows) - len(registered_train),
        }
        if expected_category_counts != actual_category_counts:
            raise ValueError(
                f"root manifest VisA {registered_category} counts mismatch"
            )
        selected_visa_by_category[registered_category] = registered_train
        for row in registered_rows:
            global_visa_hash_splits.setdefault(str(row["sha256"]), set()).add(
                str(row["split"])
            )
        total_visa_rows += len(registered_rows)
        total_visa_train += len(registered_train)
    if any(len(splits) != 1 for splits in global_visa_hash_splits.values()):
        raise ValueError("VisA content hash appears in both train and dev")
    expected_visa_totals = {
        "total": total_visa_rows,
        "train": total_visa_train,
        "dev": total_visa_rows - total_visa_train,
        "content_hash_overlap": 0,
    }
    for key, value in expected_visa_totals.items():
        if visa_summary.get(key) != value:
            raise ValueError(f"root manifest VisA {key} mismatch")

    visa_name = f"visa_{category}_split.jsonl"
    visa_rows = rows_by_name[visa_name]
    selected_visa = selected_visa_by_category[category]

    dtd_rows = rows_by_name["dtd_source_split.jsonl"]
    seen_dtd_paths: set[str] = set()
    dtd_hash_splits: dict[str, set[str]] = {}
    for row_number, row in enumerate(dtd_rows, start=1):
        label = f"DTD row {row_number}"
        if row.get("dataset") != "dtd":
            raise ValueError(f"{label} dataset must be dtd")
        row_category = row.get("category")
        if not isinstance(row_category, str) or not row_category:
            raise ValueError(f"{label} category is invalid")
        split, _, content_hash = _validate_common_row(row, label=label)
        relative = _validated_relative_path(row.get("relative_path"), label=label)
        if len(relative.parts) < 2 or relative.parts[0] != row_category:
            raise ValueError(f"{label} category/path mismatch")
        key = relative.as_posix().casefold()
        if key in seen_dtd_paths:
            raise ValueError(f"duplicate DTD relative_path: {relative}")
        seen_dtd_paths.add(key)
        dtd_hash_splits.setdefault(content_hash, set()).add(split)
    if any(len(splits) != 1 for splits in dtd_hash_splits.values()):
        raise ValueError("DTD content hash appears in both train and dev")
    expected_dtd_rows = derive_expected_source_split_rows(
        dtd_rows,
        dataset="dtd",
        dev_fraction=float(dtd_config["dev_fraction"]),
        split_salt=str(dtd_config["split_salt"]),
    )
    require_exact_canonical_rows(
        dtd_rows, expected_dtd_rows, label="DTD source split"
    )
    selected_dtd = [row for row in dtd_rows if row["split"] == "train"]

    dtd_summary = summary.get("dtd")
    actual_dtd_counts = {
        "categories": len({str(row["category"]) for row in dtd_rows}),
        "content_hash_overlap": 0,
        "dev": len(dtd_rows) - len(selected_dtd),
        "total": len(dtd_rows),
        "train": len(selected_dtd),
    }
    if dtd_summary != actual_dtd_counts:
        raise ValueError("root manifest DTD counts mismatch")

    normal_allowed_root = expected_visa_root / category / "train" / "good"
    normal_paths, normal_bytes = _verify_selected_files(
        selected_visa,
        source_root=expected_visa_root,
        allowed_root=normal_allowed_root,
        label="VisA train",
    )
    dtd_paths, dtd_bytes = _verify_selected_files(
        selected_dtd,
        source_root=expected_dtd_root,
        allowed_root=expected_dtd_root,
        label="DTD train",
    )

    evidence = {
        "input_protocol": VISA_DEVELOPMENT_PROTOCOL,
        "manifest_backed": True,
        "development_protocol_eligible": True,
        "protocol_config_path": config_path.as_posix(),
        "protocol_config_sha256": config_hash,
        "root_manifest_path": root_manifest_path.as_posix(),
        "root_manifest_sha256": actual_root_hash,
        "category_manifest": {
            "path": (manifest_root / visa_name).as_posix(),
            "sha256": artifact_entries[visa_name]["sha256"],
            "rows": len(visa_rows),
            "selected_train_rows": len(selected_visa),
            "selected_train_sha256": _selection_hash(selected_visa),
        },
        "dtd_manifest": {
            "path": (manifest_root / "dtd_source_split.jsonl").as_posix(),
            "sha256": artifact_entries["dtd_source_split.jsonl"]["sha256"],
            "rows": len(dtd_rows),
            "selected_train_rows": len(selected_dtd),
            "selected_train_sha256": _selection_hash(selected_dtd),
        },
        "registered_seed": seed,
        "registered_seeds": registered_seeds,
        "renderer_provenance": current_renderer_provenance,
        "renderer_provenance_sha256": _selection_hash(
            [current_renderer_provenance]
        ),
        "normal_training_images": len(normal_paths),
        "normal_training_bytes": normal_bytes,
        "anomaly_source_images": len(dtd_paths),
        "anomaly_source_bytes": dtd_bytes,
        "selected_file_size_and_sha256_verified": True,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "training_started": False,
    }
    return ManifestTrainingInputs(normal_paths, dtd_paths, evidence)


def _validated_relative_path_for_json(value: object, *, label: str) -> PurePosixPath:
    """Validate a project-relative JSON path whose suffix need not be an image."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} path must be a non-empty POSIX string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} path is unsafe: {value}")
    return relative
