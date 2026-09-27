"""Content-addressed, train-only input bundles for formal experiments.

The builder and resolver in this module deliberately know only the
``category/train/good`` directories of MVTec AD and VisA plus the DTD image
root used by DRAEM.  Test and ground-truth paths are neither discovered nor
accepted as bundle entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .run_artifacts import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_INPUT_MANIFEST = (
    PROJECT_ROOT / "manifests" / "formal_training_inputs" / "v1" / "manifest.json"
)
FORMAL_INPUT_CONTRACT = "formal_training_input_bundle_v1"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
SUPPORTED_METHODS = frozenset({"DRAEM", "PatchCore", "RD4AD"})
SUPPORTED_DATASETS = frozenset({"mvtec_ad", "visa"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROW_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "category",
        "split",
        "role",
        "project_relative_path",
        "size_bytes",
        "sha256",
    }
)


@dataclass(frozen=True)
class FormalTrainingInputs:
    """Authenticated paths and deterministic evidence for one formal run."""

    normal_image_paths: tuple[Path, ...]
    anomaly_source_paths: tuple[Path, ...]
    evidence: dict[str, Any]


def _json_bytes(payload: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
    else:
        text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
    return text.encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def _load_json_object(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _parse_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not read UTF-8 JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL line is forbidden: {path}:{number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row: {path}:{number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL object expected: {path}:{number}")
        rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_link_or_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _require_no_link_components(path: Path, root: Path, label: str) -> None:
    candidate = Path(os.path.abspath(path))
    anchor = Path(os.path.abspath(root))
    try:
        relative = candidate.relative_to(anchor)
    except ValueError as error:
        raise ValueError(f"{label} escapes its allowed root: {path}") from error
    current = anchor
    if current.exists() and _is_link_or_reparse(current):
        raise ValueError(f"{label} contains a linked/reparse component: {current}")
    for part in relative.parts:
        current = current / part
        if not current.exists():
            break
        if _is_link_or_reparse(current):
            raise ValueError(f"{label} contains a linked/reparse component: {current}")


def _safe_existing_path(path: Path, project_root: Path, label: str) -> Path:
    _require_no_link_components(path, project_root, label)
    resolved = path.resolve()
    if not _inside(resolved, project_root):
        raise ValueError(f"{label} escapes project root: {path}")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _validated_project_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty project-relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} is unsafe: {value}")
    return relative


def _project_relative(path: Path, project_root: Path, label: str) -> str:
    safe = _safe_existing_path(path, project_root, label)
    return safe.relative_to(project_root.resolve()).as_posix()


def _validate_name(value: str, *, label: str) -> None:
    if (
        not value
        or "\\" in value
        or "/" in value
        or PurePosixPath(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"unsafe {label}: {value!r}")


def _walk_images(directory: Path, *, project_root: Path, label: str) -> list[Path]:
    """Walk only an explicitly supplied training root without following links."""

    root = _safe_existing_path(directory, project_root, label)
    if not root.is_dir():
        raise NotADirectoryError(root)
    images: list[Path] = []
    pending = [root]
    while pending:
        current = pending.pop()
        entries = sorted(os.scandir(current), key=lambda entry: entry.name.casefold())
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or _is_link_or_reparse(path):
                raise ValueError(f"{label} contains a linked/reparse entry: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path.resolve())
    images.sort(key=lambda value: value.relative_to(project_root.resolve()).as_posix().casefold())
    if not images:
        raise ValueError(f"{label} contains no supported images: {root}")
    return images


def _file_row(
    path: Path,
    *,
    project_root: Path,
    dataset: str,
    category: str,
    split: str,
    role: str,
) -> dict[str, object]:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"formal training input is empty: {path}")
    return {
        "schema_version": 1,
        "dataset": dataset,
        "category": category,
        "split": split,
        "role": role,
        "project_relative_path": _project_relative(path, project_root, role),
        "size_bytes": size,
        "sha256": sha256_file(path),
    }


def _provenance_entries(
    provenance_files: Mapping[str, Path], project_root: Path
) -> list[dict[str, object]]:
    if not provenance_files:
        raise ValueError("builder/validator provenance files must not be empty")
    entries: list[dict[str, object]] = []
    for label, path in sorted(provenance_files.items()):
        if not isinstance(label, str) or not label:
            raise ValueError("provenance label must be a non-empty string")
        resolved = _safe_existing_path(Path(path), project_root, f"provenance {label}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        entries.append(
            {
                "label": label,
                "path": resolved.relative_to(project_root.resolve()).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return entries


def default_provenance_files() -> dict[str, Path]:
    return {
        "builder_cli": PROJECT_ROOT / "scripts" / "build_formal_training_inputs.py",
        "builder_validator_module": Path(__file__).resolve(),
    }


def load_frozen_formal_input_reference(
    *,
    project_config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Load and verify the root SHA pinned before any formal training run."""

    root = Path(project_root).resolve()
    config_path = _safe_existing_path(
        Path(project_config_path), root, "project config"
    )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config, config_sha = _load_json_object(config_path)
    reference = config.get("formal_training_inputs")
    if not isinstance(reference, dict):
        raise ValueError("project config lacks frozen formal_training_inputs")
    if reference.get("status") != "frozen_train_only_before_formal_evaluation":
        raise ValueError("formal training-input bundle is not frozen")
    if reference.get("contract") != FORMAL_INPUT_CONTRACT:
        raise ValueError("formal training-input contract is not pinned")
    relative = _validated_project_relative(
        reference.get("manifest"), label="pinned formal input manifest"
    )
    pinned_sha = reference.get("sha256")
    if not isinstance(pinned_sha, str) or not _SHA256.fullmatch(pinned_sha):
        raise ValueError("pinned formal input manifest SHA-256 is invalid")
    for flag in (
        "formal_test_content_accessed",
        "ground_truth_content_accessed",
        "test_or_ground_truth_enumerated",
        "training_started",
    ):
        if reference.get(flag) is not False:
            raise ValueError(f"pinned formal input isolation flag is not false: {flag}")
    manifest_path = _safe_existing_path(
        root / Path(*relative.parts), root, "pinned formal input manifest"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256_file(manifest_path) != pinned_sha:
        raise ValueError("pinned formal input manifest SHA-256 mismatch")
    return {
        "status": reference["status"],
        "contract": reference["contract"],
        "manifest": relative.as_posix(),
        "sha256": pinned_sha,
        "project_config_path": config_path.relative_to(root).as_posix(),
        "project_config_sha256": config_sha,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "test_or_ground_truth_enumerated": False,
        "training_started": False,
    }


def build_formal_training_input_bundle(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    dataset_roots: Mapping[str, str | Path],
    dtd_root: str | Path,
    categories: Mapping[str, Sequence[str]],
    source_manifests: Mapping[str, str | Path],
    provenance_files: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Build the complete MVTec AD/VisA train-good plus DTD bundle."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if set(dataset_roots) != SUPPORTED_DATASETS or set(categories) != SUPPORTED_DATASETS:
        raise ValueError("dataset_roots/categories must contain exactly mvtec_ad and visa")
    if set(source_manifests) != {"mvtec_ad", "visa", "dtd"}:
        raise ValueError("source_manifests must contain exactly mvtec_ad, visa and dtd")

    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    _require_no_link_components(output, root, "formal bundle output")
    if not _inside(output, root):
        raise ValueError("formal bundle output escapes project root")
    expected_output_names = {"manifest.json", "dtd__all__images.jsonl"}
    for dataset in SUPPORTED_DATASETS:
        for category in categories[dataset]:
            if not isinstance(category, str):
                raise ValueError(f"{dataset} category must be a string")
            _validate_name(category, label=f"{dataset} category")
            expected_output_names.add(f"{dataset}__{category}__train_good.jsonl")
    if output.exists() and any(output.iterdir()):
        existing_files: set[str] = set()
        for entry in os.scandir(output):
            path = Path(entry.path)
            if entry.is_symlink() or _is_link_or_reparse(path):
                raise ValueError(f"formal bundle output contains a linked/reparse entry: {path}")
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"formal bundle output contains an unexpected directory: {path}")
            existing_files.add(entry.name)
        if existing_files != expected_output_names:
            raise FileExistsError(
                "formal bundle output does not contain exactly one prior bundle; "
                f"missing={sorted(expected_output_names - existing_files)}, "
                f"extra={sorted(existing_files - expected_output_names)}"
            )
    output.mkdir(parents=True, exist_ok=True)
    _require_no_link_components(output, root, "formal bundle output")

    resolved_dataset_roots: dict[str, Path] = {}
    for dataset in sorted(SUPPORTED_DATASETS):
        source_root = _safe_existing_path(
            Path(dataset_roots[dataset]), root, f"{dataset} data root"
        )
        if not source_root.is_dir():
            raise NotADirectoryError(source_root)
        resolved_dataset_roots[dataset] = source_root
    resolved_dtd = _safe_existing_path(Path(dtd_root), root, "DTD image root")
    if not resolved_dtd.is_dir():
        raise NotADirectoryError(resolved_dtd)

    provenance = _provenance_entries(
        provenance_files or default_provenance_files(), root
    )
    source_entries: dict[str, dict[str, object]] = {}
    for name, path in sorted(source_manifests.items()):
        resolved = _safe_existing_path(Path(path), root, f"{name} source manifest")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        source_entries[name] = {
            "path": resolved.relative_to(root).as_posix(),
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }

    children: list[dict[str, object]] = []
    dataset_summary: dict[str, object] = {}
    for dataset in ("mvtec_ad", "visa"):
        declared = list(categories[dataset])
        if not declared or len(declared) != len(set(declared)):
            raise ValueError(f"{dataset} categories must be non-empty and unique")
        category_counts: dict[str, int] = {}
        dataset_total = 0
        for category in declared:
            if not isinstance(category, str):
                raise ValueError(f"{dataset} category must be a string")
            _validate_name(category, label=f"{dataset} category")
            train_good = resolved_dataset_roots[dataset] / category / "train" / "good"
            paths = _walk_images(
                train_good,
                project_root=root,
                label=f"{dataset}/{category}/train/good",
            )
            rows = [
                _file_row(
                    path,
                    project_root=root,
                    dataset=dataset,
                    category=category,
                    split="train/good",
                    role="normal_training",
                )
                for path in paths
            ]
            filename = f"{dataset}__{category}__train_good.jsonl"
            payload = _jsonl_bytes(rows)
            child_path = output / filename
            child_path.write_bytes(payload)
            children.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "role": "normal_training",
                    "path": child_path.relative_to(root).as_posix(),
                    "rows": len(rows),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            category_counts[category] = len(rows)
            dataset_total += len(rows)
        dataset_summary[dataset] = {
            "categories": category_counts,
            "normal_training_images": dataset_total,
        }

    dtd_paths = _walk_images(resolved_dtd, project_root=root, label="DTD images")
    dtd_rows: list[dict[str, object]] = []
    for path in dtd_paths:
        relative = path.relative_to(resolved_dtd)
        if len(relative.parts) < 2:
            raise ValueError(f"DTD image lacks a category directory: {path}")
        category = relative.parts[0]
        _validate_name(category, label="DTD category")
        dtd_rows.append(
            _file_row(
                path,
                project_root=root,
                dataset="dtd",
                category=category,
                split="auxiliary_texture",
                role="draem_anomaly_source",
            )
        )
    dtd_payload = _jsonl_bytes(dtd_rows)
    dtd_path = output / "dtd__all__images.jsonl"
    dtd_path.write_bytes(dtd_payload)
    children.append(
        {
            "dataset": "dtd",
            "category": None,
            "role": "draem_anomaly_source",
            "path": dtd_path.relative_to(root).as_posix(),
            "rows": len(dtd_rows),
            "size_bytes": len(dtd_payload),
            "sha256": hashlib.sha256(dtd_payload).hexdigest(),
        }
    )
    children.sort(key=lambda entry: str(entry["path"]).casefold())

    manifest: dict[str, object] = {
        "schema_version": 1,
        "contract": FORMAL_INPUT_CONTRACT,
        "path_base": "project_root",
        "project_root": ".",
        "data_roots": {
            **{
                name: path.relative_to(root).as_posix()
                for name, path in sorted(resolved_dataset_roots.items())
            },
            "dtd": resolved_dtd.relative_to(root).as_posix(),
        },
        "source_manifests": source_entries,
        "provenance": {
            "builder": provenance,
            "validator": provenance,
        },
        "datasets": dataset_summary,
        "dtd": {
            "categories": len({str(row["category"]) for row in dtd_rows}),
            "images": len(dtd_rows),
        },
        "children": children,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "test_or_ground_truth_enumerated": False,
        "training_started": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest, pretty=True))
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "normal_training_images": sum(
            int(value["normal_training_images"])
            for value in dataset_summary.values()  # type: ignore[union-attr]
        ),
        "dtd_images": len(dtd_rows),
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "test_or_ground_truth_enumerated": False,
        "training_started": False,
    }


def _validate_hash_entry(entry: object, *, label: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} must be an object")
    if set(entry) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"{label} has an unexpected schema")
    _validated_project_relative(entry.get("path"), label=f"{label} path")
    if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0:
        raise ValueError(f"{label} size_bytes is invalid")
    if not isinstance(entry.get("sha256"), str) or not _SHA256.fullmatch(entry["sha256"]):
        raise ValueError(f"{label} sha256 is invalid")
    return entry


def _validate_current_hash_entry(
    entry: object, *, project_root: Path, label: str
) -> Path:
    parsed = _validate_hash_entry(entry, label=label)
    relative = _validated_project_relative(parsed["path"], label=f"{label} path")
    path = _safe_existing_path(
        project_root / Path(*relative.parts), project_root, label
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != parsed["size_bytes"]:
        raise ValueError(f"{label} size mismatch")
    if sha256_file(path) != parsed["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _validate_row(
    row: dict[str, Any],
    *,
    project_root: Path,
    dataset: str,
    category: str | None,
    role: str,
) -> Path:
    if set(row) != _ROW_KEYS:
        raise ValueError("formal input row has an unexpected schema")
    if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
        raise ValueError("formal input row schema_version must be integer 1")
    if row.get("dataset") != dataset or row.get("role") != role:
        raise ValueError("formal input row dataset/role mismatch")
    if dataset in SUPPORTED_DATASETS:
        if row.get("category") != category or row.get("split") != "train/good":
            raise ValueError("formal normal row category/split mismatch")
    else:
        if not isinstance(row.get("category"), str) or not row["category"]:
            raise ValueError("DTD row category is invalid")
        if row.get("split") != "auxiliary_texture":
            raise ValueError("DTD row split mismatch")
    relative = _validated_project_relative(
        row.get("project_relative_path"), label="formal input row path"
    )
    if relative.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("formal input row has unsupported image suffix")
    if type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0:
        raise ValueError("formal input row size_bytes is invalid")
    if not isinstance(row.get("sha256"), str) or not _SHA256.fullmatch(row["sha256"]):
        raise ValueError("formal input row sha256 is invalid")
    path = _safe_existing_path(
        project_root / Path(*relative.parts), project_root, "formal input file"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != row["size_bytes"]:
        raise ValueError(f"formal input file-size mismatch: {relative}")
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"formal input SHA-256 mismatch: {relative}")
    return path


def _child_index(
    manifest: Mapping[str, object], *, project_root: Path, manifest_dir: Path
) -> dict[tuple[str, str | None], tuple[dict[str, Any], Path]]:
    raw_children = manifest.get("children")
    if not isinstance(raw_children, list) or not raw_children:
        raise ValueError("formal root manifest children must be a non-empty list")
    index: dict[tuple[str, str | None], tuple[dict[str, Any], Path]] = {}
    expected_files = {"manifest.json"}
    for raw in raw_children:
        if not isinstance(raw, dict):
            raise ValueError("formal root child must be an object")
        required = {"dataset", "category", "role", "path", "rows", "size_bytes", "sha256"}
        if set(raw) != required:
            raise ValueError("formal root child has an unexpected schema")
        dataset = raw.get("dataset")
        category = raw.get("category")
        if dataset not in {*SUPPORTED_DATASETS, "dtd"}:
            raise ValueError("formal root child dataset is invalid")
        if dataset == "dtd":
            if category is not None or raw.get("role") != "draem_anomaly_source":
                raise ValueError("formal DTD child identity is invalid")
        elif not isinstance(category, str) or raw.get("role") != "normal_training":
            raise ValueError("formal normal child identity is invalid")
        relative = _validated_project_relative(raw.get("path"), label="child path")
        child_path = _safe_existing_path(
            project_root / Path(*relative.parts), project_root, "formal child manifest"
        )
        if child_path.parent != manifest_dir or child_path.suffix != ".jsonl":
            raise ValueError("formal child manifest is outside its bundle directory")
        if type(raw.get("rows")) is not int or raw["rows"] <= 0:
            raise ValueError("formal child row count is invalid")
        if type(raw.get("size_bytes")) is not int or raw["size_bytes"] <= 0:
            raise ValueError("formal child size is invalid")
        if not isinstance(raw.get("sha256"), str) or not _SHA256.fullmatch(raw["sha256"]):
            raise ValueError("formal child SHA-256 is invalid")
        if child_path.stat().st_size != raw["size_bytes"]:
            raise ValueError(f"formal child size mismatch: {child_path.name}")
        if sha256_file(child_path) != raw["sha256"]:
            raise ValueError(f"formal child SHA-256 mismatch: {child_path.name}")
        key = (str(dataset), category if isinstance(category, str) else None)
        if key in index:
            raise ValueError(f"duplicate formal child identity: {key}")
        index[key] = (raw, child_path)
        expected_files.add(child_path.name)
    actual_files = set()
    for entry in os.scandir(manifest_dir):
        path = Path(entry.path)
        if entry.is_symlink() or _is_link_or_reparse(path):
            raise ValueError(f"formal bundle contains a linked/reparse entry: {path}")
        if entry.is_file(follow_symlinks=False):
            actual_files.add(entry.name)
        else:
            raise ValueError(f"formal bundle contains an unexpected directory: {path}")
    if actual_files != expected_files:
        raise ValueError(
            "formal bundle file set mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    return index


def _read_and_validate_child(
    entry: Mapping[str, object],
    path: Path,
    *,
    project_root: Path,
    dataset: str,
    category: str | None,
    role: str,
) -> tuple[list[dict[str, Any]], tuple[Path, ...], int]:
    rows, actual_sha = _parse_jsonl(path)
    if actual_sha != entry["sha256"] or len(rows) != entry["rows"]:
        raise ValueError(f"formal child content/count mismatch: {path.name}")
    paths: list[Path] = []
    total_bytes = 0
    seen: set[str] = set()
    for row in rows:
        current = _validate_row(
            row,
            project_root=project_root,
            dataset=dataset,
            category=category,
            role=role,
        )
        key = str(current).casefold()
        if key in seen:
            raise ValueError(f"duplicate formal input path: {current}")
        seen.add(key)
        paths.append(current)
        total_bytes += current.stat().st_size
    return rows, tuple(paths), total_bytes


def _expected_rows_for_current_files(
    *,
    project_root: Path,
    source_root: Path,
    dataset: str,
    category: str | None,
    role: str,
) -> list[dict[str, object]]:
    if dataset in SUPPORTED_DATASETS:
        assert category is not None
        directory = source_root / category / "train" / "good"
        split = "train/good"
    else:
        directory = source_root
        split = "auxiliary_texture"
    paths = _walk_images(directory, project_root=project_root, label=f"current {dataset} inputs")
    result: list[dict[str, object]] = []
    for path in paths:
        row_category = category
        if dataset == "dtd":
            relative = path.relative_to(source_root)
            if len(relative.parts) < 2:
                raise ValueError(f"DTD image lacks a category directory: {path}")
            row_category = relative.parts[0]
        assert row_category is not None
        result.append(
            _file_row(
                path,
                project_root=project_root,
                dataset=dataset,
                category=row_category,
                split=split,
                role=role,
            )
        )
    return result


def _evidence_selection_hash(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_jsonl_bytes(rows)).hexdigest()


def resolve_formal_training_inputs(
    *,
    method: str,
    dataset: str,
    category: str,
    dataset_root: str | Path,
    bundle_manifest_path: str | Path = DEFAULT_FORMAL_INPUT_MANIFEST,
    dtd_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    expected_manifest_sha256: str | None = None,
) -> FormalTrainingInputs:
    """Revalidate a selected category's current file set and every file byte."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"formal input bundle does not support method: {method}")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"formal input bundle does not support dataset: {dataset}")
    _validate_name(category, label="category")
    root = Path(project_root).resolve()
    manifest_path = _safe_existing_path(
        Path(bundle_manifest_path), root, "formal root manifest"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest, manifest_sha = _load_json_object(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("formal root manifest SHA-256 mismatch")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("formal root manifest schema_version must be integer 1")
    if manifest.get("contract") != FORMAL_INPUT_CONTRACT:
        raise ValueError("formal root manifest contract mismatch")
    if manifest.get("path_base") != "project_root" or manifest.get("project_root") != ".":
        raise ValueError("formal root manifest path-base contract mismatch")
    for flag in (
        "formal_test_content_accessed",
        "ground_truth_content_accessed",
        "test_or_ground_truth_enumerated",
        "training_started",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"formal root manifest isolation flag is not false: {flag}")

    source_manifests = manifest.get("source_manifests")
    if not isinstance(source_manifests, dict) or set(source_manifests) != {
        "mvtec_ad",
        "visa",
        "dtd",
    }:
        raise ValueError("formal root source manifest set mismatch")
    verified_sources: dict[str, dict[str, object]] = {}
    for name, entry in sorted(source_manifests.items()):
        path = _validate_current_hash_entry(
            entry, project_root=root, label=f"{name} source manifest"
        )
        verified_sources[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }

    raw_provenance = manifest.get("provenance")
    if not isinstance(raw_provenance, dict) or set(raw_provenance) != {"builder", "validator"}:
        raise ValueError("formal root builder/validator provenance is missing")
    verified_provenance: dict[str, list[dict[str, object]]] = {}
    for role in ("builder", "validator"):
        entries = raw_provenance.get(role)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"formal root {role} provenance is empty")
        labels: set[str] = set()
        verified: list[dict[str, object]] = []
        for number, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict) or set(entry) != {
                "label",
                "path",
                "size_bytes",
                "sha256",
            }:
                raise ValueError(f"formal {role} provenance schema mismatch")
            label = entry.get("label")
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError(f"formal {role} provenance label is invalid")
            labels.add(label)
            path = _validate_current_hash_entry(
                {key: entry[key] for key in ("path", "size_bytes", "sha256")},
                project_root=root,
                label=f"{role} provenance {number}",
            )
            verified.append(
                {
                    "label": label,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
        verified_provenance[role] = verified

    data_roots = manifest.get("data_roots")
    if not isinstance(data_roots, dict) or set(data_roots) != {"mvtec_ad", "visa", "dtd"}:
        raise ValueError("formal root data_roots set mismatch")
    resolved_roots: dict[str, Path] = {}
    for name, value in data_roots.items():
        relative = _validated_project_relative(value, label=f"{name} data root")
        resolved = _safe_existing_path(
            root / Path(*relative.parts), root, f"{name} data root"
        )
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        resolved_roots[name] = resolved
    supplied_dataset_root = Path(dataset_root).resolve()
    if supplied_dataset_root != resolved_roots[dataset]:
        raise ValueError("dataset_root does not match the formal input bundle")
    if method == "DRAEM":
        if dtd_root is None or Path(dtd_root).resolve() != resolved_roots["dtd"]:
            raise ValueError("DTD root does not match the formal input bundle")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != SUPPORTED_DATASETS:
        raise ValueError("formal root dataset summary set mismatch")
    dataset_summary = datasets.get(dataset)
    if not isinstance(dataset_summary, dict) or not isinstance(
        dataset_summary.get("categories"), dict
    ):
        raise ValueError(f"formal root {dataset} summary is invalid")
    if category not in dataset_summary["categories"]:
        raise ValueError(f"category is not registered in the formal bundle: {category}")

    index = _child_index(
        manifest, project_root=root, manifest_dir=manifest_path.parent
    )
    expected_keys = {
        (name, registered_category)
        for name in SUPPORTED_DATASETS
        for registered_category in (
            datasets[name]["categories"].keys()  # type: ignore[index,union-attr]
        )
    }
    expected_keys.add(("dtd", None))
    if set(index) != expected_keys:
        raise ValueError("formal root child identity set mismatch")

    normal_entry, normal_manifest = index[(dataset, category)]
    normal_rows, normal_paths, normal_bytes = _read_and_validate_child(
        normal_entry,
        normal_manifest,
        project_root=root,
        dataset=dataset,
        category=category,
        role="normal_training",
    )
    current_normal_rows = _expected_rows_for_current_files(
        project_root=root,
        source_root=resolved_roots[dataset],
        dataset=dataset,
        category=category,
        role="normal_training",
    )
    if normal_rows != current_normal_rows:
        raise ValueError(
            f"formal {dataset}/{category} training file set or bytes changed"
        )
    if dataset_summary["categories"].get(category) != len(normal_rows):
        raise ValueError("formal root category count mismatch")

    dtd_paths: tuple[Path, ...] = ()
    dtd_bytes = 0
    dtd_rows: list[dict[str, Any]] = []
    dtd_evidence: dict[str, object] | None = None
    if method == "DRAEM":
        dtd_entry, dtd_manifest = index[("dtd", None)]
        dtd_rows, dtd_paths, dtd_bytes = _read_and_validate_child(
            dtd_entry,
            dtd_manifest,
            project_root=root,
            dataset="dtd",
            category=None,
            role="draem_anomaly_source",
        )
        current_dtd_rows = _expected_rows_for_current_files(
            project_root=root,
            source_root=resolved_roots["dtd"],
            dataset="dtd",
            category=None,
            role="draem_anomaly_source",
        )
        if dtd_rows != current_dtd_rows:
            raise ValueError("formal DTD training file set or bytes changed")
        dtd_summary = manifest.get("dtd")
        if not isinstance(dtd_summary, dict) or dtd_summary.get("images") != len(dtd_rows):
            raise ValueError("formal root DTD count mismatch")
        dtd_evidence = {
            "manifest_path": dtd_manifest.relative_to(root).as_posix(),
            "manifest_sha256": dtd_entry["sha256"],
            "rows": len(dtd_rows),
            "selected_file_bytes": dtd_bytes,
            "selection_sha256": _evidence_selection_hash(dtd_rows),
        }

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "contract": FORMAL_INPUT_CONTRACT,
        "method": method,
        "dataset": dataset,
        "category": category,
        "path_base": "project_root",
        "project_root": ".",
        "root_manifest_path": manifest_path.relative_to(root).as_posix(),
        "root_manifest_sha256": manifest_sha,
        "dataset_root": resolved_roots[dataset].relative_to(root).as_posix(),
        "dtd_root": (
            resolved_roots["dtd"].relative_to(root).as_posix()
            if method == "DRAEM"
            else None
        ),
        "source_manifests": verified_sources,
        "builder_validator_provenance": verified_provenance,
        "category_manifest": {
            "path": normal_manifest.relative_to(root).as_posix(),
            "sha256": normal_entry["sha256"],
            "rows": len(normal_rows),
            "selected_file_bytes": normal_bytes,
            "selection_sha256": _evidence_selection_hash(normal_rows),
        },
        "dtd_manifest": dtd_evidence,
        "normal_training_images": len(normal_paths),
        "anomaly_source_images": len(dtd_paths),
        "current_file_set_and_content_sha256_verified": True,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "test_or_ground_truth_enumerated": False,
    }
    return FormalTrainingInputs(normal_paths, dtd_paths, evidence)


def revalidate_formal_training_input_evidence(
    evidence: Mapping[str, object],
    *,
    method: str,
    dataset: str,
    category: str,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Re-resolve checkpoint-embedded evidence against current source bytes."""

    for key, expected in (
        ("schema_version", 1),
        ("contract", FORMAL_INPUT_CONTRACT),
        ("method", method),
        ("dataset", dataset),
        ("category", category),
    ):
        if evidence.get(key) != expected:
            raise ValueError(f"formal training-input evidence {key} mismatch")
    if evidence.get("path_base") != "project_root" or evidence.get("project_root") != ".":
        raise ValueError("formal training-input evidence path-base contract mismatch")
    root = Path(project_root).resolve()
    required_paths = ("root_manifest_path", "dataset_root")
    relative_paths: dict[str, PurePosixPath] = {}
    for key in required_paths:
        relative_paths[key] = _validated_project_relative(
            evidence.get(key), label=f"formal training-input {key}"
        )
    expected_sha = evidence.get("root_manifest_sha256")
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        raise ValueError("formal training-input root manifest SHA-256 is invalid")
    dtd_root = evidence.get("dtd_root")
    resolved_dtd: Path | None = None
    if method == "DRAEM":
        relative_dtd = _validated_project_relative(
            dtd_root, label="formal training-input dtd_root"
        )
        resolved_dtd = root / Path(*relative_dtd.parts)
    refreshed = resolve_formal_training_inputs(
        method=method,
        dataset=dataset,
        category=category,
        dataset_root=root / Path(*relative_paths["dataset_root"].parts),
        bundle_manifest_path=root / Path(*relative_paths["root_manifest_path"].parts),
        dtd_root=resolved_dtd,
        project_root=root,
        expected_manifest_sha256=expected_sha,
    )
    if refreshed.evidence != dict(evidence):
        raise ValueError("formal training-input evidence no longer matches current bundle")
    return refreshed.evidence


def resolve_frozen_formal_training_inputs(
    *,
    method: str,
    dataset: str,
    category: str,
    dataset_root: str | Path,
    project_config_path: str | Path,
    dtd_root: str | Path | None = None,
    bundle_manifest_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> FormalTrainingInputs:
    """Resolve inputs only from the exact root manifest pinned in project config."""

    root = Path(project_root).resolve()
    pinned = load_frozen_formal_input_reference(
        project_config_path=project_config_path,
        project_root=root,
    )
    relative = _validated_project_relative(
        pinned["manifest"], label="pinned formal input manifest"
    )
    expected_path = (root / Path(*relative.parts)).resolve()
    supplied_path = (
        expected_path
        if bundle_manifest_path is None
        else Path(bundle_manifest_path).resolve()
    )
    if supplied_path != expected_path:
        raise ValueError("formal input manifest path differs from the frozen project pin")
    return resolve_formal_training_inputs(
        method=method,
        dataset=dataset,
        category=category,
        dataset_root=dataset_root,
        bundle_manifest_path=supplied_path,
        dtd_root=dtd_root,
        project_root=root,
        expected_manifest_sha256=str(pinned["sha256"]),
    )
