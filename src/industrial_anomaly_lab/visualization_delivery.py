"""Frozen, hash-gated comparison figures for formal anomaly experiments.

This module intentionally has no dataset-specific discovery code.  The F0
freeze consumes an already-created formal test inventory, and rendering only
consumes the exact ``raw_map`` and ``evaluation_mask`` arrays indexed by
completed prediction runs.  This keeps sample choice separate from model
outputs and makes every displayed array traceable to the metric source.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from .run_artifacts import prepare_output_directory, sha256_file, write_json


SELECTION_SCHEMA_VERSION = 1
FIGURE_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_ALLOWED_SCALES = {"normal", "small", "medium", "large"}
_FIXED_COHORT = "result_independent_preregistered"
_FAILURE_COHORT = "outcome_dependent_machine_selected"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_payload_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "payload_sha256" in result:
        raise ValueError("payload_sha256 is reserved")
    result["payload_sha256"] = _canonical_sha256(result)
    return result


def _verify_payload_sha256(payload: Mapping[str, Any], context: str) -> str:
    expected = payload.get("payload_sha256")
    if not _is_sha256(expected):
        raise ValueError(f"{context} is missing a valid payload_sha256")
    unsigned = dict(payload)
    del unsigned["payload_sha256"]
    actual = _canonical_sha256(unsigned)
    if actual != expected:
        raise ValueError(f"{context} payload SHA-256 mismatch")
    return actual


def _load_json(path: str | Path, context: str) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing or unknown:
        raise ValueError(f"{context} schema mismatch: missing={missing}, unknown={unknown}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase 64-character SHA-256")
    return str(value)


def _safe_relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a non-empty POSIX relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{context} is not a safe relative path: {value!r}")
    return candidate.as_posix()


def _resolve_under(root: str | Path, relative: object, context: str) -> Path:
    relative_path = _safe_relative_path(relative, context)
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{context} escapes its declared root") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_file(
    root: str | Path,
    relative: object,
    expected_sha256: object,
    context: str,
) -> Path:
    path = _resolve_under(root, relative, context)
    expected = _validate_sha256(expected_sha256, f"{context} SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{context} SHA-256 mismatch")
    return path


def _require_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _require_positive_int(value: object, context: str) -> int:
    result = _require_nonnegative_int(value, context)
    if result == 0:
        raise ValueError(f"{context} must be positive")
    return result


def _load_frozen_method_registry(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path, "method registry")
    registry = payload.get("formal_method_registry")
    if not isinstance(registry, dict):
        raise ValueError("method registry is missing formal_method_registry")
    if registry.get("status") != "frozen_before_formal_evaluation":
        raise ValueError("formal method registry is not frozen before F0")
    methods = registry.get("methods")
    baseline = registry.get("baseline")
    candidate = registry.get("candidate")
    if (
        not isinstance(methods, list)
        or not methods
        or not all(isinstance(method, str) and method for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("formal method registry has no valid unique method list")
    if not isinstance(baseline, str) or baseline not in methods:
        raise ValueError("formal baseline is missing from the method list")
    if not isinstance(candidate, str) or candidate not in methods:
        raise ValueError("formal candidate is missing from the method list")
    return {
        "registration_id": payload.get("registration_id"),
        "methods": methods,
        "baseline": baseline,
        "candidate": candidate,
    }


def _validate_inventory_entry(
    raw: object,
    *,
    dataset: str,
    dataset_root: str | Path,
    index: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"test inventory entry {index} must be an object")
    _strict_keys(
        raw,
        required=(
            "dataset",
            "category",
            "sample_id",
            "label",
            "defect_type",
            "anomaly_area_fraction",
            "anomaly_scale",
            "source_image",
            "source_image_sha256",
            "source_mask",
            "source_mask_sha256",
        ),
        context=f"test inventory entry {index}",
    )
    if raw["dataset"] != dataset:
        raise ValueError(f"test inventory entry {index} dataset mismatch")
    category = _require_nonempty_string(raw["category"], f"entry {index} category")
    defect_type = _require_nonempty_string(
        raw["defect_type"], f"entry {index} defect_type"
    )
    sample_id = _require_nonempty_string(raw["sample_id"], f"entry {index} sample_id")
    expected_prefix = f"{category}/{defect_type}/"
    if not sample_id.startswith(expected_prefix) or sample_id == expected_prefix:
        raise ValueError(
            f"entry {index} sample_id must start with category/defect_type/"
        )
    label = raw["label"]
    if isinstance(label, bool) or label not in {0, 1}:
        raise ValueError(f"entry {index} label must be integer 0 or 1")
    area = raw["anomaly_area_fraction"]
    if isinstance(area, bool) or not isinstance(area, (int, float)) or not math.isfinite(area):
        raise ValueError(f"entry {index} anomaly_area_fraction must be finite")
    area = float(area)
    scale = raw["anomaly_scale"]
    if scale not in _ALLOWED_SCALES:
        raise ValueError(f"entry {index} has invalid anomaly_scale")

    image_relative = _safe_relative_path(raw["source_image"], f"entry {index} image")
    image_sha = _validate_sha256(raw["source_image_sha256"], f"entry {index} image SHA")
    _verify_file(dataset_root, image_relative, image_sha, f"entry {index} source image")

    mask_relative = raw["source_mask"]
    mask_sha = raw["source_mask_sha256"]
    if label == 0:
        if defect_type != "good" or area != 0.0 or scale != "normal":
            raise ValueError(f"normal entry {index} has inconsistent metadata")
        if mask_relative is not None or mask_sha is not None:
            raise ValueError(f"normal entry {index} must not declare a source mask")
    else:
        if defect_type == "good" or not (0.0 < area <= 1.0) or scale == "normal":
            raise ValueError(f"anomalous entry {index} has inconsistent metadata")
        mask_relative = _safe_relative_path(mask_relative, f"entry {index} mask")
        mask_sha = _validate_sha256(mask_sha, f"entry {index} mask SHA")
        _verify_file(dataset_root, mask_relative, mask_sha, f"entry {index} source mask")

    return {
        "dataset": dataset,
        "category": category,
        "sample_id": sample_id,
        "label": int(label),
        "defect_type": defect_type,
        "anomaly_area_fraction": area,
        "anomaly_scale": scale,
        "source_image": image_relative,
        "source_image_sha256": image_sha,
        "source_mask": mask_relative,
        "source_mask_sha256": mask_sha,
    }


def _load_test_inventory(
    path: str | Path, dataset_root: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path, "formal test inventory")
    _strict_keys(
        payload,
        required=("schema_version", "kind", "dataset", "split", "entries"),
        optional=("inventory_id",),
        context="formal test inventory",
    )
    if payload["schema_version"] != 1 or payload["kind"] != "formal_test_inventory":
        raise ValueError("unsupported formal test inventory schema")
    if payload["split"] != "test":
        raise ValueError("visualization inventory must describe the formal test split")
    dataset = payload["dataset"]
    if dataset not in {"mvtec_ad", "visa"}:
        raise ValueError("inventory dataset must be mvtec_ad or visa")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("formal test inventory entries must be a non-empty list")
    entries = [
        _validate_inventory_entry(
            row, dataset=dataset, dataset_root=dataset_root, index=index
        )
        for index, row in enumerate(raw_entries)
    ]
    sample_ids = [entry["sample_id"] for entry in entries]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("formal test inventory contains duplicate sample_id values")
    return payload, entries


def _validate_failure_rule(
    raw: object, *, registry: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("failure_case_rule must be an object")
    _strict_keys(
        raw,
        required=(
            "baseline_method",
            "candidate_method",
            "candidate_filter",
            "rankings",
        ),
        context="failure_case_rule",
    )
    baseline = raw["baseline_method"]
    candidate = raw["candidate_method"]
    if baseline != registry["baseline"] or candidate != registry["candidate"]:
        raise ValueError("failure rule comparison pair differs from frozen method registry")
    if raw["candidate_filter"] != "anomalies_with_positive_evaluation_mask":
        raise ValueError("unsupported failure candidate filter")
    rankings = raw["rankings"]
    if not isinstance(rankings, list) or not rankings:
        raise ValueError("failure rule needs at least one ranking")
    allowed = {
        "candidate_absolute_failures": "candidate_pixel_average_precision",
        "candidate_regressions_vs_baseline": (
            "candidate_minus_baseline_pixel_average_precision"
        ),
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, ranking in enumerate(rankings):
        if not isinstance(ranking, dict):
            raise ValueError(f"failure ranking {index} must be an object")
        _strict_keys(
            ranking,
            required=("name", "metric", "order", "per_category", "tie_break"),
            context=f"failure ranking {index}",
        )
        name = ranking["name"]
        if name not in allowed or name in seen:
            raise ValueError(f"unsupported or duplicate failure ranking: {name!r}")
        if ranking["metric"] != allowed[name]:
            raise ValueError(f"failure ranking {name} metric is not pre-defined")
        if ranking["order"] != "ascending" or ranking["tie_break"] != "sample_id_ascending":
            raise ValueError("failure ranking order and tie-break are fixed")
        count = _require_positive_int(
            ranking["per_category"], f"failure ranking {name} per_category"
        )
        normalized.append(
            {
                "name": name,
                "metric": allowed[name],
                "order": "ascending",
                "per_category": count,
                "tie_break": "sample_id_ascending",
            }
        )
        seen.add(name)
    return {
        "status": "frozen_unresolved",
        "baseline_method": baseline,
        "candidate_method": candidate,
        "candidate_filter": raw["candidate_filter"],
        "rankings": normalized,
        "manual_selection_allowed": False,
    }


def _load_selection_config(
    path: str | Path,
    *,
    dataset: str,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path, "visualization selection config")
    _strict_keys(
        payload,
        required=(
            "schema_version",
            "kind",
            "dataset",
            "model_seed",
            "selection_salt",
            "fixed_selection",
            "failure_case_rule",
        ),
        context="visualization selection config",
    )
    if (
        payload["schema_version"] != 1
        or payload["kind"] != "formal_visualization_selection_config"
    ):
        raise ValueError("unsupported visualization selection config schema")
    if payload["dataset"] != dataset:
        raise ValueError("selection config and inventory dataset differ")
    _require_nonnegative_int(payload["model_seed"], "model_seed")
    salt = _require_nonempty_string(payload["selection_salt"], "selection_salt")
    fixed = payload["fixed_selection"]
    if not isinstance(fixed, dict):
        raise ValueError("fixed_selection must be an object")
    mode = fixed.get("mode")
    if mode == "deterministic_hash_stratified":
        _strict_keys(
            fixed,
            required=(
                "mode",
                "normal_per_category",
                "anomaly_per_defect_scale_stratum",
            ),
            context="fixed_selection",
        )
        fixed_normalized: dict[str, Any] = {
            "mode": mode,
            "normal_per_category": _require_positive_int(
                fixed["normal_per_category"], "normal_per_category"
            ),
            "anomaly_per_defect_scale_stratum": _require_positive_int(
                fixed["anomaly_per_defect_scale_stratum"],
                "anomaly_per_defect_scale_stratum",
            ),
        }
    elif mode == "explicit_sample_ids":
        _strict_keys(
            fixed,
            required=("mode", "sample_ids"),
            context="fixed_selection",
        )
        sample_ids = fixed["sample_ids"]
        if (
            not isinstance(sample_ids, list)
            or not sample_ids
            or not all(isinstance(value, str) and value for value in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)
        ):
            raise ValueError("explicit sample_ids must be a non-empty unique string list")
        fixed_normalized = {"mode": mode, "sample_ids": sample_ids}
    else:
        raise ValueError(
            "fixed selection must be deterministic_hash_stratified or explicit_sample_ids; "
            "result-driven selection is forbidden in the preregistered cohort"
        )
    failure_rule = _validate_failure_rule(payload["failure_case_rule"], registry=registry)
    return {
        "schema_version": 1,
        "kind": "formal_visualization_selection_config",
        "dataset": dataset,
        "model_seed": payload["model_seed"],
        "selection_salt": salt,
        "fixed_selection": fixed_normalized,
        "failure_case_rule": failure_rule,
    }


def _selection_key(salt: str, entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{salt}\0{entry['dataset']}\0{entry['category']}\0"
            f"{entry['sample_id']}"
        ).encode("utf-8")
    ).hexdigest()


def _frozen_entry(
    entry: Mapping[str, Any],
    *,
    cohort: str,
    group: str,
    rank: int,
    key: str,
) -> dict[str, Any]:
    return {
        **dict(entry),
        "cohort": cohort,
        "selection_group": group,
        "selection_rank": rank,
        "selection_key_sha256": key,
    }


def _choose_fixed_samples(
    entries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    fixed = config["fixed_selection"]
    salt = str(config["selection_salt"])
    by_id = {str(entry["sample_id"]): entry for entry in entries}
    if fixed["mode"] == "explicit_sample_ids":
        missing = [sample_id for sample_id in fixed["sample_ids"] if sample_id not in by_id]
        if missing:
            raise ValueError(f"explicit visualization sample IDs are absent: {missing[:5]}")
        return [
            _frozen_entry(
                by_id[sample_id],
                cohort=_FIXED_COHORT,
                group="explicit_sample_ids",
                rank=index + 1,
                key=_selection_key(salt, by_id[sample_id]),
            )
            for index, sample_id in enumerate(fixed["sample_ids"])
        ]

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    counts: dict[tuple[str, ...], int] = {}
    for entry in entries:
        if entry["label"] == 0:
            group = (str(entry["category"]), "normal")
            counts[group] = int(fixed["normal_per_category"])
        else:
            group = (
                str(entry["category"]),
                "anomaly",
                str(entry["defect_type"]),
                str(entry["anomaly_scale"]),
            )
            counts[group] = int(fixed["anomaly_per_defect_scale_stratum"])
        groups[group].append(entry)

    selected: list[dict[str, Any]] = []
    for group in sorted(groups):
        ordered = sorted(
            groups[group], key=lambda entry: (_selection_key(salt, entry), entry["sample_id"])
        )
        required = counts[group]
        if len(ordered) < required:
            raise ValueError(
                f"visualization stratum {'/'.join(group)} has {len(ordered)} samples, "
                f"fewer than required {required}"
            )
        group_name = "/".join(group)
        selected.extend(
            _frozen_entry(
                entry,
                cohort=_FIXED_COHORT,
                group=group_name,
                rank=rank,
                key=_selection_key(salt, entry),
            )
            for rank, entry in enumerate(ordered[:required], start=1)
        )
    return selected


def freeze_visualization_selection(
    *,
    test_inventory_path: str | Path,
    dataset_root: str | Path,
    selection_config_path: str | Path,
    method_registry_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the F0 visualization cohort without consulting predictions."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {output}")
    inventory, entries = _load_test_inventory(test_inventory_path, dataset_root)
    registry = _load_frozen_method_registry(method_registry_path)
    config = _load_selection_config(
        selection_config_path, dataset=str(inventory["dataset"]), registry=registry
    )
    selected = _choose_fixed_samples(entries, config)
    failure_pool = [
        {**entry, "cohort": "frozen_failure_candidate_pool"}
        for entry in sorted(entries, key=lambda row: row["sample_id"])
        if entry["label"] == 1
    ]
    if not failure_pool:
        raise ValueError("failure candidate pool contains no anomalous samples")
    payload = _with_payload_sha256(
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "kind": "formal_visualization_selection",
            "status": "frozen_before_formal_results",
            "freeze_phase": "F0_post_method_freeze_pre_inference",
            "dataset": inventory["dataset"],
            "model_seed": config["model_seed"],
            "test_inventory": {
                "sha256": sha256_file(test_inventory_path),
                "inventory_id": inventory.get("inventory_id"),
                "entry_count": len(entries),
            },
            "method_registry": {
                "sha256": sha256_file(method_registry_path),
                **registry,
            },
            "selection_config": config,
            "selection_config_sha256": sha256_file(selection_config_path),
            "result_independent_samples": selected,
            "failure_candidate_pool": failure_pool,
            "failure_case_rule": config["failure_case_rule"],
            "prediction_artifacts_accessed": False,
            "manual_post_result_selection_allowed": False,
        }
    )
    write_json(output, payload)
    return payload


def _load_selection_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path, "visualization selection manifest")
    _verify_payload_sha256(payload, "visualization selection manifest")
    if (
        payload.get("schema_version") != SELECTION_SCHEMA_VERSION
        or payload.get("kind") != "formal_visualization_selection"
        or payload.get("status") != "frozen_before_formal_results"
        or payload.get("prediction_artifacts_accessed") is not False
        or payload.get("manual_post_result_selection_allowed") is not False
    ):
        raise ValueError("visualization selection manifest is not a valid F0 freeze")
    return payload


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{context} contains blank line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{context} row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{context} is empty")
    return rows


def _load_completed_prediction_run(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    config_path = root / "run_config.json"
    state_path = root / "run_state.json"
    metrics_path = root / "metrics.json"
    index_path = root / "predictions" / "index.jsonl"
    config = _load_json(config_path, f"run config at {root}")
    state = _load_json(state_path, f"run state at {root}")
    metrics = _load_json(metrics_path, f"metrics at {root}")
    if state.get("status") != "completed":
        raise ValueError(f"prediction run is not completed: {root}")
    formal_fields = {
        "execution_scope": "full",
        "run_purpose": "formal_full",
        "claim_scope": "formal_matrix_candidate",
        "formal_matrix_eligible": True,
    }
    for field, expected in formal_fields.items():
        if config.get(field) != expected or state.get(field) != expected:
            raise ValueError(f"prediction run lacks formal {field}={expected!r}: {root}")
    if state.get("run_config_sha256") != sha256_file(config_path):
        raise ValueError(f"run_config SHA-256 mismatch: {root}")
    if state.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError(f"metrics SHA-256 mismatch: {root}")
    if state.get("predictions_index_sha256") != sha256_file(index_path):
        raise ValueError(f"prediction index SHA-256 mismatch: {root}")
    prediction_contract = config.get("prediction_contract")
    if (
        not isinstance(prediction_contract, dict)
        or prediction_contract.get("metrics_reload_saved_arrays") is not True
    ):
        raise ValueError(f"run does not attest metrics_reload_saved_arrays: {root}")
    if metrics.get("computed_from") != (
        "predictions/index.jsonl raw_map and evaluation_mask files"
    ):
        raise ValueError(f"metrics are not bound to indexed raw maps and masks: {root}")
    method = _require_nonempty_string(config.get("method"), f"method at {root}")
    settings = config.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"prediction run settings missing: {root}")
    dataset = _require_nonempty_string(settings.get("dataset"), f"dataset at {root}")
    category = _require_nonempty_string(settings.get("category"), f"category at {root}")
    seed = _require_nonnegative_int(settings.get("seed"), f"seed at {root}")
    rows = _read_jsonl(index_path, f"prediction index at {root}")
    by_sample: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = _require_nonempty_string(
            row.get("sample_id"), f"prediction row {index} sample_id"
        )
        if sample_id in by_sample:
            raise ValueError(f"duplicate prediction sample_id {sample_id!r}: {root}")
        for name in (
            "label",
            "defect_type",
            "input_image",
            "raw_map",
            "raw_map_sha256",
            "evaluation_mask",
            "evaluation_mask_sha256",
        ):
            if name not in row:
                raise ValueError(f"prediction row {sample_id!r} is missing {name}")
        by_sample[sample_id] = row
    return {
        "root": root,
        "method": method,
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "run_config_sha256": sha256_file(config_path),
        "metrics_sha256": sha256_file(metrics_path),
        "prediction_index_sha256": sha256_file(index_path),
        "rows": by_sample,
    }


def _load_prediction_runs(
    prediction_directories: Sequence[str | Path],
) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
    if not prediction_directories:
        raise ValueError("at least one prediction directory is required")
    runs = [_load_completed_prediction_run(path) for path in prediction_directories]
    method_order: list[str] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        if run["method"] not in method_order:
            method_order.append(run["method"])
        key = (run["method"], run["category"])
        if key in indexed:
            raise ValueError(f"duplicate method/category prediction run: {key}")
        indexed[key] = run
    return method_order, indexed


def _validate_entry_against_prediction(
    entry: Mapping[str, Any], row: Mapping[str, Any], *, method: str
) -> None:
    for field in ("sample_id", "label", "defect_type"):
        if row.get(field) != entry.get(field):
            raise ValueError(
                f"{method} prediction {field} mismatch for {entry.get('sample_id')!r}"
            )
    if row.get("input_image") != entry.get("source_image"):
        raise ValueError(f"{method} input image mismatch for {entry['sample_id']!r}")
    if row.get("source_ground_truth") != entry.get("source_mask"):
        raise ValueError(f"{method} source mask mismatch for {entry['sample_id']!r}")


def _load_map_and_mask(
    run: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    sample_id = str(entry["sample_id"])
    row = run["rows"].get(sample_id)
    if row is None:
        raise ValueError(
            f"{run['method']} prediction index is missing sample_id {sample_id!r}"
        )
    _validate_entry_against_prediction(entry, row, method=str(run["method"]))
    raw_path = _verify_file(
        run["root"], row["raw_map"], row["raw_map_sha256"], f"{sample_id} raw_map"
    )
    mask_path = _verify_file(
        run["root"],
        row["evaluation_mask"],
        row["evaluation_mask_sha256"],
        f"{sample_id} evaluation_mask",
    )
    raw_map = np.load(raw_path, allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False)
    if raw_map.dtype != np.float32 or raw_map.ndim != 2 or not np.isfinite(raw_map).all():
        raise ValueError(f"{sample_id} raw_map must be finite two-dimensional float32")
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError(f"{sample_id} evaluation_mask must be two-dimensional bool")
    if raw_map.shape != mask.shape:
        raise ValueError(f"{sample_id} raw_map and evaluation_mask shapes differ")
    return raw_map, mask, {
        "raw_map": row["raw_map"],
        "raw_map_sha256": row["raw_map_sha256"],
        "evaluation_mask": row["evaluation_mask"],
        "evaluation_mask_sha256": row["evaluation_mask_sha256"],
    }


def _runs_for_entry(
    entry: Mapping[str, Any],
    *,
    methods: Sequence[str],
    runs: Mapping[tuple[str, str], Mapping[str, Any]],
    dataset: str,
    seed: int,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for method in methods:
        run = runs.get((method, str(entry["category"])))
        if run is None:
            raise ValueError(
                f"missing prediction run for method={method}, category={entry['category']}"
            )
        if run["dataset"] != dataset or run["seed"] != seed:
            raise ValueError(
                f"prediction dataset/seed mismatch for method={method}, "
                f"category={entry['category']}"
            )
        selected.append(run)
    return selected


def resolve_visualization_failures(
    *,
    selection_manifest_path: str | Path,
    prediction_directories: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply the F0-frozen failure rankings without human sample choice."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite failure resolution: {output}")
    selection = _load_selection_manifest(selection_manifest_path)
    rule = selection.get("failure_case_rule")
    if not isinstance(rule, dict) or rule.get("status") != "frozen_unresolved":
        raise ValueError("selection manifest has no unresolved frozen failure rule")
    baseline = str(rule["baseline_method"])
    candidate = str(rule["candidate_method"])
    method_order, runs = _load_prediction_runs(prediction_directories)
    if set(method_order) != {baseline, candidate}:
        raise ValueError("failure resolution accepts exactly the frozen baseline and candidate")

    pool = selection.get("failure_candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("selection manifest has an empty failure candidate pool")
    audit_rows: list[dict[str, Any]] = []
    for entry in pool:
        category_runs = _runs_for_entry(
            entry,
            methods=(baseline, candidate),
            runs=runs,
            dataset=str(selection["dataset"]),
            seed=int(selection["model_seed"]),
        )
        baseline_map, baseline_mask, baseline_files = _load_map_and_mask(
            category_runs[0], entry
        )
        candidate_map, candidate_mask, candidate_files = _load_map_and_mask(
            category_runs[1], entry
        )
        if (
            baseline_mask.shape != candidate_mask.shape
            or not np.array_equal(baseline_mask, candidate_mask)
            or baseline_files["evaluation_mask_sha256"]
            != candidate_files["evaluation_mask_sha256"]
        ):
            raise ValueError(f"evaluation mask mismatch for {entry['sample_id']!r}")
        if not candidate_mask.any():
            raise ValueError(
                f"failure candidate {entry['sample_id']!r} has no positive evaluation mask"
            )
        baseline_ap = float(
            average_precision_score(baseline_mask.ravel(), baseline_map.ravel())
        )
        candidate_ap = float(
            average_precision_score(candidate_mask.ravel(), candidate_map.ravel())
        )
        if not math.isfinite(baseline_ap) or not math.isfinite(candidate_ap):
            raise ValueError(f"non-finite per-sample AP for {entry['sample_id']!r}")
        audit_rows.append(
            {
                **dict(entry),
                "baseline_method": baseline,
                "candidate_method": candidate,
                "baseline_pixel_average_precision": baseline_ap,
                "candidate_pixel_average_precision": candidate_ap,
                "candidate_minus_baseline_pixel_average_precision": (
                    candidate_ap - baseline_ap
                ),
                "baseline_raw_map_sha256": baseline_files["raw_map_sha256"],
                "candidate_raw_map_sha256": candidate_files["raw_map_sha256"],
                "evaluation_mask_sha256": candidate_files["evaluation_mask_sha256"],
            }
        )

    rankings_payload: list[dict[str, Any]] = []
    selected_cases: list[dict[str, Any]] = []
    for ranking in rule["rankings"]:
        metric_field = (
            "candidate_pixel_average_precision"
            if ranking["name"] == "candidate_absolute_failures"
            else "candidate_minus_baseline_pixel_average_precision"
        )
        category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in audit_rows:
            category_groups[str(row["category"])].append(row)
        ranked_rows: list[dict[str, Any]] = []
        for category in sorted(category_groups):
            ordered = sorted(
                category_groups[category],
                key=lambda row: (float(row[metric_field]), str(row["sample_id"])),
            )
            count = int(ranking["per_category"])
            if len(ordered) < count:
                raise ValueError(
                    f"failure pool for {category} has fewer than the frozen count {count}"
                )
            selected_ids = {row["sample_id"] for row in ordered[:count]}
            for rank, row in enumerate(ordered, start=1):
                audit = {
                    "category": category,
                    "sample_id": row["sample_id"],
                    "rank": rank,
                    "metric_value": row[metric_field],
                    "selected": row["sample_id"] in selected_ids,
                }
                ranked_rows.append(audit)
                if audit["selected"]:
                    selected_cases.append(
                        {
                            **dict(row),
                            "cohort": _FAILURE_COHORT,
                            "failure_ranking": ranking["name"],
                            "failure_rank_within_category": rank,
                            "failure_metric": ranking["metric"],
                            "failure_metric_value": row[metric_field],
                        }
                    )
        rankings_payload.append({**dict(ranking), "complete_ranking": ranked_rows})

    run_provenance = [
        {
            "method": run["method"],
            "dataset": run["dataset"],
            "category": run["category"],
            "seed": run["seed"],
            "run_config_sha256": run["run_config_sha256"],
            "metrics_sha256": run["metrics_sha256"],
            "prediction_index_sha256": run["prediction_index_sha256"],
            "run_directory": run["root"].as_posix(),
        }
        for run in sorted(runs.values(), key=lambda value: (value["method"], value["category"]))
    ]
    payload = _with_payload_sha256(
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "kind": "formal_visualization_failure_resolution",
            "status": "resolved_by_frozen_rule",
            "cohort": _FAILURE_COHORT,
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "selection_payload_sha256": selection["payload_sha256"],
            "dataset": selection["dataset"],
            "model_seed": selection["model_seed"],
            "rule": rule,
            "prediction_runs": run_provenance,
            "candidate_audit": sorted(audit_rows, key=lambda row: row["sample_id"]),
            "rankings": rankings_payload,
            "selected_cases": selected_cases,
            "manual_selection_performed": False,
        }
    )
    write_json(output, payload)
    return payload


def _load_failure_resolution(
    path: str | Path, *, selection_path: str | Path, selection: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _load_json(path, "failure resolution manifest")
    _verify_payload_sha256(payload, "failure resolution manifest")
    if (
        payload.get("kind") != "formal_visualization_failure_resolution"
        or payload.get("status") != "resolved_by_frozen_rule"
        or payload.get("manual_selection_performed") is not False
        or payload.get("selection_manifest_sha256") != sha256_file(selection_path)
        or payload.get("selection_payload_sha256") != selection.get("payload_sha256")
    ):
        raise ValueError("failure resolution is not bound to the frozen selection")
    return payload


def _slug(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value)
    normalized = normalized.strip("_")[:64]
    return normalized or "sample"


def _verify_source_entry(entry: Mapping[str, Any], dataset_root: str | Path) -> Path:
    image = _verify_file(
        dataset_root,
        entry["source_image"],
        entry["source_image_sha256"],
        f"{entry['sample_id']} source image",
    )
    if entry["source_mask"] is not None:
        _verify_file(
            dataset_root,
            entry["source_mask"],
            entry["source_mask_sha256"],
            f"{entry['sample_id']} source mask",
        )
    return image


def _render_png(
    *,
    image_path: Path,
    mask: np.ndarray,
    maps: Sequence[np.ndarray],
    methods: Sequence[str],
    title: str,
    output_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    columns = 2 + len(methods)
    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(3.0 * columns, 3.6),
        dpi=120,
        squeeze=False,
    )
    row_axes = axes[0]
    row_axes[0].imshow(image)
    row_axes[0].set_title("Input")
    row_axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    row_axes[1].set_title("GT\n(metric mask)")
    heatmap_axes = []
    image_artist = None
    for axis, method, raw_map in zip(row_axes[2:], methods, maps, strict=True):
        image_artist = axis.imshow(
            raw_map,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(f"{method}\nraw_map (metric source)")
        heatmap_axes.append(axis)
    for axis in row_axes:
        axis.set_axis_off()
    figure.suptitle(title, fontsize=10)
    figure.text(
        0.5,
        0.015,
        "SHA-verified raw_map arrays; one shared display scale; metric arrays unchanged",
        ha="center",
        fontsize=7,
    )
    if image_artist is not None:
        figure.colorbar(
            image_artist,
            ax=heatmap_axes,
            fraction=0.025,
            pad=0.02,
            label="raw anomaly score (display only)",
        )
    figure.subplots_adjust(left=0.01, right=0.96, bottom=0.08, top=0.82, wspace=0.08)
    figure.savefig(
        output_path,
        dpi=120,
        facecolor="white",
        metadata={"Software": "industrial-anomaly-lab formal visualization v1"},
    )
    plt.close(figure)


def render_visualization_comparisons(
    *,
    selection_manifest_path: str | Path,
    dataset_root: str | Path,
    prediction_directories: Sequence[str | Path],
    output_dir: str | Path,
    failure_resolution_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render frozen samples using the exact arrays consumed by metrics."""

    selection = _load_selection_manifest(selection_manifest_path)
    method_order, runs = _load_prediction_runs(prediction_directories)
    registered_methods = selection.get("method_registry", {}).get("methods")
    if not isinstance(registered_methods, list) or not set(method_order).issubset(
        set(registered_methods)
    ):
        raise ValueError("prediction methods are absent from the frozen method registry")
    fixed_entries = selection.get("result_independent_samples")
    if not isinstance(fixed_entries, list) or not fixed_entries:
        raise ValueError("frozen result-independent visualization cohort is empty")
    entries: list[dict[str, Any]] = [dict(entry) for entry in fixed_entries]
    failure_sha: str | None = None
    if failure_resolution_path is not None:
        resolution = _load_failure_resolution(
            failure_resolution_path,
            selection_path=selection_manifest_path,
            selection=selection,
        )
        selected_cases = resolution.get("selected_cases")
        if not isinstance(selected_cases, list) or not selected_cases:
            raise ValueError("failure resolution selected no cases")
        resolved_runs = resolution.get("prediction_runs")
        current_runs = [
            {
                "method": run["method"],
                "dataset": run["dataset"],
                "category": run["category"],
                "seed": run["seed"],
                "run_config_sha256": run["run_config_sha256"],
                "metrics_sha256": run["metrics_sha256"],
                "prediction_index_sha256": run["prediction_index_sha256"],
                "run_directory": run["root"].as_posix(),
            }
            for run in sorted(
                runs.values(), key=lambda value: (value["method"], value["category"])
            )
        ]
        if resolved_runs != current_runs:
            raise ValueError(
                "failure resolution was computed from different prediction runs"
            )
        entries.extend(dict(entry) for entry in selected_cases)
        failure_sha = sha256_file(failure_resolution_path)

    prepared: list[dict[str, Any]] = []
    for entry in entries:
        image_path = _verify_source_entry(entry, dataset_root)
        category_runs = _runs_for_entry(
            entry,
            methods=method_order,
            runs=runs,
            dataset=str(selection["dataset"]),
            seed=int(selection["model_seed"]),
        )
        maps: list[np.ndarray] = []
        mask: np.ndarray | None = None
        array_records: list[dict[str, Any]] = []
        expected_mask_sha: str | None = None
        for run in category_runs:
            raw_map, current_mask, files = _load_map_and_mask(run, entry)
            if mask is None:
                mask = current_mask
                expected_mask_sha = files["evaluation_mask_sha256"]
            elif (
                current_mask.shape != mask.shape
                or not np.array_equal(current_mask, mask)
                or files["evaluation_mask_sha256"] != expected_mask_sha
            ):
                raise ValueError(f"evaluation mask mismatch for {entry['sample_id']!r}")
            maps.append(raw_map)
            array_records.append(
                {
                    "method": run["method"],
                    "run_config_sha256": run["run_config_sha256"],
                    "prediction_index_sha256": run["prediction_index_sha256"],
                    **files,
                }
            )
        if mask is None:
            raise AssertionError("no evaluation mask loaded")
        raw_min = min(float(np.min(raw_map)) for raw_map in maps)
        raw_max = max(float(np.max(raw_map)) for raw_map in maps)
        display_max = raw_max if raw_max > raw_min else float(np.nextafter(raw_min, math.inf))
        prepared.append(
            {
                "entry": entry,
                "image_path": image_path,
                "mask": mask,
                "maps": maps,
                "arrays": array_records,
                "display_vmin": raw_min,
                "display_vmax": display_max,
                "raw_max_before_constant_guard": raw_max,
            }
        )

    output = prepare_output_directory(output_dir)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True)
    figure_rows: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        entry = item["entry"]
        cohort = str(entry["cohort"])
        suffix = hashlib.sha256(
            f"{cohort}\0{entry['sample_id']}".encode("utf-8")
        ).hexdigest()[:10]
        filename = (
            f"{index:04d}_{_slug(cohort)}_{_slug(str(entry['sample_id']))}_{suffix}.png"
        )
        figure_path = figures_dir / filename
        title = (
            f"{selection['dataset']} | {entry['category']} | {entry['sample_id']} | "
            f"{cohort}"
        )
        _render_png(
            image_path=item["image_path"],
            mask=item["mask"],
            maps=item["maps"],
            methods=method_order,
            title=title,
            output_path=figure_path,
            vmin=item["display_vmin"],
            vmax=item["display_vmax"],
        )
        figure_rows.append(
            {
                "cohort": cohort,
                "failure_ranking": entry.get("failure_ranking"),
                "dataset": entry["dataset"],
                "category": entry["category"],
                "sample_id": entry["sample_id"],
                "label": entry["label"],
                "defect_type": entry["defect_type"],
                "anomaly_scale": entry["anomaly_scale"],
                "source_image": entry["source_image"],
                "source_image_sha256": entry["source_image_sha256"],
                "source_mask": entry["source_mask"],
                "source_mask_sha256": entry["source_mask_sha256"],
                "evaluation_mask_sha256": item["arrays"][0][
                    "evaluation_mask_sha256"
                ],
                "methods": item["arrays"],
                "display": {
                    "colormap": "magma",
                    "normalization": "shared_per_figure_raw_minmax",
                    "vmin": item["display_vmin"],
                    "vmax": item["display_vmax"],
                    "raw_max_before_constant_guard": item[
                        "raw_max_before_constant_guard"
                    ],
                    "affects_metric_data": False,
                },
                "figure": f"figures/{filename}",
                "figure_sha256": sha256_file(figure_path),
            }
        )

    run_provenance = [
        {
            "method": run["method"],
            "dataset": run["dataset"],
            "category": run["category"],
            "seed": run["seed"],
            "run_config_sha256": run["run_config_sha256"],
            "metrics_sha256": run["metrics_sha256"],
            "prediction_index_sha256": run["prediction_index_sha256"],
            "run_directory": run["root"].as_posix(),
        }
        for run in sorted(runs.values(), key=lambda value: (value["method"], value["category"]))
    ]
    payload = _with_payload_sha256(
        {
            "schema_version": FIGURE_SCHEMA_VERSION,
            "kind": "formal_visualization_figure_manifest",
            "status": "rendered_from_sha_verified_metric_arrays",
            "dataset": selection["dataset"],
            "model_seed": selection["model_seed"],
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "selection_payload_sha256": selection["payload_sha256"],
            "failure_resolution_sha256": failure_sha,
            "cohort_definitions": {
                _FIXED_COHORT: "chosen before predictions by frozen IDs or salted hashing",
                _FAILURE_COHORT: "chosen after predictions only by the F0-frozen ranking rule",
            },
            "method_order": method_order,
            "prediction_runs": run_provenance,
            "figures": figure_rows,
            "metric_arrays_modified": False,
            "raw_maps_rewritten": False,
            "manual_sample_selection_performed": False,
        }
    )
    write_json(output / "figure_manifest.json", payload)
    return payload
