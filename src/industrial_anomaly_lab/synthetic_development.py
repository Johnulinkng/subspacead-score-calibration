"""Frozen, test-blind synthetic development inputs for DRAEM.

This module never discovers dataset files.  It accepts only paths registered in
the pinned VisA/DTD development manifests, authenticates every selected source
file by size and SHA-256, and renders each positive from fully specified recipe
fields.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset

from .development_protocol import (
    AUGMENTER_IDS,
    SYNTHETIC_RECIPE_KEYS,
    SYNTHETIC_RECIPE_SCHEMA_VERSION,
    SYNTHETIC_RENDERER_ID,
    build_synthetic_dev_recipes,
    collect_renderer_provenance,
    derive_expected_source_split_rows,
    derive_renderer_seed,
    render_perlin_mask,
    require_exact_canonical_rows,
    validate_registered_model_seeds,
    validate_source_split_config,
    validate_synthetic_development_config,
)


_IMGAUG_RENDER_LOCK = threading.Lock()
from .run_artifacts import sha256_file
from .training_inputs import (
    VISA_DEVELOPMENT_PROTOCOL,
    _artifact_index,
    _inside,
    _load_json_object_and_sha256,
    _require_no_link_components,
    _require_safe_project_path,
    _selection_hash,
    _validate_common_row,
    _validated_relative_path,
    _validated_relative_path_for_json,
)


@dataclass(frozen=True)
class VerifiedSource:
    dataset: str
    category: str
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ResolvedRecipe:
    row: dict[str, Any]
    target: VerifiedSource
    dtd_source: VerifiedSource


@dataclass(frozen=True)
class SyntheticDevelopmentInputs:
    category: str
    image_size: int
    normal_sources: tuple[VerifiedSource, ...]
    recipes: tuple[ResolvedRecipe, ...]
    evidence: dict[str, Any]


def _validate_categories(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    visa = config.get("visa")
    if not isinstance(visa, dict):
        raise ValueError("development protocol visa entry must be an object")
    categories = visa.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("development protocol categories must be a non-empty list")
    if any(
        not isinstance(item, str)
        or not item
        or "\\" in item
        or PurePosixPath(item).name != item
        for item in categories
    ):
        raise ValueError("development protocol contains an unsafe category name")
    if len(categories) != len(set(categories)):
        raise ValueError("development protocol categories must be unique")
    return visa, categories


def _validate_source_rows(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: str,
    expected_category: str | None = None,
) -> list[dict[str, Any]]:
    seen_paths: set[str] = set()
    hash_splits: dict[str, set[str]] = {}
    for row_number, row in enumerate(rows, start=1):
        label = f"{dataset} row {row_number}"
        if row.get("dataset") != dataset:
            raise ValueError(f"{label} dataset mismatch")
        category = row.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{label} category is invalid")
        if expected_category is not None and category != expected_category:
            raise ValueError(f"{label} category mismatch")
        split, _, content_hash = _validate_common_row(row, label=label)
        relative = _validated_relative_path(row.get("relative_path"), label=label)
        if dataset == "visa":
            if relative.parts[:3] != (category, "train", "good") or len(relative.parts) < 4:
                raise ValueError(f"{label} path is outside {category}/train/good")
        elif len(relative.parts) < 2 or relative.parts[0] != category:
            raise ValueError(f"{label} category/path mismatch")
        key = relative.as_posix().casefold()
        if key in seen_paths:
            raise ValueError(f"duplicate {dataset} relative_path: {relative}")
        seen_paths.add(key)
        hash_splits.setdefault(content_hash, set()).add(split)
    if any(len(splits) != 1 for splits in hash_splits.values()):
        raise ValueError(f"{dataset} content hash appears in both train and dev")
    selected = [row for row in rows if row["split"] == "dev"]
    if not selected:
        raise ValueError(f"{dataset} selected development set is empty")
    return selected


def _verify_dev_sources(
    rows: Sequence[dict[str, Any]],
    *,
    source_root: Path,
    allowed_root: Path,
    dataset: str,
) -> tuple[tuple[VerifiedSource, ...], int]:
    verified: list[VerifiedSource] = []
    total_bytes = 0
    for row_number, row in enumerate(rows, start=1):
        relative = _validated_relative_path(
            row.get("relative_path"), label=f"{dataset} dev row {row_number}"
        )
        candidate = source_root / Path(*relative.parts)
        _require_no_link_components(candidate, allowed_root, f"{dataset} dev")
        path = candidate.resolve()
        if not _inside(path, allowed_root):
            raise ValueError(f"{dataset} dev path escapes its allowed root: {relative}")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{dataset} dev file is missing or linked: {path}")
        expected_size = int(row["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"{dataset} dev file-size mismatch: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"{dataset} dev source SHA-256 mismatch: {relative}")
        verified.append(
            VerifiedSource(
                dataset=dataset,
                category=str(row["category"]),
                relative_path=relative.as_posix(),
                path=path,
                size_bytes=actual_size,
                sha256=actual_hash,
            )
        )
        total_bytes += actual_size
    return tuple(verified), total_bytes


def _as_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _validate_recipes(
    rows: Sequence[dict[str, Any]],
    *,
    category: str,
    protocol_id: str,
    synthetic_config: dict[str, Any],
    normal_sources: Sequence[VerifiedSource],
    dtd_sources: Sequence[VerifiedSource],
) -> tuple[ResolvedRecipe, ...]:
    if synthetic_config.get("renderer_id") != SYNTHETIC_RENDERER_ID:
        raise ValueError("synthetic renderer_id is missing or unsupported")
    if synthetic_config.get("recipe_schema_version") != SYNTHETIC_RECIPE_SCHEMA_VERSION:
        raise ValueError("synthetic recipe schema version is missing or unsupported")
    image_size = int(synthetic_config["image_size"])
    variants = int(synthetic_config["positive_variants_per_dev_image"])
    augmenters_per_positive = int(synthetic_config["augmenters_per_positive"])
    if image_size <= 0 or variants <= 0 or augmenters_per_positive <= 0:
        raise ValueError("synthetic development counts and image size must be positive")
    permitted_scales = {int(item) for item in synthetic_config["perlin_scales"]}
    if not permitted_scales or any(
        scale <= 0 or scale >= image_size or image_size % scale
        for scale in permitted_scales
    ):
        raise ValueError(
            "registered Perlin scales must be positive proper divisors of image_size"
        )
    configured_threshold = _as_number(
        synthetic_config["perlin_threshold"], label="configured Perlin threshold"
    )
    beta_range = synthetic_config["blend_beta_range"]
    if not isinstance(beta_range, list) or len(beta_range) != 2:
        raise ValueError("blend_beta_range must contain two values")
    beta_low, beta_high = (
        _as_number(beta_range[0], label="blend beta lower bound"),
        _as_number(beta_range[1], label="blend beta upper bound"),
    )
    if not 0.0 <= beta_low <= beta_high <= 1.0:
        raise ValueError("blend_beta_range must be within [0, 1]")

    target_by_key = {
        (source.relative_path, source.sha256): source for source in normal_sources
    }
    dtd_by_key = {(source.relative_path, source.sha256): source for source in dtd_sources}
    expected_pairs = {
        (source.relative_path, variant)
        for source in normal_sources
        for variant in range(variants)
    }
    seen_pairs: set[tuple[str, int]] = set()
    seen_ids: set[str] = set()
    resolved: list[ResolvedRecipe] = []
    for row_number, row in enumerate(rows, start=1):
        if row.get("category") != category:
            continue
        label = f"synthetic recipe row {row_number}"
        if set(row) != SYNTHETIC_RECIPE_KEYS:
            raise ValueError(f"{label} exact schema mismatch")
        if row.get("recipe_schema_version") != SYNTHETIC_RECIPE_SCHEMA_VERSION:
            raise ValueError(f"{label} recipe schema version mismatch")
        if row.get("protocol_id") != protocol_id:
            raise ValueError(f"{label} protocol_id mismatch")
        if row.get("renderer_id") != SYNTHETIC_RENDERER_ID:
            raise ValueError(f"{label} renderer_id mismatch")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.startswith(f"{category}/"):
            raise ValueError(f"{label} sample_id is invalid")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate synthetic sample_id: {sample_id}")
        seen_ids.add(sample_id)

        target_relative = _validated_relative_path(
            row.get("target_relative_path"), label=f"{label} target"
        ).as_posix()
        dtd_relative = _validated_relative_path(
            row.get("dtd_relative_path"), label=f"{label} DTD source"
        ).as_posix()
        target_key = (target_relative, row.get("target_sha256"))
        dtd_key = (dtd_relative, row.get("dtd_sha256"))
        if target_key not in target_by_key:
            raise ValueError(f"{label} target is not a verified VisA-dev source")
        if dtd_key not in dtd_by_key:
            raise ValueError(f"{label} source is not a verified DTD-dev source")
        variant = row.get("variant")
        if isinstance(variant, bool) or not isinstance(variant, int) or not 0 <= variant < variants:
            raise ValueError(f"{label} variant is invalid")
        pair = (target_relative, variant)
        if pair in seen_pairs:
            raise ValueError(f"duplicate synthetic target/variant pair: {pair}")
        seen_pairs.add(pair)
        recipe_seed = row.get("recipe_seed")
        perlin_seed = row.get("perlin_seed")
        perlin_attempt = row.get("perlin_seed_attempt")
        if (
            isinstance(recipe_seed, bool)
            or not isinstance(recipe_seed, int)
            or not 0 <= recipe_seed <= 0xFFFFFFFF
        ):
            raise ValueError(f"{label} recipe_seed must be uint32")
        if (
            isinstance(perlin_attempt, bool)
            or not isinstance(perlin_attempt, int)
            or not 0 <= perlin_attempt <= 1024
        ):
            raise ValueError(f"{label} perlin_seed_attempt is invalid")
        if perlin_seed != derive_renderer_seed(recipe_seed, "perlin", perlin_attempt):
            raise ValueError(f"{label} perlin_seed derivation mismatch")
        if row.get("image_size") != image_size:
            raise ValueError(f"{label} image_size mismatch")
        target_rotation_applied = row.get("target_rotation_applied")
        target_rotation_degrees = row.get("target_rotation_degrees")
        if not isinstance(target_rotation_applied, bool):
            raise ValueError(f"{label} target rotation flag is invalid")
        if target_rotation_applied:
            rotation_value = _as_number(
                target_rotation_degrees, label=f"{label} target rotation"
            )
            if not -90.0 <= rotation_value <= 90.0:
                raise ValueError(f"{label} target rotation is outside [-90, 90]")
        elif target_rotation_degrees is not None:
            raise ValueError(f"{label} disabled target rotation must be null")
        augmenter_ids = row.get("augmenter_ids")
        augmenter_seeds = row.get("augmenter_seeds")
        if (
            not isinstance(augmenter_ids, list)
            or len(augmenter_ids) != augmenters_per_positive
            or len(set(augmenter_ids)) != len(augmenter_ids)
            or any(item not in AUGMENTER_IDS for item in augmenter_ids)
        ):
            raise ValueError(f"{label} augmenter_ids are invalid")
        expected_augmenter_seeds = [
            derive_renderer_seed(recipe_seed, "augmenter", position, augmenter_id)
            for position, augmenter_id in enumerate(augmenter_ids)
        ]
        if augmenter_seeds != expected_augmenter_seeds:
            raise ValueError(f"{label} augmenter_seeds derivation mismatch")
        for axis in ("x", "y"):
            scale = row.get(f"perlin_scale_{axis}")
            if scale not in permitted_scales:
                raise ValueError(f"{label} perlin_scale_{axis} is not registered")
        rotation = row.get("perlin_rotation_degrees")
        if type(rotation) is not float or not -90.0 <= rotation <= 90.0:
            raise ValueError(f"{label} perlin rotation is invalid")
        threshold = _as_number(row.get("perlin_threshold"), label=f"{label} threshold")
        if threshold != configured_threshold:
            raise ValueError(f"{label} perlin threshold mismatch")
        beta = _as_number(row.get("blend_beta"), label=f"{label} blend_beta")
        if not beta_low <= beta <= beta_high:
            raise ValueError(f"{label} blend_beta is outside the registered range")
        if row.get("label") != 1 or row.get("rendered") is not False:
            raise ValueError(f"{label} label/rendered contract mismatch")
        for earlier_attempt in range(perlin_attempt):
            earlier_mask = render_perlin_mask(
                image_size=image_size,
                scale_x=int(row["perlin_scale_x"]),
                scale_y=int(row["perlin_scale_y"]),
                rotation_degrees=float(row["perlin_rotation_degrees"]),
                threshold=threshold,
                seed=derive_renderer_seed(recipe_seed, "perlin", earlier_attempt),
            )
            if bool(earlier_mask.any()):
                raise ValueError(f"{label} skipped an earlier valid Perlin seed")
        selected_mask = render_perlin_mask(
            image_size=image_size,
            scale_x=int(row["perlin_scale_x"]),
            scale_y=int(row["perlin_scale_y"]),
            rotation_degrees=float(row["perlin_rotation_degrees"]),
            threshold=threshold,
            seed=int(perlin_seed),
        )
        if not bool(selected_mask.any()):
            raise ValueError(f"{label} declares a positive with an empty mask")
        resolved.append(
            ResolvedRecipe(
                row=dict(row),
                target=target_by_key[target_key],
                dtd_source=dtd_by_key[dtd_key],
            )
        )
    if seen_pairs != expected_pairs:
        missing = len(expected_pairs - seen_pairs)
        extra = len(seen_pairs - expected_pairs)
        raise ValueError(f"synthetic recipe coverage mismatch; missing={missing}, extra={extra}")
    resolved.sort(key=lambda item: str(item.row["sample_id"]))
    return tuple(resolved)


def resolve_synthetic_development_inputs(
    *,
    category: str,
    dataset_root: Path,
    anomaly_source_root: Path,
    seed: int,
    protocol_config_path: Path,
) -> SyntheticDevelopmentInputs:
    """Authenticate the frozen dev bundle without discovering or reading test data."""

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
    visa_config, categories = _validate_categories(config)
    if category not in categories:
        raise ValueError(f"category is not registered by the development protocol: {category}")
    dtd_config = config.get("dtd")
    synthetic_config = config.get("synthetic_development")
    frozen = config.get("frozen_manifest")
    if not isinstance(dtd_config, dict) or not isinstance(synthetic_config, dict):
        raise ValueError("development protocol dtd and synthetic entries must be objects")
    if not isinstance(frozen, dict):
        raise ValueError("development protocol frozen_manifest entry is required")
    validate_source_split_config(visa_config, label="VisA")
    validate_source_split_config(dtd_config, label="DTD")
    validate_synthetic_development_config(synthetic_config)
    false_positive_threshold = _as_number(
        synthetic_config.get("normal_false_positive_threshold"),
        label="normal false-positive threshold",
    )
    if not 0.0 <= false_positive_threshold <= 1.0:
        raise ValueError("normal false-positive threshold must be in [0, 1]")

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
    pinned_hash = frozen.get("sha256")
    summary, actual_hash = _load_json_object_and_sha256(root_manifest_path)
    if not isinstance(pinned_hash, str) or actual_hash != pinned_hash:
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
        summary=summary, manifest_root=manifest_root, expected_names=expected_names
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
    all_visa_rows: list[dict[str, Any]] = []
    all_visa_dev_rows: list[dict[str, Any]] = []
    global_visa_hash_splits: dict[str, set[str]] = {}
    visa_dev_by_category: dict[str, list[dict[str, Any]]] = {}
    for registered_category in categories:
        registered_rows = rows_by_name[f"visa_{registered_category}_split.jsonl"]
        registered_dev = _validate_source_rows(
            registered_rows, dataset="visa", expected_category=registered_category
        )
        expected_registered_rows = derive_expected_source_split_rows(
            registered_rows,
            dataset="visa",
            dev_fraction=float(visa_config["dev_fraction"]),
            split_salt=str(visa_config["split_salt"]),
        )
        require_exact_canonical_rows(
            registered_rows,
            expected_registered_rows,
            label=f"VisA {registered_category} source split",
        )
        actual_counts = {
            "total": len(registered_rows),
            "train": len(registered_rows) - len(registered_dev),
            "dev": len(registered_dev),
        }
        if visa_summary["categories"].get(registered_category) != actual_counts:
            raise ValueError(
                f"root manifest VisA {registered_category} counts mismatch"
            )
        for row in registered_rows:
            global_visa_hash_splits.setdefault(str(row["sha256"]), set()).add(
                str(row["split"])
            )
        all_visa_rows.extend(registered_rows)
        all_visa_dev_rows.extend(registered_dev)
        visa_dev_by_category[registered_category] = registered_dev
    if any(len(splits) != 1 for splits in global_visa_hash_splits.values()):
        raise ValueError("VisA content hash appears in both train and dev")
    expected_visa_totals = {
        "total": len(all_visa_rows),
        "train": len(all_visa_rows) - len(all_visa_dev_rows),
        "dev": len(all_visa_dev_rows),
        "content_hash_overlap": 0,
    }
    for key, value in expected_visa_totals.items():
        if visa_summary.get(key) != value:
            raise ValueError(f"root manifest VisA {key} mismatch")
    visa_rows = rows_by_name[f"visa_{category}_split.jsonl"]
    visa_dev_rows = visa_dev_by_category[category]
    dtd_rows = rows_by_name["dtd_source_split.jsonl"]
    dtd_dev_rows = _validate_source_rows(dtd_rows, dataset="dtd")
    expected_dtd_rows = derive_expected_source_split_rows(
        dtd_rows,
        dataset="dtd",
        dev_fraction=float(dtd_config["dev_fraction"]),
        split_salt=str(dtd_config["split_salt"]),
    )
    require_exact_canonical_rows(dtd_rows, expected_dtd_rows, label="DTD source split")
    dtd_summary = summary.get("dtd")
    expected_dtd_summary = {
        "categories": len({str(row["category"]) for row in dtd_rows}),
        "total": len(dtd_rows),
        "train": len(dtd_rows) - len(dtd_dev_rows),
        "dev": len(dtd_dev_rows),
        "content_hash_overlap": 0,
    }
    if dtd_summary != expected_dtd_summary:
        raise ValueError("root manifest DTD counts mismatch")

    expected_recipe_rows = build_synthetic_dev_recipes(
        all_visa_rows,
        dtd_rows,
        protocol_id=VISA_DEVELOPMENT_PROTOCOL,
        recipe_salt=str(synthetic_config["recipe_salt"]),
        variants_per_image=int(synthetic_config["positive_variants_per_dev_image"]),
        image_size=int(synthetic_config["image_size"]),
        augmenters_per_positive=int(synthetic_config["augmenters_per_positive"]),
        perlin_scales=[int(value) for value in synthetic_config["perlin_scales"]],
        perlin_threshold=float(synthetic_config["perlin_threshold"]),
        blend_beta_range=(
            float(synthetic_config["blend_beta_range"][0]),
            float(synthetic_config["blend_beta_range"][1]),
        ),
        target_rotation_probability=float(
            synthetic_config["target_rotation_probability"]
        ),
        recipe_schema_version=int(synthetic_config["recipe_schema_version"]),
    )
    actual_recipe_rows = rows_by_name["synthetic_dev_recipes.jsonl"]
    require_exact_canonical_rows(
        actual_recipe_rows,
        expected_recipe_rows,
        label="synthetic recipe manifest",
    )
    synthetic_summary = summary.get("synthetic_development")
    expected_synthetic_summary = {
        "renderer_id": SYNTHETIC_RENDERER_ID,
        "recipe_schema_version": SYNTHETIC_RECIPE_SCHEMA_VERSION,
        "normal_samples": len(all_visa_dev_rows),
        "synthetic_positive_recipes": len(expected_recipe_rows),
        "empty_mask_recipes": 0,
        "images_rendered": 0,
    }
    if synthetic_summary != expected_synthetic_summary:
        raise ValueError("root manifest synthetic-development summary mismatch")
    normal_sources, normal_bytes = _verify_dev_sources(
        visa_dev_rows,
        source_root=expected_visa_root,
        allowed_root=expected_visa_root / category / "train" / "good",
        dataset="VisA",
    )
    dtd_sources, dtd_bytes = _verify_dev_sources(
        dtd_dev_rows,
        source_root=expected_dtd_root,
        allowed_root=expected_dtd_root,
        dataset="DTD",
    )
    recipes = _validate_recipes(
        actual_recipe_rows,
        category=category,
        protocol_id=VISA_DEVELOPMENT_PROTOCOL,
        synthetic_config=synthetic_config,
        normal_sources=normal_sources,
        dtd_sources=dtd_sources,
    )

    recipe_rows = [item.row for item in recipes]
    evidence = {
        "schema_version": 1,
        "evaluation_scope": "synthetic-development-only",
        "claim_eligible": False,
        "manifest_backed": True,
        "protocol_config_path": config_path.as_posix(),
        "protocol_config_sha256": config_hash,
        "root_manifest_path": root_manifest_path.as_posix(),
        "root_manifest_sha256": actual_hash,
        "category_manifest": {
            "path": (manifest_root / f"visa_{category}_split.jsonl").as_posix(),
            "sha256": artifact_entries[f"visa_{category}_split.jsonl"]["sha256"],
            "selected_dev_rows": len(visa_dev_rows),
            "selected_dev_sha256": _selection_hash(visa_dev_rows),
        },
        "dtd_manifest": {
            "path": (manifest_root / "dtd_source_split.jsonl").as_posix(),
            "sha256": artifact_entries["dtd_source_split.jsonl"]["sha256"],
            "selected_dev_rows": len(dtd_dev_rows),
            "selected_dev_sha256": _selection_hash(dtd_dev_rows),
        },
        "recipe_manifest": {
            "path": (manifest_root / "synthetic_dev_recipes.jsonl").as_posix(),
            "sha256": artifact_entries["synthetic_dev_recipes.jsonl"]["sha256"],
            "selected_rows": len(recipe_rows),
            "selected_sha256": _selection_hash(recipe_rows),
        },
        "registered_seed": seed,
        "registered_seeds": registered_seeds,
        "normal_dev_images": len(normal_sources),
        "normal_dev_bytes": normal_bytes,
        "dtd_dev_images": len(dtd_sources),
        "dtd_dev_bytes": dtd_bytes,
        "synthetic_positive_recipes": len(recipes),
        "recipe_schema_version": SYNTHETIC_RECIPE_SCHEMA_VERSION,
        "renderer_id": SYNTHETIC_RENDERER_ID,
        "renderer_provenance": current_renderer_provenance,
        "renderer_provenance_sha256": _selection_hash(
            [current_renderer_provenance]
        ),
        "normal_false_positive_threshold": false_positive_threshold,
        "global_recipe_derivation_verified": True,
        "global_recipe_rows": len(actual_recipe_rows),
        "selected_file_size_and_sha256_verified": True,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "training_started": False,
        "performance_evaluated": False,
    }
    return SyntheticDevelopmentInputs(
        category=category,
        image_size=int(synthetic_config["image_size"]),
        normal_sources=normal_sources,
        recipes=recipes,
        evidence=evidence,
    )


def _decode_verified(source: VerifiedSource) -> np.ndarray:
    _require_no_link_components(source.path, source.path.parent, "verified source")
    payload = source.path.read_bytes()
    if len(payload) != source.size_bytes:
        raise ValueError(f"source changed after validation (size): {source.relative_path}")
    if hashlib.sha256(payload).hexdigest() != source.sha256:
        raise ValueError(f"source changed after validation (SHA-256): {source.relative_path}")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode verified source: {source.relative_path}")
    return image


def _make_augmenter(augmenter_id: str, seed: int) -> iaa.Augmenter:
    # Some composite imgaug operators construct internal children that retain
    # the library-global RNG.  Reset it from the frozen sub-seed before
    # construction, then recursively localize every child below.
    imgaug.seed(seed)
    constructors: dict[str, Any] = {
        "gamma_contrast": lambda: iaa.GammaContrast((0.5, 2.0), per_channel=True),
        "brightness_multiply_add": lambda: iaa.MultiplyAndAddToBrightness(
            mul=(0.8, 1.2), add=(-30, 30)
        ),
        "sharpness": lambda: iaa.pillike.EnhanceSharpness(),
        "hue_saturation": lambda: iaa.AddToHueAndSaturation(
            (-50, 50), per_channel=True
        ),
        "solarize": lambda: iaa.Solarize(0.5, threshold=(32, 128)),
        "posterize": lambda: iaa.Posterize(),
        "invert": lambda: iaa.Invert(),
        "autocontrast": lambda: iaa.pillike.Autocontrast(),
        "equalize": lambda: iaa.pillike.Equalize(),
        "affine_rotate": lambda: iaa.Affine(rotate=(-45, 45)),
    }
    try:
        augmenter = constructors[augmenter_id]()
    except KeyError as error:
        raise ValueError(f"unsupported augmenter_id: {augmenter_id}") from error
    # ``seed=`` on the outer imgaug augmenter leaves some internal children on
    # the global RNG.  Recursively localize them, then freeze the one-shot draw.
    sequence = iaa.Sequential([augmenter], random_order=False)
    sequence.seed_(seed, deterministic_too=True)
    return sequence.to_deterministic()


def blend_synthetic_anomaly(
    target: np.ndarray,
    augmented_source: np.ndarray,
    mask_2d: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Apply the author's blend formula and return the renderer float32 contract."""

    mask = mask_2d.astype(np.float32)[:, :, None]
    source_region = augmented_source.astype(np.float32) * mask / 255.0
    augmented = (
        target * (1.0 - mask)
        + (1.0 - beta) * source_region
        + beta * target * mask
    )
    return (mask * augmented + (1.0 - mask) * target).astype(
        np.float32, copy=False
    )


def render_synthetic_recipe(recipe: ResolvedRecipe) -> tuple[np.ndarray, np.ndarray]:
    """Render one frozen recipe as BGR float32 input and a binary mask."""

    row = recipe.row
    size = int(row["image_size"])
    target_uint8 = cv2.resize(
        _decode_verified(recipe.target), (size, size), interpolation=cv2.INTER_LINEAR
    )
    source = cv2.resize(
        _decode_verified(recipe.dtd_source), (size, size), interpolation=cv2.INTER_LINEAR
    )
    # imgaug 0.4.0 has composite operators whose child constructors can touch
    # its process-global RNG.  Serialize that short section so direct threaded
    # callers retain the renderer-v2 byte contract. DataLoader workers are
    # processes and therefore keep independent locks/RNGs.
    with _IMGAUG_RENDER_LOCK:
        if bool(row["target_rotation_applied"]):
            target_uint8 = iaa.Affine(
                rotate=float(row["target_rotation_degrees"]),
                order=1,
                cval=0,
                mode="constant",
            )(image=target_uint8)
        for augmenter_id, augmenter_seed in zip(
            row["augmenter_ids"], row["augmenter_seeds"], strict=True
        ):
            source = _make_augmenter(str(augmenter_id), int(augmenter_seed))(
                image=source
            )
        mask_2d = render_perlin_mask(
            image_size=size,
            scale_x=int(row["perlin_scale_x"]),
            scale_y=int(row["perlin_scale_y"]),
            rotation_degrees=float(row["perlin_rotation_degrees"]),
            threshold=float(row["perlin_threshold"]),
            seed=int(row["perlin_seed"]),
        )
    target = target_uint8.astype(np.float32) / 255.0
    if not bool(mask_2d.any()):
        raise ValueError(f"synthetic positive rendered an empty mask: {row['sample_id']}")
    beta = float(row["blend_beta"])
    augmented = blend_synthetic_anomaly(
        target, source, mask_2d, beta=beta
    )
    if not np.isfinite(augmented).all():
        raise ValueError(f"synthetic render is non-finite: {row['sample_id']}")
    return augmented.astype(np.float32, copy=False), mask_2d


class SyntheticDevelopmentDataset(Dataset[dict[str, object]]):
    """Normal VisA-dev samples plus every frozen synthetic positive recipe."""

    def __init__(self, inputs: SyntheticDevelopmentInputs) -> None:
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.inputs.normal_sources) + len(self.inputs.recipes)

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.inputs.normal_sources):
            source = self.inputs.normal_sources[index]
            image = cv2.resize(
                _decode_verified(source),
                (self.inputs.image_size, self.inputs.image_size),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32) / 255.0
            mask = np.zeros((self.inputs.image_size, self.inputs.image_size), dtype=bool)
            sample_id = f"{self.inputs.category}/{Path(source.relative_path).stem}/normal_dev"
            recipe_id = ""
            dtd_relative_path = ""
            label = 0
            input_relative_path = source.relative_path
        else:
            recipe = self.inputs.recipes[index - len(self.inputs.normal_sources)]
            image, mask = render_synthetic_recipe(recipe)
            sample_id = str(recipe.row["sample_id"])
            recipe_id = sample_id
            dtd_relative_path = recipe.dtd_source.relative_path
            label = 1
            input_relative_path = recipe.target.relative_path
        return {
            "image": torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()),
            "mask": torch.from_numpy(mask[None, :, :].astype(np.float32)),
            "label": torch.tensor(label, dtype=torch.int64),
            "sample_id": sample_id,
            "recipe_id": recipe_id,
            "input_relative_path": input_relative_path,
            "dtd_relative_path": dtd_relative_path,
        }
