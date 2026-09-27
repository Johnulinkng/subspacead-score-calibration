"""Fail-closed run-purpose and formal provenance contracts.

The contract deliberately separates a complete run from a formal run.  A
complete author reproduction or development run can be useful, but neither is
eligible for the registered F0 matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from .run_artifacts import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "configs" / "project.json"
DEFAULT_REGISTRATION = PROJECT_ROOT / "configs" / "preregistered_experiments.json"
DEFAULT_DEVELOPMENT_PROTOCOL = PROJECT_ROOT / "configs" / "development_protocol.json"

RunPurpose = Literal[
    "debug_limited", "development_full", "formal_full", "author_reproduction"
]
RUN_PURPOSES = frozenset(
    {"debug_limited", "development_full", "formal_full", "author_reproduction"}
)
CLAIM_SCOPE_BY_PURPOSE = {
    "debug_limited": "debug_only",
    "development_full": "development_only",
    "formal_full": "formal_matrix_candidate",
    "author_reproduction": "author_reproduction_only",
}
FORMAL_INPUT_PROVENANCE_METHODS = frozenset({"DRAEM", "PatchCore", "RD4AD"})
EXPECTED_FORMAL_VENDOR_SOURCES = {
    "DRAEM": (
        "third_party/DRAEM",
        "2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef",
    ),
    "PatchCore": (
        "third_party/PatchCore",
        "fcaa92f124fb1ad74a7acf56726decd4b27cbcad",
    ),
    "RD4AD": (
        "third_party/RD4AD",
        "6554076872c65f8784f6ece8cfb39ce77e1aee12",
    ),
}


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_mapping(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return resolved, payload


def _registered_seeds(
    registration: Mapping[str, object], project: Mapping[str, object]
) -> list[int]:
    seeds = registration.get("model_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("formal/development authority requires at least 3 unique registered seeds")
    evaluation = project.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("project evaluation configuration is missing")
    if evaluation.get("minimum_seeds") != len(seeds):
        raise ValueError("project minimum_seeds does not match registered seed count")
    if evaluation.get("seeds") != seeds:
        raise ValueError("project seeds do not exactly match registration model_seeds")
    return seeds


def classify_run_purpose(run_purpose: str, *, max_steps: int | None) -> dict[str, object]:
    """Validate purpose/completeness without inferring formal eligibility."""

    if run_purpose not in RUN_PURPOSES:
        raise ValueError(f"unsupported run_purpose: {run_purpose}")
    if run_purpose == "debug_limited":
        if max_steps is None or max_steps <= 0:
            raise ValueError("debug_limited requires a positive max_steps")
        execution_scope = "debug_limited"
    else:
        if max_steps is not None:
            raise ValueError(f"max_steps is forbidden for {run_purpose}")
        execution_scope = "full"
    return {
        "run_purpose": run_purpose,
        "execution_scope": execution_scope,
        "claim_scope": CLAIM_SCOPE_BY_PURPOSE[run_purpose],
        "formal_matrix_eligible": run_purpose == "formal_full",
    }


def validate_development_authority(
    *,
    method: str,
    dataset: str,
    category: str,
    seed: int,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    registration_path: str | Path = DEFAULT_REGISTRATION,
    development_protocol_path: str | Path = DEFAULT_DEVELOPMENT_PROTOCOL,
) -> dict[str, object]:
    """Authorize claim-ineligible full development without a formal-method freeze."""

    project_path, project = _load_mapping(project_config_path, "project config")
    register_path, registration = _load_mapping(registration_path, "registration")
    development_path, development = _load_mapping(
        development_protocol_path, "development protocol"
    )
    seeds = _registered_seeds(registration, project)
    if seed not in seeds:
        raise ValueError(f"development seed is not preregistered: {seed}")
    if dataset != "visa":
        raise ValueError("development_full is restricted to the frozen VisA development protocol")
    datasets = project.get("datasets")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get(dataset), Mapping):
        raise ValueError(f"dataset is not registered: {dataset}")
    categories = datasets[dataset].get("categories")  # type: ignore[index]
    if not isinstance(categories, list) or category not in categories:
        raise ValueError(f"category is not registered for {dataset}: {category}")
    if development.get("status") != "frozen_before_model_development":
        raise ValueError("development protocol is not frozen")
    return {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "registered_seed_count": len(seeds),
        "registration_id": registration.get("registration_id"),
        "registration_path": str(register_path),
        "registration_sha256": sha256_file(register_path),
        "project_config_path": str(project_path),
        "project_config_sha256": sha256_file(project_path),
        "development_protocol_path": str(development_path),
        "development_protocol_sha256": sha256_file(development_path),
        "run_purpose": "development_full",
        "claim_scope": "development_only",
        "formal_matrix_eligible": False,
    }


def validate_formal_authority(
    *,
    method: str,
    dataset: str,
    category: str,
    seed: int,
    phase: Literal["training", "inference"],
    recipe: Mapping[str, object],
    training_input_evidence: Mapping[str, object] | None = None,
    formal_input_project_root: str | Path | None = None,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    registration_path: str | Path = DEFAULT_REGISTRATION,
) -> dict[str, object]:
    """Authorize one F0 action only when the exact recipe is preregistered."""

    project_path, project = _load_mapping(project_config_path, "project config")
    register_path, registration = _load_mapping(registration_path, "registration")
    evaluation = project.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("formal_protocol_frozen") is not True:
        raise ValueError("formal evaluation protocol is not frozen")
    seeds = _registered_seeds(registration, project)
    if seed not in seeds:
        raise ValueError(f"formal seed is not preregistered: {seed}")

    datasets = project.get("datasets")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get(dataset), Mapping):
        raise ValueError(f"dataset is not registered: {dataset}")
    categories = datasets[dataset].get("categories")  # type: ignore[index]
    if not isinstance(categories, list) or category not in categories:
        raise ValueError(f"category is not registered for {dataset}: {category}")

    registry = registration.get("formal_method_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("formal method registry is missing")
    if registry.get("status") != "frozen_before_formal_evaluation":
        raise ValueError("formal method registry is not frozen")
    methods = registry.get("methods")
    if not isinstance(methods, list) or method not in methods:
        raise ValueError(f"method is not in the frozen formal registry: {method}")
    recipes = registry.get("recipes")
    if not isinstance(recipes, Mapping) or not isinstance(recipes.get(method), Mapping):
        raise ValueError(f"formal recipe is not frozen for method: {method}")
    method_recipe = recipes[method]  # type: ignore[index]
    recipe_id = method_recipe.get("recipe_id")
    registered_recipe = method_recipe.get(phase)
    if not isinstance(recipe_id, str) or not recipe_id:
        raise ValueError(f"formal recipe_id is missing for method: {method}")
    if not isinstance(registered_recipe, Mapping):
        raise ValueError(f"formal {phase} recipe is missing for method: {method}")
    if dict(registered_recipe) != dict(recipe):
        raise ValueError(
            f"formal {phase} recipe differs from registration: "
            f"expected_sha256={canonical_sha256(registered_recipe)}, "
            f"observed_sha256={canonical_sha256(recipe)}"
        )
    if method == "DRAEM":
        baseline = project.get("baseline")
        if not isinstance(baseline, Mapping) or baseline.get("formal_training_recipe_frozen") is not True:
            raise ValueError("DRAEM formal training recipe is not frozen in project config")
    training_input_validation: dict[str, object] | None = None
    if method in FORMAL_INPUT_PROVENANCE_METHODS and phase == "training":
        if not isinstance(training_input_evidence, Mapping):
            raise ValueError(
                f"{method} formal_full is blocked: manifest-backed training-input "
                "provenance is required"
            )
        from .formal_training_inputs import (  # local import avoids a module cycle
            load_frozen_formal_input_reference,
            revalidate_formal_training_input_evidence,
        )

        input_project_root = (
            Path(formal_input_project_root).resolve()
            if formal_input_project_root is not None
            else (
                project_path.parent.parent
                if project_path.parent.name == "configs"
                else PROJECT_ROOT
            )
        )
        refreshed = revalidate_formal_training_input_evidence(
            training_input_evidence,
            method=method,
            dataset=dataset,
            category=category,
            project_root=input_project_root,
        )
        pinned = load_frozen_formal_input_reference(
            project_config_path=project_path,
            project_root=input_project_root,
        )
        if refreshed.get("root_manifest_path") != pinned.get("manifest"):
            raise ValueError(
                "formal training-input evidence manifest differs from project pin"
            )
        if refreshed.get("root_manifest_sha256") != pinned.get("sha256"):
            raise ValueError(
                "formal training-input evidence root SHA-256 differs from project pin"
            )
        training_input_validation = {
            "contract": refreshed["contract"],
            "root_manifest_path": refreshed["root_manifest_path"],
            "root_manifest_sha256": refreshed["root_manifest_sha256"],
            "pinned_status": pinned["status"],
            "pinned_in_project_config": True,
            "category_manifest_sha256": refreshed["category_manifest"]["sha256"],
            "dtd_manifest_sha256": (
                None
                if refreshed["dtd_manifest"] is None
                else refreshed["dtd_manifest"]["manifest_sha256"]
            ),
            "current_file_set_and_content_sha256_verified": True,
            "formal_test_content_accessed": False,
            "ground_truth_content_accessed": False,
        }

    return {
        "schema_version": 1,
        "method": method,
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "phase": phase,
        "recipe_id": recipe_id,
        "recipe_sha256": canonical_sha256(recipe),
        "registered_seed_count": len(seeds),
        "registration_id": registration.get("registration_id"),
        "registration_path": str(register_path),
        "registration_sha256": sha256_file(register_path),
        "project_config_path": str(project_path),
        "project_config_sha256": sha256_file(project_path),
        "formal_registry_status": registry.get("status"),
        "run_purpose": "formal_full",
        "claim_scope": "formal_matrix_candidate",
        "formal_matrix_eligible": True,
        "training_input_validation": training_input_validation,
    }


def validate_formal_checkpoint_manifest(
    manifest_path: str | Path,
    *,
    method: str,
    dataset: str,
    category: str,
    seed: int,
    checkpoint_files: Mapping[str, str | Path],
    external_hashes: Mapping[str, str] | None = None,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    registration_path: str | Path = DEFAULT_REGISTRATION,
) -> dict[str, object]:
    """Bind formal inference to an exact project-trained checkpoint manifest."""

    resolved, manifest = _load_mapping(manifest_path, "checkpoint manifest")
    if manifest.get("schema_version") != 2:
        raise ValueError("formal checkpoint manifest must use schema_version 2")
    expected_identity = {
        "method": method,
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "run_purpose": "formal_full",
        "execution_scope": "full",
        "claim_scope": "formal_matrix_candidate",
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"checkpoint manifest {key} mismatch: expected={expected!r}, "
                f"observed={manifest.get(key)!r}"
            )
    if manifest.get("checkpoint_origin") != "project_training_run":
        raise ValueError("formal checkpoint was not produced by a project training run")
    if manifest.get("final_checkpoint_only") is not True or manifest.get("test_selected") is not False:
        raise ValueError("formal checkpoint manifest lacks final-only/no-test-selection proof")
    training_recipe = manifest.get("training_recipe")
    if not isinstance(training_recipe, Mapping):
        raise ValueError("checkpoint manifest is missing its training recipe")
    formal_training_inputs = manifest.get("formal_training_inputs")
    if method in FORMAL_INPUT_PROVENANCE_METHODS and not isinstance(
        formal_training_inputs, Mapping
    ):
        raise ValueError(
            "formal checkpoint manifest is missing training-input provenance"
        )
    authority = validate_formal_authority(
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        phase="training",
        recipe=training_recipe,
        training_input_evidence=(
            formal_training_inputs
            if isinstance(formal_training_inputs, Mapping)
            else None
        ),
        project_config_path=project_config_path,
        registration_path=registration_path,
    )
    embedded = manifest.get("formal_provenance")
    if not isinstance(embedded, Mapping):
        raise ValueError("checkpoint manifest is missing formal provenance")
    for key in (
        "registration_id",
        "registration_sha256",
        "project_config_sha256",
        "recipe_id",
        "recipe_sha256",
    ):
        if embedded.get(key) != authority.get(key):
            raise ValueError(f"checkpoint manifest formal provenance mismatch: {key}")
    if method in FORMAL_INPUT_PROVENANCE_METHODS and embedded.get(
        "training_input_validation"
    ) != authority.get("training_input_validation"):
        raise ValueError(
            "checkpoint manifest formal provenance mismatch: training_input_validation"
        )

    entries = manifest.get("checkpoints")
    if not isinstance(entries, Mapping) or set(entries) != set(checkpoint_files):
        raise ValueError("checkpoint manifest roles do not match inference checkpoints")
    checked: dict[str, object] = {}
    for role, supplied in checkpoint_files.items():
        entry = entries.get(role)
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid checkpoint manifest entry: {role}")
        supplied_path = Path(supplied).resolve()
        if not supplied_path.is_file():
            raise FileNotFoundError(supplied_path)
        recorded_path = entry.get("path")
        if (
            not isinstance(recorded_path, str)
            or not recorded_path
            or "/" in recorded_path
            or "\\" in recorded_path
            or Path(recorded_path).name != recorded_path
            or recorded_path != supplied_path.name
        ):
            raise ValueError(f"checkpoint filename mismatch: {role}")
        actual_sha = sha256_file(supplied_path)
        if entry.get("sha256") != actual_sha:
            raise ValueError(f"checkpoint SHA-256 mismatch: {role}")
        if entry.get("size_bytes") != supplied_path.stat().st_size:
            raise ValueError(f"checkpoint size mismatch: {role}")
        checked[role] = {
            "path": str(supplied_path),
            "sha256": actual_sha,
            "size_bytes": supplied_path.stat().st_size,
        }

    recorded_external = manifest.get("external_hashes", {})
    if not isinstance(recorded_external, Mapping):
        raise ValueError("checkpoint manifest external_hashes must be an object")
    supplied_external = dict(external_hashes or {})
    if set(recorded_external) != set(supplied_external):
        raise ValueError(
            "checkpoint manifest external_hashes roles do not match supplied hashes"
        )
    for role, expected_sha in supplied_external.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or recorded_external.get(role) != expected_sha
        ):
            raise ValueError(f"external checkpoint hash mismatch: {role}")

    if method == "PatchCore":
        if manifest.get("artifact_type") != "patchcore_safe_fitted_state":
            raise ValueError("PatchCore formal manifest has the wrong artifact_type")
        if set(checkpoint_files) != {"memory_bank", "faiss_index", "backbone"}:
            raise ValueError(
                "PatchCore formal manifest must bind memory_bank, faiss_index and backbone"
            )

    source_evidence = manifest.get("formal_source_provenance")
    if not isinstance(source_evidence, Mapping):
        raise ValueError("formal checkpoint manifest is missing formal source provenance")
    from .formal_source_provenance import (  # local import avoids a module cycle
        validate_formal_source_provenance,
    )

    config_path = Path(project_config_path).resolve()
    source_root = (
        config_path.parent.parent
        if config_path.parent.name == "configs"
        else PROJECT_ROOT
    )
    source_validation = validate_formal_source_provenance(
        source_evidence,
        project_root=source_root,
    )
    repositories = source_validation.get("repositories")
    if not isinstance(repositories, list):
        raise ValueError("formal source provenance has no repository inventory")
    by_name = {
        str(entry.get("name")): entry
        for entry in repositories
        if isinstance(entry, Mapping)
    }
    for vendor, (expected_path, expected_commit) in EXPECTED_FORMAL_VENDOR_SOURCES.items():
        entry = by_name.get(vendor)
        if (
            not isinstance(entry, Mapping)
            or entry.get("path") != expected_path
            or entry.get("head_commit") != expected_commit
            or entry.get("expected_commit") != expected_commit
            or entry.get("worktree_clean") is not True
        ):
            raise ValueError(
                f"formal source provenance lacks the frozen clean {vendor} vendor"
            )

    return {
        "manifest_path": str(resolved),
        "manifest_sha256": sha256_file(resolved),
        "manifest_schema_version": manifest.get("schema_version"),
        "checkpoint_origin": "project_training_run",
        "checkpoints": checked,
        "external_hashes": supplied_external,
        "formal_training_inputs": (
            None
            if not isinstance(formal_training_inputs, Mapping)
            else dict(formal_training_inputs)
        ),
        "training_input_validation": authority.get("training_input_validation"),
        "training_authority": authority,
        "formal_source_validation": source_validation,
        "validated": True,
    }


def validate_embedded_formal_result(
    provenance: Mapping[str, object],
    *,
    method: str,
    dataset: str,
    category: str,
    seed: int,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    registration_path: str | Path = DEFAULT_REGISTRATION,
) -> None:
    """Recheck a formal inference record before it enters aggregation."""

    if provenance.get("claim_scope") != "formal_matrix_candidate":
        raise ValueError("result claim_scope is not formal_matrix_candidate")
    inference_recipe = provenance.get("inference_recipe")
    if not isinstance(inference_recipe, Mapping):
        raise ValueError("formal result is missing its inference recipe")
    current = validate_formal_authority(
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        phase="inference",
        recipe=inference_recipe,
        project_config_path=project_config_path,
        registration_path=registration_path,
    )
    embedded_authority = provenance.get("inference_authority")
    if not isinstance(embedded_authority, Mapping):
        raise ValueError("formal result is missing inference authority")
    for key in (
        "registration_id",
        "registration_sha256",
        "project_config_sha256",
        "recipe_id",
        "recipe_sha256",
    ):
        if embedded_authority.get(key) != current.get(key):
            raise ValueError(f"formal result authority mismatch: {key}")

    checkpoint_validation = provenance.get("checkpoint_validation")
    if not isinstance(checkpoint_validation, Mapping):
        raise ValueError("formal result is missing checkpoint validation")
    manifest_path = checkpoint_validation.get("manifest_path")
    manifest_sha = checkpoint_validation.get("manifest_sha256")
    if not isinstance(manifest_path, str) or not isinstance(manifest_sha, str):
        raise ValueError("formal result has invalid checkpoint manifest provenance")
    path = Path(manifest_path)
    if not path.is_file() or sha256_file(path) != manifest_sha:
        raise ValueError("formal result checkpoint manifest is missing or changed")
    if checkpoint_validation.get("checkpoint_origin") != "project_training_run":
        raise ValueError("formal result checkpoint origin is not project_training_run")
    checkpoint_entries = checkpoint_validation.get("checkpoints")
    if not isinstance(checkpoint_entries, Mapping) or not checkpoint_entries:
        raise ValueError("formal result has no validated checkpoint files")
    checkpoint_files: dict[str, str] = {}
    for role, entry in checkpoint_entries.items():
        if not isinstance(role, str) or not isinstance(entry, Mapping):
            raise ValueError("formal result checkpoint entry is invalid")
        checkpoint_path = entry.get("path")
        if not isinstance(checkpoint_path, str):
            raise ValueError("formal result checkpoint path is invalid")
        checkpoint_files[role] = checkpoint_path
    external_hashes = checkpoint_validation.get("external_hashes", {})
    if not isinstance(external_hashes, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in external_hashes.items()
    ):
        raise ValueError("formal result external checkpoint hashes are invalid")
    refreshed = validate_formal_checkpoint_manifest(
        path,
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        checkpoint_files=checkpoint_files,
        external_hashes=external_hashes,  # type: ignore[arg-type]
        project_config_path=project_config_path,
        registration_path=registration_path,
    )
    if refreshed["manifest_sha256"] != manifest_sha:
        raise ValueError("formal result checkpoint validation no longer matches")
