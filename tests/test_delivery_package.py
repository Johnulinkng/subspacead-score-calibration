from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
import warnings
import zipfile
from pathlib import Path

import pytest

from industrial_anomaly_lab.delivery_package import (
    build_delivery_archive,
    validate_delivery_spec,
    verify_delivery_archive,
)
from industrial_anomaly_lab.metrics import EVALUATION_PROTOCOL, METRICS_SCHEMA_VERSION
from industrial_anomaly_lab.result_aggregation import ResultRow, macro_by_seed, method_summary
from industrial_anomaly_lab.run_artifacts import sha256_file


_METHODS = ["DRAEM", "Candidate", "AblationNoX", "PatchCore", "RD4AD"]
_DATASETS = {"mvtec_ad": ["bottle"], "visa": ["candle"]}
_SEED = 7
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _init_repo(root: Path, files: dict[str, str] | None = None) -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Delivery Fixture")
    for relative, content in (files or {"README.md": "fixture\n"}).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD")


def _synthetic_source_evidence(root: Path) -> dict[str, object]:
    repositories: list[dict[str, object]] = []
    for name, relative in (
        ("project", "."),
        ("DRAEM", "third_party/DRAEM"),
        ("PatchCore", "third_party/PatchCore"),
        ("RD4AD", "third_party/RD4AD"),
    ):
        repository = root if relative == "." else root / relative
        head = _git(repository, "rev-parse", "HEAD")
        repositories.append(
            {
                "name": name,
                "path": relative,
                "head_commit": head,
                "head_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
                "expected_commit": None if relative == "." else head,
                "worktree_clean": True,
                "untracked_files_checked": True,
                "submodule_changes_checked": True,
            }
        )
    authority_paths = [
        "configs/source_scope.json",
        "configs/project.json",
        "configs/registration.json",
        "configs/delivery_spec.json",
    ]
    scope_path = root / "configs/source_scope.json"
    return {
        "schema_version": 1,
        "contract": "clean_git_source_provenance_v1",
        "scope": "configs/source_scope.json",
        "scope_sha256": sha256_file(scope_path),
        "repositories": repositories,
        "authority_files": [
            {
                "path": relative,
                "size_bytes": (root / relative).stat().st_size,
                "sha256": sha256_file(root / relative),
            }
            for relative in authority_paths
        ],
        "all_registered_repositories_clean": True,
        "project_head_binds_all_tracked_source_files": True,
        "untracked_source_files_forbidden": True,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _canonical_payload_sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _formal_runs(root: Path) -> tuple[list[ResultRow], dict[tuple[str, str], Path]]:
    rows: list[ResultRow] = []
    paths: dict[tuple[str, str], Path] = {}
    for method_index, method in enumerate(_METHODS):
        for dataset, categories in _DATASETS.items():
            category = categories[0]
            run = root / "artifacts" / "formal_runs" / method / dataset / category / f"seed_{_SEED}"
            inventory_dir = root / "manifests" / "formal_evaluation" / f"{dataset}_v1"
            inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
            receipt = json.loads((inventory_dir / "receipt.json").read_text(encoding="utf-8"))
            category_counts = receipt["category_counts"][category]
            inventory_evidence = {
                "schema_version": 1,
                "contract": "formal_test_inventory_category_selection_v1",
                "bundle": inventory_dir.relative_to(root).as_posix(),
                "inventory_id": inventory["inventory_id"],
                "dataset": dataset,
                "split": "test",
                "category": category,
                "inventory_sha256": sha256_file(inventory_dir / "inventory.json"),
                "receipt_sha256": sha256_file(inventory_dir / "receipt.json"),
                "entry_count": len(inventory["entries"]),
                "category_entry_count": category_counts["normal"]
                + category_counts["anomaly"],
                "dataset_root": receipt["dataset_root"],
            }
            (run / "predictions").mkdir(parents=True, exist_ok=True)
            (run / "predictions" / "index.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "index": index,
                            "sample_id": entry["sample_id"],
                            "label": entry["label"],
                            "defect_type": entry["defect_type"],
                            "input_image": entry["source_image"],
                            "source_ground_truth": entry["source_mask"],
                            "source_image_sha256": entry["source_image_sha256"],
                            "source_ground_truth_sha256": entry["source_mask_sha256"],
                            "formal_test_inventory_id": inventory_evidence["inventory_id"],
                            "formal_test_inventory_sha256": inventory_evidence["inventory_sha256"],
                            "formal_test_inventory_receipt_sha256": inventory_evidence["receipt_sha256"],
                            "raw_map_max_score": float(entry["label"]),
                            "method_native_image_score": float(entry["label"]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for index, entry in enumerate(inventory["entries"])
                    if entry["category"] == category
                ),
                encoding="utf-8",
            )
            config = {
                "schema_version": 2,
                "method": method,
                "execution_scope": "full",
                "run_purpose": "formal_full",
                "claim_scope": "formal_matrix_candidate",
                "formal_matrix_eligible": True,
                "formal_provenance": {
                    "fixture": True,
                    "formal_test_inventory": inventory_evidence,
                },
                "formal_test_inventory": inventory_evidence,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "settings": {"dataset": dataset, "category": category, "seed": _SEED},
            }
            base = 0.70 + method_index * 0.02
            metric_values = {
                "image_auroc": base,
                "pixel_auroc": base + 0.01,
                "pixel_ap": base - 0.01,
                "aupro_0_30": base + 0.015,
            }
            metrics = {
                "schema_version": METRICS_SCHEMA_VERSION,
                "dataset": dataset,
                "category": category,
                "num_images": inventory_evidence["category_entry_count"],
                "protocol": EVALUATION_PROTOCOL,
                "metrics": metric_values,
                "formal_test_inventory": inventory_evidence,
            }
            _write_json(run / "run_config.json", config)
            _write_json(run / "metrics.json", metrics)
            state = {
                "status": "completed",
                "execution_scope": "full",
                "run_purpose": "formal_full",
                "claim_scope": "formal_matrix_candidate",
                "formal_matrix_eligible": True,
                "num_images": inventory_evidence["category_entry_count"],
                "formal_test_inventory": inventory_evidence,
                "run_config_sha256": sha256_file(run / "run_config.json"),
                "metrics_sha256": sha256_file(run / "metrics.json"),
                "predictions_index_sha256": sha256_file(run / "predictions" / "index.jsonl"),
            }
            _write_json(run / "run_state.json", state)
            rows.append(
                ResultRow(
                    method=method,
                    dataset=dataset,
                    category=category,
                    seed=_SEED,
                    run_id=run.resolve().as_posix(),
                    **metric_values,
                )
            )
            paths[(method, dataset)] = run
    return rows, paths


def _visualization(
    root: Path,
    dataset: str,
    category: str,
    run_paths: dict[tuple[str, str], Path],
) -> dict[str, str]:
    visual_root = root / "artifacts" / "formal_visualizations" / dataset
    figure_records: list[dict[str, object]] = []
    for name, cohort in (
        ("fixed.png", "result_independent_preregistered"),
        ("failure.png", "outcome_dependent_machine_selected"),
    ):
        path = visual_root / "figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_PNG)
        sample_id = (
            f"{category}/good/000"
            if cohort == "result_independent_preregistered"
            else f"{category}/crack/001"
        )
        figure_records.append(
            {
                "cohort": cohort,
                "sample_id": sample_id,
                "failure_ranking": (
                    None
                    if cohort == "result_independent_preregistered"
                    else "candidate_absolute_failures"
                ),
                "figure": f"figures/{name}",
                "figure_sha256": sha256_file(path),
                "methods": [{"method": "DRAEM"}, {"method": "Candidate"}],
                "display": {"affects_metric_data": False},
            }
        )
    prediction_runs: list[dict[str, object]] = []
    for method in ("DRAEM", "Candidate"):
        run = run_paths[(method, dataset)]
        prediction_runs.append(
            {
                "method": method,
                "dataset": dataset,
                "category": category,
                "seed": _SEED,
                "run_config_sha256": sha256_file(run / "run_config.json"),
                "metrics_sha256": sha256_file(run / "metrics.json"),
                "prediction_index_sha256": sha256_file(run / "predictions" / "index.jsonl"),
                "run_directory": run.resolve().as_posix(),
            }
        )
    inventory_dir = root / "manifests" / "formal_evaluation" / f"{dataset}_v1"
    inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
    registration_path = root / "configs" / "registration.json"
    selection_config_path = root / "configs" / f"formal_visualization_{dataset}.json"
    raw_selection_config = json.loads(selection_config_path.read_text(encoding="utf-8"))
    raw_failure_rule = raw_selection_config["failure_case_rule"]
    selection_config = {
        **raw_selection_config,
        "failure_case_rule": {
            "status": "frozen_unresolved",
            **raw_failure_rule,
            "manual_selection_allowed": False,
        },
    }
    fixed_inventory_entry = next(
        entry for entry in inventory["entries"] if entry["label"] == 0
    )
    failure_inventory_entry = next(
        entry for entry in inventory["entries"] if entry["label"] == 1
    )
    fixed_sample = {
        **fixed_inventory_entry,
        "cohort": "result_independent_preregistered",
    }
    failure_pool_sample = {
        **failure_inventory_entry,
        "cohort": "frozen_failure_candidate_pool",
    }
    failure_sample = {
        **failure_inventory_entry,
        "cohort": "outcome_dependent_machine_selected",
        "failure_ranking": "candidate_absolute_failures",
    }
    selection: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_visualization_selection",
        "status": "frozen_before_formal_results",
        "freeze_phase": "F0_post_method_freeze_pre_inference",
        "dataset": dataset,
        "model_seed": _SEED,
        "test_inventory": {
            "sha256": sha256_file(inventory_dir / "inventory.json"),
            "inventory_id": inventory["inventory_id"],
            "entry_count": len(inventory["entries"]),
        },
        "method_registry": {
            "sha256": sha256_file(registration_path),
            "registration_id": "fixture-formal-v1",
            "methods": _METHODS,
            "baseline": "DRAEM",
            "candidate": "Candidate",
        },
        "selection_config": selection_config,
        "selection_config_sha256": sha256_file(selection_config_path),
        "result_independent_samples": [fixed_sample],
        "failure_candidate_pool": [failure_pool_sample],
        "failure_case_rule": {"fixture": True},
        "prediction_artifacts_accessed": False,
        "manual_post_result_selection_allowed": False,
    }
    selection["payload_sha256"] = _canonical_payload_sha(selection)
    selection_path = inventory_dir / "visualization_selection.json"
    _write_json(selection_path, selection)
    failure: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_visualization_failure_resolution",
        "status": "resolved_by_frozen_rule",
        "cohort": "outcome_dependent_machine_selected",
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_payload_sha256": selection["payload_sha256"],
        "dataset": dataset,
        "model_seed": _SEED,
        "rule": selection["failure_case_rule"],
        "prediction_runs": prediction_runs,
        "candidate_audit": [failure_sample],
        "rankings": [{"fixture": True}],
        "selected_cases": [failure_sample],
        "manual_selection_performed": False,
    }
    failure["payload_sha256"] = _canonical_payload_sha(failure)
    failure_path = inventory_dir / "visualization_failures.json"
    _write_json(failure_path, failure)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_visualization_figure_manifest",
        "status": "rendered_from_sha_verified_metric_arrays",
        "dataset": dataset,
        "model_seed": _SEED,
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_payload_sha256": selection["payload_sha256"],
        "failure_resolution_sha256": sha256_file(failure_path),
        "method_order": ["DRAEM", "Candidate"],
        "prediction_runs": prediction_runs,
        "figures": figure_records,
        "metric_arrays_modified": False,
        "raw_maps_rewritten": False,
        "manual_sample_selection_performed": False,
    }
    manifest["payload_sha256"] = _canonical_payload_sha(manifest)
    manifest_path = visual_root / "figure_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "dataset": dataset,
        "selection_config": selection_config_path.relative_to(root).as_posix(),
        "selection_manifest": selection_path.relative_to(root).as_posix(),
        "failure_resolution_manifest": failure_path.relative_to(root).as_posix(),
        "figure_manifest": manifest_path.relative_to(root).as_posix(),
    }


def _formal_inventory_bundle(root: Path, dataset: str, category: str) -> dict[str, str]:
    bundle = root / "manifests" / "formal_evaluation" / f"{dataset}_v1"
    entries = [
        {
            "dataset": dataset,
            "category": category,
            "sample_id": f"{category}/good/000",
            "label": 0,
            "defect_type": "good",
            "anomaly_area_fraction": 0.0,
            "anomaly_scale": "normal",
            "source_image": f"{category}/test/good/000.png",
            "source_image_sha256": "1" * 64,
            "source_mask": None,
            "source_mask_sha256": None,
        },
        {
            "dataset": dataset,
            "category": category,
            "sample_id": f"{category}/crack/001",
            "label": 1,
            "defect_type": "crack",
            "anomaly_area_fraction": 0.01,
            "anomaly_scale": "small",
            "source_image": f"{category}/test/crack/001.png",
            "source_image_sha256": "2" * 64,
            "source_mask": f"{category}/ground_truth/crack/001_mask.png",
            "source_mask_sha256": "3" * 64,
        },
    ]
    inventory = {
        "schema_version": 1,
        "kind": "formal_test_inventory",
        "inventory_id": f"{dataset}-fixture",
        "dataset": dataset,
        "split": "test",
        "entries": entries,
    }
    inventory_path = bundle / "inventory.json"
    _write_json(inventory_path, inventory)
    prepared = f"manifests/{dataset}_prepared.json"
    receipt = {
        "schema_version": 1,
        "kind": "formal_test_inventory_receipt",
        "contract": "formal_test_inventory_v1",
        "dataset": dataset,
        "split": "test",
        "inventory_id": inventory["inventory_id"],
        "inventory": "inventory.json",
        "inventory_sha256": sha256_file(inventory_path),
        "dataset_root": f"data/{dataset}",
        "project_config": "configs/project.json",
        "project_config_sha256": sha256_file(root / "configs/project.json"),
        "registration": "configs/registration.json",
        "registration_sha256": sha256_file(root / "configs/registration.json"),
        "protocol": "configs/inventory_protocol.json",
        "protocol_sha256": sha256_file(root / "configs/inventory_protocol.json"),
        "prepared_manifest": prepared,
        "prepared_manifest_sha256": sha256_file(root / prepared),
        "entry_count": len(entries),
        "category_counts": {category: {"normal": 1, "anomaly": 1, "masks": 1}},
        "formal_test_content_accessed": True,
        "ground_truth_masks_decoded_for_preregistered_scale_strata": True,
        "model_predictions_accessed": False,
        "result_driven_selection_performed": False,
    }
    receipt_path = bundle / "receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "dataset": dataset,
        "inventory": inventory_path.relative_to(root).as_posix(),
        "receipt": receipt_path.relative_to(root).as_posix(),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "project"
    root.mkdir(parents=True)
    third_commits: dict[str, str] = {}
    for name in ("DRAEM", "PatchCore", "RD4AD"):
        third_commits[name] = _init_repo(root / "third_party" / name)

    project = {
        "schema_version": 1,
        "baseline": {"formal_training_recipe_frozen": True},
        "evaluation": {"formal_protocol_frozen": True},
        "datasets": {
            dataset: {"categories": categories} for dataset, categories in _DATASETS.items()
        },
    }
    registration = {
        "schema_version": 1,
        "registration_id": "fixture-formal-v1",
        "model_seeds": [_SEED],
        "formal_method_registry": {
            "status": "frozen_before_formal_evaluation",
            "methods": _METHODS,
            "baseline": "DRAEM",
            "candidate": "Candidate",
        },
    }
    _write_json(root / "configs" / "project.json", project)
    _write_json(root / "configs" / "registration.json", registration)
    _write_json(
        root / "configs" / "inventory_protocol.json",
        {"schema_version": 1, "kind": "formal_test_inventory_protocol"},
    )
    _write_json(
        root / "configs" / "source_scope.json",
        {
            "schema_version": 1,
            "kind": "formal_source_scope",
            "status": "required_clean_at_formal_launch",
        },
    )
    (root / "docs").mkdir()
    (root / "docs" / "reproduction.md").write_text("fixture reproduction\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "methods.py").write_text("METHODS = True\n", encoding="utf-8")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    source_manifests: dict[str, str] = {}
    for name, commit in third_commits.items():
        relative = f"manifests/{name.lower()}_source.json"
        _write_json(root / relative, {"name": name, "commit": commit})
        source_manifests[name] = relative
    for dataset in _DATASETS:
        _write_json(
            root / "manifests" / f"{dataset}_prepared.json",
            {"schema_version": 1, "dataset": dataset},
        )

    formal_inventories = [
        _formal_inventory_bundle(root, dataset, categories[0])
        for dataset, categories in _DATASETS.items()
    ]
    for dataset in _DATASETS:
        _write_json(
            root / "configs" / f"formal_visualization_{dataset}.json",
            {
                "schema_version": 1,
                "kind": "formal_visualization_selection_config",
                "dataset": dataset,
                "model_seed": _SEED,
                "selection_salt": "delivery-fixture-salt",
                "fixed_selection": {
                    "mode": "explicit_sample_ids",
                    "sample_ids": [
                        f"{_DATASETS[dataset][0]}/good/000",
                    ],
                },
                "failure_case_rule": {
                    "baseline_method": "DRAEM",
                    "candidate_method": "Candidate",
                    "candidate_filter": "anomalies_with_positive_evaluation_mask",
                    "rankings": [
                        {
                            "name": "candidate_absolute_failures",
                            "metric": "candidate_pixel_average_precision",
                            "order": "ascending",
                            "per_category": 1,
                            "tie_break": "sample_id_ascending",
                        }
                    ],
                },
            },
        )
    planned_visualizations = [
        {
            "dataset": dataset,
            "selection_config": f"configs/formal_visualization_{dataset}.json",
            "selection_manifest": (
                f"manifests/formal_evaluation/{dataset}_v1/visualization_selection.json"
            ),
            "failure_resolution_manifest": (
                f"manifests/formal_evaluation/{dataset}_v1/visualization_failures.json"
            ),
            "figure_manifest": (
                f"artifacts/formal_visualizations/{dataset}/figure_manifest.json"
            ),
        }
        for dataset in _DATASETS
    ]

    spec: dict[str, object] = {
        "schema_version": 1,
        "kind": "industrial_anomaly_final_delivery",
        "package_name": "fixture-final",
        "project_config": "configs/project.json",
        "registration": "configs/registration.json",
        "formal_test_inventories": formal_inventories,
        "formal_results_root": "artifacts/formal_runs",
        "expected": {
            "datasets": ["mvtec_ad", "visa"],
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "claim_decision": "pass",
        },
        "experiment_package": {
            "files": [
                "README.md",
                "configs/project.json",
                "configs/registration.json",
                "configs/inventory_protocol.json",
                "configs/source_scope.json",
                "manifests/mvtec_ad_prepared.json",
                "manifests/visa_prepared.json",
                *source_manifests.values(),
            ],
            "method_roles": {
                "baseline": "DRAEM",
                "candidate": "Candidate",
                "ablations": ["AblationNoX"],
                "strong_controls": ["PatchCore", "RD4AD"],
            },
            "method_implementation_files": {
                method: ["src/methods.py"] for method in _METHODS
            },
        },
        "comparison_tables": {
            "aggregation_manifest": "artifacts/final_results/aggregation_manifest.json",
            "per_category_seed": "artifacts/final_results/per_category_seed.csv",
            "macro_by_seed": "artifacts/final_results/macro_by_seed.csv",
            "method_summary": "artifacts/final_results/method_summary.csv",
        },
        "visualizations": planned_visualizations,
        "reproduction_document": "docs/reproduction.md",
        "third_party_sources": [
            {
                "name": name,
                "path": f"third_party/{name}",
                "commit": third_commits[name],
                "source_manifest": source_manifests[name],
            }
            for name in ("DRAEM", "PatchCore", "RD4AD")
        ],
    }
    spec_path = root / "configs" / "delivery_spec.json"
    _write_json(spec_path, spec)

    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Delivery Fixture")
    tracked = [
        "README.md",
        "configs/project.json",
        "configs/registration.json",
        "configs/inventory_protocol.json",
        "configs/source_scope.json",
        "configs/delivery_spec.json",
        *[f"configs/formal_visualization_{dataset}.json" for dataset in _DATASETS],
        "docs/reproduction.md",
        "src/methods.py",
        "manifests/mvtec_ad_prepared.json",
        "manifests/visa_prepared.json",
        *[
            item[key]
            for item in formal_inventories
            for key in ("inventory", "receipt")
        ],
        *source_manifests.values(),
    ]
    _git(root, "add", "--", *tracked)
    _git(root, "commit", "--quiet", "-m", "freeze fixture sources")

    rows, run_paths = _formal_runs(root)
    result_dir = root / "artifacts" / "final_results"
    _write_csv(result_dir / "per_category_seed.csv", [row.to_dict() for row in rows])
    _write_csv(result_dir / "macro_by_seed.csv", macro_by_seed(rows))
    _write_csv(result_dir / "method_summary.csv", method_summary(rows))
    aggregation = {
        "schema_version": 1,
        "runs_root": (root / "artifacts" / "formal_runs").resolve().as_posix(),
        "methods": _METHODS,
        "seeds": [_SEED],
        "dataset_categories": _DATASETS,
        "registration": {
            "path": "configs/registration.json",
            "sha256": sha256_file(root / "configs" / "registration.json"),
            "registration_id": "fixture-formal-v1",
        },
        "matrix_validation": "passed",
        "row_count": len(rows),
        "comparisons": [{"fixture": True}],
        "claim_adjudication": {
            "schema_version": 1,
            "decision": "pass",
            "gates": {
                "primary_positive_each_dataset": True,
                "equal_dataset_average_gain": True,
                "nonnegative_category_fraction_each_dataset": True,
                "bootstrap_ci_rule": True,
                "secondary_metric_regression_limit": True,
            },
        },
    }
    _write_json(result_dir / "aggregation_manifest.json", aggregation)

    for dataset, categories in _DATASETS.items():
        assert _visualization(root, dataset, categories[0], run_paths) == next(
            item for item in planned_visualizations if item["dataset"] == dataset
        )
    _git(
        root,
        "add",
        "--",
        *[item["selection_manifest"] for item in planned_visualizations],
    )
    _git(root, "commit", "--quiet", "-m", "freeze F0 visualization selections")
    source_evidence = _synthetic_source_evidence(root)
    for run in run_paths.values():
        config_path = run / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["formal_provenance"]["checkpoint_validation"] = {
            "formal_source_validation": source_evidence,
        }
        _write_json(config_path, config)
        state_path = run / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["run_config_sha256"] = sha256_file(config_path)
        _write_json(state_path, state)
    for dataset, categories in _DATASETS.items():
        _visualization(root, dataset, categories[0], run_paths)
    return {"root": root, "spec": spec_path}


def _rewrite_state_hash(run: Path) -> None:
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    state["metrics_sha256"] = sha256_file(run / "metrics.json")
    _write_json(run / "run_state.json", state)


def test_synthetic_complete_delivery_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    first = root / "delivery" / "first.zip"
    second = root / "delivery" / "second.zip"
    report_one = build_delivery_archive(root, fixture["spec"], first)
    report_two = build_delivery_archive(root, fixture["spec"], second)
    assert first.read_bytes() == second.read_bytes()
    assert report_one["archive_sha256"] == report_two["archive_sha256"]
    assert report_one["status"] == "verified"
    assert verify_delivery_archive(first)["payload_tree_sha256"] == report_one["payload_tree_sha256"]
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names[-1] == "DELIVERY-MANIFEST.json"
        assert not any(name.startswith(("data/", "runs/", ".cache/")) for name in names)
        manifest = json.loads(archive.read("DELIVERY-MANIFEST.json"))
        assert manifest["deliverables"]["experiment_package"]["present"] is True
        assert manifest["deliverables"]["comparison_tables"]["present"] is True
        assert manifest["deliverables"]["visualizations"]["present"] is True
        assert manifest["deliverables"]["reproduction_document"]["present"] is True


def test_validate_only_has_no_archive_side_effect(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = validate_delivery_spec(fixture["root"], fixture["spec"])
    assert report["delivery_status"] == "complete_formal_claim_passed"
    assert not list(fixture["root"].rglob("*.zip"))


@pytest.mark.parametrize(
    "failure",
    [
        "missing_table",
        "missing_png",
        "missing_inventory",
        "missing_reproduction",
        "missing_experiment_source",
        "incomplete_run",
    ],
)
def test_missing_or_incomplete_deliverables_fail_closed(tmp_path: Path, failure: str) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    if failure == "missing_table":
        (root / "artifacts/final_results/method_summary.csv").unlink()
    elif failure == "missing_png":
        (root / "artifacts/formal_visualizations/visa/figures/failure.png").unlink()
    elif failure == "missing_inventory":
        (root / "manifests/formal_evaluation/visa_v1/inventory.json").unlink()
    elif failure == "missing_reproduction":
        (root / "docs/reproduction.md").unlink()
    elif failure == "missing_experiment_source":
        (root / "src/methods.py").unlink()
    else:
        run = root / f"artifacts/formal_runs/DRAEM/visa/candle/seed_{_SEED}"
        state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
        state["status"] = "failed"
        _write_json(run / "run_state.json", state)
    output = root / "delivery" / f"{failure}.zip"
    with pytest.raises((FileNotFoundError, ValueError)):
        build_delivery_archive(root, fixture["spec"], output)
    assert not output.exists()


def test_partial_prediction_coverage_and_source_identity_drift_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    run = root / f"artifacts/formal_runs/DRAEM/visa/candle/seed_{_SEED}"
    index_path = run / "predictions/index.jsonl"
    index_path.write_text(index_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["num_images"] = 1
    _write_json(run / "metrics.json", metrics)
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    state["num_images"] = 1
    state["metrics_sha256"] = sha256_file(run / "metrics.json")
    state["predictions_index_sha256"] = sha256_file(index_path)
    _write_json(run / "run_state.json", state)
    with pytest.raises(ValueError, match="complete inventory category"):
        validate_delivery_spec(root, fixture["spec"])

    fixture = _fixture(tmp_path / "source-drift")
    root = fixture["root"]
    run = root / f"artifacts/formal_runs/DRAEM/visa/candle/seed_{_SEED}"
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    repositories = config["formal_provenance"]["checkpoint_validation"][
        "formal_source_validation"
    ]["repositories"]
    next(item for item in repositories if item["name"] == "project")["head_commit"] = "0" * 40
    _write_json(run / "run_config.json", config)
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    state["run_config_sha256"] = sha256_file(run / "run_config.json")
    _write_json(run / "run_state.json", state)
    with pytest.raises(ValueError, match="source repository identity drifted"):
        validate_delivery_spec(root, fixture["spec"])


def test_f0_visualization_sample_outside_inventory_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    selection_path = root / "manifests/formal_evaluation/visa_v1/visualization_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["result_independent_samples"][0]["sample_id"] = "candle/good/not-inventory"
    selection.pop("payload_sha256")
    selection["payload_sha256"] = _canonical_payload_sha(selection)
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="outside the F0 inventory"):
        validate_delivery_spec(root, fixture["spec"])


def test_generalized_dataset_cache_and_weight_paths_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    for forbidden in (
        "weights/model.safetensors",
        "cache/item.bin",
        "datasets/sample.txt",
        "models/model.h5",
    ):
        spec = json.loads(json.dumps(original))
        spec["experiment_package"]["files"].append(forbidden)
        _write_json(fixture["spec"], spec)
        with pytest.raises(ValueError, match="forbidden"):
            validate_delivery_spec(fixture["root"], fixture["spec"])


def test_legacy_metrics_schema_is_rejected_even_when_state_hash_is_updated(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    run = root / f"artifacts/formal_runs/DRAEM/mvtec_ad/bottle/seed_{_SEED}"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["schema_version"] = 1
    _write_json(run / "metrics.json", metrics)
    _rewrite_state_hash(run)
    with pytest.raises(ValueError, match="legacy metrics schema"):
        validate_delivery_spec(root, fixture["spec"])


def test_tampered_table_and_path_escape_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    table = root / "artifacts/final_results/per_category_seed.csv"
    rows = _read_csv_for_test(table)
    rows[0]["aupro_0_30"] = "0.0"
    _write_csv(table, rows)
    with pytest.raises(ValueError, match="disagrees with formal runs"):
        validate_delivery_spec(root, fixture["spec"])

    fixture = _fixture(tmp_path / "escape")
    spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    spec["experiment_package"]["files"].append("../outside.txt")
    _write_json(fixture["spec"], spec)
    with pytest.raises(ValueError, match="escape"):
        validate_delivery_spec(fixture["root"], fixture["spec"])


def test_absolute_and_linked_allowlist_paths_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    spec["experiment_package"]["files"].append(
        (fixture["root"] / "README.md").resolve().as_posix()
    )
    _write_json(fixture["spec"], spec)
    with pytest.raises(ValueError, match="absolute or escape"):
        validate_delivery_spec(fixture["root"], fixture["spec"])

    fixture = _fixture(tmp_path / "linked")
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = True\n", encoding="utf-8")
    link = fixture["root"] / "src" / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable on this Windows host")
    spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    spec["experiment_package"]["files"].append("src/linked.py")
    _write_json(fixture["spec"], spec)
    with pytest.raises(ValueError, match="links/reparse points"):
        validate_delivery_spec(fixture["root"], fixture["spec"])


def _read_csv_for_test(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rewrite_zip(
    source: Path,
    destination: Path,
    *,
    member_order: list[str] | None = None,
    compression: int = zipfile.ZIP_DEFLATED,
    mutate_manifest=None,
) -> None:
    with zipfile.ZipFile(source, "r") as original:
        names = original.namelist()
        payloads = {name: original.read(name) for name in names}
    if mutate_manifest is not None:
        manifest = json.loads(payloads["DELIVERY-MANIFEST.json"])
        mutate_manifest(manifest)
        payloads["DELIVERY-MANIFEST.json"] = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    with zipfile.ZipFile(destination, "w", compression=compression) as output:
        for name in member_order or names:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100644 << 16)
            info.compress_type = compression
            output.writestr(info, payloads[name], compress_type=compression)


def test_tampered_archive_duplicate_member_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    archive = fixture["root"] / "delivery" / "final.zip"
    build_delivery_archive(fixture["root"], fixture["spec"], archive)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as handle:
            handle.writestr("README.md", b"tampered\n")
    with pytest.raises(ValueError, match="duplicate"):
        verify_delivery_archive(archive)


def test_archive_verifier_requires_four_deliverables_order_and_deflate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    archive = fixture["root"] / "delivery" / "final.zip"
    build_delivery_archive(fixture["root"], fixture["spec"], archive)

    no_reproduction = fixture["root"] / "delivery" / "no-reproduction-record.zip"

    def remove_reproduction_classification(manifest: dict[str, object]) -> None:
        record = next(
            item
            for item in manifest["files"]
            if item["deliverable"] == "reproduction_document"
        )
        record["deliverable"] = "experiment_package"

    _rewrite_zip(archive, no_reproduction, mutate_manifest=remove_reproduction_classification)
    with pytest.raises(ValueError, match="missing one of the four deliverables"):
        verify_delivery_archive(no_reproduction)

    with zipfile.ZipFile(archive, "r") as handle:
        names = handle.namelist()
    out_of_order = fixture["root"] / "delivery" / "out-of-order.zip"
    _rewrite_zip(
        archive,
        out_of_order,
        member_order=list(reversed(names[:-1])) + ["DELIVERY-MANIFEST.json"],
    )
    with pytest.raises(ValueError, match="not deterministically ordered"):
        verify_delivery_archive(out_of_order)

    stored = fixture["root"] / "delivery" / "stored.zip"
    _rewrite_zip(archive, stored, compression=zipfile.ZIP_STORED)
    with pytest.raises(ValueError, match="not deflate-compressed"):
        verify_delivery_archive(stored)
