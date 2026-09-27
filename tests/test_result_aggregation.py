import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.formal_provenance import (
    validate_formal_authority,
    validate_formal_checkpoint_manifest,
)
from industrial_anomaly_lab.formal_test_inventory import (
    build_formal_test_inventory_bundle,
    resolve_formal_test_inventory_category,
)
from industrial_anomaly_lab.metrics import EVALUATION_PROTOCOL, image_score_contract
from industrial_anomaly_lab.result_aggregation import (
    ResultRow,
    adjudicate_registered_claim,
    load_completed_run,
    macro_by_seed,
    paired_comparison,
    validate_complete_matrix,
    validate_registered_invocation,
)
from industrial_anomaly_lab.run_artifacts import sha256_file
import industrial_anomaly_lab.formal_source_provenance as source_provenance_module


FORMAL_FIXTURE_METHOD = "SYNTHETIC_FORMAL"


@pytest.fixture(autouse=True)
def _synthetic_clean_source(monkeypatch: pytest.MonkeyPatch) -> None:
    commits = {
        "DRAEM": ("third_party/DRAEM", "2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef"),
        "PatchCore": ("third_party/PatchCore", "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"),
        "RD4AD": ("third_party/RD4AD", "6554076872c65f8784f6ece8cfb39ce77e1aee12"),
    }
    validation = {
        "schema_version": 1,
        "contract": "clean_git_source_provenance_v1",
        "repositories": [
            {
                "name": name,
                "path": path,
                "head_commit": commit,
                "expected_commit": commit,
                "worktree_clean": True,
            }
            for name, (path, commit) in commits.items()
        ],
    }
    monkeypatch.setattr(
        source_provenance_module,
        "validate_formal_source_provenance",
        lambda evidence, *, project_root: validation,
    )


def _row(method: str, category: str, seed: int, value: float) -> ResultRow:
    return ResultRow(method, "toy", category, seed, f"{method}-{category}-{seed}", value, value, value, value)


def test_complete_matrix_rejects_missing_and_duplicate_rows():
    rows = [_row("base", "a", 1, 0.5), _row("base", "a", 1, 0.5)]
    with pytest.raises(ValueError, match="duplicates=.*missing="):
        validate_complete_matrix(
            rows,
            methods=["base"],
            dataset_categories={"toy": ["a", "b"]},
            seeds=[1],
        )


def test_macro_and_paired_statistics_use_category_seed_pairs():
    rows = []
    for category, gain in (("a", 0.10), ("b", 0.20)):
        for seed in (1, 2):
            rows.append(_row("base", category, seed, 0.5))
            rows.append(_row("new", category, seed, 0.5 + gain))
    validate_complete_matrix(
        rows,
        methods=["base", "new"],
        dataset_categories={"toy": ["a", "b"]},
        seeds=[1, 2],
    )
    macros = macro_by_seed(rows)
    new_values = [row["aupro_0_30"] for row in macros if row["method"] == "new"]
    assert new_values == pytest.approx([0.65, 0.65])
    comparison = paired_comparison(
        rows,
        baseline="base",
        candidate="new",
        dataset="toy",
        metric="aupro_0_30",
        bootstrap_samples=1000,
        bootstrap_seed=7,
    )
    assert comparison["mean_difference"] == pytest.approx(0.15)
    assert comparison["improved_category_fraction"] == 1.0
    assert comparison["nonnegative_category_fraction"] == 1.0
    assert comparison["two_sided_exact_sign_flip_p"] == pytest.approx(0.5)


def test_paired_statistics_distinguish_improvement_from_no_regression():
    rows = [
        _row("base", "a", 1, 0.5),
        _row("new", "a", 1, 0.6),
        _row("base", "b", 1, 0.5),
        _row("new", "b", 1, 0.5),
    ]
    comparison = paired_comparison(
        rows,
        baseline="base",
        candidate="new",
        dataset="toy",
        metric="aupro_0_30",
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert comparison["improved_category_fraction"] == 0.5
    assert comparison["nonnegative_category_fraction"] == 1.0


def test_registered_claim_adjudication_preserves_each_gate():
    comparisons = []
    for dataset, primary_gain, lower in (("mvtec_ad", 0.012, 0.002), ("visa", 0.009, -0.001)):
        for metric in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30"):
            comparisons.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "mean_difference": primary_gain if metric == "aupro_0_30" else -0.004,
                    "nonnegative_category_fraction": 0.7,
                    "ci95_percentile": [lower, 0.02],
                }
            )
    result = adjudicate_registered_claim(
        comparisons, datasets=["mvtec_ad", "visa"]
    )
    assert result["decision"] == "pass"
    assert result["observed"]["equal_dataset_average_gain"] == pytest.approx(0.0105)
    assert all(result["gates"].values())

    failed = [dict(item) for item in comparisons]
    for item in failed:
        if item["dataset"] == "visa" and item["metric"] == "pixel_ap":
            item["mean_difference"] = -0.006
    failed_result = adjudicate_registered_claim(
        failed, datasets=["mvtec_ad", "visa"]
    )
    assert failed_result["decision"] == "fail"
    assert failed_result["gates"]["secondary_metric_regression_limit"] is False


def test_load_completed_run_rejects_unclassified_author_and_debug_runs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps({"method": "DRAEM", "execution_scope": "full", "settings": {"seed": 42}}),
        encoding="utf-8",
    )
    (run / "run_state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "protocol": "draem_256_unified_v1",
                "dataset": "visa",
                "category": "candle",
                "metrics": {name: 0.75 for name in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_purpose is not formal_full"):
        load_completed_run(run)

    payload = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "run_purpose": "author_reproduction",
            "claim_scope": "author_reproduction_only",
            "formal_matrix_eligible": False,
        }
    )
    (run / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="run_purpose is not formal_full"):
        load_completed_run(run)

    payload["execution_scope"] = "debug_limited"
    payload["run_purpose"] = "debug_limited"
    payload["claim_scope"] = "debug_only"
    (run / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="debug-limited"):
        load_completed_run(run)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(mask, mode="L").save(path)


def _synthetic_formal_run(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    seeds = [11, 22, 33]
    training_recipe = {
        "schema_version": 1,
        "method": FORMAL_FIXTURE_METHOD,
        "epochs": 700,
    }
    inference_recipe = {
        "schema_version": 2,
        "method": FORMAL_FIXTURE_METHOD,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "common_image_score": "max_of_persisted_raw_anomaly_map",
        "method_native_image_score": "synthetic_fixture_score",
    }
    project_root = tmp_path / "project"
    project_path = project_root / "configs" / "project.json"
    registration_path = project_root / "configs" / "registration.json"
    prepared_path = project_root / "manifests" / "visa_prepared.json"
    protocol_path = project_root / "configs" / "inventory_protocol.json"
    dataset_root = project_root / "data" / "visa"
    _write_rgb(dataset_root / "candle/test/good/000.JPG", 40)
    _write_rgb(dataset_root / "candle/test/bad/001.JPG", 200)
    _write_mask(dataset_root / "candle/ground_truth/bad/001.png")
    _write_json(
        prepared_path,
        {
            "schema_version": 1,
            "dataset": "VisA",
            "prepared": {"root": "data/visa"},
            "split": {
                "partition_counts": {
                    "candle": {
                        "train_normal": 1,
                        "test_normal": 1,
                        "test_anomaly": 1,
                    }
                }
            },
        },
    )
    _write_json(
        protocol_path,
        {
            "schema_version": 1,
            "kind": "formal_test_inventory_protocol",
            "status": "frozen_before_formal_test_enumeration",
            "datasets": ["mvtec_ad", "visa"],
            "mask_interpretation": {
                "positive_rule": "pixel_value_gt_0",
                "area_denominator": "all_native_mask_pixels",
            },
            "anomaly_scale_thresholds": {
                "small_max_fraction": 0.01,
                "medium_max_fraction": 0.1,
            },
        },
    )
    _write_json(
        project_path,
        {
            "baseline": {"formal_training_recipe_frozen": True},
            "evaluation": {
                "minimum_seeds": 3,
                "seeds": seeds,
                "formal_protocol_frozen": True,
            },
            "datasets": {
                "visa": {
                    "categories": ["candle"],
                    "prepared_manifest": "manifests/visa_prepared.json",
                }
            },
        },
    )
    candidate_method = "SYNTHETIC_CANDIDATE"
    candidate_training_recipe = {
        "schema_version": 1,
        "method": candidate_method,
        "epochs": 700,
    }
    candidate_inference_recipe = {
        "schema_version": 2,
        "method": candidate_method,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    _write_json(
        registration_path,
        {
            "registration_id": "synthetic-formal-v1",
            "model_seeds": seeds,
            "formal_method_registry": {
                "status": "frozen_before_formal_evaluation",
                "methods": [FORMAL_FIXTURE_METHOD, candidate_method],
                "baseline": FORMAL_FIXTURE_METHOD,
                "candidate": candidate_method,
                "recipes": {
                    FORMAL_FIXTURE_METHOD: {
                        "recipe_id": "synthetic-formal-v1",
                        "training": training_recipe,
                        "inference": inference_recipe,
                    },
                    candidate_method: {
                        "recipe_id": "synthetic-candidate-v1",
                        "training": candidate_training_recipe,
                        "inference": candidate_inference_recipe,
                    },
                },
            },
        },
    )
    inventory_dir = project_root / "manifests" / "formal" / "visa"
    build_formal_test_inventory_bundle(
        dataset="visa",
        dataset_root=dataset_root,
        output_dir=inventory_dir,
        project_root=project_root,
        project_config_path=project_path,
        registration_path=registration_path,
        protocol_path=protocol_path,
        prepared_manifest_path=prepared_path,
    )
    inventory_selection = resolve_formal_test_inventory_category(
        output_dir=inventory_dir,
        dataset="visa",
        category="candle",
        dataset_root=dataset_root,
        project_root=project_root,
    )
    inventory_evidence = inventory_selection["evidence"]
    run = project_root / "formal-run"
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    reconstructive = checkpoint_dir / "reconstructive.pt"
    discriminative = checkpoint_dir / "discriminative.pt"
    reconstructive.write_bytes(b"synthetic-reconstructive")
    discriminative.write_bytes(b"synthetic-discriminative")
    training_authority = validate_formal_authority(
        method=FORMAL_FIXTURE_METHOD,
        dataset="visa",
        category="candle",
        seed=seeds[0],
        phase="training",
        recipe=training_recipe,
        project_config_path=project_path,
        registration_path=registration_path,
    )
    manifest_path = checkpoint_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 2,
            "method": FORMAL_FIXTURE_METHOD,
            "dataset": "visa",
            "category": "candle",
            "seed": seeds[0],
            "run_purpose": "formal_full",
            "execution_scope": "full",
            "claim_scope": "formal_matrix_candidate",
            "checkpoint_origin": "project_training_run",
            "final_checkpoint_only": True,
            "test_selected": False,
            "training_recipe": training_recipe,
            "formal_provenance": training_authority,
            "formal_source_provenance": {"synthetic_fixture": True},
            "checkpoints": {
                "reconstructive": {
                    "path": reconstructive.name,
                    "size_bytes": reconstructive.stat().st_size,
                    "sha256": sha256_file(reconstructive),
                },
                "discriminative": {
                    "path": discriminative.name,
                    "size_bytes": discriminative.stat().st_size,
                    "sha256": sha256_file(discriminative),
                },
            },
            "external_hashes": {},
        },
    )
    checkpoint_validation = validate_formal_checkpoint_manifest(
        manifest_path,
        method=FORMAL_FIXTURE_METHOD,
        dataset="visa",
        category="candle",
        seed=seeds[0],
        checkpoint_files={
            "reconstructive": reconstructive,
            "discriminative": discriminative,
        },
        project_config_path=project_path,
        registration_path=registration_path,
    )
    inference_authority = validate_formal_authority(
        method=FORMAL_FIXTURE_METHOD,
        dataset="visa",
        category="candle",
        seed=seeds[0],
        phase="inference",
        recipe=inference_recipe,
        project_config_path=project_path,
        registration_path=registration_path,
    )
    _write_json(
        run / "run_config.json",
        {
            "method": FORMAL_FIXTURE_METHOD,
            "execution_scope": "full",
            "run_purpose": "formal_full",
            "claim_scope": "formal_matrix_candidate",
            "formal_matrix_eligible": True,
            "prediction_contract": {
                "image_scores": image_score_contract("synthetic_fixture_score")
            },
            "settings": {
                "dataset": "visa",
                "category": "candle",
                "seed": seeds[0],
            },
            "formal_provenance": {
                "claim_scope": "formal_matrix_candidate",
                "inference_recipe": inference_recipe,
                "inference_authority": inference_authority,
                "checkpoint_validation": checkpoint_validation,
                "formal_test_inventory": inventory_evidence,
            },
            "formal_test_inventory": inventory_evidence,
        },
    )
    _write_json(
        run / "metrics.json",
        {
            "schema_version": 2,
            "protocol": EVALUATION_PROTOCOL,
            "dataset": "visa",
            "category": "candle",
            "num_images": 2,
            "image_score_contract": image_score_contract(
                "synthetic_fixture_score"
            ),
            "metrics": {
                "image_auroc": 1.0,
                "image_auroc_method_native": 1.0,
                "pixel_auroc": 1.0,
                "pixel_ap": 1.0,
                "aupro_0_30": 1.0,
            },
            "formal_test_inventory": inventory_evidence,
        },
    )
    predictions = run / "predictions"
    raw_maps = predictions / "raw_maps"
    evaluation_masks = predictions / "evaluation_masks"
    raw_maps.mkdir(parents=True)
    evaluation_masks.mkdir(parents=True)
    prediction_rows = []
    for index, entry in enumerate(inventory_selection["entries"]):
        value = float(entry["label"])
        raw_path = raw_maps / f"{index:06d}.npy"
        mask_path = evaluation_masks / f"{index:06d}.npy"
        raw_map = np.zeros((2, 2), dtype=np.float32)
        evaluation_mask = np.zeros((2, 2), dtype=bool)
        if entry["label"] == 1:
            raw_map[0, 0] = 1.0
            evaluation_mask[0, 0] = True
        np.save(raw_path, raw_map, allow_pickle=False)
        np.save(mask_path, evaluation_mask, allow_pickle=False)
        prediction_rows.append(
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
                "formal_test_inventory_sha256": inventory_evidence[
                    "inventory_sha256"
                ],
                "formal_test_inventory_receipt_sha256": inventory_evidence[
                    "receipt_sha256"
                ],
                "raw_map": raw_path.relative_to(run).as_posix(),
                "raw_map_sha256": sha256_file(raw_path),
                "evaluation_mask": mask_path.relative_to(run).as_posix(),
                "evaluation_mask_sha256": sha256_file(mask_path),
                "raw_map_max_score": value,
                "method_native_image_score": value,
            }
        )
    (predictions / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    _write_json(
        run / "run_state.json",
        {
            "status": "completed",
            "execution_scope": "full",
            "run_purpose": "formal_full",
            "claim_scope": "formal_matrix_candidate",
            "formal_matrix_eligible": True,
            "num_images": 2,
            "run_config_sha256": sha256_file(run / "run_config.json"),
            "metrics_sha256": sha256_file(run / "metrics.json"),
            "predictions_index_sha256": sha256_file(predictions / "index.jsonl"),
            "formal_test_inventory": inventory_evidence,
        },
    )
    return run, project_path, registration_path, discriminative


def test_load_completed_run_revalidates_formal_authority_and_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    run, project, registration, discriminative = _synthetic_formal_run(tmp_path)
    row = load_completed_run(
        run, project_config_path=project, registration_path=registration
    )
    assert (row.method, row.dataset, row.category, row.seed) == (
        FORMAL_FIXTURE_METHOD,
        "visa",
        "candle",
        11,
    )

    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run_purpose"] = "author_reproduction"
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="run_purpose is not formal_full"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )

    config["run_purpose"] = "formal_full"
    _write_json(config_path, config)

    metrics_path = run / "metrics.json"
    original_metrics = metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(original_metrics)
    metrics["metrics"]["aupro_0_30"] = 0.99
    _write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="metrics_sha256"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )
    metrics_path.write_text(original_metrics, encoding="utf-8")

    discriminative.write_bytes(b"tampered-after-inference")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )


@pytest.mark.parametrize("metric_name", ["pixel_auroc", "pixel_ap", "aupro_0_30"])
def test_load_completed_run_recomputes_pixel_metrics(
    tmp_path: Path, metric_name: str,
) -> None:
    run, project, registration, _ = _synthetic_formal_run(tmp_path)
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"][metric_name] = 0.5
    _write_json(metrics_path, metrics)
    state_path = run / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["metrics_sha256"] = sha256_file(metrics_path)
    _write_json(state_path, state)

    with pytest.raises(ValueError, match=rf"{metric_name} does not match persisted"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )


def test_load_completed_run_revalidates_evaluation_mask_hash(tmp_path: Path) -> None:
    mask_run, mask_project, mask_registration, _ = _synthetic_formal_run(tmp_path)
    mask_path = mask_run / "predictions/evaluation_masks/000000.npy"
    np.save(mask_path, np.ones((2, 2), dtype=bool), allow_pickle=False)
    with pytest.raises(ValueError, match="evaluation_mask hash mismatch"):
        load_completed_run(
            mask_run,
            project_config_path=mask_project,
            registration_path=mask_registration,
        )


def test_load_completed_run_rejects_boolean_metric_values(tmp_path: Path) -> None:
    run, project, registration, _ = _synthetic_formal_run(tmp_path)
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"]["aupro_0_30"] = True
    _write_json(metrics_path, metrics)
    state_path = run / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["metrics_sha256"] = sha256_file(metrics_path)
    _write_json(state_path, state)

    with pytest.raises(ValueError, match="missing or non-finite aupro_0_30"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )


def test_load_completed_run_revalidates_inventory_source_and_prediction_identity(
    tmp_path: Path,
) -> None:
    run, project, registration, _ = _synthetic_formal_run(tmp_path)
    source_image = project.parent.parent / "data/visa/candle/test/good/000.JPG"
    _write_rgb(source_image, 250)
    with pytest.raises(ValueError, match="differs from current complete test split"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )

    clean_run, clean_project, clean_registration, _ = _synthetic_formal_run(
        tmp_path / "clean"
    )
    index_path = clean_run / "predictions/index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["input_image"] = "candle/test/good/forged.JPG"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    state_path = clean_run / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predictions_index_sha256"] = sha256_file(index_path)
    _write_json(state_path, state)
    with pytest.raises(ValueError, match="differs from inventory field input_image"):
        load_completed_run(
            clean_run,
            project_config_path=clean_project,
            registration_path=clean_registration,
        )


def test_load_completed_run_rejects_legacy_generic_image_score(tmp_path: Path) -> None:
    run, project, registration, _ = _synthetic_formal_run(tmp_path)
    index_path = run / "predictions/index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    for row in rows:
        row["image_score"] = row.pop("raw_map_max_score")
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    state_path = run / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predictions_index_sha256"] = sha256_file(index_path)
    _write_json(state_path, state)

    with pytest.raises(ValueError, match="legacy generic image_score is forbidden"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )


def test_load_completed_run_recomputes_raw_map_max_score(tmp_path: Path) -> None:
    run, project, registration, _ = _synthetic_formal_run(tmp_path)
    index_path = run / "predictions/index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["raw_map_max_score"] = 0.25
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    state_path = run / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predictions_index_sha256"] = sha256_file(index_path)
    _write_json(state_path, state)

    with pytest.raises(ValueError, match="does not match persisted raw_map"):
        load_completed_run(
            run, project_config_path=project, registration_path=registration
        )


def test_load_completed_run_rejects_missing_execution_scope(tmp_path):
    run = tmp_path / "unscoped"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps({"method": "PatchCore", "settings": {"seed": 7}}),
        encoding="utf-8",
    )
    (run / "run_state.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "protocol": "draem_256_unified_v1",
                "dataset": "visa",
                "category": "candle",
                "metrics": {
                    name: 0.5
                    for name in (
                        "image_auroc",
                        "pixel_auroc",
                        "pixel_ap",
                        "aupro_0_30",
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing execution_scope"):
        load_completed_run(run)


def _registration() -> dict[str, object]:
    return {
        "model_seeds": [11, 22, 33],
        "formal_statistics": {
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260927,
        },
        "formal_method_registry": {
            "status": "frozen_before_formal_evaluation",
            "methods": ["DRAEM", "candidate_v1"],
            "baseline": "DRAEM",
            "candidate": "candidate_v1",
        },
    }


def test_registered_invocation_requires_exact_frozen_matrix():
    registration = _registration()
    validate_registered_invocation(
        registration,
        methods=["DRAEM", "candidate_v1"],
        seeds=[11, 22, 33],
        baseline="DRAEM",
        candidate="candidate_v1",
        bootstrap_samples=10000,
        bootstrap_seed=20260927,
    )

    with pytest.raises(ValueError, match="formal seeds differ"):
        validate_registered_invocation(
            registration,
            methods=["DRAEM", "candidate_v1"],
            seeds=[11],
            baseline="DRAEM",
            candidate="candidate_v1",
            bootstrap_samples=10000,
            bootstrap_seed=20260927,
        )

    boolean_seed_registration = _registration()
    boolean_seed_registration["model_seeds"] = [True, 22, 33]
    with pytest.raises(ValueError, match="registration is missing model_seeds"):
        validate_registered_invocation(
            boolean_seed_registration,
            methods=["DRAEM", "candidate_v1"],
            seeds=[1, 22, 33],
            baseline="DRAEM",
            candidate="candidate_v1",
            bootstrap_samples=10000,
            bootstrap_seed=20260927,
        )
    with pytest.raises(ValueError, match="bootstrap settings differ"):
        validate_registered_invocation(
            registration,
            methods=["DRAEM", "candidate_v1"],
            seeds=[11, 22, 33],
            baseline="DRAEM",
            candidate="candidate_v1",
            bootstrap_samples=9999,
            bootstrap_seed=20260927,
        )


def test_registered_invocation_refuses_unfrozen_methods():
    registration = _registration()
    registration["formal_method_registry"] = {
        "status": "pending_candidate_and_ablation_freeze",
        "methods": None,
        "baseline": "DRAEM",
        "candidate": None,
    }
    with pytest.raises(ValueError, match="not frozen"):
        validate_registered_invocation(
            registration,
            methods=["DRAEM"],
            seeds=[11, 22, 33],
            baseline=None,
            candidate=None,
            bootstrap_samples=10000,
            bootstrap_seed=20260927,
        )
