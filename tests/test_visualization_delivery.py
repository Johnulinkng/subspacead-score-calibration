from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.run_artifacts import sha256_file, write_json, write_jsonl
from industrial_anomaly_lab.visualization_delivery import (
    freeze_visualization_selection,
    render_visualization_comparisons,
    resolve_visualization_failures,
)


SEED = 7
METHODS = ("DRAEM", "Improved")


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    image[:, :, 1] = (value + 20) % 255
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path, region: tuple[slice, slice]) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[region] = 255
    Image.fromarray(mask, mode="L").save(path)
    return mask.astype(bool)


def _fixture_files(tmp_path: Path) -> dict[str, object]:
    dataset_root = tmp_path / "dataset"
    specs = [
        ("bottle/good/000", 0, "normal", 0.0, None),
        ("bottle/crack/001", 1, "small", 4 / 256, (slice(2, 4), slice(2, 4))),
        ("bottle/crack/002", 1, "large", 64 / 256, (slice(4, 12), slice(4, 12))),
        ("bottle/scratch/003", 1, "medium", 24 / 256, (slice(6, 9), slice(3, 11))),
    ]
    entries: list[dict[str, object]] = []
    masks: dict[str, np.ndarray] = {}
    for index, (sample_id, label, scale, area, region) in enumerate(specs):
        category, defect, stem = sample_id.split("/")
        image_relative = f"{category}/test/{defect}/{stem}.png"
        image_path = dataset_root / image_relative
        _write_rgb(image_path, 40 + index * 35)
        mask_relative: str | None = None
        mask_sha: str | None = None
        if region is not None:
            mask_relative = f"{category}/ground_truth/{defect}/{stem}_mask.png"
            mask_path = dataset_root / mask_relative
            masks[sample_id] = _write_mask(mask_path, region)
            mask_sha = sha256_file(mask_path)
        else:
            masks[sample_id] = np.zeros((16, 16), dtype=bool)
        entries.append(
            {
                "dataset": "mvtec_ad",
                "category": category,
                "sample_id": sample_id,
                "label": label,
                "defect_type": defect,
                "anomaly_area_fraction": area,
                "anomaly_scale": scale,
                "source_image": image_relative,
                "source_image_sha256": sha256_file(image_path),
                "source_mask": mask_relative,
                "source_mask_sha256": mask_sha,
            }
        )

    inventory_path = tmp_path / "formal_test_inventory.json"
    write_json(
        inventory_path,
        {
            "schema_version": 1,
            "kind": "formal_test_inventory",
            "inventory_id": "synthetic-only",
            "dataset": "mvtec_ad",
            "split": "test",
            "entries": entries,
        },
    )
    registry_path = tmp_path / "registration.json"
    write_json(
        registry_path,
        {
            "registration_id": "synthetic-f0",
            "formal_method_registry": {
                "status": "frozen_before_formal_evaluation",
                "methods": list(METHODS),
                "baseline": METHODS[0],
                "candidate": METHODS[1],
            },
        },
    )
    config_path = tmp_path / "visualization_config.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "kind": "formal_visualization_selection_config",
            "dataset": "mvtec_ad",
            "model_seed": SEED,
            "selection_salt": "synthetic-visualization-v1",
            "fixed_selection": {
                "mode": "deterministic_hash_stratified",
                "normal_per_category": 1,
                "anomaly_per_defect_scale_stratum": 1,
            },
            "failure_case_rule": {
                "baseline_method": METHODS[0],
                "candidate_method": METHODS[1],
                "candidate_filter": "anomalies_with_positive_evaluation_mask",
                "rankings": [
                    {
                        "name": "candidate_absolute_failures",
                        "metric": "candidate_pixel_average_precision",
                        "order": "ascending",
                        "per_category": 1,
                        "tie_break": "sample_id_ascending",
                    },
                    {
                        "name": "candidate_regressions_vs_baseline",
                        "metric": (
                            "candidate_minus_baseline_pixel_average_precision"
                        ),
                        "order": "ascending",
                        "per_category": 1,
                        "tie_break": "sample_id_ascending",
                    },
                ],
            },
        },
    )
    return {
        "dataset_root": dataset_root,
        "entries": entries,
        "masks": masks,
        "inventory_path": inventory_path,
        "registry_path": registry_path,
        "config_path": config_path,
    }


def _freeze(tmp_path: Path, fixture: dict[str, object]) -> Path:
    output = tmp_path / "selection.json"
    freeze_visualization_selection(
        test_inventory_path=fixture["inventory_path"],
        dataset_root=fixture["dataset_root"],
        selection_config_path=fixture["config_path"],
        method_registry_path=fixture["registry_path"],
        output_path=output,
    )
    return output


def _map_for(method: str, sample_id: str, mask: np.ndarray) -> np.ndarray:
    if method == "DRAEM":
        return np.where(mask, 0.9, 0.1).astype(np.float32)
    if sample_id == "bottle/crack/001":
        return np.where(mask, 0.05, 0.95).astype(np.float32)
    if sample_id == "bottle/crack/002":
        y, x = np.indices(mask.shape)
        return ((x + 2 * y) / 47.0).astype(np.float32)
    return np.where(mask, 0.8, 0.2).astype(np.float32)


def _write_prediction_run(
    root: Path,
    *,
    method: str,
    entries: list[dict[str, object]],
    masks: dict[str, np.ndarray],
) -> Path:
    run = root / method
    raw_dir = run / "predictions" / "raw_maps"
    mask_dir = run / "predictions" / "evaluation_masks"
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        sample_id = str(entry["sample_id"])
        raw_path = raw_dir / f"{index:06d}.npy"
        eval_mask_path = mask_dir / f"{index:06d}.npy"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        eval_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("wb") as handle:
            np.save(handle, _map_for(method, sample_id, masks[sample_id]), allow_pickle=False)
        with eval_mask_path.open("wb") as handle:
            np.save(handle, masks[sample_id], allow_pickle=False)
        rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "label": entry["label"],
                "defect_type": entry["defect_type"],
                "input_image": entry["source_image"],
                "source_ground_truth": entry["source_mask"],
                "raw_map": raw_path.relative_to(run).as_posix(),
                "raw_map_sha256": sha256_file(raw_path),
                "evaluation_mask": eval_mask_path.relative_to(run).as_posix(),
                "evaluation_mask_sha256": sha256_file(eval_mask_path),
            }
        )
    config_path = run / "run_config.json"
    index_path = run / "predictions" / "index.jsonl"
    write_json(
        config_path,
        {
            "method": method,
            "execution_scope": "full",
            "run_purpose": "formal_full",
            "claim_scope": "formal_matrix_candidate",
            "formal_matrix_eligible": True,
            "prediction_contract": {"metrics_reload_saved_arrays": True},
            "settings": {
                "dataset": "mvtec_ad",
                "category": "bottle",
                "seed": SEED,
            },
        },
    )
    write_jsonl(index_path, rows)
    metrics_path = run / "metrics.json"
    write_json(
        metrics_path,
        {
            "dataset": "mvtec_ad",
            "category": "bottle",
            "computed_from": (
                "predictions/index.jsonl raw_map and evaluation_mask files"
            ),
            "metrics": {},
        },
    )
    write_json(
        run / "run_state.json",
        {
            "status": "completed",
            "execution_scope": "full",
            "run_purpose": "formal_full",
            "claim_scope": "formal_matrix_candidate",
            "formal_matrix_eligible": True,
            "run_config_sha256": sha256_file(config_path),
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_index_sha256": sha256_file(index_path),
        },
    )
    return run


def _prediction_runs(tmp_path: Path, fixture: dict[str, object]) -> list[Path]:
    return [
        _write_prediction_run(
            tmp_path / "runs",
            method=method,
            entries=fixture["entries"],
            masks=fixture["masks"],
        )
        for method in METHODS
    ]


def test_freeze_is_result_independent_and_stratified(tmp_path: Path) -> None:
    fixture = _fixture_files(tmp_path)
    first = _freeze(tmp_path, fixture)
    second = tmp_path / "selection_repeat.json"
    freeze_visualization_selection(
        test_inventory_path=fixture["inventory_path"],
        dataset_root=fixture["dataset_root"],
        selection_config_path=fixture["config_path"],
        method_registry_path=fixture["registry_path"],
        output_path=second,
    )
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert first.read_bytes() == second.read_bytes()
    assert payload["prediction_artifacts_accessed"] is False
    assert payload["manual_post_result_selection_allowed"] is False
    assert len(payload["result_independent_samples"]) == 4
    assert {row["anomaly_scale"] for row in payload["result_independent_samples"]} == {
        "normal",
        "small",
        "medium",
        "large",
    }
    assert len(payload["failure_candidate_pool"]) == 3


def test_freeze_rejects_result_driven_fixed_selection(tmp_path: Path) -> None:
    fixture = _fixture_files(tmp_path)
    config = json.loads(Path(fixture["config_path"]).read_text(encoding="utf-8"))
    config["fixed_selection"] = {
        "mode": "result_ranked",
        "metric": "best_candidate_score",
    }
    write_json(fixture["config_path"], config)
    with pytest.raises(ValueError, match="result-driven selection is forbidden"):
        _freeze(tmp_path, fixture)


def test_freeze_rejects_source_hash_tamper(tmp_path: Path) -> None:
    fixture = _fixture_files(tmp_path)
    image = Path(fixture["dataset_root"]) / "bottle/test/good/000.png"
    _write_rgb(image, 250)
    with pytest.raises(ValueError, match="source image SHA-256 mismatch"):
        _freeze(tmp_path, fixture)


def test_prediction_hash_tamper_and_sample_mismatch_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture_files(tmp_path)
    selection = _freeze(tmp_path, fixture)
    runs = _prediction_runs(tmp_path, fixture)
    raw_path = runs[1] / "predictions/raw_maps/000001.npy"
    with raw_path.open("wb") as handle:
        np.save(handle, np.zeros((16, 16), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="raw_map SHA-256 mismatch"):
        render_visualization_comparisons(
            selection_manifest_path=selection,
            dataset_root=fixture["dataset_root"],
            prediction_directories=runs,
            output_dir=tmp_path / "tampered_render",
        )

    # Rebuild clean runs, then make one index point the shared ID at a different image.
    clean_root = tmp_path / "clean"
    clean_runs = [
        _write_prediction_run(
            clean_root,
            method=method,
            entries=fixture["entries"],
            masks=fixture["masks"],
        )
        for method in METHODS
    ]
    index_path = clean_runs[1] / "predictions/index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["input_image"] = rows[1]["input_image"]
    write_jsonl(index_path, rows)
    state_path = clean_runs[1] / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predictions_index_sha256"] = sha256_file(index_path)
    write_json(state_path, state)
    with pytest.raises(ValueError, match="input image mismatch"):
        render_visualization_comparisons(
            selection_manifest_path=selection,
            dataset_root=fixture["dataset_root"],
            prediction_directories=clean_runs,
            output_dir=tmp_path / "mismatched_render",
        )


def test_machine_failure_resolution_and_render_are_reproducible(tmp_path: Path) -> None:
    fixture = _fixture_files(tmp_path)
    selection = _freeze(tmp_path, fixture)
    runs = _prediction_runs(tmp_path, fixture)
    resolution_path = tmp_path / "failure_resolution.json"
    resolution = resolve_visualization_failures(
        selection_manifest_path=selection,
        prediction_directories=runs,
        output_path=resolution_path,
    )
    assert resolution["manual_selection_performed"] is False
    assert len(resolution["candidate_audit"]) == 3
    assert len(resolution["selected_cases"]) == 2
    assert all(
        row["sample_id"] == "bottle/crack/001"
        for row in resolution["selected_cases"]
    )
    assert all(
        len(ranking["complete_ranking"]) == 3
        for ranking in resolution["rankings"]
    )

    raw_hashes_before = {
        path: sha256_file(path)
        for run in runs
        for path in (run / "predictions/raw_maps").glob("*.npy")
    }
    first = render_visualization_comparisons(
        selection_manifest_path=selection,
        dataset_root=fixture["dataset_root"],
        prediction_directories=runs,
        failure_resolution_path=resolution_path,
        output_dir=tmp_path / "render_one",
    )
    second = render_visualization_comparisons(
        selection_manifest_path=selection,
        dataset_root=fixture["dataset_root"],
        prediction_directories=runs,
        failure_resolution_path=resolution_path,
        output_dir=tmp_path / "render_two",
    )
    first_pngs = sorted((tmp_path / "render_one/figures").glob("*.png"))
    second_pngs = sorted((tmp_path / "render_two/figures").glob("*.png"))
    assert len(first_pngs) == len(second_pngs) == 6
    assert [sha256_file(path) for path in first_pngs] == [
        sha256_file(path) for path in second_pngs
    ]
    assert first == second
    assert first["metric_arrays_modified"] is False
    assert first["raw_maps_rewritten"] is False
    assert {
        figure["cohort"] for figure in first["figures"]
    } == {
        "result_independent_preregistered",
        "outcome_dependent_machine_selected",
    }
    assert all(
        figure["display"]["normalization"] == "shared_per_figure_raw_minmax"
        and figure["display"]["affects_metric_data"] is False
        for figure in first["figures"]
    )
    assert raw_hashes_before == {
        path: sha256_file(path)
        for run in runs
        for path in (run / "predictions/raw_maps").glob("*.npy")
    }
