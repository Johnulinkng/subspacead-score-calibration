"""Verify one fixed image-score aggregation on sealed cached PaDiM maps.

No backbone, model, training, pixel-metric recomputation or network access is used.
The previous test set was already inspected; all outputs remain exploratory.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

# These are fixed resource settings, not scientific tuning parameters.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
sys.dont_write_bytecode = True

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot emit an empty result table")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_scores(raw: np.ndarray) -> tuple[float, float, int]:
    """Return exact max and the float64 mean of the fixed top 1% (floor)."""
    values = np.asarray(raw).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Scores must be nonempty and finite")
    count = max(math.floor(values.size * 0.01), 1)
    # np.partition allocates at most one bounded map-sized array. Boundary ties
    # have equal values, so their internal order cannot affect this mean.
    top = np.partition(values, values.size - count)[-count:]
    return float(values.max()), float(top.mean(dtype=np.float64)), count


def process_memory() -> dict:
    if os.name != "nt":
        return {"working_set_bytes": None, "peak_working_set_bytes": None}

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current = ctypes.windll.kernel32.GetCurrentProcess
    get_current.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    if not get_memory(get_current(), ctypes.byref(counters), counters.cb):
        raise OSError("Cannot collect process memory counters")
    return {"working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize)}


def guarded_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Unsafe or missing run input: {relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/cached_score_top1p_20261012.json")
    args = parser.parse_args()
    source = Path(__file__).resolve()
    run = args.run_root.resolve()
    output = args.output_dir.resolve()
    protocol_path = args.config.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.is_relative_to(run):
        raise ValueError("The new score run must be outside the original frozen run")
    protocol = read_json(protocol_path)
    if (protocol["protocol_id"] != "cached_padim_image_score_top1p_20261012_v1"
            or protocol["top_fraction"] != 0.01 or protocol["top_count"] != 655
            or protocol["map_shape"] != [256, 256]
            or protocol["expected_categories"] != 27
            or protocol["expected_test_images"] != 3887):
        raise ValueError("The fixed scoring protocol was changed")
    state_path = run / "run_state.json"
    original_config_path = run / "run_config.json"
    metric_path = run / "results/category_metrics.jsonl"
    sample_path = run / "results/samples.jsonl"
    original_state = read_json(state_path)
    if original_state["status"] != "completed" or original_state["completed_categories"] != 27:
        raise ValueError("The original complete 27-category run is required")
    original = read_json(original_config_path)
    original_protocol = original["protocol"]
    categories = [(dataset, category) for dataset, spec in original_protocol["datasets"].items()
                  for category in spec["categories"]]
    if len(categories) != 27 or len(set(categories)) != 27 or len(original_protocol["seeds"]) != 1:
        raise ValueError("Unexpected original category or seed coverage")
    seed = original_protocol["seeds"][0]
    metrics = read_jsonl(metric_path)
    expected_matrix = {(d, c, v, seed) for d, c in categories for v in original_protocol["variants"]}
    if len(metrics) != 108 or {(r["dataset"], r["category"], r["variant"], r["seed"]) for r in metrics} != expected_matrix:
        raise ValueError("The original 108-row matrix is incomplete or duplicated")
    baseline = {(r["dataset"], r["category"]): r for r in metrics if r["variant"] == "baseline"}
    if len(baseline) != 27 or any(r["status"] != "completed" or not np.isfinite(r["image_auroc"]) for r in baseline.values()):
        raise ValueError("Original baseline metrics are incomplete or invalid")
    samples = read_jsonl(sample_path)
    sample_keys = {(s["dataset"], s["category"], s["sample_id"]) for s in samples}
    if len(samples) != 3887 or len(sample_keys) != 3887:
        raise ValueError("Expected exactly 3887 unique full-test sample identities")
    by_category: dict[tuple[str, str], list[dict]] = {key: [] for key in categories}
    for sample in samples:
        key = (sample["dataset"], sample["category"])
        if key not in by_category or sample["seed"] != seed or type(sample["label"]) is not int or sample["label"] not in (0, 1):
            raise ValueError("Invalid sample category, label or seed")
        by_category[key].append(sample)
    if {d: sum(len(by_category[(d, c)]) for ds, c in categories if ds == d) for d in original_protocol["datasets"]} != protocol["expected_dataset_test_images"]:
        raise ValueError("Dataset test counts differ from the fixed complete inventory")

    input_seals = {str(p.relative_to(run)): digest(p) for p in (state_path, original_config_path, metric_path, sample_path)}
    integrity_by_sample: dict[tuple[str, str, str], dict] = {}
    for key in categories:
        dataset, category = key
        selected = by_category[key]
        if len(selected) != baseline[key]["test_images"] or {s["label"] for s in selected} != {0, 1}:
            raise ValueError("Category count or label coverage differs from the original baseline")
        inventory_path = run / "models" / dataset / category / "test_inventory.json"
        integrity_path = run / "models" / dataset / category / "prediction_integrity.jsonl"
        inventory = read_json(inventory_path)
        inventory_labels = {row["sample_id"]: row["label"] for row in inventory}
        if len(inventory_labels) != len(inventory) or inventory_labels != {s["sample_id"]: s["label"] for s in selected}:
            raise ValueError("Stored labels do not match the complete original test inventory")
        checked = [row for row in read_jsonl(integrity_path) if row.get("variant") == "baseline"]
        if len(checked) != len(selected) or len({row["sample_id"] for row in checked}) != len(selected):
            raise ValueError("Baseline prediction integrity coverage is duplicated or incomplete")
        check_by_id = {row["sample_id"]: row for row in checked}
        if set(check_by_id) != {s["sample_id"] for s in selected}:
            raise ValueError("Baseline integrity sample IDs do not match the full test inventory")
        for sample in selected:
            sealed = check_by_id[sample["sample_id"]]
            if sealed["path"] != sample["raw_maps"]["baseline"] or len(sealed["sha256"]) != 64:
                raise ValueError("Baseline prediction seal and sample path disagree")
            guarded_path(run, sealed["path"])
            integrity_by_sample[(dataset, category, sample["sample_id"])] = sealed
        for path in (inventory_path, integrity_path):
            input_seals[str(path.relative_to(run))] = digest(path)

    # Everything above reads metadata only. Freeze all parameters, current source
    # and metadata hashes before opening the first prediction map.
    output.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    frozen = {"schema_version": 1, "status": "frozen_before_cached_map_reads",
              "created_at_utc": started, "protocol": protocol,
              "protocol_file_sha256": digest(protocol_path), "script_sha256": digest(source),
              "original_run_root": str(run), "original_input_sha256": input_seals,
              "prediction_sha256_source": "All per-category prediction_integrity.jsonl inputs are sealed above; every baseline map is checked before scoring.",
              "source_file": str(source), "protocol_file": str(protocol_path),
              "scientific_claim_eligible": False, "previous_test_results_known": True}
    write_json(output / "run_config.json", frozen)
    start = time.perf_counter()
    write_json(output / "run_state.json", {"status": "running", "started_at_utc": started,
                                            "map_read_count": 0, "completed_categories": 0})
    sample_scores: list[dict] = []
    category_rows: list[dict] = []
    comparisons: list[dict] = []
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.metrics import roc_auc_score
        for dataset, category in categories:
            labels, max_scores, mean_scores = [], [], []
            for sample in by_category[(dataset, category)]:
                sealed = integrity_by_sample[(dataset, category, sample["sample_id"])]
                map_path = guarded_path(run, sealed["path"])
                data = map_path.read_bytes()
                current_sha = hashlib.sha256(data).hexdigest()
                if current_sha != sealed["sha256"]:
                    raise ValueError(f"Baseline raw map seal mismatch: {map_path}")
                raw = np.load(io.BytesIO(data), allow_pickle=False)
                if raw.dtype != np.float32 or raw.shape != (256, 256) or not np.isfinite(raw).all():
                    raise ValueError(f"Raw baseline map violates float32 finite 256x256 contract: {map_path}")
                maximum, top_mean, count = aggregate_scores(raw)
                if count != 655 or maximum != sealed["raw_map_max"]:
                    raise ValueError("Top count or baseline max differs from the fixed integrity contract")
                labels.append(sample["label"])
                max_scores.append(maximum)
                mean_scores.append(top_mean)
                sample_scores.append({"dataset": dataset, "category": category, "sample_id": sample["sample_id"],
                                      "label": sample["label"], "map_path": sealed["path"], "map_sha256": current_sha,
                                      "baseline_max": maximum, "baseline_top1p_mean": top_mean, "top_count": count})
            control = float(roc_auc_score(labels, max_scores))
            candidate = float(roc_auc_score(labels, mean_scores))
            expected_control = baseline[(dataset, category)]["image_auroc"]
            if abs(control - expected_control) > 1e-12:
                raise ValueError(f"Original maximum-score AUROC was not reproduced for {dataset}/{category}")
            for method, auroc in (("baseline_max", control), ("baseline_top1p_mean", candidate)):
                category_rows.append({"dataset": dataset, "category": category, "method": method, "seed": seed,
                                      "test_images": len(labels), "normal_images": labels.count(0),
                                      "anomalous_images": labels.count(1), "image_auroc": auroc})
            comparisons.append({"dataset": dataset, "category": category, "baseline_max_image_auroc": control,
                                "top1p_mean_image_auroc": candidate, "delta_percentage_points": 100 * (candidate - control),
                                "original_control_absolute_error": abs(control - expected_control)})
            write_json(output / "run_state.json", {"status": "running", "started_at_utc": started,
                       "updated_at_utc": utc_now(), "map_read_count": len(sample_scores),
                       "completed_categories": len(comparisons)}, exclusive=False)
            print(json.dumps({"category": f"{dataset}/{category}", "completed_categories": len(comparisons),
                              "baseline_max_image_auroc": control, "top1p_mean_image_auroc": candidate}, allow_nan=False), flush=True)

        macro_rows, macro_comparisons = [], []
        for dataset in original_protocol["datasets"]:
            selected = [r for r in comparisons if r["dataset"] == dataset]
            baseline_macro = float(np.mean([r["baseline_max_image_auroc"] for r in selected]))
            candidate_macro = float(np.mean([r["top1p_mean_image_auroc"] for r in selected]))
            for method, macro in (("baseline_max", baseline_macro), ("baseline_top1p_mean", candidate_macro)):
                macro_rows.append({"dataset": dataset, "method": method, "categories": len(selected), "image_auroc": macro})
            macro_comparisons.append({"dataset": dataset, "categories": len(selected),
                "baseline_max_image_auroc": baseline_macro, "top1p_mean_image_auroc": candidate_macro,
                "delta_percentage_points": 100 * (candidate_macro - baseline_macro),
                "improved_categories": sum(r["delta_percentage_points"] > 0 for r in selected),
                "regressed_categories": sum(r["delta_percentage_points"] < 0 for r in selected),
                "tied_categories": sum(r["delta_percentage_points"] == 0 for r in selected)})
        if digest(source) != frozen["script_sha256"] or digest(protocol_path) != frozen["protocol_file_sha256"]:
            raise ValueError("Scoring source or protocol changed during execution")
        if any(digest(run / relative) != checksum for relative, checksum in input_seals.items()):
            raise ValueError("Original metadata changed during execution")
        write_csv(output / "sample_scores.csv", sample_scores)
        write_csv(output / "category_metrics.csv", category_rows)
        write_csv(output / "category_comparison.csv", comparisons)
        write_csv(output / "dataset_macro_metrics.csv", macro_rows)
        write_csv(output / "dataset_macro_comparison.csv", macro_comparisons)
        summary = {"status": "completed", "started_at_utc": started, "finished_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - start, "completed_categories": 27,
            "test_images": len(sample_scores), "map_read_count": len(sample_scores), "category_metric_rows": len(category_rows),
            "numpy_version": np.__version__, "sklearn_version": sklearn_version,
            "device": "cpu", "cpu_threads": 2, "training": False, "gpu_used": False,
            "model_imports": False, "process_memory": process_memory(),
            "all_baseline_map_sha256_verified": True, "all_max_controls_reproduce_original": True,
            "max_control_absolute_error": max(r["original_control_absolute_error"] for r in comparisons),
            "dataset_comparisons": macro_comparisons, "pixel_metrics_unchanged": True,
            "scientific_claim_eligible": False, "previous_test_results_known": True,
            "interpretation": "Fixed image-score aggregation ablation only. Pixel ranking/maps are unchanged. Report all negative and positive outcomes. No original-method or SCI Q3 claim is established."}
        write_json(output / "summary.json", summary)
        write_json(output / "run_state.json", summary, exclusive=False)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        write_json(output / "run_state.json", {"status": "failed", "started_at_utc": started,
                   "updated_at_utc": utc_now(), "elapsed_seconds": time.perf_counter() - start,
                   "map_read_count": len(sample_scores), "completed_categories": len(comparisons),
                   "error": f"{type(error).__name__}: {error}"}, exclusive=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
