"""Independently audit a completed OAS/ML matrix without training or inference.

Only persisted files are consumed. The default audit recomputes both image
score aggregations and tied-rank AUROC from all raw maps. Pixel-metric
recomputation is optional and uses the same source-sealed exact evaluator.
The audit receipt and temporary sorting storage stay outside both input runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("ml_max", "ml_top1p", "oas_max", "oas_top1p")
PIXEL_METRICS = ("pixel_auroc", "pixel_ap", "aupro_0_30")


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)


def read_jsonl(path: Path):
    return [json.loads(line, object_pairs_hook=_reject_duplicates)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inside(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (not relative or posix.is_absolute() or ".." in posix.parts
            or "\\" in relative or ":" in relative):
        raise ValueError(f"Unsafe member path: {relative!r}")
    path = (root / Path(*posix.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping input file: {relative}")
    return path


class HashCache:
    def __init__(self):
        self.values = {}

    def digest(self, path: Path) -> str:
        path = path.resolve()
        before = path.stat()
        stamp = (before.st_size, before.st_mtime_ns)
        cached = self.values.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        after = path.stat()
        if stamp != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while hashing: {path}")
        value = hasher.hexdigest()
        self.values[path] = (stamp, value)
        return value

    def verify(self, path: Path, expected: str):
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"Invalid SHA-256 seal: {path}")
        if self.digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")


def unique_by(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate {label}: {key}")
        result[key] = row
    return result


def number(value, label, *, unit_interval=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Invalid finite number: {label}")
    if unit_interval and not 0 <= value <= 1:
        raise ValueError(f"Metric outside [0,1]: {label}")
    return float(value)


def close(actual, expected, label, *, tolerance=1e-12):
    actual, expected = number(actual, label), number(expected, label)
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"Numerical mismatch for {label}: {actual} != {expected}")


def rank_auroc(labels, scores, np):
    """Independent Mann-Whitney AUROC with average ranks for exact ties."""
    labels = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    if (labels.shape != values.shape or labels.ndim != 1 or not values.size
            or not np.isfinite(values).all() or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Invalid image labels or scores")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both labels")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = (starts + 1 + ends) / 2.0
    positive_counts = np.add.reduceat(labels[order].astype(np.int64), starts)
    positive_rank_sum = float(np.sum(ranks * positive_counts))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def audit(run: Path, baseline: Path, protocol_path: Path, pixel_recompute: bool,
          temporary_parent: Path):
    started = time.monotonic()
    state, config = read_json(run / "run_state.json"), read_json(run / "run_config.json")
    if state.get("status") != "completed" or config.get("benchmark_train_only"):
        raise ValueError("Only a completed full-matrix run can be audited")
    if state.get("completed_categories") != 27 or state.get("test_images") != 3887 or state.get("metric_rows") != 108:
        raise ValueError("Completed state has unexpected counts")
    if state.get("scientific_claim_eligible") is not False:
        raise ValueError("The exploratory claim boundary was changed")
    import numpy as np

    hashes = HashCache()
    hashes.verify(protocol_path, config["protocol_sha256"])
    protocol = read_json(protocol_path)
    if protocol != config["protocol"]:
        raise ValueError("Recorded and external protocols differ")
    if (protocol.get("protocol_id") != "padim_oas_paired_20261012_v1"
            or protocol.get("expected_categories") != 27 or protocol.get("expected_test_images") != 3887
            or protocol.get("scientific_claim_eligible") is not False):
        raise ValueError("Unexpected comparison protocol")
    fixed = {"regularization": 0.01, "sigma": 4.0, "model_size": 224,
             "output_size": 256, "dimensions": 100, "position_chunk_size": 64, "cpu_threads": 2}
    if any(protocol.get(key) != value for key, value in fixed.items()) or protocol.get("estimators") != ["ml", "oas"]:
        raise ValueError("Frozen estimator or rendering settings differ")
    if (PROJECT / protocol["baseline_run"]).resolve() != baseline:
        raise ValueError("Supplied baseline differs from the frozen baseline")
    for relative, expected in config["source_sha256"].items():
        hashes.verify(inside(PROJECT, relative), expected)
    for relative, expected in config["input_metadata_sha256"].items():
        hashes.verify(inside(baseline, relative), expected)
    original = read_json(baseline / "run_config.json")
    if read_json(baseline / "run_state.json")["status"] != "completed":
        raise ValueError("Baseline is not completed")
    if protocol["seed"] != original["protocol"]["seeds"][0] or len(original["protocol"]["seeds"]) != 1:
        raise ValueError("Comparison and baseline seeds differ")
    for relative in ("src/industrial_anomaly_lab/quick_features.py", "src/industrial_anomaly_lab/quick_metrics.py"):
        if config["source_sha256"][relative] != original["source_sha256"][relative]:
            raise ValueError(f"Common geometry/evaluator source differs from baseline: {relative}")
        hashes.verify(inside(PROJECT, relative), original["source_sha256"][relative])
    hashes.verify(inside(PROJECT, original["weight_path"]), original["weight_sha256"])
    tasks = [(d, c) for d, settings in original["protocol"]["datasets"].items() for c in settings["categories"]]
    if len(tasks) != 27 or len(set(tasks)) != 27:
        raise ValueError("Baseline category matrix is invalid")
    if config["expected_matrix"] != {"categories": [list(t) for t in tasks], "variants": list(VARIANTS)}:
        raise ValueError("New expected matrix does not match baseline categories")
    expected_metadata = {"run_config.json", "run_state.json", "results/category_metrics.jsonl", "results/samples.jsonl"}
    for d, c in tasks:
        expected_metadata.update(f"models/{d}/{c}/{name}" for name in
                                 ("statistics_integrity.json", "train_manifest.json", "test_inventory.json", "prediction_integrity.jsonl"))
    if set(config["input_metadata_sha256"]) != expected_metadata:
        raise ValueError("Missing or unexpected sealed baseline metadata")
    rows = read_jsonl(run / "results/category_metrics.jsonl")
    samples = read_jsonl(run / "results/samples.jsonl")
    if len(rows) != 108 or len(samples) != 3887:
        raise ValueError("Full result counts differ")
    row_index = unique_by(rows, ("dataset", "category", "variant", "seed"), "metric identity")
    sample_index = unique_by(samples, ("dataset", "category", "sample_id"), "sample identity")
    expected_identities = {(d, c, variant, protocol["seed"]) for d, c in tasks for variant in VARIANTS}
    if set(row_index) != expected_identities:
        raise ValueError("Incomplete or unregistered metric identities")
    old_samples = unique_by(read_jsonl(baseline / "results/samples.jsonl"),
                            ("dataset", "category", "sample_id"), "baseline sample identity")
    if set(sample_index) != set(old_samples):
        raise ValueError("New sample identities differ from the full baseline")
    old_rows = unique_by(read_jsonl(baseline / "results/category_metrics.jsonl"),
                         ("dataset", "category", "variant", "seed"), "baseline metric identity")
    selected_new = {key for key, item in sample_index.items() if item.get("selected") is True}
    selected_old = {key for key, item in old_samples.items() if item.get("selected") is True}
    if selected_new != selected_old or len(selected_new) != 54:
        raise ValueError("Selected sample IDs differ from the baseline")

    results = []
    map_count = mask_count = train_total = test_total = completion_members = 0
    with TemporaryDirectory(prefix=".oas-audit-metrics-", dir=temporary_parent) as temporary:
        if pixel_recompute:
            sys.path.insert(0, str(PROJECT / "src"))
            from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        for d, c in tasks:
            cat, pred, old_model = run / "categories" / d / c, run / "predictions" / d / c, baseline / "models" / d / c
            completion = read_json(cat / "completion.json")
            if completion.get("status") != "completed":
                raise ValueError(f"Category is not completed: {d}/{c}")
            files = completion["sha256"]
            actual = {p.relative_to(run).as_posix() for directory in (cat, pred) for p in directory.rglob("*")
                      if p.is_file() and p != cat / "completion.json"}
            if set(files) != actual:
                raise ValueError(f"Category completion coverage differs: {d}/{c}")
            required = {f"categories/{d}/{c}/{name}" for name in
                        ("metrics.json", "samples.jsonl", "scorers.json", "oas_alpha.npy", "statistics_input.json", "test_inventory.json")}
            if not required.issubset(files):
                raise ValueError(f"Category completion lacks metadata: {d}/{c}")
            for relative, expected in files.items():
                hashes.verify(inside(run, relative), expected)
            completion_members += len(files)
            seal = read_json(old_model / "statistics_integrity.json")
            statistics_path = old_model / "statistics.npz"
            hashes.verify(statistics_path, seal["sha256"])
            if statistics_path.stat().st_size != seal["size_bytes"]:
                raise ValueError(f"Baseline statistics size differs: {d}/{c}")
            train, inventory = read_json(old_model / "train_manifest.json"), read_json(old_model / "test_inventory.json")
            expected_category_files = required | {
                f"predictions/{d}/{c}/{model}/{index:06d}.npy"
                for index in range(len(inventory)) for model in ("ml", "oas", "masks")
            }
            if set(files) != expected_category_files:
                raise ValueError(f"Unregistered or missing category artifact: {d}/{c}")
            if read_json(cat / "test_inventory.json") != inventory:
                raise ValueError(f"Copied test inventory differs: {d}/{c}")
            recorded_input = read_json(cat / "statistics_input.json")
            if (recorded_input["statistics_sha256"] != seal["sha256"] or recorded_input["original_seal"] != seal
                    or Path(recorded_input["original_model"]).resolve() != old_model.resolve()
                    or recorded_input["train_images"] != len(train)
                    or recorded_input["train_manifest_sha256"] != hashes.digest(old_model / "train_manifest.json")):
                raise ValueError(f"Statistics provenance differs: {d}/{c}")
            baseline_row = old_rows[(d, c, "baseline", protocol["seed"])]
            if baseline_row["train_images"] != len(train) or baseline_row["test_images"] != len(inventory):
                raise ValueError(f"Baseline training/test count differs: {d}/{c}")
            cat_rows = read_json(cat / "metrics.json")
            cat_index = unique_by(cat_rows, ("variant",), "category variant")
            if set(cat_index) != {(v,) for v in VARIANTS}:
                raise ValueError(f"Category variants differ: {d}/{c}")
            for variant in VARIANTS:
                row = cat_index[(variant,)]
                if row != row_index[(d, c, variant, protocol["seed"])]:
                    raise ValueError(f"Combined and category metrics differ: {d}/{c}/{variant}")
                if row["status"] != "completed" or row["train_images"] != len(train) or row["test_images"] != len(inventory):
                    raise ValueError(f"Metric status/count differs: {d}/{c}/{variant}")
                score = variant.split("_", 1)[1]
                if row["score_definition"] != protocol["score_definitions"][score]:
                    raise ValueError(f"Image score definition differs: {d}/{c}/{variant}")
                for metric in ("image_auroc", "image_auroc_raw_max", "image_auroc_top1p_mean", *PIXEL_METRICS):
                    number(row[metric], f"{d}/{c}/{variant}/{metric}", unit_interval=True)
            alpha = np.load(cat / "oas_alpha.npy", allow_pickle=False)
            if alpha.shape != (3136,) or alpha.dtype != np.float64 or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
                raise ValueError(f"OAS coefficient contract differs: {d}/{c}")
            scorers = read_json(cat / "scorers.json")
            if set(scorers) != {"ml", "oas"}:
                raise ValueError(f"Scorer metadata coverage differs: {d}/{c}")
            summary = {"count": 3136, "minimum": float(alpha.min()), "mean": float(alpha.mean()),
                       "median": float(np.median(alpha)), "maximum": float(alpha.max()),
                       "standard_deviation": float(alpha.std()), "zero_count": int(np.count_nonzero(alpha == 0)),
                       "one_count": int(np.count_nonzero(alpha == 1))}
            if set(scorers["oas"]["alpha_summary"]) != set(summary):
                raise ValueError(f"OAS alpha summary schema differs: {d}/{c}")
            for name, value in summary.items():
                close(scorers["oas"]["alpha_summary"][name], value, f"{d}/{c}/alpha/{name}")
            for model in ("ml", "oas"):
                metadata = scorers[model]
                expected_mode = "oas" if model == "oas" else "ml_no_shrinkage"
                if (metadata["mode"] != expected_mode or metadata["sample_count"] != len(train)
                        or metadata["regularization"] != 0.01 or metadata["dimensions"] != 100
                        or metadata["oas_covariance_denominator"] != "n" or metadata["input_covariance_denominator"] != "n-1"
                        or metadata["per_image_normalization"] is not False
                        or metadata["spatial_smoothing"] != {"sigma": 4.0, "plane_size": 224, "mode": "reflect"}
                        or metadata["output_size"] != 256):
                    raise ValueError(f"Scorer scientific metadata differs: {d}/{c}/{model}")
                close(metadata["ml_covariance_factor"], (len(train)-1)/len(train), f"{d}/{c}/ML factor")
                if model == "ml":
                    expected_ml = {"count": 3136, "minimum": 0.0, "mean": 0.0, "median": 0.0, "maximum": 0.0,
                                   "standard_deviation": 0.0, "zero_count": 3136, "one_count": 0}
                    if metadata["alpha_summary"] != expected_ml:
                        raise ValueError(f"ML control has nonzero coefficients: {d}/{c}")
            cat_samples = read_jsonl(cat / "samples.jsonl")
            if len(cat_samples) != len(inventory):
                raise ValueError(f"Category test count differs: {d}/{c}")
            local_ids = set()
            maps = {"ml": [], "oas": []}
            image_scores = {"ml": [], "oas": []}
            masks, labels = [], []
            for index, (sample, input_row) in enumerate(zip(cat_samples, inventory, strict=True)):
                key = (d, c, sample["sample_id"])
                if key in local_ids or sample != sample_index[key]:
                    raise ValueError(f"Duplicate or inconsistent category sample: {key}")
                local_ids.add(key)
                old = old_samples[key]
                if (sample["sample_id"] != input_row["sample_id"] or sample["seed"] != protocol["seed"]
                        or sample["dataset"] != d or sample["category"] != c or sample["label"] != input_row["label"]
                        or sample["label"] != old["label"] or sample["selected"] != old["selected"]
                        or sample["image_sha256"] != input_row["image_sha256"] or sample["image_sha256"] != old["image_sha256"]
                        or Path(sample["image_path"]).resolve() != Path(input_row["image_path"]).resolve()):
                    raise ValueError(f"Input identity or selection differs: {key}")
                if sample["mask_path"] != f"predictions/{d}/{c}/masks/{index:06d}.npy":
                    raise ValueError(f"Mask storage order differs: {key}")
                mask_path = inside(run, sample["mask_path"])
                hashes.verify(mask_path, sample["mask_sha256"])
                mask = np.load(mask_path, allow_pickle=False)
                old_mask = np.load(inside(baseline, old["mask_path"]), allow_pickle=False)
                if (mask.shape != (256, 256) or mask.dtype != np.bool_ or old_mask.dtype != np.bool_
                        or not np.array_equal(mask, old_mask) or (sample["label"] == 0 and mask.any())):
                    raise ValueError(f"Evaluation mask differs: {key}")
                masks.append(mask_path)
                labels.append(sample["label"])
                mask_count += 1
                if set(sample["raw_maps"]) != {"ml", "oas"} or set(sample["raw_map_sha256"]) != {"ml", "oas"} or set(sample["image_scores"]) != {"ml", "oas"}:
                    raise ValueError(f"Prediction coverage differs: {key}")
                for model in ("ml", "oas"):
                    relative = sample["raw_maps"][model]
                    if relative != f"predictions/{d}/{c}/{model}/{index:06d}.npy":
                        raise ValueError(f"Map storage order differs: {key}/{model}")
                    path = inside(run, relative)
                    hashes.verify(path, sample["raw_map_sha256"][model])
                    raw = np.load(path, allow_pickle=False)
                    if raw.shape != (256, 256) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                        raise ValueError(f"Raw map contract differs: {key}/{model}")
                    flat = raw.reshape(-1)
                    count = max(math.floor(flat.size * 0.01), 1)
                    if count != 655:
                        raise ValueError("Top-percent pixel count differs")
                    maximum = float(flat.max())
                    # Independent complete-map sorting, rather than importing
                    # the runner's partition-based aggregate_scores helper.
                    top = float(np.sort(flat, kind="quicksort")[-count:].mean(dtype=np.float64))
                    recorded = sample["image_scores"][model]
                    if set(recorded) != {"max", "top1p"}:
                        raise ValueError(f"Image score coverage differs: {key}/{model}")
                    if number(recorded["max"], "raw max") != maximum:
                        raise ValueError(f"Saved maximum differs: {key}/{model}")
                    close(recorded["top1p"], top, f"{key}/{model}/top1p")
                    image_scores[model].append((maximum, top))
                    maps[model].append(path)
                    map_count += 1
            expected_local = {key for key in old_samples if key[:2] == (d, c)}
            if local_ids != expected_local:
                raise ValueError(f"Missing or unregistered category samples: {d}/{c}")
            for model in ("ml", "oas"):
                maximum_row, top_row = cat_index[(f"{model}_max",)], cat_index[(f"{model}_top1p",)]
                for metric in PIXEL_METRICS:
                    if np.float64(maximum_row[metric]).tobytes() != np.float64(top_row[metric]).tobytes():
                        raise ValueError(f"Aggregation changes a pixel metric: {d}/{c}/{model}/{metric}")
                scores = np.asarray(image_scores[model], dtype=np.float64)
                recomputed = [rank_auroc(labels, scores[:, i], np) for i in (0, 1)]
                for row, aggregation in ((maximum_row, 0), (top_row, 1)):
                    close(row["image_auroc"], recomputed[aggregation], f"{d}/{c}/{row['variant']}/AUROC")
                    close(row["image_auroc_raw_max"], recomputed[0], f"{d}/{c}/{row['variant']}/raw max AUROC")
                    close(row["image_auroc_top1p_mean"], recomputed[1], f"{d}/{c}/{row['variant']}/top1p AUROC")
                if pixel_recompute:
                    recalculated = evaluate_from_files(maps[model], masks, labels, fpr_limit=0.30, temp_root=temporary)
                    close(recalculated["image_auroc"], recomputed[0], f"{d}/{c}/{model}/evaluator max AUROC")
                    for metric in PIXEL_METRICS:
                        close(recalculated[metric], maximum_row[metric], f"{d}/{c}/{model}/{metric}")
            train_total += len(train)
            test_total += len(inventory)
            results.append({"dataset": d, "category": c, "status": "passed", "train_images": len(train),
                            "test_images": len(inventory), "metric_rows": 4, "maps": 2*len(inventory),
                            "alpha_summary": summary, "pixel_metrics_recomputed": pixel_recompute})
            print(json.dumps({"audited_categories": len(results), "dataset": d, "category": c,
                              "test_images": len(inventory), "status": "passed"}), flush=True)
    if test_total != 3887 or train_total != 12288 or map_count != 7774 or mask_count != 3887:
        raise ValueError("Final sample, training or prediction totals differ")
    dataset_categories = Counter(d for d, _ in tasks)
    if dataset_categories != {"mvtec_ad": 15, "visa": 12}:
        raise ValueError("Dataset category split differs")
    dataset_counts = {
        d: {"categories": dataset_categories[d],
            "normal_training_images": sum(item["train_images"] for item in results if item["dataset"] == d),
            "test_images": sum(item["test_images"] for item in results if item["dataset"] == d)}
        for d in dataset_categories
    }
    if dataset_counts != {
        "mvtec_ad": {"categories": 15, "normal_training_images": 3629, "test_images": 1725},
        "visa": {"categories": 12, "normal_training_images": 8659, "test_images": 2162},
    }:
        raise ValueError("Frozen per-dataset sample counts differ")
    return {"schema_version": 1, "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(run), "baseline_root": str(baseline), "protocol_sha256": config["protocol_sha256"],
            "auditor_sha256": hashes.digest(Path(__file__)), "category_count": 27, "metric_rows": 108,
            "normal_training_images": train_total, "test_images": test_total, "raw_maps": map_count,
            "evaluation_masks": mask_count, "selected_samples": len(selected_new),
            "dataset_counts": dataset_counts,
            "completion_members_verified": completion_members, "source_files_verified": len(config["source_sha256"]),
            "baseline_metadata_files_verified": len(config["input_metadata_sha256"]),
            "image_auroc_recomputed": True, "pixel_metrics_recomputed": pixel_recompute,
            "score_aggregation": {"max": "persisted float32 raw map maximum", "top1p": "float64 mean of largest 655 persisted pixels"},
            "auroc_recomputation": "independent average tied-rank Mann-Whitney statistic",
            "metric_tolerance": 1e-12, "elapsed_seconds": time.monotonic()-started,
            "training_performed": False, "feature_inference_performed": False,
            "scientific_improvement_confirmed": False, "categories": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/quick/oas-r18-20261012")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT / "runs/quick/padim-r18-20261012T022650Z")
    parser.add_argument("--protocol", type=Path, default=PROJECT / "configs/oas_comparison_20261012.json")
    parser.add_argument("--receipt", "--output", dest="receipt", type=Path)
    parser.add_argument("--recompute-pixel-metrics", action="store_true")
    args = parser.parse_args()
    run, baseline = args.run_root.resolve(), args.baseline_root.resolve()
    output = args.receipt.resolve() if args.receipt else run.parent / (run.name + "-independent-audit.json")
    if output.is_relative_to(run) or output.is_relative_to(baseline):
        raise ValueError("Audit receipt must stay outside both input runs")
    if output.exists():
        raise FileExistsError("Refusing to replace an existing audit receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit(run, baseline, args.protocol.resolve(), args.recompute_pixel_metrics, output.parent)
        exit_code = 0
    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError, ImportError, RuntimeError) as error:
        result = {"schema_version": 1, "status": "failed", "run_root": str(run), "baseline_root": str(baseline),
                  "error": f"{type(error).__name__}: {error}", "training_performed": False,
                  "feature_inference_performed": False, "audited_at_utc": datetime.now(timezone.utc).isoformat()}
        exit_code = 2
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "categories"}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
