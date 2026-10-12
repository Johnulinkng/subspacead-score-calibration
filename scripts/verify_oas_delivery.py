"""Verify OAS cache, selected model inference, and optional full pixel metrics.

All predictions and masks are read from the delivered folder. No network or
training is performed. Metric sorting and the receipt stay outside that folder
so its master file manifest remains valid after verification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True
EXTENSION = "优化实验/OAS验证"
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def member(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if not relative or value.is_absolute() or ".." in value.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative package member: {relative!r}")
    path = (root / Path(*value.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping package member: {relative}")
    return path


def verify_cache(package: Path) -> tuple[dict[str, Any], Path, list[tuple[str, str]]]:
    extension = package / EXTENSION
    info = read(extension / "extension_manifest.json")
    if info.get("status") != "completed" or info.get("test_images") != 3887:
        raise ValueError("Completed full OAS extension required")
    run = extension / "run"
    config = read(run / "run_config.json")
    if read(run / "run_state.json").get("status") != "completed":
        raise ValueError("Delivered OAS run is incomplete")
    for relative, expected in config["source_sha256"].items():
        if digest(member(package / "code", relative)) != expected:
            raise ValueError(f"Executed source differs from the delivered source: {relative}")
    tasks = [(str(d), str(c)) for d, c in config["expected_matrix"]["categories"]]
    if len(tasks) != 27 or len(set(tasks)) != 27:
        raise ValueError("Invalid frozen category matrix")
    samples, seals = 0, 0
    for dataset, category in tasks:
        directory = run / f"categories/{dataset}/{category}"
        seal = read(directory / "completion.json")
        if seal.get("status") != "completed":
            raise ValueError("Incomplete category seal")
        for relative, expected in seal["sha256"].items():
            if digest(member(run, relative)) != expected:
                raise ValueError(f"OAS category evidence hash mismatch: {relative}")
            seals += 1
        statistics = read(directory / "statistics_input.json")
        if digest(member(package, f"models/{dataset}/{category}/statistics.npz")) != statistics["statistics_sha256"]:
            raise ValueError("Reused original model statistics digest mismatch")
        inventory = read_lines(directory / "samples.jsonl")
        samples += len(inventory)
    if samples != 3887 or len(read_lines(run / "results/category_metrics.jsonl")) != 108:
        raise ValueError("Full sample or metric-row coverage mismatch")
    return {"status": "verified", "categories": len(tasks), "test_images": samples, "category_evidence_hashes": seals,
            "model_statistic_hashes": 27, "all_oas_cache_and_mask_hashes_verified": True}, run, tasks


def verify_inference(package: Path, run: Path, sample_count: int) -> list[dict[str, Any]]:
    if sample_count < 0 or sample_count > 54:
        raise ValueError("Inference sample count must be in 0..54")
    if not sample_count:
        return []
    import numpy as np
    from industrial_anomaly_lab.quick_features import FeatureExtractor, load_statistics, render_maps
    from industrial_anomaly_lab.oas_features import prepare_ml_scorer, prepare_oas_scorer

    original = read(package / "run_config.json")
    protocol = read(run / "run_config.json")["protocol"]
    backbone = original["backbone"]
    weight = member(package, "assets/weights/" + Path(backbone["weight_path"]).name)
    extractor = FeatureExtractor(weight, expected_sha256=backbone["sha256"], seed=int(protocol["seed"]), dimensions=100, device="cpu", cpu_threads=2)
    selected = read(package / EXTENSION / "visualization_manifest.json")["samples"][:sample_count]
    output, scorers, active = [], {}, None
    settings = {key: protocol[key] for key in ("regularization", "sigma", "model_size", "output_size", "position_chunk_size", "cpu_threads")}
    if settings["cpu_threads"] != 2:
        raise ValueError("Delivered verification requires the frozen two-thread protocol")
    for sample in selected:
        category = (sample["dataset"], sample["category"])
        if category != active:
            scorers.clear()
            gc.collect()
            stats = load_statistics(member(package, f"models/{category[0]}/{category[1]}/statistics.npz"))
            if not np.array_equal(stats.channel_indices, extractor.channel_indices):
                raise ValueError("Frozen feature channel mismatch")
            scorers = {"ml": prepare_ml_scorer(stats, **settings), "oas": prepare_oas_scorer(stats, **settings)}
            saved_alpha = np.load(member(run, f"categories/{category[0]}/{category[1]}/oas_alpha.npy"), allow_pickle=False)
            if not np.array_equal(saved_alpha, scorers["oas"].alphas):
                raise ValueError("OAS training-only coefficients do not reproduce exactly")
            del stats
            gc.collect()
            active = category
        image_path = member(package, sample["source_image_copy"])
        if digest(image_path) != sample["source_image_sha256"]:
            raise ValueError("Selected original source-image hash mismatch")
        features = extractor(image_path)
        item = {"dataset": category[0], "category": category[1], "sample_id": sample["sample_id"], "variants": {}}
        for estimator in ("ml", "oas"):
            old = np.load(member(run, sample["raw_maps"][estimator]["path"]), allow_pickle=False)
            regenerated = np.asarray(render_maps(scorers[estimator].distance_maps(features), sigma=float(protocol["sigma"]), output_size=int(protocol["output_size"])), dtype=np.float32).reshape(old.shape)
            passed = bool(np.allclose(regenerated, old, atol=1e-5, rtol=1e-6))
            item["variants"][estimator] = {"passed": passed, "maximum_absolute_error": float(np.max(np.abs(regenerated - old))), "atol": 1e-5, "rtol": 1e-6}
            if not passed:
                raise ValueError(f"Selected inference mismatch: {sample['sample_id']}/{estimator}")
        item["status"] = "verified"
        output.append(item)
        print(json.dumps({"inference_completed": len(output), "sample_id": sample["sample_id"]}), flush=True)
    scorers.clear()
    del extractor
    gc.collect()
    return output


def verify_metrics(run: Path, tasks: list[tuple[str, str]], count: int, work: Path) -> list[dict[str, Any]]:
    if count < 0 or count > 27:
        raise ValueError("Metrics category count must be in 0..27")
    if not count:
        return []
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    from evaluate_cached_top1p import aggregate_scores

    output = []
    for dataset, category in tasks[:count]:
        samples = read_lines(run / f"categories/{dataset}/{category}/samples.jsonl")
        rows = {r["variant"]: r for r in read(run / f"categories/{dataset}/{category}/metrics.json")}
        masks = [member(run, r["mask_path"]) for r in samples]
        labels = [int(r["label"]) for r in samples]
        for estimator in ("ml", "oas"):
            maps = [member(run, r["raw_maps"][estimator]) for r in samples]
            measured = evaluate_from_files(maps, masks, labels, temp_root=work)
            scores = [aggregate_scores(np.load(path, allow_pickle=False)) for path in maps]
            if any(k != 655 for _, _, k in scores):
                raise ValueError("Fixed top1-percent pixel count mismatch")
            top = float(roc_auc_score(labels, [v[1] for v in scores]))
            for score_method in ("max", "top1p"):
                regenerated = dict(measured)
                if score_method == "top1p":
                    regenerated["image_auroc"] = top
                variant = estimator + "_" + score_method
                errors = {key: abs(float(regenerated[key]) - float(rows[variant][key])) for key in METRICS}
                if any(v > 1e-12 for v in errors.values()):
                    raise ValueError(f"Cached metric replay mismatch: {dataset}/{category}/{variant}: {errors}")
                output.append({"dataset": dataset, "category": category, "variant": variant, "status": "verified", "absolute_errors": errors})
        print(json.dumps({"metrics_completed_category": category, "dataset": dataset, "metric_rows_verified": len(output)}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--inference-samples", type=int, default=2)
    parser.add_argument("--metrics-categories", type=int, default=0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    package = args.package_root.resolve()
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.gettempdir()) / "oas-delivery-metric-replay"
    if work.is_relative_to(package) or args.receipt and args.receipt.resolve().is_relative_to(package):
        parser.error("Work directory and receipt must remain outside the immutable delivery folder")
    work.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(package / "code/src"), str(package / "code/scripts")]
    started = time.monotonic()
    try:
        integrity, run, tasks = verify_cache(package)
        print(json.dumps({"cache_integrity": integrity}), flush=True)
        inference = verify_inference(package, run, args.inference_samples)
        metric_rows = verify_metrics(run, tasks, args.metrics_categories, work)
        result = {"status": "verified", "cache_integrity": integrity, "inference": inference, "metric_replay": metric_rows,
                  "elapsed_seconds": time.monotonic() - started, "training_performed": False, "network_access_required": False, "gpu_used": False, "cpu_threads": 2}
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(json.dumps({"status": "verification_failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
