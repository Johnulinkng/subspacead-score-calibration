"""Frozen full-matrix OAS comparison, reusing sealed normal-training statistics."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_cached_top1p import digest, read_json, read_jsonl, process_memory, aggregate_scores, utc_now


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def checked(root, path, expected):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError(f"Input path or content drift: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oas_comparison_20261012.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-train-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if not run.is_relative_to(ROOT / "runs/quick"):
        raise ValueError("Output must stay under runs/quick")
    if args.resume and args.benchmark_train_only:
        raise ValueError("Benchmark cannot resume")
    if run.exists() != args.resume:
        raise ValueError("Use a fresh directory, or explicitly resume an existing run")
    run.mkdir(parents=True, exist_ok=args.resume)
    (run / "results").mkdir(exist_ok=args.resume)
    protocol = read_json(args.config)
    baseline = (ROOT / protocol["baseline_run"]).resolve()
    original = read_json(baseline / "run_config.json")
    if read_json(baseline / "run_state.json")["status"] != "completed":
        raise ValueError("Baseline must be completed")
    sources = [Path(__file__), ROOT / "scripts/evaluate_cached_top1p.py",
               ROOT / "src/industrial_anomaly_lab/quick_features.py",
               ROOT / "src/industrial_anomaly_lab/oas_features.py",
               ROOT / "src/industrial_anomaly_lab/quick_metrics.py"]
    frozen = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    tasks = [(d, c) for d, v in original["protocol"]["datasets"].items() for c in v["categories"]]
    if len(tasks) != 27 or protocol["expected_test_images"] != 3887:
        raise ValueError("Unexpected frozen matrix")
    metadata_paths = [baseline / "run_config.json", baseline / "run_state.json",
                      baseline / "results/category_metrics.jsonl", baseline / "results/samples.jsonl"]
    for d, c in tasks:
        metadata_paths.extend(baseline / "models" / d / c / p for p in
            ("statistics_integrity.json", "train_manifest.json", "test_inventory.json", "prediction_integrity.jsonl"))
    inputs = {p.relative_to(baseline).as_posix(): digest(p) for p in metadata_paths}
    config = {"schema_version": 1, "protocol": protocol, "protocol_sha256": digest(args.config),
              "source_sha256": frozen, "input_metadata_sha256": inputs, "baseline_root": str(baseline),
              "project_root": str(ROOT), "started_at_utc": utc_now(),
              "benchmark_train_only": args.benchmark_train_only,
              "expected_matrix": {"categories": tasks, "variants": [f"{m}_{s}" for m in ("ml", "oas") for s in ("max", "top1p")]}}
    state = {"status": "initializing", "started_at_utc": utc_now(), "pid": os.getpid(),
             "completed_categories": 0, "test_content_accessed": False, "scientific_claim_eligible": False}
    completed = set()
    if args.resume:
        prior = read_json(run / "run_config.json")
        previous_state = read_json(run / "run_state.json")
        if previous_state["status"] == "completed" or prior["benchmark_train_only"]:
            raise ValueError("Completed run or benchmark cannot resume")
        for k in ("protocol_sha256", "source_sha256", "input_metadata_sha256"):
            if prior[k] != config[k]:
                raise ValueError(f"Resume drift: {k}")
        config = prior
        history = run / "execution_history" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        history.mkdir(parents=True)
        import shutil
        shutil.copy2(run / "run_state.json", history / "run_state.json")
        state.update(started_at_utc=previous_state["started_at_utc"], test_content_accessed=previous_state["test_content_accessed"])
        for d, c in tasks:
            cat = run / "categories" / d / c
            if (cat / "completion.json").exists():
                seal = read_json(cat / "completion.json")
                for rel, h in seal["sha256"].items():
                    checked(run, run / rel, h)
                completed.add((d, c))
            else:
                for partial in (cat, run / "predictions" / d / c):
                    if partial.exists():
                        destination = history / partial.relative_to(run)
                        if not partial.resolve().is_relative_to(run) or not destination.resolve().is_relative_to(run):
                            raise ValueError("Partial archive outside run")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        partial.rename(destination)
    write_json(run / "run_config.json", config)
    state["completed_categories"] = len(completed)
    write_json(run / "run_state.json", state)
    started = time.monotonic()
    try:
        import numpy as np
        from PIL import Image
        from sklearn.metrics import roc_auc_score
        from industrial_anomaly_lab.quick_features import FeatureExtractor, load_statistics, render_maps, select_channels
        from industrial_anomaly_lab.oas_features import prepare_oas_scorer, prepare_ml_scorer
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        extractor = FeatureExtractor(ROOT / original["weight_path"], expected_sha256=original["weight_sha256"],
            seed=protocol["seed"], dimensions=100, device="cpu", cpu_threads=2)
        old_samples = {(s["dataset"], s["category"], s["sample_id"]): s for s in read_jsonl(baseline / "results/samples.jsonl")}
        for d, c in tasks:
            if (d, c) in completed:
                continue
            if digest(args.config) != config["protocol_sha256"] or any(digest(ROOT / rel) != h for rel, h in frozen.items()):
                raise ValueError("Frozen protocol or source changed")
            old_model = baseline / "models" / d / c
            for name in ("statistics_integrity.json", "train_manifest.json", "test_inventory.json", "prediction_integrity.jsonl"):
                checked(baseline, old_model / name, inputs[(old_model / name).relative_to(baseline).as_posix()])
            state.update(status="running", current_task={"dataset": d, "category": c, "phase": "prepare"}, updated_at_utc=utc_now())
            write_json(run / "run_state.json", state)
            cat = run / "categories" / d / c
            cat.mkdir(parents=True)
            seal = read_json(old_model / "statistics_integrity.json")
            stats = load_statistics(checked(baseline, old_model / "statistics.npz", seal["sha256"]))
            train = read_json(old_model / "train_manifest.json")
            if (stats.count != len(train) or stats.metadata["train_manifest_sha256"] != digest(old_model / "train_manifest.json")
                or stats.metadata["checkpoint_sha256"] != original["weight_sha256"] or stats.metadata["seed"] != protocol["seed"]
                or stats.metadata["dataset"] != d or stats.metadata["category"] != c or stats.feature_shape != (100,56,56)
                or not np.array_equal(stats.channel_indices, select_channels(protocol["seed"], 100))):
                raise ValueError("Training statistics identity mismatch")
            train_count = stats.count
            settings = {k: protocol[k] for k in ("regularization", "sigma", "model_size", "output_size", "position_chunk_size", "cpu_threads")}
            scorers = {"ml": prepare_ml_scorer(stats, **settings), "oas": prepare_oas_scorer(stats, **settings)}
            write_json(cat / "scorers.json", {m: s.metadata for m, s in scorers.items()})
            np.save(cat / "oas_alpha.npy", scorers["oas"].alphas, allow_pickle=False)
            write_json(cat / "statistics_input.json", {"original_model": str(old_model), "statistics_sha256": seal["sha256"], "original_seal": seal,
                "train_images": train_count, "train_manifest_sha256": digest(old_model / "train_manifest.json")})
            del stats
            gc.collect()
            if args.benchmark_train_only:
                for item in train[:16]:
                    feature = extractor(checked(ROOT, ROOT / item["path"], item["sha256"]))
                    for s in scorers.values():
                        values = render_maps(s.distance_maps(feature), sigma=4, output_size=256)
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite benchmark map")
                write_json(run / "benchmark.json", {"normal_train_images": min(16, len(train)), "full_statistics_train_images": train_count,
                    "elapsed_seconds": time.monotonic()-started, "test_content_accessed": False, "gpu": False, "cpu_threads": 2, **process_memory()})
                state.update(status="benchmark_completed", elapsed_seconds=time.monotonic()-started, updated_at_utc=utc_now())
                write_json(run / "run_state.json", state)
                return 0
            inventory = read_json(old_model / "test_inventory.json")
            write_json(cat / "test_inventory.json", inventory)
            pred = run / "predictions" / d / c
            for m in ("ml", "oas", "masks"):
                (pred / m).mkdir(parents=True)
            maps = {m: [] for m in scorers}
            scores = {m: [] for m in scorers}
            masks, labels = [], []
            state.update(test_content_accessed=True, current_task={"dataset": d, "category": c, "phase": "predict", "images": 0})
            write_json(run / "run_state.json", state)
            for index, sample in enumerate(inventory):
                old = old_samples[(d,c,sample["sample_id"])]
                image = checked(ROOT / original["protocol"]["datasets"][d]["root"], Path(sample["image_path"]), sample["image_sha256"])
                if sample["mask_path"] is None:
                    mask = np.zeros((256,256), dtype=bool)
                else:
                    source_mask = checked(ROOT / original["protocol"]["datasets"][d]["root"], Path(sample["mask_path"]), sample["mask_sha256"])
                    with Image.open(source_mask) as handle:
                        mask = np.asarray(handle.convert("L").resize((256,256), Image.Resampling.NEAREST)) > 0
                if not np.array_equal(mask, np.load(baseline / old["mask_path"], allow_pickle=False)):
                    raise ValueError("Old saved GT differs from authenticated source GT")
                mask_path = pred / "masks" / f"{index:06d}.npy"
                np.save(mask_path, mask, allow_pickle=False)
                masks.append(mask_path)
                labels.append(sample["label"])
                feature = extractor(image)
                row = {"dataset": d, "category": c, "seed": protocol["seed"], "sample_id": sample["sample_id"],
                    "label": sample["label"], "image_path": str(image), "image_sha256": sample["image_sha256"],
                    "mask_path": mask_path.relative_to(run).as_posix(), "mask_sha256": digest(mask_path),
                    "raw_maps": {}, "raw_map_sha256": {}, "image_scores": {}, "selected": old["selected"]}
                for m, scorer in scorers.items():
                    raw = np.asarray(render_maps(scorer.distance_maps(feature), sigma=4, output_size=256), dtype=np.float32).reshape(256,256)
                    mx, top, count = aggregate_scores(raw)
                    if count != 655:
                        raise ValueError("Top-percent pixel count drift")
                    path = pred / m / f"{index:06d}.npy"
                    np.save(path, raw, allow_pickle=False)
                    maps[m].append(path)
                    scores[m].append((mx, top))
                    row["raw_maps"][m] = path.relative_to(run).as_posix()
                    row["raw_map_sha256"][m] = digest(path)
                    row["image_scores"][m] = {"max": mx, "top1p": top}
                append(cat / "samples.jsonl", row)
                if (index+1) % 25 == 0 or index+1 == len(inventory):
                    state.update(current_task={"dataset": d, "category": c, "phase": "predict", "images": index+1}, updated_at_utc=utc_now())
                    write_json(run / "run_state.json", state)
            del scorers, scorer, feature
            gc.collect()
            state.update(current_task={"dataset": d, "category": c, "phase": "metrics"}, updated_at_utc=utc_now())
            write_json(run / "run_state.json", state)
            rows = []
            for m in ("ml", "oas"):
                metrics = evaluate_from_files(maps[m], masks, labels, temp_root=run / "metric_tmp")
                aurocs = [float(roc_auc_score(labels, np.asarray(scores[m])[:,i])) for i in (0,1)]
                if abs(metrics["image_auroc"]-aurocs[0]) > 1e-12:
                    raise ValueError("Max score/evaluator drift")
                for i, score in enumerate(("max", "top1p")):
                    rows.append({"dataset": d, "category": c, "variant": f"{m}_{score}", "seed": protocol["seed"],
                        "status": "completed", "train_images": train_count, "test_images": len(inventory),
                        **metrics, "image_auroc": aurocs[i], "image_auroc_raw_max": aurocs[0],
                        "image_auroc_top1p_mean": aurocs[1], "score_definition": protocol["score_definitions"][score]})
            write_json(cat / "metrics.json", rows)
            category_files = [p for p in cat.rglob("*") if p.is_file()] + [p for p in pred.rglob("*") if p.is_file()]
            write_json(cat / "completion.json", {"status": "completed", "time": utc_now(),
                "sha256": {p.relative_to(run).as_posix(): digest(p) for p in category_files}})
            state["completed_categories"] += 1
            state.update(elapsed_seconds=time.monotonic()-started, updated_at_utc=utc_now(), **process_memory())
            write_json(run / "run_state.json", state)
            print(json.dumps({"completed": state["completed_categories"], "dataset": d, "category": c, "metrics": rows}), flush=True)
        allrows = [row for d,c in tasks for row in read_json(run / "categories" / d / c / "metrics.json")]
        allsamples = [s for d,c in tasks for s in read_jsonl(run / "categories" / d / c / "samples.jsonl")]
        if len(allrows) != 108 or len(allsamples) != 3887:
            raise ValueError("Incomplete full matrix")
        (run / "results/category_metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in allrows), encoding="utf-8")
        (run / "results/samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in allsamples), encoding="utf-8")
        state.update(status="completed", current_task=None, completed_categories=27, test_images=3887,
            metric_rows=108, elapsed_seconds=time.monotonic()-started, finished_at_utc=utc_now(), **process_memory())
        write_json(run / "run_state.json", state)
        return 0
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started, updated_at_utc=utc_now())
        write_json(run / "run_state.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
