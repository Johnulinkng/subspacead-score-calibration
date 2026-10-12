"""Run a frozen, low-resource PaDiM-style reproduction and ablation matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/quick_reproduction_20261012.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-train-only", action="store_true")
    parser.add_argument("--benchmark-images", type=int, default=16)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    run = args.output_dir.resolve()
    if not run.is_relative_to(ROOT / "runs/quick"):
        raise ValueError("Quick runs must stay below runs/quick")
    if run.exists():
        raise ValueError("Refusing to replace an existing quick run")
    run.mkdir(parents=True)
    (run / "results").mkdir()
    state_path = run / "run_state.json"
    state = {"status": "initializing", "started_at_utc": now(), "pid": os.getpid(), "completed_categories": 0,
             "scientific_claim_eligible": False, "test_content_accessed": False, "current_task": None}
    write_json(state_path, state)
    from industrial_anomaly_lab.quick_features import FeatureExtractor, fit_statistics, prepare_scorer, render_maps, save_statistics
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    from industrial_anomaly_lab.data_protocols import build_test_manifest, validate_train_layout
    import numpy as np
    from PIL import Image

    source_paths = [Path(__file__), ROOT / "src/industrial_anomaly_lab/quick_features.py",
                    ROOT / "src/industrial_anomaly_lab/quick_metrics.py", ROOT / "src/industrial_anomaly_lab/data_protocols.py"]
    frozen_sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths}
    frozen_config_hash = digest(config_path)
    variant_names = list(protocol["variants"])
    config = {"schema_version": 1, "protocol": protocol, "protocol_sha256": frozen_config_hash,
              "source_sha256": frozen_sources, "project_root": str(ROOT), "started_at_utc": now(),
              "weight_path": protocol["model"]["weights"], "weight_sha256": protocol["model"]["weights_sha256"],
              "backbone": {"architecture": "resnet18", "weight_path": protocol["model"]["weights"],
                           "sha256": protocol["model"]["weights_sha256"], "official_url": protocol["model"]["weights_url"]},
              "expected_matrix": {"datasets": {name: d["categories"] for name, d in protocol["datasets"].items()},
                                  "variants": variant_names, "seeds": protocol["seeds"]},
              "samples_index": "results/samples.jsonl", "benchmark_train_only": args.benchmark_train_only,
              "sources": protocol["sources"], "new_work": protocol["new_work"]}
    write_json(run / "run_config.json", config)
    started = time.monotonic()
    try:
        extractor = FeatureExtractor(ROOT / protocol["model"]["weights"],
            expected_sha256=protocol["model"]["weights_sha256"], seed=protocol["seeds"][0],
            dimensions=protocol["model"]["dimensions"], device=protocol["resources"]["device"], cpu_threads=2)
        config["extractor"] = extractor.metadata
        write_json(run / "run_config.json", config)
        for dataset_name, dataset_config in protocol["datasets"].items():
            dataset_root = ROOT / dataset_config["root"]
            for category in dataset_config["categories"]:
                if digest(config_path) != frozen_config_hash or any(digest(ROOT / rel) != h for rel, h in frozen_sources.items()):
                    raise RuntimeError("Frozen quick configuration or execution source changed")
                state.update(status="running", current_task={"dataset": dataset_name, "category": category, "phase": "fit"}, updated_at_utc=now())
                write_json(state_path, state)
                training_paths = validate_train_layout(dataset_root, category)
                if args.benchmark_train_only:
                    training_paths = training_paths[:args.benchmark_images]
                category_started = time.monotonic()
                model_root = run / "models" / dataset_name / category
                model_root.mkdir(parents=True)
                append(run / "results/events.jsonl", {"event": "fit_started", "dataset": dataset_name, "category": category, "train_images": len(training_paths), "time": now()})
                train_rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "size_bytes": p.stat().st_size} for p in training_paths]
                write_json(model_root / "train_manifest.json", train_rows)
                def train_features():
                    for index, path in enumerate(training_paths):
                        feature = extractor(path)
                        if index % 25 == 0 or index + 1 == len(training_paths):
                            append(run / "results/events.jsonl", {"event": "fit_progress", "dataset": dataset_name, "category": category, "images": index + 1, "elapsed_seconds": time.monotonic() - category_started})
                        yield feature
                stats = fit_statistics(train_features(), channel_indices=extractor.channel_indices,
                    metadata={**extractor.metadata, "dataset": dataset_name, "category": category, "train_manifest_sha256": digest(model_root / "train_manifest.json")})
                save_statistics(stats, model_root / "statistics.npz")
                fit_seconds = time.monotonic() - category_started
                scorers = {alpha: prepare_scorer(stats, shrinkage_alpha=alpha) for alpha in (0.0, 0.1)}
                if args.benchmark_train_only:
                    feature = extractor(training_paths[0])
                    distance_maps = {alpha: scorer.distance_maps(feature) for alpha, scorer in scorers.items()}
                    maps = {name: render_maps(distance_maps[v["shrinkage_alpha"]], sigma=v["sigma"], output_size=256)
                        for name, v in protocol["variants"].items()}
                    write_json(run / "benchmark.json", {"training_images": len(training_paths), "fit_seconds": fit_seconds,
                        "total_seconds": time.monotonic() - started, "maps_finite": all(np.isfinite(m).all() for m in maps.values()),
                        "test_content_accessed": False, "claim_eligible": False})
                    state.update(status="benchmark_completed", elapsed_seconds=time.monotonic()-started, updated_at_utc=now())
                    write_json(state_path, state)
                    return 0
                # This is the first point at which the new, frozen exploratory
                # round enumerates test/GT. Fitting above consumes only train/good.
                samples = build_test_manifest(dataset_name, dataset_root, category)
                state.update(test_content_accessed=True, current_task={"dataset": dataset_name, "category": category, "phase": "predict"})
                write_json(state_path, state)
                frozen_inventory = [{**s.to_dict(), "image_sha256": digest(s.image_path),
                                     "mask_sha256": None if s.mask_path is None else digest(s.mask_path)} for s in samples]
                write_json(model_root / "test_inventory.json", frozen_inventory)
                prediction_root = run / "predictions" / dataset_name / category
                mask_root = prediction_root / "masks"
                mask_root.mkdir(parents=True)
                for name in variant_names:
                    (prediction_root / name).mkdir()
                selected_ids = {next((s.sample_id for s in samples if s.label == label), "") for label in (0, 1)}
                map_files = {name: [] for name in variant_names}
                mask_files = []
                for index, sample in enumerate(samples):
                    feature = extractor(sample.image_path)
                    if sample.mask_path is None:
                        mask = np.zeros((256, 256), dtype=bool)
                    else:
                        with Image.open(sample.mask_path) as image:
                            mask = np.asarray(image.convert("L").resize((256,256), Image.Resampling.NEAREST)) > 0
                    mask_path = mask_root / f"{index:06d}.npy"
                    np.save(mask_path, mask, allow_pickle=False)
                    mask_files.append(mask_path)
                    raw_maps = {}
                    distance_maps = {alpha: scorer.distance_maps(feature) for alpha, scorer in scorers.items()}
                    for name, variant in protocol["variants"].items():
                        values = render_maps(distance_maps[variant["shrinkage_alpha"]], sigma=variant["sigma"], output_size=256)
                        values = np.asarray(values, dtype=np.float32).reshape(256,256)
                        if not np.isfinite(values).all():
                            raise FloatingPointError("Non-finite raw anomaly map")
                        path = prediction_root / name / f"{index:06d}.npy"
                        np.save(path, values, allow_pickle=False)
                        map_files[name].append(path)
                        raw_maps[name] = path.relative_to(run).as_posix()
                    append(run / "results/samples.jsonl", {"dataset": dataset_name, "category": category,
                        "seed": protocol["seeds"][0], "sample_id": sample.sample_id, "label": sample.label,
                        "image_path": str(sample.image_path), "mask_path": mask_path.relative_to(run).as_posix(),
                        "raw_maps": raw_maps, "selected": sample.sample_id in selected_ids, "image_sha256": frozen_inventory[index]["image_sha256"]})
                    if (index+1) % 25 == 0:
                        append(run / "results/events.jsonl", {"event": "prediction_progress", "dataset": dataset_name, "category": category, "images": index+1, "time": now()})
                state.update(current_task={"dataset": dataset_name, "category": category, "phase": "metrics"})
                write_json(state_path, state)
                del scorers
                for name in variant_names:
                    metric_started = time.monotonic()
                    metrics = evaluate_from_files(map_files[name], mask_files, [s.label for s in samples])
                    row = {"dataset": dataset_name, "category": category, "variant": name, "seed": protocol["seeds"][0],
                           "status": "completed", "train_images": len(training_paths), "test_images": len(samples),
                           "fit_seconds": fit_seconds, "metrics_seconds": time.monotonic()-metric_started,
                           "category_elapsed_seconds": time.monotonic()-category_started, **metrics}
                    append(run / "results/category_metrics.jsonl", row)
                    print(json.dumps(row), flush=True)
                state["completed_categories"] += 1
                state.update(updated_at_utc=now(), elapsed_seconds=time.monotonic()-started)
                write_json(state_path, state)
                del stats
        state.update(status="completed", current_task=None, finished_at_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
