"""Frozen four-group full-dataset experiment for normal content calibration."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from run_score_calibration import read, lines, sha, write, now, verify_seal
from run_route1_resource_trial import Monitor
from context_result_summary import summarize
from dataclasses import replace


def seal_inputs(config, cfg, tasks):
    original = ROOT / cfg["source_run"]
    normal = ROOT / cfg["normal_context_run"]
    paths = [config, Path(__file__), ROOT / "scripts/context_result_summary.py", ROOT / "scripts/run_score_calibration.py",
        ROOT / "scripts/run_route1_resource_trial.py", ROOT / "src/industrial_anomaly_lab/context_calibration.py",
        ROOT / "src/industrial_anomaly_lab/subspace_features.py", ROOT / "src/industrial_anomaly_lab/quick_metrics.py",
        ROOT / "src/industrial_anomaly_lab/metrics.py", ROOT / "configs/route1_controls_20261014.json",
        normal / "frozen_protocol.json", normal / "audit_receipt.json", normal / "decision.json"]
    for d, c in tasks:
        nc = normal / "categories" / d / c
        oc = original / "categories" / d / c
        paths += [nc / name for name in ("normal_context.npz", "normal_records.json", "completion.json", "diagnostics.json")]
        paths += [oc / name for name in ("models/official_cpu_two_pass.npz", "models/current_cpu_one_pass.npz", "samples.jsonl", "metrics.json")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}



def load_aligned_single_pass(directory):
    import numpy as np
    from industrial_anomaly_lab.subspace_features import load_subspace_model
    reference = load_subspace_model(directory / "official_cpu_two_pass.npz")
    current = load_subspace_model(directory / "current_cpu_one_pass.npz")
    if current.k != reference.k or current.count != reference.count:
        raise ValueError("Single-pass/reference PCA rank or input count differs")
    dots = np.sum(current.components * reference.components, axis=0)
    if np.min(np.abs(dots)) < 1 - 1e-8:
        raise ValueError("PCA axes cannot be matched to the frozen content frame")
    signs = np.where(dots < 0, -1., 1.)
    return replace(current, components=current.components * signs,
        metadata={**current.metadata, "content_axis_orientation": "column sign aligned to frozen prototype reference", "actual_fit_variant": "current_cpu_one_pass"})


def save_model(path, model):
    import numpy as np
    np.savez_compressed(path, centers=model.centers, tails=model.tails, global_tail=model.global_tail,
        global_center=model.global_center, global_scale=model.global_scale)


def load_model(path):
    import numpy as np
    from industrial_anomaly_lab.context_calibration import ContextCalibration
    with np.load(path, allow_pickle=False) as a:
        return ContextCalibration(a["centers"].copy(), a["tails"].copy(), float(a["global_tail"]),
            float(a["global_center"]), float(a["global_scale"]))


def normal_thresholds(model, cache, variant):
    import numpy as np
    part = cache["statistics_partition"].astype(bool)
    held = np.stack([model.apply(r, z, variant) for r, z in zip(cache["residual_grid"][~part], cache["latent"][~part], strict=True)])
    return {"pixel_threshold": float(np.quantile(held, .99, method="higher")),
        "image_threshold": float(np.quantile(held.max(axis=(1, 2)), .95, method="higher"))}


def score_category(run, cfg, category, model, samples, frames, monitor):
    import numpy as np
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    d, c = samples[0]["dataset"], samples[0]["category"]
    output = run / "categories" / d / c
    masks = [ROOT / cfg["source_run"] / s["mask_path"] for s in samples]
    labels = [s["label"] for s in samples]
    rows = []
    with np.load(ROOT / cfg["normal_context_run"] / "categories" / d / c / "normal_context.npz", allow_pickle=False) as archive:
        cache = {k: archive[k].copy() for k in archive.files}
    for v in cfg["variants"]:
        monitor.set_stage(f"score:{d}/{c}/{v}")
        thresholds = normal_thresholds(model, cache, v)
        staging = output / "evaluation_tmp" / v
        staging.mkdir(parents=True, exist_ok=True)
        map_files = []
        counters = {k: 0 for k in ("normal_pixels", "normal_fp", "background_pixels", "background_fp", "foreground_pixels", "foreground_tp", "normal_images", "normal_image_fp", "anomaly_images", "anomaly_image_tp")}
        timings = []
        for j, (sample, frame) in enumerate(zip(samples, frames, strict=True)):
            monitor.check()
            with np.load(output / frame["cache"], allow_pickle=False) as a:
                raw, z = a["residual_grid"], a["descriptor"]
            tick = time.perf_counter()
            scores = model.apply(raw, z, v)
            timings.append(time.perf_counter() - tick)
            mask = np.load(masks[j], allow_pickle=False).astype(bool)
            cutoff = thresholds["pixel_threshold"]
            counters["background_pixels"] += int((~mask).sum())
            counters["background_fp"] += int(np.count_nonzero(scores[~mask] > cutoff))
            counters["foreground_pixels"] += int(mask.sum())
            counters["foreground_tp"] += int(np.count_nonzero(scores[mask] > cutoff))
            if sample["label"] == 0:
                counters["normal_pixels"] += scores.size
                counters["normal_fp"] += int(np.count_nonzero(scores > cutoff))
                counters["normal_images"] += 1
                counters["normal_image_fp"] += int(scores.max() > thresholds["image_threshold"])
            else:
                counters["anomaly_images"] += 1
                counters["anomaly_image_tp"] += int(scores.max() > thresholds["image_threshold"])
            if shutil.disk_usage(run).free < cfg["resource_policy"]["minimum_disk_gib"] * 1024**3:
                raise RuntimeError("Disk reserve reached during staged scoring")
            path = staging / f"{j:06d}.npy"
            np.save(path, scores, allow_pickle=False)
            map_files.append(path)
        monitor.set_stage(f"metrics:{d}/{c}/{v}")
        metric = evaluate_from_files(map_files, masks, labels, temp_root=run / "metric_tmp")
        if v == "global":
            delta = max(abs(metric[k] - rows[0][k]) for k in metric)
            if delta > 1e-10:
                raise ValueError(f"Global affine ranking changed: {delta}")
        baseline_apply = rows[0]["apply_median_ms"] / 1000 if rows else float(np.median(timings))
        forward = float(np.median([f["forward_seconds"] + f["residual_seconds"] for f in frames]))
        context = float(np.median([f["descriptor_seconds"] for f in frames]))
        added = max(0., float(np.median(timings)) - baseline_apply) + (context if v == "conditional_tail" else 0.)
        row = {"dataset": d, "category": c, "variant": v, "test_images": len(samples), **metric,
            "normal_pixel_fpr": counters["normal_fp"] / counters["normal_pixels"],
            "background_pixel_fpr": counters["background_fp"] / counters["background_pixels"],
            "anomaly_pixel_recall": counters["foreground_tp"] / counters["foreground_pixels"],
            "normal_image_fpr": counters["normal_image_fp"] / counters["normal_images"],
            "anomaly_image_tpr": counters["anomaly_image_tp"] / counters["anomaly_images"],
            **thresholds, "raw_counts": counters, "apply_median_ms": float(np.median(timings) * 1000),
            "apply_p95_ms": float(np.quantile(timings, .95) * 1000), "paired_gpu_extra_fraction": added / (forward + baseline_apply),
            "forward_and_residual_median_ms": forward * 1000, "context_descriptor_median_ms": context * 1000,
            "calibration_parameter_bytes": model.parameter_bytes if v == "conditional_tail" else (16 if v == "global" else 8 if v == "conditioning_removed" else 0)}
        rows.append(row)
        write(output / f"metrics_{v}.json", row)
        resolved = staging.resolve()
        if not resolved.is_relative_to(run.resolve()) or "evaluation_tmp" not in resolved.parts:
            raise ValueError("Unsafe cleanup path")
        for path in map_files:
            if path.resolve().parent != resolved:
                raise ValueError("Unexpected generated map path")
            path.unlink()
        staging.rmdir()
        print(f"metrics {d}/{c}/{v}: I={metric['image_auroc']:.5f} AP={metric['pixel_ap']:.5f} PRO={metric['aupro_0_30']:.5f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/context_calibration_single_pass_20261015.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = read(args.config)
    run = args.output_dir.resolve()
    normal = ROOT / cfg["normal_context_run"]
    original = ROOT / cfg["source_run"]
    normal_frozen = read(normal / "frozen_protocol.json")
    verify_seal(normal_frozen["sha256"])
    aliases = {"normal_pilot_protocol": normal / "frozen_protocol.json", "normal_pilot_audit": normal / "audit_receipt.json", "normal_pilot_decision": normal / "decision.json"}
    for rel, expected in cfg["source_sha256"].items():
        if sha(aliases.get(rel, ROOT / rel)) != expected:
            raise ValueError("Prepared method source changed")
    if not read(normal / "decision.json")["normal_only_feasibility_passed"]:
        raise ValueError("Normal-only feasibility failed")
    tasks = normal_frozen["tasks"]
    if len(tasks) != 27 or not run.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Invalid matrix or output location")
    if run.exists() and not args.resume:
        raise ValueError("Fresh output or explicit resume required")
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "run.lock"
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "w") as f:
        f.write(str(os.getpid()))
    monitor = Monitor(cfg["resource_policy"])
    started = time.perf_counter()
    try:
        initial = monitor.sample()
        p = cfg["resource_policy"]
        if initial["system_free_ram_bytes"] < p["initial_free_ram_gib"] * 1024**3 or initial["system_free_commit_bytes"] < p["initial_free_commit_gib"] * 1024**3 or shutil.disk_usage(run).free < p["minimum_disk_gib"] * 1024**3:
            raise RuntimeError("Initial resource admission not met")
        source_seal = seal_inputs(args.config, cfg, tasks)
        if (run / "frozen_protocol.json").exists():
            frozen = read(run / "frozen_protocol.json")
            if frozen["sha256"] != source_seal or frozen["protocol"] != cfg:
                raise ValueError("Frozen execution changed")
        else:
            frozen = {"protocol": cfg, "sha256": source_seal, "tasks": tasks, "frozen_at": now(), "before_current_test_evaluation": True}
            write(run / "frozen_protocol.json", frozen)
        monitor.thread.start()
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.context_calibration import quantized_descriptor, fit_context_from_cache
        from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model
        threadpool_limits(4)
        pca_protocol = read(ROOT / "configs/route1_controls_20261014.json")
        monitor.set_stage("load_backbone")
        extractor = LocalDinoExtractor(ROOT / pca_protocol["weights_dir"], expected_checkpoint_sha256=pca_protocol["checkpoint_sha256"],
            expected_artifact_sha256=pca_protocol["weight_files_sha256"], device="cuda:0", cpu_threads=4, max_cuda_memory_gib=2.)
        all_rows = []
        for i, (d, c) in enumerate(tasks):
            verify_seal(frozen["sha256"])
            output = run / "categories" / d / c
            output.mkdir(parents=True, exist_ok=True)
            if (output / "completion.json").exists():
                for rel, expected in read(output / "completion.json")["sha256"].items():
                    if sha(output / rel) != expected:
                        raise ValueError("Completed category changed")
                for frame in read(output / "frames.json"):
                    if sha(output / frame["cache"]) != frame["cache_sha256"]:
                        raise ValueError("Completed frame cache changed")
                all_rows.extend(read(output / "metrics.json"))
                continue
            with np.load(normal / "categories" / d / c / "normal_context.npz", allow_pickle=False) as a:
                normal_cache = {key: a[key].copy() for key in a.files}
            tick = time.perf_counter()
            calibration, diagnostics = fit_context_from_cache(normal_cache, cfg["minimum_group_tokens"], cfg["tail_quantile"])
            diagnostics["calibration_fit_seconds"] = time.perf_counter() - tick
            save_model(output / "calibration.npz", calibration)
            write(output / "fit_diagnostics.json", diagnostics)
            pca = load_aligned_single_pass(original / "categories" / d / c / "models")
            samples = lines(original / "categories" / d / c / "samples.jsonl")
            frame_dir = output / "frames"
            frame_dir.mkdir(exist_ok=True)
            frames = []
            for j, sample in enumerate(samples):
                monitor.set_stage(f"forward:{d}/{c}")
                if shutil.disk_usage(run).free < cfg["resource_policy"]["minimum_disk_gib"] * 1024**3:
                    raise RuntimeError("Disk reserve reached")
                free, _ = torch.cuda.mem_get_info()
                if free < 1024**3:
                    raise RuntimeError("GPU free memory below reserve")
                image = Path(sample["image_path"]).resolve(strict=True)
                if not image.is_relative_to((ROOT / "data").resolve()) or sha(image) != sample["image_sha256"]:
                    raise ValueError("Test image content/path drift")
                tick = time.perf_counter()
                tokens = extractor(image).reshape(-1, 384)
                forward_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                raw = pca.reconstruction_scores(tokens).reshape(32, 32)
                residual_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                z = quantized_descriptor(tokens, pca, cfg["context_dim"])
                descriptor_seconds = time.perf_counter() - tick
                prior = original / sample["raw_maps"]["official_cpu_two_pass"]
                if sha(prior) != sample["raw_map_sha256"]["official_cpu_two_pass"] or sha(original / sample["mask_path"]) != sample["mask_sha256"]:
                    raise ValueError("Original CPU map/mask drift")
                new_baseline = calibration.apply(raw, z, "baseline")
                old_baseline = np.load(prior, allow_pickle=False)
                max_error = float(np.max(np.abs(new_baseline - old_baseline)))
                if not np.allclose(new_baseline, old_baseline, atol=1e-4, rtol=3e-5):
                    raise ValueError(f"GPU/CPU baseline equivalence failed: {d}/{c}/{j} {max_error}")
                cache_path = frame_dir / f"{j:06d}.npz"
                np.savez_compressed(cache_path, residual_grid=raw, descriptor=z)
                frames.append({"sample_id": sample["sample_id"], "cache": cache_path.relative_to(output).as_posix(), "cache_sha256": sha(cache_path),
                    "forward_seconds": forward_seconds, "residual_seconds": residual_seconds, "descriptor_seconds": descriptor_seconds,
                    "cpu_baseline_max_abs_error": max_error, "image_sha256": sample["image_sha256"]})
                write(output / "frames.json", frames)
                write(run / "run_state.json", {"status": "running", "pid": os.getpid(), "completed_categories": i,
                    "current_category": f"{d}/{c}", "test_image": j + 1, "test_images": len(samples), "updated_at": now()})
            rows = score_category(run, cfg, (d, c), calibration, samples, frames, monitor)
            old_metrics = read(original / "categories" / d / c / "metrics.json")
            original_metric = next(r for r in old_metrics if r["variant"] == "official_cpu_two_pass_max")
            write(output / "cpu_gpu_baseline_comparison.json", {"max_map_abs_error": max(r["cpu_baseline_max_abs_error"] for r in frames),
                "metric_delta_gpu_minus_cpu": {k: rows[0][k] - original_metric[k] for k in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")},
                "notice": "All four groups use the fresh GPU baseline, not old CPU metrics."})
            write(output / "metrics.json", rows)
            files = [output / name for name in ("calibration.npz", "fit_diagnostics.json", "frames.json", "metrics.json", "cpu_gpu_baseline_comparison.json")]
            files += list(output.glob("metrics_*.json"))
            write(output / "completion.json", {"sha256": {p.relative_to(output).as_posix(): sha(p) for p in files}, "finished_at": now()})
            all_rows.extend(rows)
            gc.collect()
        summarize(run, cfg, all_rows)
        verify_seal(frozen["sha256"])
        write(run / "run_state.json", {"status": "completed", "pid": os.getpid(), "completed_categories": 27, "test_images": 3887,
            "metric_rows": 108, "elapsed_seconds": time.perf_counter() - started, "finished_at": now()})
    except BaseException as error:
        write(run / f"failure_{int(time.time())}.json", {"error": repr(error), "traceback": traceback.format_exc(), "at": now()})
        write(run / "run_state.json", {"status": "failed", "pid": os.getpid(), "error": repr(error), "at": now()})
        raise
    finally:
        write(run / f"resources_{int(time.time())}.json", monitor.close())
        lock.unlink()


if __name__ == "__main__":
    main()
