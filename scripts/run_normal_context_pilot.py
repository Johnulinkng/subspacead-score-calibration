"""Normal-only feasibility study for content-conditioned residual envelopes.

No Test image, mask or score is read. Normal features are regenerated with
the pinned extractor, checked against the existing normal residual maps,
and stored as compact, explicitly quantized 16D content descriptors.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_score_calibration import read, sha, write, verify_seal, now
from run_route1_resource_trial import Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if run.exists() or not run.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Use a fresh project run")
    normal_run = ROOT / "runs/quick/score-calibration-20261015-v2"
    frozen_normal = read(normal_run / "frozen_protocol.json")
    verify_seal(frozen_normal["sha256"])
    protocol = {"protocol_id": "normal_content_pilot_20261015_v1", "normal_source": str(normal_run),
        "pca_source": "runs/quick/route1-controls-20261014", "clusters": 8, "latent_components": 16,
        "latent_storage_and_assignment": "Project raw float32 features onto first min(16,rank) PCA axes; divide by sqrt(eigenvalue+1e-6); round to float16 and convert back to float32 before clustering and assignment. Reconstruction residuals remain original float32.",
        "tail_quantile": .95, "tail_floor": "global normal statistics-partition Q95", "seed": 1887537248,
        "kmeans": {"batch_size": 1024, "n_init": 1, "max_iter": 30, "reassignment_ratio": .01},
        "scope": "normal training images only, disjoint statistics/holdout partitions already selected before earlier Test evaluations; no new category selection or parameter search",
        "support_gate": "In each dataset, >=60% of categories with >=2 occupied hot clusters (training tail > global Q95, train and holdout >=256 tokens each) must have lower mean hot-cluster |heldout exceedance - .05| than a global threshold. This is normal-model feasibility, not anomaly-detection effectiveness.",
        "resource_policy": frozen_normal["protocol"]["resource_policy"]}
    run.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src/industrial_anomaly_lab/subspace_features.py", ROOT / "configs/route1_controls_20261014.json",
        normal_run / "frozen_protocol.json", normal_run / "audit_receipt.json"]
    frozen = {"protocol": protocol, "tasks": frozen_normal["tasks"], "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}, "frozen_at": now()}
    write(run / "frozen_protocol.json", frozen)
    monitor = Monitor(protocol["resource_policy"])
    started = time.perf_counter()
    try:
        initial = monitor.sample()
        if initial["system_free_ram_bytes"] < protocol["resource_policy"]["initial_free_ram_gib"] * 1024**3 or initial["system_free_commit_bytes"] < protocol["resource_policy"]["initial_free_commit_gib"] * 1024**3:
            raise RuntimeError("Initial RAM/commit admission not met")
        monitor.thread.start()
        import numpy as np
        import torch
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model, render_subspace_map
        threadpool_limits(4)
        p = read(ROOT / "configs/route1_controls_20261014.json")
        extractor = LocalDinoExtractor(ROOT / p["weights_dir"], expected_checkpoint_sha256=p["checkpoint_sha256"],
            expected_artifact_sha256=p["weight_files_sha256"], device="cuda:0", cpu_threads=4, max_cuda_memory_gib=2.)
        report = []
        for i, (d, c) in enumerate(frozen["tasks"]):
            verify_seal(frozen["sha256"])
            monitor.set_stage(f"normal_context:{d}/{c}")
            nd = normal_run / "normal" / d / c
            old_records = read(nd / "records.json")
            completion = read(nd / "completion.json")
            for rel, digest in completion["sha256"].items():
                if sha(nd / rel) != digest:
                    raise ValueError("Normal-only cache changed")
            model_path = ROOT / protocol["pca_source"] / "categories" / d / c / "models/official_cpu_two_pass.npz"
            model = load_subspace_model(model_path)
            dim = min(protocol["latent_components"], model.k)
            mean = model.mean.astype(np.float32)
            basis = model.components[:, :dim].astype(np.float32)
            scale = np.sqrt(model.eigenvalues[:dim].astype(np.float32) + np.float32(1e-6))
            contexts, residuals = [], []
            max_normal_map_error = 0.
            for j, record in enumerate(old_records):
                monitor.check()
                free, _ = torch.cuda.mem_get_info()
                if free < 1024**3:
                    raise RuntimeError("GPU free-memory reserve reached")
                image = ROOT / record["path"]
                if sha(image) != record["sha256"]:
                    raise ValueError("Normal image content changed")
                features = extractor(image).reshape(-1, 384)
                z = ((features - mean) @ basis) / scale
                if not np.isfinite(z).all() or np.max(np.abs(z)) > np.finfo(np.float16).max:
                    raise ValueError("Invalid content descriptor before quantization")
                contexts.append(z.astype(np.float16))
                raw = model.reconstruction_scores(features).reshape(32, 32)
                residuals.append(raw)
                current_map = render_subspace_map(raw)
                previous_map = np.load(nd / record["raw_map"], allow_pickle=False)
                error = float(np.max(np.abs(current_map - previous_map)))
                max_normal_map_error = max(max_normal_map_error, error)
                if not np.allclose(current_map, previous_map, atol=1e-4, rtol=3e-5):
                    raise ValueError("Regenerated normal residuals differ from pinned reference")
                write(run / "run_state.json", {"status": "running", "pid": os.getpid(), "completed_categories": i,
                    "current_category": f"{d}/{c}", "normal_image": j + 1, "normal_images": len(old_records), "updated_at": now()})
            z = np.stack(contexts)
            r = np.stack(residuals)
            is_fit = np.array([record["partition"] == "statistics" for record in old_records])
            fit_z = z[is_fit].astype(np.float32).reshape(-1, dim)
            hold_z = z[~is_fit].astype(np.float32).reshape(-1, dim)
            fit_r, hold_r = r[is_fit].ravel(), r[~is_fit].ravel()
            kmeans = MiniBatchKMeans(n_clusters=protocol["clusters"], random_state=protocol["seed"], **protocol["kmeans"])
            tick = time.perf_counter()
            kmeans.fit(fit_z)
            cluster_fit_seconds = time.perf_counter() - tick
            fit_labels, hold_labels = kmeans.predict(fit_z), kmeans.predict(hold_z)
            global_tail = max(float(np.quantile(fit_r, protocol["tail_quantile"], method="higher")), 1e-12)
            tails, detail = [], []
            for cluster in range(protocol["clusters"]):
                train_scores, hold_scores = fit_r[fit_labels == cluster], hold_r[hold_labels == cluster]
                tail = float(np.quantile(train_scores, .95, method="higher")) if train_scores.size >= 256 else global_tail
                tail = max(tail, global_tail)
                tails.append(tail)
                eligible = train_scores.size >= 256 and hold_scores.size >= 256 and tail > global_tail
                global_exceed = float(np.mean(hold_scores > global_tail)) if hold_scores.size else None
                conditional_exceed = float(np.mean(hold_scores > tail)) if hold_scores.size else None
                detail.append({"cluster": cluster, "train_tokens": int(train_scores.size), "hold_tokens": int(hold_scores.size),
                    "tail": tail, "eligible_hot_cluster": bool(eligible), "global_holdout_exceedance": global_exceed,
                    "conditional_holdout_exceedance": conditional_exceed})
            eligible = [row for row in detail if row["eligible_hot_cluster"]]
            global_error = float(np.mean([abs(row["global_holdout_exceedance"] - .05) for row in eligible])) if eligible else None
            conditional_error = float(np.mean([abs(row["conditional_holdout_exceedance"] - .05) for row in eligible])) if eligible else None
            supports = len(eligible) >= 2 and conditional_error < global_error
            cat = run / "categories" / d / c
            cat.mkdir(parents=True)
            np.savez_compressed(cat / "normal_context.npz", latent=z, residual_grid=r, statistics_partition=is_fit,
                centers=kmeans.cluster_centers_, tails=np.array(tails, dtype=np.float64), global_tail=np.array(global_tail),
                fit_labels=fit_labels.astype(np.uint8), hold_labels=hold_labels.astype(np.uint8))
            write(cat / "normal_records.json", old_records)
            row = {"dataset": d, "category": c, "normal_images": len(old_records), "content_dim": dim,
                "pca_model_sha256": sha(model_path), "max_regenerated_normal_map_error": max_normal_map_error,
                "cluster_fit_seconds": cluster_fit_seconds, "global_tail": global_tail, "eligible_hot_clusters": len(eligible),
                "hot_group_global_coverage_error": global_error, "hot_group_conditional_coverage_error": conditional_error,
                "normal_only_support": bool(supports), "clusters": detail}
            write(cat / "diagnostics.json", row)
            write(cat / "completion.json", {"sha256": {p.name: sha(p) for p in (cat / "normal_context.npz", cat / "normal_records.json", cat / "diagnostics.json")}})
            report.append(row)
            print(f"normal context {i + 1}/27 {d}/{c}: hot={len(eligible)}, global_error={global_error}, conditional_error={conditional_error}, support={supports}", flush=True)
        summaries = []
        for d in ("mvtec_ad", "visa"):
            selected = [row for row in report if row["dataset"] == d]
            fraction = sum(row["normal_only_support"] for row in selected) / len(selected)
            summaries.append({"dataset": d, "supporting_categories": sum(row["normal_only_support"] for row in selected),
                "categories": len(selected), "support_fraction": fraction, "normal_only_feasibility_passed": fraction >= .6})
        passed = all(row["normal_only_feasibility_passed"] for row in summaries)
        write(run / "normal_diagnostics.json", report)
        write(run / "decision.json", {"normal_only_feasibility_passed": passed, "datasets": summaries,
            "anomaly_detection_effectiveness": "not evaluated", "goal_achieved": False,
            "next": "Freeze a full paired four-group Test protocol before evaluating effectiveness" if passed else "Do not proceed to Test benchmark without a normal-data-supported mechanism",
            "no_test_content_accessed": True})
        verify_seal(frozen["sha256"])
        write(run / "run_state.json", {"status": "completed", "pid": os.getpid(), "completed_categories": 27,
            "normal_images": sum(row["normal_images"] for row in report), "no_test_content_accessed": True,
            "elapsed_seconds": time.perf_counter() - started, "finished_at": now()})
    except BaseException as error:
        write(run / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(), "at": now()})
        write(run / "run_state.json", {"status": "failed", "pid": os.getpid(), "error": repr(error), "at": now()})
        raise
    finally:
        write(run / "resources.json", monitor.close())


if __name__ == "__main__":
    main()
