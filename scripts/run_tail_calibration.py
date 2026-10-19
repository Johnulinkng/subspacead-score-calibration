"""Normal-tail-protected score calibration, frozen before full Test evaluation."""
from __future__ import annotations
import argparse
import csv
import gc
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from run_score_calibration import read, lines, sha, write, now, verify_seal
from run_route1_resource_trial import Monitor


def freeze_inputs(config, source, normals, tasks):
    files = [config, Path(__file__), ROOT / "src/industrial_anomaly_lab/tail_calibration.py",
        ROOT / "src/industrial_anomaly_lab/quick_metrics.py", ROOT / "src/industrial_anomaly_lab/metrics.py",
        ROOT / "scripts/run_score_calibration.py", ROOT / "scripts/run_route1_resource_trial.py",
        normals / "frozen_protocol.json", normals / "audit_receipt.json", normals / "category_metrics.json",
        normals / "decision.json", ROOT / "artifacts/audits/normal-calibration-diagnosis-20261015.json"]
    for d, c in tasks:
        normal = normals / "normal" / d / c
        cat = normals / "categories" / d / c
        files += [normal / "completion.json", normal / "records.json", normal / "selection.json",
                  cat / "maps_global.json", cat / "calibration.npz",
                  source / "categories" / d / c / "samples.jsonl",
                  source / "categories" / d / c / "metrics.json"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def operating_rates(model, variant, held, samples, source, base_variant, monitor, map_sink=None):
    import numpy as np
    normal_scores = np.stack([model.apply(x, variant) for x in held])
    pixel_cut = float(np.quantile(normal_scores, .99, method="higher"))
    image_cut = float(np.quantile(normal_scores.max(axis=(1, 2)), .95, method="higher"))
    counts = {k: 0 for k in ("normal_pixels", "normal_fp", "background_pixels", "background_fp",
        "foreground_pixels", "foreground_tp", "normal_images", "normal_image_fp", "anomaly_images", "anomaly_image_tp")}
    timings, manifest = [], []
    identity_max_error = 0.
    amplified_count = 0
    for j, sample in enumerate(samples):
        monitor.check()
        raw = np.load(source / sample["raw_maps"][base_variant], allow_pickle=False)
        start = time.perf_counter()
        scores = model.apply(raw, variant)
        timings.append(time.perf_counter() - start)
        mask = np.load(source / sample["mask_path"], allow_pickle=False).astype(bool)
        counts["background_pixels"] += int((~mask).sum())
        counts["background_fp"] += int(np.count_nonzero(scores[~mask] > pixel_cut))
        counts["foreground_pixels"] += int(mask.sum())
        counts["foreground_tp"] += int(np.count_nonzero(scores[mask] > pixel_cut))
        if sample["label"] == 0:
            counts["normal_pixels"] += scores.size
            counts["normal_fp"] += int(np.count_nonzero(scores > pixel_cut))
            counts["normal_images"] += 1
            counts["normal_image_fp"] += int(scores.max() > image_cut)
        else:
            counts["anomaly_images"] += 1
            counts["anomaly_image_tp"] += int(scores.max() > image_cut)
        if variant == "local_guarded":
            high = raw >= model.envelope
            if high.any():
                identity_max_error = max(identity_max_error, float(np.max(np.abs(scores[high] - raw[high]))))
            amplified_count += int(np.count_nonzero(scores > raw))
        if map_sink is not None:
            item = map_sink(j, sample, scores)
            manifest.append(item)
    rates = {"normal_pixel_fpr": counts["normal_fp"] / counts["normal_pixels"],
        "background_pixel_fpr": counts["background_fp"] / counts["background_pixels"],
        "anomaly_pixel_recall": counts["foreground_tp"] / counts["foreground_pixels"],
        "normal_image_fpr": counts["normal_image_fp"] / counts["normal_images"],
        "anomaly_image_tpr": counts["anomaly_image_tp"] / counts["anomaly_images"],
        "pixel_threshold": pixel_cut, "image_threshold": image_cut,
        "apply_median_ms": float(np.median(timings) * 1000), "apply_p95_ms": float(np.quantile(timings, .95) * 1000),
        "raw_counts": counts, "protected_high_score_max_error": identity_max_error,
        "amplified_pixel_count": amplified_count}
    return rates, manifest


def evaluate(run, cfg, frozen, source, normals, tasks, monitor):
    import numpy as np
    from industrial_anomaly_lab.tail_calibration import VARIANTS, fit_tail_calibration, save_tail_calibration
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    predecessor = {(r["dataset"], r["category"], r["variant"]): r for r in read(normals / "category_metrics.json")}
    all_rows = []
    for i, (d, c) in enumerate(tasks):
        verify_seal(frozen["sha256"])
        monitor.set_stage(f"calibration:{d}/{c}")
        cat = run / "categories" / d / c
        cat.mkdir(parents=True, exist_ok=True)
        if (cat / "completion.json").exists():
            for rel, expected in read(cat / "completion.json")["sha256"].items():
                if sha(cat / rel) != expected:
                    raise ValueError("Completed category changed")
            for v in ("local_guarded", "protection_removed"):
                for r in read(cat / f"maps_{v}.json"):
                    if sha(cat / r["file"]) != r["sha256"]:
                        raise ValueError("Completed compressed prediction changed")
            all_rows.extend(read(cat / "metrics.json"))
            print(f"cached {i + 1}/27 {d}/{c}", flush=True)
            continue
        nd = normals / "normal" / d / c
        for rel, expected in read(nd / "completion.json")["sha256"].items():
            if sha(nd / rel) != expected:
                raise ValueError("Normal map cache mismatch")
        records = read(nd / "records.json")
        stats = np.stack([np.load(nd / r["raw_map"], allow_pickle=False) for r in records if r["partition"] == "statistics"])
        held = np.stack([np.load(nd / r["raw_map"], allow_pickle=False) for r in records if r["partition"] == "threshold"])
        tick = time.perf_counter()
        model = fit_tail_calibration(stats, tail_quantile=cfg["tail_quantile"], smoothing_sigma=cfg["smoothing_sigma"])
        fit_seconds = time.perf_counter() - tick
        save_tail_calibration(cat / "calibration.npz", model)
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        for sample in samples:
            for p, expected in ((sample["raw_maps"][cfg["baseline_model"]], sample["raw_map_sha256"][cfg["baseline_model"]]), (sample["mask_path"], sample["mask_sha256"])):
                if sha(source / p) != expected:
                    raise ValueError("Baseline Test map/mask mismatch")
        old_global = read(normals / "categories" / d / c / "maps_global.json")
        for j, (record, sample) in enumerate(zip(old_global, samples, strict=True)):
            path = normals / "categories" / d / c / record["file"]
            if sha(path) != record["sha256"] or record["sample_id"] != sample["sample_id"]:
                raise ValueError("Reused global map mismatch")
            raw = np.load(source / sample["raw_maps"][cfg["baseline_model"]], allow_pickle=False)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), model.apply(raw, "global"))
        rows = []
        for variant in VARIANTS:
            monitor.set_stage(f"evaluation:{d}/{c}/{variant}")
            current_files = []
            if variant in ("baseline", "global"):
                def sink(j, sample, scores):
                    return {"sample_id": sample["sample_id"], "reused": True,
                        "source": str(source / sample["raw_maps"][cfg["baseline_model"]]) if variant == "baseline" else str(normals / "categories" / d / c / old_global[j]["file"])}
            else:
                staging = cat / "evaluation_tmp" / variant
                staging.mkdir(parents=True, exist_ok=True)
                target = cat / "maps" / variant
                target.mkdir(parents=True, exist_ok=True)
                def sink(j, sample, scores):
                    if shutil.disk_usage(run).free < cfg["resource_policy"]["minimum_disk_gib"] * 1024**3:
                        raise RuntimeError("Disk reserve reached; no input/result deletion performed")
                    temporary = staging / f"{j:06d}.npy"
                    np.save(temporary, scores, allow_pickle=False)
                    current_files.append(temporary)
                    compressed = target / f"{j:06d}.npz"
                    np.savez_compressed(compressed, scores=scores)
                    return {"sample_id": sample["sample_id"], "file": compressed.relative_to(cat).as_posix(), "sha256": sha(compressed), "shape": list(scores.shape), "dtype": str(scores.dtype)}
            rates, manifest = operating_rates(model, variant, held, samples, source, cfg["baseline_model"], monitor, sink)
            metric_keys = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
            if variant in ("baseline", "global"):
                old = predecessor[d, c, variant]
                metrics = {k: old[k] for k in metric_keys}
                for k in ("normal_pixel_fpr", "anomaly_pixel_recall", "normal_image_fpr", "anomaly_image_tpr", "pixel_threshold", "image_threshold"):
                    if rates[k] != old[k]:
                        raise ValueError("Reused baseline/global operating point changed")
            else:
                metrics = evaluate_from_files(current_files, [source / s["mask_path"] for s in samples], [s["label"] for s in samples], temp_root=run / "metric_tmp")
                # Only generated uncompressed staging files are removed. Full
                # outputs persist losslessly; resolved paths stay in this run.
                resolved = staging.resolve()
                if not resolved.is_relative_to(run.resolve()) or "evaluation_tmp" not in resolved.parts:
                    raise ValueError("Unsafe staging cleanup path")
                for p in current_files:
                    if p.resolve().parent != resolved:
                        raise ValueError("Unexpected staging file")
                    p.unlink()
                staging.rmdir()
            row = {"dataset": d, "category": c, "variant": variant, "test_images": len(samples), **metrics, **rates,
                "calibration_fit_seconds": fit_seconds if variant == "local_guarded" else None,
                "calibration_parameter_bytes": model.parameter_bytes if variant in ("local_guarded", "protection_removed") else (16 if variant == "global" else 0)}
            if variant == "local_guarded" and (rates["amplified_pixel_count"] or rates["protected_high_score_max_error"]):
                raise ValueError("Candidate violates bounded suppression/high-tail identity")
            rows.append(row)
            write(cat / f"maps_{variant}.json", manifest)
            write(cat / f"metrics_{variant}.json", row)
            print(f"metrics {i + 1}/27 {d}/{c}/{variant}: I={metrics['image_auroc']:.5f} AP={metrics['pixel_ap']:.5f} PRO={metrics['aupro_0_30']:.5f}", flush=True)
        write(cat / "metrics.json", rows)
        paths = [cat / "calibration.npz", cat / "metrics.json"] + list(cat.glob("maps_*.json")) + list(cat.glob("metrics_*.json"))
        write(cat / "completion.json", {"sha256": {p.relative_to(cat).as_posix(): sha(p) for p in paths}, "finished_at": now()})
        all_rows.extend(rows)
        write(run / "run_state.json", {"status": "running", "pid": os.getpid(), "completed_categories": i + 1, "updated_at": now()})
        gc.collect()
    finish(run, cfg, all_rows, normals)


def finish(run, cfg, rows, normals):
    import numpy as np
    from industrial_anomaly_lab.tail_calibration import VARIANTS
    keys = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30", "normal_pixel_fpr", "background_pixel_fpr", "anomaly_pixel_recall", "normal_image_fpr", "anomaly_image_tpr")
    macro = []
    for d, expected in (("mvtec_ad", 15), ("visa", 12)):
        for v in VARIANTS:
            subset = [r for r in rows if r["dataset"] == d and r["variant"] == v]
            if len(subset) != expected:
                raise ValueError("Incomplete matrix")
            macro.append({"dataset": d, "variant": v, "categories": len(subset), **{k: float(np.mean([r[k] for r in subset])) for k in keys}})
    a = cfg["acceptance"]
    checks = []
    for d in ("mvtec_ad", "visa"):
        get = lambda v: next(r for r in macro if (r["dataset"], r["variant"]) == (d, v))
        base, candidate, ablation = get("baseline"), get("local_guarded"), get("protection_removed")
        pairs = [(r, next(b for b in rows if (b["dataset"], b["category"], b["variant"]) == (d, r["category"], "baseline"))) for r in rows if (r["dataset"], r["variant"]) == (d, "local_guarded")]
        gains = {k: [r[k] - b[k] for r, b in pairs] for k in ("pixel_ap", "aupro_0_30")}
        stable = {k: candidate[k] - base[k] >= a["localization_gain_minimum"] and np.mean(np.array(v) > 0) >= a["category_improvement_fraction_minimum"] and np.median(v) > a["paired_median_gain_minimum"] for k, v in gains.items()}
        false_gains = [b["normal_pixel_fpr"] - r["normal_pixel_fpr"] for r, b in pairs]
        fpr = (candidate["normal_pixel_fpr"] <= base["normal_pixel_fpr"] * (1 - a["false_positive_relative_reduction_minimum"]) and candidate["anomaly_pixel_recall"] >= base["anomaly_pixel_recall"] - a["pixel_recall_maximum_drop"] and candidate["anomaly_image_tpr"] >= base["anomaly_image_tpr"] - a["image_tpr_maximum_drop"] and np.mean(np.array(false_gains) > 0) >= a["category_improvement_fraction_minimum"] and np.median(false_gains) > 0)
        checks.append({"dataset": d, "image_noninferiority": candidate["image_auroc"] >= base["image_auroc"] - a["image_auroc_maximum_drop"], "pixel_ap_stable_gain": bool(stable["pixel_ap"]), "aupro_stable_gain": bool(stable["aupro_0_30"]), "false_positive_gain_without_recall_loss": bool(fpr), "delta": {k: candidate[k] - base[k] for k in keys}, "candidate_minus_ablation": {k: candidate[k] - ablation[k] for k in keys}, "category_gains": {k: {"positive": int(np.sum(np.array(v) > 0)), "total": len(v), "median": float(np.median(v))} for k, v in gains.items()}})
    candidate_rows = [r for r in rows if r["variant"] == "local_guarded"]
    cpu_forward = read(normals / "device_equivalence.json")["cpu_forward_median_seconds"]
    overhead = max(r["apply_p95_ms"] for r in candidate_rows) / (cpu_forward * 1000)
    parameters = max(r["calibration_parameter_bytes"] for r in candidate_rows)
    common_branch = any(all(r[k] for r in checks) for k in ("pixel_ap_stable_gain", "aupro_stable_gain", "false_positive_gain_without_recall_loss"))
    supported = all(r["image_noninferiority"] for r in checks) and common_branch and overhead <= a["apply_overhead_fraction_of_measured_cpu_forward_maximum"] and parameters <= a["calibration_parameter_max_mib"] * 1024**2
    decision = {"status": "supported" if supported else "rejected", "goal_achieved": False, "effectiveness_gate_passed": bool(supported), "completion_notice": "An independent full replay and resource/component audit are required before goal completion, even if this effectiveness gate passes.", "checks": checks, "same_gain_branch_both_datasets": common_branch, "apply_p95_over_measured_cpu_forward": overhead, "parameter_max_bytes": parameters, "test_notice": cfg["test_notice"], "normal_fit_notice": cfg["normal_fit_notice"], "novelty_notice": cfg["novelty_notice"]}
    write(run / "decision.json", decision)
    write(run / "dataset_macro_metrics.json", macro)
    write(run / "category_metrics.json", rows)
    for name, values in (("dataset_macro_metrics", macro), ("category_metrics", rows)):
        with (run / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    text = ["保护异常上尾的正常波动抑制", "", "本轮效果判断：" + ("通过预先固定的效果要求，仍需独立核对。" if supported else "未通过预先固定的效果要求，候选不成立。"), "",
        "上一轮位置中位数/MAD除法校准已否定。本轮改变机制：正常统计确定局部上界T(p)，原始分数r低于上界时输出r*r/T(p)，达到或超过上界时保留r。T(p)至少等于正常训练图全局95%分位数；位置上界平滑尺度为一个特征网格间距（8像素）。",
        "消融保持同一上界，只去掉按当前分数触发的上尾保护，改成固定位置权重r*global_tail/T(p)。全局对照仍为普通正仿射校准，类别内排序不变。",
        "本轮没有重新训练、提特征或拟合PCA。复用上轮1724张正常图的统计/阈值分区、原版模型和3887张测试图的原始异常图。单遍统计收益沿用已完成的同骨干实验，不计作本校准组件收益。", ""]
    for r in checks:
        text.append(f"{r['dataset']}：图像AUROC变化{r['delta']['image_auroc']*100:+.4f}个百分点，像素AP变化{r['delta']['pixel_ap']*100:+.4f}个百分点，AUPRO变化{r['delta']['aupro_0_30']*100:+.4f}个百分点，正常图像素误报率变化{r['delta']['normal_pixel_fpr']*100:+.4f}个百分点。")
    text += ["", f"校准参数约{parameters/1024**2:.3f} MiB。最慢类别P95校准应用时间为旧四张固定正常图CPU前向中位数的{overhead*100:.3f}%。不代表整机峰值或整套流程加速。", cfg["normal_fit_notice"], cfg["test_notice"], cfg["novelty_notice"], "", "方法来源：DINOv2 https://github.com/facebookresearch/dinov2；SubspaceAD https://github.com/CLendering/SubspaceAD 。正常分数校准、分位数与尾部抑制都有既有技术基础，本轮不声称论文原创性或录用资格。"]
    report = "\n".join(text) + "\n"
    (run / "实验报告.txt").write_text(report, encoding="utf-8")
    columns = ("dataset", "variant", *keys)
    table = "<table><tr>" + "".join(f"<th>{k}</th>" for k in columns) + "</tr>"
    for r in macro:
        table += "<tr>" + "".join(f"<td>{html.escape(r[k]) if isinstance(r[k], str) else format(100*r[k], '.4f')}</td>" for k in columns) + "</tr>"
    table += "</table>"
    (run / "report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>保护异常上尾的校准实验</title><style>body{font:16px/1.7 sans-serif;margin:40px}pre{white-space:pre-wrap}table{border-collapse:collapse;font-size:13px}th,td{padding:6px;border:1px solid #ccc}</style><h1>保护异常上尾的校准实验</h1><pre>' + html.escape(report) + '</pre><p>下表均为百分比。</p>' + table + '<p><a href="category_metrics.csv">完整逐类指标</a> · <a href="decision.json">预定效果要求</a></p></html>', encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/tail_calibration_20261015.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = read(args.config)
    run = args.output_dir.resolve()
    source, normals = ROOT / cfg["source_run"], ROOT / cfg["normal_source_run"]
    predecessor = read(normals / "frozen_protocol.json")
    verify_seal(predecessor["sha256"])
    if read(normals / "run_state.json")["status"] != "completed" or not read(normals / "audit_receipt.json")["all_27_categories"]:
        raise ValueError("Audited normal cache required")
    tasks = predecessor["tasks"]
    if len(tasks) != 27 or not run.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Invalid matrix or output root")
    if run.exists() and not args.resume:
        raise ValueError("Fresh run or explicit resume required")
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "run.lock"
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "w") as f:
        f.write(str(os.getpid()))
    monitor = Monitor(cfg["resource_policy"])
    started = time.perf_counter()
    try:
        initial = monitor.sample()
        policy = cfg["resource_policy"]
        if initial["system_free_ram_bytes"] < policy["initial_free_ram_gib"] * 1024**3 or initial["system_free_commit_bytes"] < policy["initial_free_commit_gib"] * 1024**3:
            raise RuntimeError("Initial RAM/commit reserve not met")
        if shutil.disk_usage(run).free < policy["minimum_disk_gib"] * 1024**3:
            raise RuntimeError("Disk reserve not met")
        seals = freeze_inputs(args.config, source, normals, tasks)
        if (run / "frozen_protocol.json").exists():
            frozen = read(run / "frozen_protocol.json")
            if frozen["sha256"] != seals or frozen["protocol"] != cfg:
                raise ValueError("Frozen protocol/source changed")
        else:
            frozen = {"protocol": cfg, "sha256": seals, "tasks": tasks, "frozen_at": now(), "before_current_test_evaluation": True}
            write(run / "frozen_protocol.json", frozen)
        monitor.thread.start()
        evaluate(run, cfg, frozen, source, normals, tasks, monitor)
        verify_seal(frozen["sha256"])
        write(run / "run_state.json", {"status": "completed", "pid": os.getpid(), "completed_categories": 27, "metric_rows": 108, "elapsed_seconds": time.perf_counter() - started, "finished_at": now()})
    except BaseException as e:
        write(run / f"failure_{int(time.time())}.json", {"error": repr(e), "traceback": traceback.format_exc(), "at": now()})
        write(run / "run_state.json", {"status": "failed", "pid": os.getpid(), "error": repr(e), "at": now()})
        raise
    finally:
        write(run / f"resources_{int(time.time())}.json", monitor.close())
        lock.unlink()


if __name__ == "__main__":
    main()
