"""Frozen normal-only calibration experiment using sealed SubspaceAD results."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from run_route1_resource_trial import Monitor


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def seal_sources(config_path, source, tasks):
    paths = [config_path, Path(__file__), ROOT / "src/industrial_anomaly_lab/score_calibration.py",
             ROOT / "src/industrial_anomaly_lab/subspace_features.py",
             ROOT / "src/industrial_anomaly_lab/quick_metrics.py",
             ROOT / "src/industrial_anomaly_lab/metrics.py",
             ROOT / "scripts/run_route1_resource_trial.py",
             ROOT / "configs/route1_controls_20261014.json", source / "run_config.json", source / "run_state.json"]
    for d, c in tasks:
        cat = source / "categories" / d / c
        paths.extend(cat / p for p in ("samples.jsonl", "train_manifest.json", "metrics.json", "models/official_cpu_two_pass.npz"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def verify_seal(seal):
    for p, h in seal.items():
        if sha(ROOT / p) != h:
            raise ValueError(f"Frozen source or evidence changed: {p}")


def selected_normals(cfg, source, d, c):
    train = read(source / "categories" / d / c / "train_manifest.json")
    if len(train) < 16:
        raise ValueError("Not enough normal training images for disjoint statistics/threshold sets")
    def key(row):
        text = f'{cfg["selection_seed"]}|{d}|{c}|{row["path"]}'
        return hashlib.sha256(text.encode()).hexdigest()
    selected = sorted(train, key=key)[:cfg["normal_sample_maximum"]]
    for i, row in enumerate(selected):
        row = dict(row)
        row["partition"] = "statistics" if i < len(selected) // 2 else "threshold"
        selected[i] = row
    return selected


def checked_normal(row):
    path = (ROOT / row["path"]).resolve(strict=True)
    if not path.is_relative_to((ROOT / "data").resolve()) or sha(path) != row["sha256"]:
        raise ValueError("Normal image path/content mismatch")
    return path


def collect_normals(cfg, frozen, run, source, tasks, monitor):
    import numpy as np
    import torch
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model, render_subspace_map
    p = read(ROOT / cfg["source_config"])
    def extractor(device):
        return LocalDinoExtractor(ROOT / p["weights_dir"], expected_checkpoint_sha256=p["checkpoint_sha256"],
            expected_artifact_sha256=p["weight_files_sha256"], device=device, cpu_threads=cfg["cpu_threads"],
            max_cuda_memory_gib=cfg["gpu_memory_limit_gib"])
    monitor.set_stage("normal_extraction")
    gpu = extractor("cuda:0") if torch.cuda.is_available() else None
    cpu = extractor("cpu")
    equivalence = []
    cpu_times = []
    for d, c in (("mvtec_ad", "bottle"), ("visa", "cashew")):
        model = load_subspace_model(source / "categories" / d / c / "models" / "official_cpu_two_pass.npz")
        for row in selected_normals(cfg, source, d, c)[:2]:
            monitor.check()
            path = checked_normal(row)
            start = time.perf_counter()
            tokens = cpu(path)
            cpu_times.append(time.perf_counter() - start)
            a = render_subspace_map(model.reconstruction_scores(tokens.reshape(-1, 384)).reshape(32, 32))
            entry = {"dataset": d, "category": c, "normal_path": row["path"], "cpu_forward_seconds": cpu_times[-1]}
            if gpu is not None:
                b = render_subspace_map(model.reconstruction_scores(gpu(path).reshape(-1, 384)).reshape(32, 32))
                entry.update(max_abs_error=float(np.max(np.abs(a - b))),
                    passed=bool(np.allclose(a, b, atol=cfg["equivalence_atol"], rtol=cfg["equivalence_rtol"])))
            else:
                entry.update(passed=True, gpu_available=False)
            equivalence.append(entry)
    use_gpu = gpu is not None and all(r["passed"] for r in equivalence)
    chosen = gpu if use_gpu else cpu
    if use_gpu:
        del cpu
    else:
        del gpu
        torch.cuda.empty_cache()
    gc.collect()
    write(run / "device_equivalence.json", {"gate": equivalence, "selected_device": str(chosen.device),
        "cpu_forward_median_seconds": float(np.median(cpu_times)), "scope": "Four fixed normal images; numerical compatibility check, not all-image equivalence proof"})
    for index, (d, c) in enumerate(tasks):
        verify_seal(frozen["sha256"])
        cat = run / "normal" / d / c
        cat.mkdir(parents=True, exist_ok=True)
        completed = cat / "completion.json"
        if completed.exists():
            for rel, h in read(completed)["sha256"].items():
                if sha(cat / rel) != h:
                    raise ValueError("Normal cache changed")
            print(f"normal cached {index + 1}/27 {d}/{c}", flush=True)
            continue
        selected = selected_normals(cfg, source, d, c)
        write(cat / "selection.json", selected)
        model = load_subspace_model(source / "categories" / d / c / "models" / "official_cpu_two_pass.npz")
        started = time.perf_counter()
        records = []
        for j, row in enumerate(selected):
            monitor.check()
            if shutil.disk_usage(run).free < cfg["resource_policy"]["minimum_disk_gib"] * 1024**3:
                raise RuntimeError("Disk reserve reached")
            if use_gpu:
                free, _ = torch.cuda.mem_get_info()
                if free < 1024**3:
                    raise RuntimeError("CUDA free memory below 1 GiB reserve")
            path = checked_normal(row)
            tick = time.perf_counter()
            features = chosen(path)
            forward_seconds = time.perf_counter() - tick
            scores = model.reconstruction_scores(features.reshape(-1, 384)).reshape(32, 32)
            raw = render_subspace_map(scores)
            outfile = cat / f"{j:03d}.npy"
            np.save(outfile, raw, allow_pickle=False)
            records.append({**row, "raw_map": outfile.name, "raw_map_sha256": sha(outfile), "forward_seconds": forward_seconds})
            write(run / "run_state.json", {"status": "running", "phase": "normal_extraction", "pid": os.getpid(),
                "category": f"{d}/{c}", "normal_image": j + 1, "normal_images": len(selected),
                "completed_categories": index, "updated_at": now()})
        write(cat / "records.json", records)
        write(completed, {"dataset": d, "category": c, "normal_count": len(records), "device": str(chosen.device),
            "elapsed_seconds": time.perf_counter() - started,
            "sha256": {p.name: sha(p) for p in [cat / "selection.json", cat / "records.json"] + sorted(cat.glob("*.npy"))}})
        print(f"normal completed {index + 1}/27 {d}/{c} ({len(records)} images)", flush=True)
    del chosen
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(cfg, frozen, run, source, tasks, monitor):
    import numpy as np
    from scipy.stats import spearmanr
    from industrial_anomaly_lab.score_calibration import VARIANTS, fit_normal_calibration, save_calibration
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    monitor.set_stage("evaluation")
    all_rows = []
    for index, (d, c) in enumerate(tasks):
        verify_seal(frozen["sha256"])
        output = run / "categories" / d / c
        output.mkdir(parents=True, exist_ok=True)
        completion = output / "completion.json"
        if completion.exists():
            receipt = read(completion)
            for rel, h in receipt["sha256"].items():
                if sha(output / rel) != h:
                    raise ValueError("Completed evaluation changed")
            all_rows.extend(read(output / "metrics.json"))
            continue
        normal_dir = run / "normal" / d / c
        receipt = read(normal_dir / "completion.json")
        for rel, h in receipt["sha256"].items():
            if sha(normal_dir / rel) != h:
                raise ValueError("Normal cache authentication failed")
        records = read(normal_dir / "records.json")
        statistics = np.stack([np.load(normal_dir / r["raw_map"], allow_pickle=False) for r in records if r["partition"] == "statistics"])
        held = np.stack([np.load(normal_dir / r["raw_map"], allow_pickle=False) for r in records if r["partition"] == "threshold"])
        tick = time.perf_counter()
        model = fit_normal_calibration(statistics, prior_count=cfg["prior_count"], smoothing_sigma=cfg["smoothing_sigma"], floor_ratio=cfg["floor_ratio"])
        fit_seconds = time.perf_counter() - tick
        save_calibration(output / "calibration.npz", model)
        halves = np.array_split(statistics, 2)
        rho = float(spearmanr(np.median(halves[0], axis=0).ravel(), np.median(halves[1], axis=0).ravel()).statistic)
        if not np.isfinite(rho):
            rho = 0.0
        hot = model.local_center >= np.quantile(model.local_center, .8)
        raw_cut = float(np.quantile(held, cfg["pixel_threshold_quantile"], method=cfg["quantile_method"]))
        exceeding = held > raw_cut
        concentration = float(exceeding[:, hot].mean() / max(exceeding.mean(), 1e-12))
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        for sample in samples:
            for p, h in ((sample["raw_maps"][cfg["baseline_model"]], sample["raw_map_sha256"][cfg["baseline_model"]]),
                         (sample["mask_path"], sample["mask_sha256"])):
                if sha(source / p) != h:
                    raise ValueError("Sealed Test map or mask mismatch")
        category_rows = []
        baseline_metrics = next(r for r in read(source / "categories" / d / c / "metrics.json") if r["variant"] == cfg["baseline_model"] + "_max")
        for variant in VARIANTS:
            monitor.check()
            transformed_normal = np.stack([model.apply(x, variant) for x in held])
            pixel_cut = float(np.quantile(transformed_normal, cfg["pixel_threshold_quantile"], method=cfg["quantile_method"]))
            image_cut = float(np.quantile(transformed_normal.max(axis=(1, 2)), cfg["image_threshold_quantile"], method=cfg["quantile_method"]))
            maps, masks, labels = [], [], []
            timings = []
            normal_pixels = normal_fp = background_pixels = background_fp = foreground_pixels = foreground_tp = 0
            normal_images = normal_image_fp = anomalous_images = anomalous_image_tp = 0
            map_manifest = []
            for j, sample in enumerate(samples):
                monitor.check()
                raw_path = source / sample["raw_maps"][cfg["baseline_model"]]
                raw = np.load(raw_path, allow_pickle=False)
                tick = time.perf_counter()
                changed = model.apply(raw, variant)
                timings.append(time.perf_counter() - tick)
                mask_path = source / sample["mask_path"]
                mask = np.load(mask_path, allow_pickle=False).astype(bool)
                label = int(sample["label"])
                background_pixels += int((~mask).sum())
                background_fp += int(np.count_nonzero(changed[~mask] > pixel_cut))
                foreground_pixels += int(mask.sum())
                foreground_tp += int(np.count_nonzero(changed[mask] > pixel_cut))
                if label == 0:
                    normal_pixels += changed.size
                    normal_fp += int(np.count_nonzero(changed > pixel_cut))
                    normal_images += 1
                    normal_image_fp += int(changed.max() > image_cut)
                else:
                    anomalous_images += 1
                    anomalous_image_tp += int(changed.max() > image_cut)
                if variant == "baseline":
                    outfile = raw_path
                else:
                    outdir = output / "maps" / variant
                    outdir.mkdir(parents=True, exist_ok=True)
                    outfile = outdir / f"{j:06d}.npy"
                    np.save(outfile, changed, allow_pickle=False)
                    map_manifest.append({"sample_id": sample["sample_id"], "file": outfile.relative_to(output).as_posix(), "sha256": sha(outfile)})
                maps.append(outfile)
                masks.append(mask_path)
                labels.append(label)
            if variant == "baseline":
                metrics = {k: float(baseline_metrics[k]) for k in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")}
            else:
                monitor.set_stage(f"metrics:{d}/{c}/{variant}")
                metrics = evaluate_from_files(maps, masks, labels, temp_root=run / "metric_tmp")
                monitor.set_stage("evaluation")
            row = {"dataset": d, "category": c, "variant": variant, "test_images": len(samples), **metrics,
                "normal_pixel_fpr": normal_fp / normal_pixels, "background_pixel_fpr": background_fp / background_pixels,
                "anomaly_pixel_recall": foreground_tp / foreground_pixels,
                "normal_image_fpr": normal_image_fp / normal_images,
                "anomaly_image_tpr": anomalous_image_tp / anomalous_images,
                "pixel_threshold": pixel_cut, "image_threshold": image_cut,
                "apply_median_ms": float(np.median(timings) * 1000), "apply_p95_ms": float(np.quantile(timings, .95) * 1000),
                "calibration_fit_seconds": fit_seconds if variant == "local_shrunk" else None,
                "calibration_parameter_bytes": model.parameter_bytes if variant.startswith("local") else (16 if variant == "global" else 0),
                "normal_spatial_alignment_spearman": rho, "normal_error_concentration_top20pct": concentration}
            if variant == "global":
                delta = max(abs(metrics[k] - baseline_metrics[k]) for k in metrics)
                if delta > 1e-10:
                    raise ValueError(f"Global affine ranking invariance violated: {delta}")
                row["global_metric_max_absolute_error"] = delta
            write(output / f"maps_{variant}.json", map_manifest)
            write(output / f"metrics_{variant}.json", row)
            category_rows.append(row)
            print(f"metrics {index + 1}/27 {d}/{c}/{variant}: I={metrics['image_auroc']:.4f} PRO={metrics['aupro_0_30']:.4f}", flush=True)
        write(output / "metrics.json", category_rows)
        files = [output / "calibration.npz", output / "metrics.json"] + list(output.glob("metrics_*.json")) + list(output.glob("maps_*.json"))
        write(completion, {"sha256": {p.relative_to(output).as_posix(): sha(p) for p in files},
            "normal_cache_completion_sha256": sha(normal_dir / "completion.json"), "finished_at": now()})
        all_rows.extend(category_rows)
        write(run / "run_state.json", {"status": "running", "phase": "evaluation", "pid": os.getpid(),
            "completed_categories": index + 1, "metric_rows": len(all_rows), "updated_at": now()})
        gc.collect()
    finish_report(cfg, frozen, run, all_rows)


def finish_report(cfg, frozen, run, rows):
    import numpy as np
    from industrial_anomaly_lab.score_calibration import VARIANTS
    keys = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30", "normal_pixel_fpr", "background_pixel_fpr",
            "anomaly_pixel_recall", "normal_image_fpr", "anomaly_image_tpr")
    macros = []
    for d in ("mvtec_ad", "visa"):
        for v in VARIANTS:
            subset = [r for r in rows if r["dataset"] == d and r["variant"] == v]
            expected = 15 if d == "mvtec_ad" else 12
            if len(subset) != expected:
                raise ValueError("Incomplete full-category matrix")
            macros.append({"dataset": d, "variant": v, "categories": len(subset),
                **{k: float(np.mean([r[k] for r in subset])) for k in keys}})
    a = cfg["acceptance"]
    datasets = []
    for d in ("mvtec_ad", "visa"):
        base = next(r for r in macros if r["dataset"] == d and r["variant"] == "baseline")
        local = next(r for r in macros if r["dataset"] == d and r["variant"] == "local_shrunk")
        pairs = [(r, next(b for b in rows if b["dataset"] == d and b["category"] == r["category"] and b["variant"] == "baseline"))
                 for r in rows if r["dataset"] == d and r["variant"] == "local_shrunk"]
        gains = {k: [r[k] - b[k] for r, b in pairs] for k in ("pixel_ap", "aupro_0_30")}
        stable = {k: local[k] - base[k] >= a["localization_gain_minimum"] and
                  np.mean(np.array(gains[k]) > 0) >= a["category_improvement_fraction_minimum"] and
                  np.median(gains[k]) > a["paired_median_gain_minimum"] for k in gains}
        fpr_gains = [(b["normal_pixel_fpr"] - r["normal_pixel_fpr"]) for r, b in pairs]
        fpr_pass = (local["normal_pixel_fpr"] <= base["normal_pixel_fpr"] * (1 - a["false_positive_relative_reduction_minimum"]) and
                    local["anomaly_pixel_recall"] >= base["anomaly_pixel_recall"] - a["pixel_recall_maximum_drop"] and
                    local["anomaly_image_tpr"] >= base["anomaly_image_tpr"] - a["image_tpr_maximum_drop"] and
                    np.mean(np.array(fpr_gains) > 0) >= a["category_improvement_fraction_minimum"] and np.median(fpr_gains) > 0)
        datasets.append({"dataset": d, "image_noninferiority": local["image_auroc"] >= base["image_auroc"] - a["image_auroc_maximum_drop"],
            "pixel_ap_stable_gain": bool(stable["pixel_ap"]), "aupro_stable_gain": bool(stable["aupro_0_30"]),
            "false_positive_gain_without_recall_loss": bool(fpr_pass),
            "delta": {k: local[k] - base[k] for k in keys},
            "category_gains": {k: {"positive": int(np.sum(np.array(v) > 0)), "total": len(v), "median": float(np.median(v))} for k, v in gains.items()},
            "normal_structure_supported": bool(np.median([r["normal_spatial_alignment_spearman"] for r, b in pairs]) >= a["normal_alignment_minimum_spearman"] and
                np.median([r["normal_error_concentration_top20pct"] for r, b in pairs]) >= a["normal_error_concentration_minimum"])})
    device = read(run / "device_equivalence.json")
    local_rows = [r for r in rows if r["variant"] == "local_shrunk"]
    overhead_ratio = max(r["apply_p95_ms"] for r in local_rows) / (device["cpu_forward_median_seconds"] * 1000)
    parameter_max = max(r["calibration_parameter_bytes"] for r in local_rows)
    same_branch = any(all(r[k] for r in datasets) for k in ("pixel_ap_stable_gain", "aupro_stable_gain", "false_positive_gain_without_recall_loss"))
    passed = all(r["image_noninferiority"] and r["normal_structure_supported"] for r in datasets) and same_branch and (
        overhead_ratio <= a["apply_overhead_fraction_of_measured_cpu_forward_maximum"] and parameter_max <= a["calibration_parameter_max_mib"] * 1024**2)
    decision = {"protocol_id": cfg["protocol_id"], "status": "supported" if passed else "rejected",
        "objective_satisfied": bool(passed), "datasets": datasets, "same_gain_branch_both_datasets": same_branch,
        "apply_p95_over_measured_cpu_forward": overhead_ratio, "parameter_max_bytes": parameter_max,
        "normal_fit_notice": cfg["normal_fit_notice"], "test_notice": cfg["test_notice"], "novelty_notice": cfg["novelty_notice"],
        "remaining_if_rejected": "No effectiveness claim. Preserve all four controls and regressions. Do not select a different parameter from Test scores."}
    write(run / "decision.json", decision)
    write(run / "dataset_macro_metrics.json", macros)
    write(run / "category_metrics.json", rows)
    for name, values in (("dataset_macro_metrics", macros), ("category_metrics", rows)):
        with (run / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = list(dict.fromkeys(k for row in values for k in row))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    text = ["正常结构波动校准：完整配对实验", "",
        "结论：" + ("达到本轮预先固定的工程效果要求。" if passed else "候选方案未达到本轮预先固定的效果要求，不作为有效改进交付。"), "",
        "四组使用同样的DINOv2-S、原版SubspaceAD模型和测试异常图。校准统计与操作阈值分别由正常训练图的两个分区确定。原PCA模型见过这两个分区，因此不能声称严格的独立校准或误报率保证。",
        "全局校准是正仿射变换，理论上不改变类别内排序；全四指标实际复算核对。局部候选将平滑的位置中位数和MAD向全局统计收缩，消融只去掉收缩。参数在新测试评测之前冻结。", ""]
    for d in datasets:
        text.append(f"{d['dataset']}：图像AUROC变化{d['delta']['image_auroc'] * 100:+.4f}个百分点，像素AP变化{d['delta']['pixel_ap'] * 100:+.4f}个百分点，AUPRO变化{d['delta']['aupro_0_30'] * 100:+.4f}个百分点；正常图像素误报率变化{d['delta']['normal_pixel_fpr'] * 100:+.4f}个百分点。")
    text += ["", f"局部校准参数最大{parameter_max / 1024**2:.3f} MiB；最慢类别P95处理耗时为4张固定正常图CPU前向中位数的{overhead_ratio * 100:.3f}%。这不是整机独立进程峰值或全流程加速比例。",
        "单遍统计的精度和执行成本证据沿用已完成的同骨干Subspace对照，不把其耗时节省算成校准组件收益。",
        "", cfg["normal_fit_notice"], cfg["test_notice"], cfg["novelty_notice"],
        "", "原始资料：SubspaceAD https://github.com/CLendering/SubspaceAD；DINOv2 https://github.com/facebookresearch/dinov2；相关正常校准先例 https://www.mdpi.com/2079-9292/15/16/3654 。保留来源；本实验没有证明方法原创或期刊录用资格。"]
    report = "\n".join(text) + "\n"
    (run / "实验报告.txt").write_text(report, encoding="utf-8")
    columns = ("dataset", "variant", *keys)
    table = "<table><thead><tr>" + "".join(f"<th>{k}</th>" for k in columns) + "</tr></thead><tbody>"
    for r in macros:
        table += "<tr>" + "".join(f"<td>{html.escape(str(r[k])) if isinstance(r[k], str) else format(r[k] * 100, '.4f')}</td>" for k in columns) + "</tr>"
    table += "</tbody></table>"
    (run / "report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>正常评分校准实验</title><style>body{font:16px/1.7 sans-serif;margin:40px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}pre{white-space:pre-wrap}</style><h1>正常评分校准实验</h1><pre>' + html.escape(report) + '</pre><p>下表指标均为百分比。</p>' + table + '<p><a href="category_metrics.csv">逐类指标（含所有退步项）</a> · <a href="decision.json">预定要求逐项判断</a></p></html>', encoding="utf-8")
    verify_seal(frozen["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/score_calibration_20261015.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("normal", "evaluate"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if not run.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Output must be in runs/quick")
    cfg = read(args.config)
    source = ROOT / cfg["source_run"]
    if read(source / "run_state.json")["status"] != "completed":
        raise ValueError("Original controls must be completed")
    categories = (ROOT / "runs/quick/padim-r18-20261012T022650Z/run_config.json")
    dataset_cfg = read(categories)["protocol"]["datasets"]
    tasks = [(d, c) for d, s in dataset_cfg.items() for c in s["categories"]]
    if len(tasks) != 27:
        raise ValueError("Expected all 27 categories")
    if not args.resume and run.exists():
        raise ValueError("Use a fresh run or explicit resume")
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    monitor = Monitor(cfg["resource_policy"])
    start = time.perf_counter()
    try:
        initial = monitor.sample()
        policy = cfg["resource_policy"]
        if initial["system_free_ram_bytes"] < policy["initial_free_ram_gib"] * 1024**3 or initial["system_free_commit_bytes"] < policy["initial_free_commit_gib"] * 1024**3:
            raise RuntimeError("Initial RAM/commit admission requirement not met")
        if shutil.disk_usage(run).free < policy["minimum_disk_gib"] * 1024**3:
            raise RuntimeError("Disk reserve not met")
        seal = seal_sources(args.config, source, tasks)
        if (run / "frozen_protocol.json").exists():
            frozen = read(run / "frozen_protocol.json")
            if frozen["sha256"] != seal or frozen["protocol"] != cfg:
                raise ValueError("Frozen experiment changed")
        else:
            frozen = {"frozen_at": now(), "protocol": cfg, "sha256": seal, "tasks": tasks}
            write(run / "frozen_protocol.json", frozen)
        monitor.thread.start()
        if args.phase == "normal":
            collect_normals(cfg, frozen, run, source, tasks, monitor)
            status = "normal_statistics_ready"
        else:
            evaluate(cfg, frozen, run, source, tasks, monitor)
            status = "completed"
        write(run / "run_state.json", {"status": status, "phase": args.phase, "pid": os.getpid(), "finished_at": now(),
            "elapsed_seconds": time.perf_counter() - start, "completed_categories": 27})
        print(status, flush=True)
    except BaseException as e:
        write(run / f"failure_{args.phase}_{int(time.time())}.json", {"error": repr(e), "traceback": traceback.format_exc(), "at": now(), "pid": os.getpid()})
        write(run / "run_state.json", {"status": "failed", "phase": args.phase, "pid": os.getpid(), "error": repr(e), "at": now()})
        raise
    finally:
        write(run / f"resources_{args.phase}_{int(time.time())}.json", monitor.close())
        lock.unlink()


if __name__ == "__main__":
    main()
