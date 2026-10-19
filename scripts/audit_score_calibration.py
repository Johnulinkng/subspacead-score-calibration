"""Replay calibration outputs, measure isolated overhead, and render fixed examples."""
from __future__ import annotations
import argparse
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_score_calibration import read, lines, sha, write, selected_normals, verify_seal


def worker(run, variant, output):
    import numpy as np
    from industrial_anomaly_lab.score_calibration import fit_normal_calibration
    from run_route1_resource_trial import Monitor
    cfg = read(run / "frozen_protocol.json")["protocol"]
    mon = Monitor(cfg["resource_policy"])
    mon.sample()
    mon.thread.start()
    cat = run / "normal/mvtec_ad/bottle"
    rows = [r for r in read(cat / "records.json") if r["partition"] == "statistics"]
    values = np.stack([np.load(cat / r["raw_map"], allow_pickle=False) for r in rows])
    query = values[0]
    mon.set_stage("fit")
    tick = time.perf_counter()
    model = fit_normal_calibration(values, prior_count=cfg["prior_count"], smoothing_sigma=cfg["smoothing_sigma"], floor_ratio=cfg["floor_ratio"]) if variant != "baseline" else None
    fit_seconds = time.perf_counter() - tick
    mon.set_stage("apply")
    timings = []
    checksum = 0.0
    for _ in range(100):
        mon.check()
        tick = time.perf_counter()
        changed = query.astype(np.float64, copy=True) if model is None else model.apply(query, variant)
        timings.append(time.perf_counter() - tick)
        checksum += float(changed.mean())
    resource = mon.close()
    write(output, {"variant": variant, "fit_seconds": fit_seconds,
        "apply_median_ms": float(np.median(timings) * 1000), "apply_p95_ms": float(np.quantile(timings, .95) * 1000),
        "checksum": checksum, "parameters_bytes": model.parameter_bytes if model else 0,
        "peak_working_set_bytes": resource["process_peak_working_set_bytes"],
        "peak_commit_bytes": resource["process_peak_commit_bytes"],
        "scope": "Independent process; same 32 bottle normal residual maps; imports/data loading/fit included in OS peaks; timed fit excludes imports/data load; 100 CPU apply repetitions; no DINO model", "resources": resource})


def audit(run):
    import numpy as np
    from industrial_anomaly_lab.score_calibration import VARIANTS, fit_normal_calibration, load_calibration
    frozen = read(run / "frozen_protocol.json")
    cfg = frozen["protocol"]
    if read(run / "run_state.json")["status"] != "completed":
        raise ValueError("Complete evaluation required")
    verify_seal(frozen["sha256"])
    source = ROOT / cfg["source_run"]
    records = read(run / "category_metrics.json")
    if len(records) != 108 or len({(r["dataset"], r["category"], r["variant"]) for r in records}) != 108:
        raise ValueError("Expected all 27 x 4 rows")
    replayed_maps = normal_count = sample_count = 0
    max_error = 0.0
    for d, c in frozen["tasks"]:
        cat = run / "categories" / d / c
        normal = run / "normal" / d / c
        expected = selected_normals(cfg, source, d, c)
        if read(normal / "selection.json") != expected:
            raise ValueError("Normal-only selection differs from frozen procedure")
        for folder in (cat, normal):
            for rel, digest in read(folder / "completion.json")["sha256"].items():
                if sha(folder / rel) != digest:
                    raise ValueError("Artifact seal mismatch")
        normal_records = read(normal / "records.json")
        normal_count += len(normal_records)
        stat_rows = [r for r in normal_records if r["partition"] == "statistics"]
        threshold_rows = [r for r in normal_records if r["partition"] == "threshold"]
        if {r["path"] for r in stat_rows} & {r["path"] for r in threshold_rows}:
            raise ValueError("Statistics/threshold image overlap")
        for row in normal_records:
            if sha(ROOT / row["path"]) != row["sha256"] or sha(normal / row["raw_map"]) != row["raw_map_sha256"]:
                raise ValueError("Normal input/cache hash mismatch")
        fit_maps = np.stack([np.load(normal / r["raw_map"], allow_pickle=False) for r in stat_rows])
        fitted = fit_normal_calibration(fit_maps, prior_count=cfg["prior_count"], smoothing_sigma=cfg["smoothing_sigma"], floor_ratio=cfg["floor_ratio"])
        saved = load_calibration(cat / "calibration.npz")
        for key in ("global_center", "global_scale", "local_center", "local_scale", "reliability"):
            np.testing.assert_array_equal(getattr(fitted, key), getattr(saved, key))
        held = np.stack([np.load(normal / r["raw_map"], allow_pickle=False) for r in threshold_rows])
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        sample_count += len(samples)
        for variant in VARIANTS:
            row = next(r for r in records if (r["dataset"], r["category"], r["variant"]) == (d, c, variant))
            calibrated = np.stack([saved.apply(m, variant) for m in held])
            pcut = float(np.quantile(calibrated, cfg["pixel_threshold_quantile"], method=cfg["quantile_method"]))
            icut = float(np.quantile(calibrated.max(axis=(1, 2)), cfg["image_threshold_quantile"], method=cfg["quantile_method"]))
            if pcut != row["pixel_threshold"] or icut != row["image_threshold"]:
                raise ValueError("Threshold replay mismatch")
            predictions = read(cat / f"maps_{variant}.json")
            if variant != "baseline" and len(predictions) != len(samples):
                raise ValueError("Prediction matrix incomplete")
            normal_pixels = fp = foreground = tp = normal_images = image_fp = anomalies = image_tp = 0
            for j, sample in enumerate(samples):
                raw = np.load(source / sample["raw_maps"][cfg["baseline_model"]], allow_pickle=False)
                expected_map = saved.apply(raw, variant)
                if variant != "baseline":
                    rec = predictions[j]
                    if rec["sample_id"] != sample["sample_id"] or sha(cat / rec["file"]) != rec["sha256"]:
                        raise ValueError("Test prediction hash/order mismatch")
                    actual = np.load(cat / rec["file"], allow_pickle=False)
                    error = float(np.max(np.abs(actual - expected_map)))
                    max_error = max(max_error, error)
                    if error != 0:
                        raise ValueError("Calibration prediction replay mismatch")
                    replayed_maps += 1
                mask = np.load(source / sample["mask_path"], allow_pickle=False).astype(bool)
                foreground += int(mask.sum())
                tp += int(np.count_nonzero(expected_map[mask] > pcut))
                if sample["label"] == 0:
                    normal_pixels += expected_map.size
                    fp += int(np.count_nonzero(expected_map > pcut))
                    normal_images += 1
                    image_fp += int(expected_map.max() > icut)
                else:
                    anomalies += 1
                    image_tp += int(expected_map.max() > icut)
            actual_rates = {"normal_pixel_fpr": fp / normal_pixels, "anomaly_pixel_recall": tp / foreground,
                "normal_image_fpr": image_fp / normal_images, "anomaly_image_tpr": image_tp / anomalies}
            for key, value in actual_rates.items():
                if value != row[key]:
                    raise ValueError("Operating-point replay mismatch")
        print(f"audit {d}/{c}", flush=True)
    macro = read(run / "dataset_macro_metrics.json")
    for row in macro:
        selected = [r for r in records if r["dataset"] == row["dataset"] and r["variant"] == row["variant"]]
        for key in row:
            if key not in ("dataset", "variant", "categories"):
                if row[key] != float(np.mean([r[key] for r in selected])):
                    raise ValueError("Macro replay mismatch")
    resources = run / "isolated_resources"
    resources.mkdir(exist_ok=True)
    for repeat in range(3):
        for variant in ("baseline", "local_shrunk"):
            outfile = resources / f"{variant}_{repeat}.json"
            if not outfile.exists():
                subprocess.run([sys.executable, str(Path(__file__)), "--run", str(run), "--worker", variant, "--output", str(outfile)], check=True, cwd=ROOT)
    resource_summary = {}
    for variant in ("baseline", "local_shrunk"):
        trial_rows = [read(resources / f"{variant}_{i}.json") for i in range(3)]
        if any(r["resources"]["resource_violation"] for r in trial_rows):
            raise ValueError("Isolated resource trial violated reserve")
        resource_summary[variant] = {key: float(np.median([r[key] for r in trial_rows])) for key in (
            "fit_seconds", "apply_median_ms", "apply_p95_ms", "peak_working_set_bytes", "peak_commit_bytes")}
    resource_summary["extra_peak_working_set_mib"] = (resource_summary["local_shrunk"]["peak_working_set_bytes"] - resource_summary["baseline"]["peak_working_set_bytes"]) / 1024**2
    resource_summary["extra_peak_commit_mib"] = (resource_summary["local_shrunk"]["peak_commit_bytes"] - resource_summary["baseline"]["peak_commit_bytes"]) / 1024**2
    resource_summary["scope"] = "Differences of three-process medians, fixed bottle/32-normal-map calibration stage; not whole-dataset or full DINO process peak differences"
    write(run / "isolated_resource_summary.json", resource_summary)
    receipt = {"verified_at": datetime.now(timezone.utc).isoformat(), "all_27_categories": True, "metric_rows": len(records),
        "normal_images": normal_count, "test_images": sample_count, "replayed_calibrated_maps": replayed_maps,
        "max_map_replay_error": max_error, "threshold_and_operating_point_replay": "passed",
        "normal_selection_and_disjoint_partition_replay": "passed", "calibration_parameter_replay": "passed",
        "input_and_source_hashes": "passed", "macro_replay": "passed",
        "metric_scope": "Full pixel metrics computed by original sealed exact evaluator for all 81 transformed category groups; baseline metrics reuse authenticated original full evaluation. This audit replays all transformed maps, operating points, macros and parameters; no redundant full pixel-sort replay.",
        "sha256": {p.name: sha(p) for p in (run / "frozen_protocol.json", run / "decision.json", run / "category_metrics.json", run / "dataset_macro_metrics.json", run / "isolated_resource_summary.json")}}
    write(run / "audit_receipt.json", receipt)
    render(run, source, frozen)
    notes = "\n独立资源补测：固定bottle的32张正常图，在六个独立进程中比较原始评分与局部校准，各三次。校准阶段额外工作集峰值中位数差为%.3f MiB，额外提交内存峰值差为%.3f MiB。该结果不代表整套数据的完整模型进程差值。\n完整重放核对：%d张正常图、%d张测试图、%d张校准异常图；异常图、参数、阈值与操作点重放误差为0。\n" % (resource_summary["extra_peak_working_set_mib"], resource_summary["extra_peak_commit_mib"], normal_count, sample_count, replayed_maps)
    with (run / "实验报告.txt").open("a", encoding="utf-8") as f:
        f.write(notes)
    reproduce = f'''复现这轮正常评分校准实验

实验位置：{run}
固定配置：{ROOT / 'configs/score_calibration_20261015.json'}
运行环境：{ROOT / '.venv/Scripts/python.exe'}（项目现有环境）

在E:\\3k目录运行，OUTPUT换成不存在的新目录：
.venv\\Scripts\\python.exe -u scripts\\run_score_calibration.py --output-dir runs/quick/OUTPUT --phase normal
.venv\\Scripts\\python.exe -u scripts\\run_score_calibration.py --output-dir runs/quick/OUTPUT --phase evaluate --resume
.venv\\Scripts\\python.exe -u scripts\\audit_score_calibration.py --run runs/quick/OUTPUT

依赖现有完整同骨干实验及原始数据；独立复制代码而不复制这些输入不能直接复现。正常图统计最多64张/类，统计与阈值两分区不重叠，但复用的PCA见过全部训练正常图。测试图仅使用既有未经显示归一化的异常图，没有重训DINO。

脚本、原始模型、评测器、输入索引在frozen_protocol.json中固定。恢复时检查其摘要，已有完成项校验后复用。旧目录score-calibration-20261015保留首次CUDA设备索引错误记录；修复为cuda:0后使用本目录，未根据测试指标更改方法。

全局正仿射校准是排序不变的对照，不声称其能提升AUROC/AP/AUPRO。局部候选的关键组件为向全局统计收缩；local_no_shrink只移除收缩，其他设置相同。其收益和退步完整保存在108条逐类结果中。

SubspaceAD与DINOv2来源、原版许可证继续保留。正常校准和平滑收缩均有既有技术基础，本实验不证明方法原创，也不证明达到任何期刊录用标准。
'''
    (run / "复现说明.txt").write_text(reproduce, encoding="utf-8")
    seal = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
    write(run / "artifact_manifest.json", {"sha256": seal, "source_root": str(ROOT), "failed_initialization": str(ROOT / "runs/quick/score-calibration-20261015")})


def render(run, source, frozen):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    cfg = frozen["protocol"]
    folder = run / "visualizations"
    folder.mkdir(exist_ok=True)
    links = []
    for d, c in frozen["tasks"]:
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        indices = [i for i, r in enumerate(samples) if r["selected"]]
        if len(indices) != 2 or {samples[i]["label"] for i in indices} != {0, 1}:
            raise ValueError("Expected original fixed normal/defect visual pair")
        for i in indices:
            row = samples[i]
            maps = [np.load(source / row["raw_maps"][cfg["baseline_model"]], allow_pickle=False)]
            maps += [np.load(run / "categories" / d / c / "maps" / v / f"{i:06d}.npy", allow_pickle=False) for v in ("global", "local_shrunk", "local_no_shrink")]
            with Image.open(row["image_path"]) as im:
                image = np.asarray(im.convert("RGB").resize((256, 256)))
            mask = np.load(source / row["mask_path"], allow_pickle=False)
            fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
            axes[0].imshow(image)
            axes[0].set_title("Input")
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            for ax, values, name in zip(axes[2:], maps, ("Baseline", "Global affine", "Local + shrink", "Local, no shrink")):
                low, high = np.quantile(values, (.01, .99))
                if high <= low:
                    high = low + 1e-9
                ax.imshow(values, cmap="turbo", vmin=low, vmax=high)
                ax.set_title(f"{name}\nraw [{values.min():.3g}, {values.max():.3g}]", fontsize=10)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{d}/{row['sample_id']} (label={row['label']})", fontsize=11)
            fig.text(.5, .015, "Each heatmap uses its own 1%/99% display range. Metrics use unnormalized raw maps; color is not a calibrated probability.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .9))
            file = folder / f"{d}_{c}_{row['label']}.png"
            fig.savefig(file, dpi=140)
            plt.close(fig)
            links.append(f'<details><summary>{html.escape(d + "/" + row["sample_id"])}</summary><img loading="lazy" style="width:100%" src="visualizations/{file.name}"></details>')
    report = (run / "report.html").read_text(encoding="utf-8")
    addition = '<h2>固定的54组对比图</h2><p>沿用原实验每类第一张正常图与第一张异常图，未按本轮效果选择样例。颜色采用各图独立1%/99%显示范围，不代表概率，数值评测使用原始分数。</p>' + "".join(links)
    addition += '<p><a href="audit_receipt.json">全量重放核对</a> · <a href="isolated_resource_summary.json">独立资源补测</a> · <a href="复现说明.txt">复现说明</a></p>'
    (run / "report.html").write_text(report.replace("</html>", addition + "</html>"), encoding="utf-8")
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--worker", choices=("baseline", "local_shrunk"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    resolved = args.run.resolve()
    if not resolved.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Run must be in project runs/quick")
    if args.worker:
        worker(resolved, args.worker, args.output)
    else:
        audit(resolved)
