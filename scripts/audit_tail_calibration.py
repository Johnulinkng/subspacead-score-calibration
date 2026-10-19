"""Replay all tail-calibrated outputs and independently measure the added stage."""
from __future__ import annotations
import argparse
import gc
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_score_calibration import read, lines, sha, write, now, verify_seal
from run_tail_calibration import operating_rates
from run_route1_resource_trial import Monitor


def worker(run, variant, output):
    import numpy as np
    from industrial_anomaly_lab.tail_calibration import fit_tail_calibration
    cfg = read(run / "frozen_protocol.json")["protocol"]
    normal = ROOT / cfg["normal_source_run"] / "normal/mvtec_ad/bottle"
    rows = [r for r in read(normal / "records.json") if r["partition"] == "statistics"]
    monitor = Monitor(cfg["resource_policy"])
    monitor.sample()
    monitor.thread.start()
    values = np.stack([np.load(normal / r["raw_map"], allow_pickle=False) for r in rows])
    monitor.set_stage("fit")
    started = time.perf_counter()
    model = fit_tail_calibration(values, tail_quantile=cfg["tail_quantile"], smoothing_sigma=cfg["smoothing_sigma"]) if variant == "local_guarded" else None
    fitting = time.perf_counter() - started
    monitor.set_stage("apply")
    timings = []
    checksum = 0.
    for _ in range(100):
        monitor.check()
        tick = time.perf_counter()
        scores = model.apply(values[0], "local_guarded") if model else values[0].astype(np.float64)
        timings.append(time.perf_counter() - tick)
        checksum += float(scores.mean())
    resources = monitor.close()
    write(output, {"variant": variant, "fit_seconds": fitting, "apply_median_ms": float(np.median(timings) * 1000),
        "apply_p95_ms": float(np.quantile(timings, .95) * 1000), "checksum": checksum,
        "peak_working_set_bytes": resources["process_peak_working_set_bytes"], "peak_commit_bytes": resources["process_peak_commit_bytes"],
        "resources": resources, "scope": "Independent CPU scoring-stage process, same fixed 32 bottle normal maps; loading/imports included in OS peaks, excluded from fit/apply timing; no backbone"})


def render(run, cfg, tasks):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    source = ROOT / cfg["source_run"]
    normals = ROOT / cfg["normal_source_run"]
    directory = run / "visualizations"
    directory.mkdir(exist_ok=True)
    links = []
    for d, c in tasks:
        cat = run / "categories" / d / c
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        selected = [j for j, sample in enumerate(samples) if sample["selected"]]
        if len(selected) != 2 or {samples[j]["label"] for j in selected} != {0, 1}:
            raise ValueError("Original fixed visual selection changed")
        for j in selected:
            sample = samples[j]
            baseline = np.load(source / sample["raw_maps"][cfg["baseline_model"]], allow_pickle=False)
            global_record = read(cat / "maps_global.json")[j]
            global_map = np.load(global_record["source"], allow_pickle=False)
            maps = [baseline, global_map]
            for v in ("local_guarded", "protection_removed"):
                record = read(cat / f"maps_{v}.json")[j]
                with np.load(cat / record["file"], allow_pickle=False) as a:
                    maps.append(a["scores"].copy())
            with Image.open(sample["image_path"]) as im:
                image = np.asarray(im.convert("RGB").resize((256, 256)))
            mask = np.load(source / sample["mask_path"], allow_pickle=False)
            fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
            axes[0].imshow(image)
            axes[0].set_title("Input")
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            # Candidate and ablation retain the raw residual unit; use the
            # same baseline-derived range for these three panels.
            low, high = np.quantile(baseline, (.01, .99))
            if high <= low:
                high = low + 1e-9
            for axis, values, name in zip(axes[2:], maps, ("Baseline", "Global affine", "Protected tail", "Protection removed")):
                if name == "Global affine":
                    own_low, own_high = np.quantile(values, (.01, .99))
                else:
                    own_low, own_high = low, high
                axis.imshow(values, cmap="turbo", vmin=own_low, vmax=max(own_high, own_low + 1e-12))
                axis.set_title(f"{name}\nraw [{values.min():.3g}, {values.max():.3g}]", fontsize=10)
            for axis in axes:
                axis.axis("off")
            fig.suptitle(f"{d}/{sample['sample_id']} (label={sample['label']})", fontsize=11)
            fig.text(.5, .012, "Baseline/candidate/ablation share the baseline 1%-99% display range; global affine uses its own range. Raw scores are not probabilities.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .05, 1, .9))
            file = directory / f"{d}_{c}_{sample['label']}.png"
            fig.savefig(file, dpi=140)
            plt.close(fig)
            links.append(f'<details><summary>{html.escape(d + "/" + sample["sample_id"])}</summary><img loading="lazy" style="width:100%" src="visualizations/{file.name}"></details>')
    path = run / "report.html"
    report = path.read_text(encoding="utf-8")
    addition = '<h2>固定54组对比图</h2><p>沿用原实验每类第一张正常图与第一张异常图，没有按本轮效果挑图。基线、候选和消融使用同一个基线1%/99%显示范围；全局仿射图单位不同，独立换算显示。</p>' + "".join(links)
    addition += '<p><a href="audit_receipt.json">全量重放核对</a> · <a href="isolated_resource_summary.json">独立资源补测</a> · <a href="completion_audit.json">目标逐项核对</a> · <a href="复现说明.txt">复现说明</a></p>'
    path.write_text(report.replace("</html>", addition + "</html>"), encoding="utf-8")


def audit(run):
    import numpy as np
    from industrial_anomaly_lab.tail_calibration import VARIANTS, fit_tail_calibration, load_tail_calibration
    frozen = read(run / "frozen_protocol.json")
    cfg = frozen["protocol"]
    verify_seal(frozen["sha256"])
    if read(run / "run_state.json")["status"] != "completed":
        raise ValueError("Full experiment must finish first")
    source = ROOT / cfg["source_run"]
    normals = ROOT / cfg["normal_source_run"]
    rows = read(run / "category_metrics.json")
    if len(rows) != 108 or len({(r["dataset"], r["category"], r["variant"]) for r in rows}) != 108:
        raise ValueError("Incomplete 27 x 4 result matrix")
    monitor = Monitor(cfg["resource_policy"])
    monitor.sample()
    monitor.thread.start()
    maps = samples_count = normals_count = 0
    max_error = 0.
    try:
        for d, c in frozen["tasks"]:
            monitor.set_stage(f"replay:{d}/{c}")
            cat = run / "categories" / d / c
            nd = normals / "normal" / d / c
            for directory in (cat, nd):
                for rel, expected in read(directory / "completion.json")["sha256"].items():
                    if sha(directory / rel) != expected:
                        raise ValueError("Category/cache artifact hash changed")
            records = read(nd / "records.json")
            normals_count += len(records)
            fit_records = [r for r in records if r["partition"] == "statistics"]
            threshold_records = [r for r in records if r["partition"] == "threshold"]
            if {r["path"] for r in fit_records} & {r["path"] for r in threshold_records}:
                raise ValueError("Normal partition overlap")
            for r in records:
                if sha(ROOT / r["path"]) != r["sha256"]:
                    raise ValueError("Normal source image changed")
            stats = np.stack([np.load(nd / r["raw_map"], allow_pickle=False) for r in fit_records])
            held = np.stack([np.load(nd / r["raw_map"], allow_pickle=False) for r in threshold_records])
            reconstructed = fit_tail_calibration(stats, tail_quantile=cfg["tail_quantile"], smoothing_sigma=cfg["smoothing_sigma"])
            model = load_tail_calibration(cat / "calibration.npz")
            for k in ("global_center", "global_scale", "global_tail", "envelope", "count"):
                np.testing.assert_array_equal(getattr(reconstructed, k), getattr(model, k))
            samples = lines(source / "categories" / d / c / "samples.jsonl")
            samples_count += len(samples)
            for s in samples:
                if sha(source / s["raw_maps"][cfg["baseline_model"]]) != s["raw_map_sha256"][cfg["baseline_model"]] or sha(source / s["mask_path"]) != s["mask_sha256"]:
                    raise ValueError("Source Test map/mask changed")
            for v in VARIANTS:
                manifest = read(cat / f"maps_{v}.json")
                if len(manifest) != len(samples):
                    raise ValueError("Incomplete output maps")
                def replay(j, sample, scores):
                    nonlocal maps, max_error
                    record = manifest[j]
                    if record["sample_id"] != sample["sample_id"]:
                        raise ValueError("Output map order mismatch")
                    if record.get("reused"):
                        if v == "baseline":
                            actual = np.load(record["source"], allow_pickle=False)
                        else:
                            actual = np.load(record["source"], allow_pickle=False)
                    else:
                        if sha(cat / record["file"]) != record["sha256"]:
                            raise ValueError("Compressed map hash mismatch")
                        with np.load(cat / record["file"], allow_pickle=False) as archive:
                            if archive.files != ["scores"]:
                                raise ValueError("Unexpected compressed-map fields")
                            actual = archive["scores"]
                    error = float(np.max(np.abs(actual - scores)))
                    max_error = max(max_error, error)
                    if error != 0:
                        raise ValueError("Full score replay mismatch")
                    maps += 1
                    return {}
                rates, _ = operating_rates(model, v, held, samples, source, cfg["baseline_model"], monitor, replay)
                expected_row = next(r for r in rows if (r["dataset"], r["category"], r["variant"]) == (d, c, v))
                for k, value in rates.items():
                    if k not in ("apply_median_ms", "apply_p95_ms") and value != expected_row[k]:
                        raise ValueError(f"Operating point/property replay mismatch: {k}")
            print(f"audit {d}/{c}", flush=True)
        macros = read(run / "dataset_macro_metrics.json")
        for row in macros:
            subset = [r for r in rows if (r["dataset"], r["variant"]) == (row["dataset"], row["variant"])]
            for key in row:
                if key not in ("dataset", "variant", "categories") and row[key] != float(np.mean([r[key] for r in subset])):
                    raise ValueError("Macro replay mismatch")
    finally:
        write(run / "resources_audit.json", monitor.close())
    directory = run / "isolated_resources"
    directory.mkdir(exist_ok=True)
    for rep in range(3):
        for v in ("baseline", "local_guarded"):
            out = directory / f"{v}_{rep}.json"
            if not out.exists():
                subprocess.run([sys.executable, str(Path(__file__)), "--run", str(run), "--worker", v, "--output", str(out)], check=True, cwd=ROOT)
    resource = {}
    for v in ("baseline", "local_guarded"):
        trials = [read(directory / f"{v}_{rep}.json") for rep in range(3)]
        if any(r["resources"]["resource_violation"] for r in trials):
            raise ValueError("Resource trial failed reserve")
        resource[v] = {k: float(np.median([r[k] for r in trials])) for k in ("fit_seconds", "apply_median_ms", "apply_p95_ms", "peak_working_set_bytes", "peak_commit_bytes")}
    resource["extra_peak_working_set_mib"] = (resource["local_guarded"]["peak_working_set_bytes"] - resource["baseline"]["peak_working_set_bytes"]) / 1024**2
    resource["extra_peak_commit_mib"] = (resource["local_guarded"]["peak_commit_bytes"] - resource["baseline"]["peak_commit_bytes"]) / 1024**2
    resource["scope"] = "Differences of three independent CPU-process medians for fixed bottle/32-normal-map added calibration stage; no whole-dataset or full-backbone peak claim"
    write(run / "isolated_resource_summary.json", resource)
    decision = read(run / "decision.json")
    component_support = all(r["candidate_minus_ablation"]["image_auroc"] >= 0 and (r["candidate_minus_ablation"]["pixel_ap"] > 0 or r["candidate_minus_ablation"]["aupro_0_30"] > 0) for r in decision["checks"])
    small_memory = resource["extra_peak_working_set_mib"] <= 128 and resource["extra_peak_commit_mib"] <= 128
    success = decision["effectiveness_gate_passed"] and component_support and small_memory
    write(run / "audit_receipt.json", {"verified_at": now(), "all_27_categories": True, "metric_rows": len(rows),
        "normal_images_reused": normals_count, "test_images": samples_count, "all_variant_maps_replayed": maps,
        "max_map_replay_error": max_error, "normal_parameter_and_partition_replay": "passed",
        "operating_point_and_counts_replay": "passed", "no_amplification_and_high_tail_identity": "passed",
        "source_and_input_authentication": "passed", "macro_replay": "passed",
        "metric_scope": "Both new variants fully evaluated with the unchanged exact evaluator on losslessly staged float64 maps. Baseline/global reuse authenticated complete earlier metrics and every map is replayed here. No redundant full pixel-sort repeat.",
        "key_component_supported_in_both_dataset_macros": component_support, "added_stage_memory_within_128_mib": small_memory})
    write(run / "completion_audit.json", {"goal_achieved": bool(success), "candidate_effectiveness": decision["status"],
        "effectiveness_gate_passed": decision["effectiveness_gate_passed"], "component_attribution_supported": component_support,
        "small_added_memory": small_memory, "replay_error": max_error, "datasets": decision["checks"],
        "single_pass_fitting_evidence": "Previous full same-backbone controls and independent resource trials; no PCA refit in this candidate run.",
        "remaining": [] if success else ["Two-dataset consistent benefit and/or component/resource requirement not met. Preserve all regressions; do not select new parameters from these Test scores."]})
    render(run, cfg, frozen["tasks"])
    with (run / "实验报告.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"\n独立核对：{maps}张四组异常图逐张重放误差为0。独立瓶子类别资源对照各3次，校准阶段新增工作集峰值中位数差{resource['extra_peak_working_set_mib']:.3f} MiB，提交内存峰值差{resource['extra_peak_commit_mib']:.3f} MiB。只代表该校准阶段，不代表完整骨干进程。\n总目标逐项核对：{'已达到本轮预定工程要求' if success else '尚未达到；候选不作为成功改进交付'}。\n")
    text = f'''复现说明

当前实验：{run}
配置：{ROOT / 'configs/tail_calibration_20261015.json'}
代码：scripts/run_tail_calibration.py、scripts/audit_tail_calibration.py、src/industrial_anomaly_lab/tail_calibration.py。

在E:\\3k项目目录、现有环境中，OUTPUT替换成不存在的目录：
.venv\\Scripts\\python.exe -u scripts\\run_tail_calibration.py --output-dir runs/quick/OUTPUT
.venv\\Scripts\\python.exe -u scripts\\audit_tail_calibration.py --run runs/quick/OUTPUT

输入依赖已完成的route1-controls-20261014和score-calibration-20261015-v2。后者提供1724张正常训练异常图及原始全局对照，前者提供原版模型和3887张测试图的原始异常图。它们与配置、代码在frozen_protocol.json中固定。复现需要这些输入，代码快照单独复制不能独立运行。

本轮只改变评分机制，没有新DINO前向或PCA拟合。完整结果含所有27类与退步项，图像AUROC使用输出异常图的最大值，不借用另一张图的分数。像素AP、AUROC与AUPRO使用原有精确256平面评测器。新的预测保存为无损float64 NPZ，评测时暂存NPY，之后只清理本轮生成的暂存文件。

正常统计/阈值分区不重叠，但原PCA已经见过全部正常训练数据，没有严格独立校准覆盖率保证。测试集此前已访问，方案在本轮评测前冻结，但不能声称新盲测。既有分位数、正常校准和尾部抑制技术的出处保留，不宣称已证明论文原创性。
'''
    (run / "复现说明.txt").write_text(text, encoding="utf-8")
    for rel, digest in frozen["sha256"].items():
        if rel.startswith(("scripts/", "src/", "configs/")):
            target = run / "code" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
            if sha(target) != digest:
                raise ValueError("Code snapshot mismatch")
    for rel in ("scripts/audit_tail_calibration.py", "tests/test_tail_calibration.py"):
        target = run / "code" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    provenance = run / "public_sources"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "vendor/SubspaceAD/LICENSE", provenance / "SubspaceAD-LICENSE")
    (provenance / "sources.txt").write_text("SubspaceAD https://github.com/CLendering/SubspaceAD\nDINOv2 https://github.com/facebookresearch/dinov2\nRelated normal calibration https://www.mdpi.com/2079-9292/15/16/3654\nRelated normal-tail suppression https://www.mdpi.com/2313-433X/12/10/466\nThese references do not prove this adapter novel.\n", encoding="utf-8")
    verify_seal(frozen["sha256"])
    write(run / "artifact_manifest.json", {"sha256": {p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker", choices=("baseline", "local_guarded"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Run must stay in project")
    if args.worker:
        worker(run, args.worker, args.output)
    else:
        audit(run)
