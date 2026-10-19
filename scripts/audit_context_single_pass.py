"""Independent full metric replay, fresh-image checks and isolated resources."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_score_calibration import read, lines, sha, write, now, verify_seal
from run_context_single_pass import load_model, normal_thresholds, load_aligned_single_pass
from run_route1_resource_trial import Monitor


def worker(run, variant, output):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from industrial_anomaly_lab.context_calibration import ContextCalibration, fit_context_from_cache, quantized_descriptor
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model
    threadpool_limits(4)
    cfg = read(run / "frozen_protocol.json")["protocol"]
    p = read(ROOT / "configs/route1_controls_20261014.json")
    monitor = Monitor(cfg["resource_policy"])
    monitor.sample()
    monitor.thread.start()
    monitor.set_stage("fit_calibration")
    tick = time.perf_counter()
    if variant == "conditional_tail":
        with np.load(ROOT / cfg["normal_context_run"] / "categories/mvtec_ad/bottle/normal_context.npz", allow_pickle=False) as a:
            data = {k: a[k].copy() for k in a.files}
        model, _ = fit_context_from_cache(data, cfg["minimum_group_tokens"], cfg["tail_quantile"])
        del data
    else:
        model = ContextCalibration(np.zeros((8, 16), np.float32), np.ones(8), 1., 0., 1.)
    fit_seconds = time.perf_counter() - tick
    monitor.set_stage("load_backbone")
    extractor = LocalDinoExtractor(ROOT / p["weights_dir"], expected_checkpoint_sha256=p["checkpoint_sha256"],
        expected_artifact_sha256=p["weight_files_sha256"], device="cpu", cpu_threads=4)
    pca = load_aligned_single_pass(ROOT / cfg["source_run"] / "categories/mvtec_ad/bottle/models")
    image = ROOT / read(ROOT / cfg["normal_context_run"] / "categories/mvtec_ad/bottle/normal_records.json")[0]["path"]
    monitor.set_stage("forward")
    tick = time.perf_counter()
    tokens = extractor(image).reshape(-1, 384)
    forward = time.perf_counter() - tick
    raw = pca.reconstruction_scores(tokens).reshape(32, 32)
    timings = []
    descriptor_timings = []
    checksum = 0.
    monitor.set_stage("apply")
    for _ in range(100):
        monitor.check()
        tick = time.perf_counter()
        z = quantized_descriptor(tokens, pca, cfg["context_dim"]) if variant == "conditional_tail" else None
        descriptor_timings.append(time.perf_counter() - tick)
        tick = time.perf_counter()
        score = model.apply(raw, z, variant)
        timings.append(time.perf_counter() - tick)
        checksum += float(score.mean())
    resource = monitor.close()
    write(output, {"variant": variant, "calibration_fit_seconds": fit_seconds, "cpu_forward_seconds": forward,
        "descriptor_median_ms": float(np.median(descriptor_timings) * 1000), "apply_median_ms": float(np.median(timings) * 1000),
        "apply_p95_ms": float(np.quantile(timings, .95) * 1000), "checksum": checksum,
        "peak_working_set_bytes": resource["process_peak_working_set_bytes"], "peak_commit_bytes": resource["process_peak_commit_bytes"],
        "resources": resource, "scope": "Own independent full CPU-backbone process, fixed first selected bottle normal image; normal-cache calibration fit included only in candidate; 100 scoring repetitions. CPU worker measurements are separate from paired GPU benchmark timings."})


def render(run, cfg, tasks):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    source = ROOT / cfg["source_run"]
    directory = run / "visualizations"
    directory.mkdir(exist_ok=True)
    links = []
    for d, c in tasks:
        cat = run / "categories" / d / c
        model = load_model(cat / "calibration.npz")
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        frames = read(cat / "frames.json")
        chosen = [i for i, s in enumerate(samples) if s["selected"]]
        if len(chosen) != 2 or {samples[j]["label"] for j in chosen} != {0, 1}:
            raise ValueError("Fixed normal/anomalous visual selection changed")
        for j in chosen:
            sample = samples[j]
            with np.load(cat / frames[j]["cache"], allow_pickle=False) as a:
                raw, z = a["residual_grid"], a["descriptor"]
            maps = [model.apply(raw, z, v) for v in cfg["variants"]]
            with Image.open(sample["image_path"]) as im:
                image = np.asarray(im.convert("RGB").resize((256, 256)))
            mask = np.load(source / sample["mask_path"], allow_pickle=False)
            low, high = np.quantile(maps[0], (.01, .99))
            fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
            axes[0].imshow(image)
            axes[0].set_title("Input")
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            for axis, values, title in zip(axes[2:], maps, ("Baseline", "Global affine", "Content conditioned", "Conditioning removed")):
                display_low, display_high = np.quantile(values, (.01, .99)) if title == "Global affine" else (low, high)
                axis.imshow(values, cmap="turbo", vmin=display_low, vmax=max(display_high, display_low + 1e-12))
                axis.set_title(f"{title}\nraw [{values.min():.3g}, {values.max():.3g}]", fontsize=10)
            for axis in axes:
                axis.axis("off")
            fig.suptitle(f"{d}/{sample['sample_id']} (label={sample['label']})", fontsize=11)
            fig.text(.5, .012, "Baseline/candidate/ablation share baseline display limits; global affine uses its own. Metrics use actual raw maps, never display-normalized values.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .05, 1, .9))
            path = directory / f"{d}_{c}_{sample['label']}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            links.append(f'<details><summary>{html.escape(d + "/" + sample["sample_id"])}</summary><img loading="lazy" style="width:100%" src="visualizations/{path.name}"></details>')
    path = run / "report.html"
    text = path.read_text(encoding="utf-8")
    appendix = '<h2>固定54组可视化</h2><p>沿用每类固定的正常图和异常图，包含下降项。基线、候选与消融使用相同显示范围，全局仿射图独立换算；颜色不是概率。</p>' + "".join(links)
    appendix += '<p><a href="audit_receipt.json">全量指标独立复算</a> · <a href="fresh_image_inference.json">原图重新推理</a> · <a href="isolated_resource_summary.json">独立资源</a> · <a href="completion_audit.json">目标逐项判断</a> · <a href="复现说明.txt">复现说明</a></p>'
    path.write_text(text.replace("</html>", appendix + "</html>"), encoding="utf-8")


def audit(run):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from industrial_anomaly_lab.context_calibration import fit_context_from_cache, quantized_descriptor
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    threadpool_limits(4)
    frozen = read(run / "frozen_protocol.json")
    cfg = frozen["protocol"]
    if read(run / "run_state.json")["status"] != "completed":
        raise ValueError("Full run must finish first")
    verify_seal(frozen["sha256"])
    source = ROOT / cfg["source_run"]
    normal_root = ROOT / cfg["normal_context_run"]
    all_rows = read(run / "category_metrics.json")
    if len(all_rows) != 108 or len({(r["dataset"], r["category"], r["variant"]) for r in all_rows}) != 108:
        raise ValueError("Incomplete four-group matrix")
    monitor = Monitor(cfg["resource_policy"])
    monitor.sample()
    monitor.thread.start()
    replayed = []
    image_count = normal_count = map_count = 0
    max_metric_error = max_cpu_gpu_error = 0.
    try:
        for d, c in frozen["tasks"]:
            monitor.set_stage(f"replay:{d}/{c}")
            cat = run / "categories" / d / c
            for rel, digest in read(cat / "completion.json")["sha256"].items():
                if sha(cat / rel) != digest:
                    raise ValueError("Completed artifact changed")
            normal_cat = normal_root / "categories" / d / c
            for rel, digest in read(normal_cat / "completion.json")["sha256"].items():
                if sha(normal_cat / rel) != digest:
                    raise ValueError("Normal content input changed")
            normal_records = read(normal_cat / "normal_records.json")
            train_manifest = read(source / "categories" / d / c / "train_manifest.json")
            normal_training = {r["path"]: r["sha256"] for r in train_manifest}
            for record in normal_records:
                if normal_training.get(record["path"]) != record["sha256"] or sha(ROOT / record["path"]) != record["sha256"]:
                    raise ValueError("Calibration inputs are not authenticated normal training images")
            normal_count += len(normal_records)
            with np.load(normal_cat / "normal_context.npz", allow_pickle=False) as a:
                cache = {k: a[k].copy() for k in a.files}
            refitted, _ = fit_context_from_cache(cache, cfg["minimum_group_tokens"], cfg["tail_quantile"])
            model = load_model(cat / "calibration.npz")
            for field in ("centers", "tails", "global_tail", "global_center", "global_scale"):
                np.testing.assert_array_equal(getattr(model, field), getattr(refitted, field))
            samples = lines(source / "categories" / d / c / "samples.jsonl")
            frames = read(cat / "frames.json")
            if len(frames) != len(samples):
                raise ValueError("Incomplete shared-frame cache")
            image_count += len(samples)
            masks = [source / sample["mask_path"] for sample in samples]
            for sample, frame in zip(samples, frames, strict=True):
                if frame["sample_id"] != sample["sample_id"] or sha(cat / frame["cache"]) != frame["cache_sha256"]:
                    raise ValueError("Frame cache content or order changed")
                if sha(Path(sample["image_path"])) != sample["image_sha256"] or sha(source / sample["mask_path"]) != sample["mask_sha256"]:
                    raise ValueError("Original Test image/mask changed")
            for v in cfg["variants"]:
                expected = next(row for row in all_rows if (row["dataset"], row["category"], row["variant"]) == (d, c, v))
                threshold = normal_thresholds(model, cache, v)
                if any(threshold[k] != expected[k] for k in threshold):
                    raise ValueError("Normal threshold replay mismatch")
                staging = run / "audit" / "staging" / d / c / v
                staging.mkdir(parents=True, exist_ok=True)
                files = []
                counts = {k: 0 for k in expected["raw_counts"]}
                for j, (sample, frame) in enumerate(zip(samples, frames, strict=True)):
                    monitor.check()
                    with np.load(cat / frame["cache"], allow_pickle=False) as a:
                        raw, z = a["residual_grid"], a["descriptor"]
                    if raw.dtype != np.float32 or z.dtype != np.float16 or raw.shape != (32, 32) or z.shape != (1024, cfg["context_dim"]):
                        raise ValueError("Frame precision/geometry changed")
                    scores = model.apply(raw, z, v)
                    if v == "baseline":
                        prior = source / sample["raw_maps"]["official_cpu_two_pass"]
                        if sha(prior) != sample["raw_map_sha256"]["official_cpu_two_pass"]:
                            raise ValueError("Original CPU map changed")
                        old = np.load(prior, allow_pickle=False)
                        max_cpu_gpu_error = max(max_cpu_gpu_error, float(np.max(np.abs(old - scores))))
                        if not np.allclose(old, scores, atol=1e-4, rtol=3e-5):
                            raise ValueError("Whole-matrix CPU/GPU numerical control failed")
                    mask = np.load(masks[j], allow_pickle=False).astype(bool)
                    counts["background_pixels"] += int((~mask).sum())
                    counts["background_fp"] += int(np.count_nonzero(scores[~mask] > threshold["pixel_threshold"]))
                    counts["foreground_pixels"] += int(mask.sum())
                    counts["foreground_tp"] += int(np.count_nonzero(scores[mask] > threshold["pixel_threshold"]))
                    if sample["label"] == 0:
                        counts["normal_pixels"] += scores.size
                        counts["normal_fp"] += int(np.count_nonzero(scores > threshold["pixel_threshold"]))
                        counts["normal_images"] += 1
                        counts["normal_image_fp"] += int(scores.max() > threshold["image_threshold"])
                    else:
                        counts["anomaly_images"] += 1
                        counts["anomaly_image_tp"] += int(scores.max() > threshold["image_threshold"])
                    if shutil.disk_usage(run).free < cfg["resource_policy"]["minimum_disk_gib"] * 1024**3:
                        raise RuntimeError("Audit disk reserve reached")
                    outfile = staging / f"{j:06d}.npy"
                    np.save(outfile, scores, allow_pickle=False)
                    files.append(outfile)
                    map_count += 1
                metrics = evaluate_from_files(files, masks, [sample["label"] for sample in samples], temp_root=run / "audit" / "metric_tmp")
                error = max(abs(metrics[k] - expected[k]) for k in metrics)
                max_metric_error = max(max_metric_error, error)
                if error != 0 or counts != expected["raw_counts"]:
                    raise ValueError("Full exact metric or operating-point replay mismatch")
                replayed.append({"dataset": d, "category": c, "variant": v, **metrics, "max_metric_error": error})
                if not staging.resolve().is_relative_to((run / "audit" / "staging").resolve()):
                    raise ValueError("Unsafe audit staging cleanup")
                for path in files:
                    if path.resolve().parent != staging.resolve():
                        raise ValueError("Unexpected audit staging path")
                    path.unlink()
                staging.rmdir()
            print(f"full replay {d}/{c}", flush=True)
            gc.collect()
        write(run / "audit/replayed_metrics.json", replayed)
        verify_seal(frozen["sha256"])
    finally:
        write(run / "audit/resource_receipt.json", monitor.close())
    # Fresh original-image inference uses the same GPU backend as this trial.
    p = read(ROOT / "configs/route1_controls_20261014.json")
    extractor = LocalDinoExtractor(ROOT / p["weights_dir"], expected_checkpoint_sha256=p["checkpoint_sha256"],
        expected_artifact_sha256=p["weight_files_sha256"], device="cuda:0", cpu_threads=4, max_cuda_memory_gib=2.)
    fresh = []
    for d, c in (("mvtec_ad", "bottle"), ("visa", "cashew")):
        cat = run / "categories" / d / c
        model = load_model(cat / "calibration.npz")
        pca = load_aligned_single_pass(source / "categories" / d / c / "models")
        samples = lines(source / "categories" / d / c / "samples.jsonl")
        frames = read(cat / "frames.json")
        for j in [j for j, sample in enumerate(samples) if sample["selected"]]:
            tokens = extractor(samples[j]["image_path"]).reshape(-1, 384)
            raw = pca.reconstruction_scores(tokens).reshape(32, 32)
            z = quantized_descriptor(tokens, pca, cfg["context_dim"])
            with np.load(cat / frames[j]["cache"], allow_pickle=False) as a:
                cached_raw, cached_z = a["residual_grid"], a["descriptor"]
            if not np.array_equal(z, cached_z) or not np.array_equal(raw, cached_raw):
                raise ValueError("Fresh original-image frame differs from the actual trial")
            for v in cfg["variants"]:
                a, b = model.apply(raw, z, v), model.apply(cached_raw, cached_z, v)
                error = float(np.max(np.abs(a - b)))
                if error:
                    raise ValueError("Fresh original-image calibrated score mismatch")
                fresh.append({"dataset": d, "category": c, "sample_id": samples[j]["sample_id"], "variant": v, "max_score_map_error": error})
    del extractor
    gc.collect()
    write(run / "fresh_image_inference.json", {"original_images": 4, "paired_maps": 16, "rows": fresh, "scope": "Four fixed original normal/defect images, actual GPU backbone rerun; complements full-cache metric replay"})
    directory = run / "isolated_resources"
    directory.mkdir(exist_ok=True)
    for repeat in range(3):
        def launch(v):
            out = directory / f"{v}_{repeat}.json"
            if not out.exists():
                subprocess.run([sys.executable, str(Path(__file__)), "--run", str(run), "--worker", v, "--output", str(out)], check=True, cwd=ROOT)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, v) for v in ("baseline", "conditional_tail")]
            for future in futures:
                future.result()
    resource = {}
    for v in ("baseline", "conditional_tail"):
        trials = [read(directory / f"{v}_{r}.json") for r in range(3)]
        if any(r["resources"]["resource_violation"] for r in trials):
            raise ValueError("Resource trial violated reserve")
        resource[v] = {k: float(np.median([r[k] for r in trials])) for k in ("calibration_fit_seconds", "descriptor_median_ms", "apply_median_ms", "apply_p95_ms", "peak_working_set_bytes", "peak_commit_bytes")}
    resource["extra_peak_working_set_mib"] = (resource["conditional_tail"]["peak_working_set_bytes"] - resource["baseline"]["peak_working_set_bytes"]) / 1024**2
    resource["extra_peak_commit_mib"] = (resource["conditional_tail"]["peak_commit_bytes"] - resource["baseline"]["peak_commit_bytes"]) / 1024**2
    resource["scope"] = "Differences of three independent-process medians, fixed bottle image, full CPU backbone; paired worker scheduling may introduce contention. Whole-matrix GPU timings are reported separately."
    write(run / "isolated_resource_summary.json", resource)
    small_memory = resource["extra_peak_working_set_mib"] <= 128 and resource["extra_peak_commit_mib"] <= 128
    decision = read(run / "decision.json")
    achieved = decision["effectiveness_gate_passed"] and small_memory
    write(run / "audit_receipt.json", {"verified_at": now(), "all_27_categories": True, "normal_images": normal_count,
        "test_images": image_count, "four_group_maps_regenerated": map_count, "exact_metric_rows_recomputed": len(replayed),
        "max_metric_replay_error": max_metric_error, "max_cpu_gpu_baseline_map_error": max_cpu_gpu_error,
        "fresh_original_images": 4, "fresh_calibrated_maps": 16, "fresh_image_map_error": 0,
        "normal_only_fit_and_threshold_input_authentication": "passed", "normal_parameters_and_thresholds": "passed",
        "all_frame_and_source_hashes": "passed", "operating_point_counts": "passed", "isolated_resource_trials": 6})
    write(run / "completion_audit.json", {"goal_achieved": bool(achieved), "effectiveness_gate_passed": decision["effectiveness_gate_passed"],
        "same_gain_branch_both_datasets": decision["same_gain_branch_both_datasets"],
        "image_detection_noninferiority": all(r["image_noninferiority"] for r in decision["checks"]),
        "component_attribution": decision["component_support_both_datasets"], "small_runtime_and_parameter_state": decision["resource_timing_and_parameter_gate"],
        "small_added_process_memory": small_memory, "full_exact_replay": max_metric_error == 0,
        "single_pass_pca": "Already validated by the complete same-backbone experiment and independent resource matrix. Current score experiment reuses fitted PCA; no new fit speed is claimed.",
        "remaining": [] if achieved else ["Consistent two-dataset effect and/or resource/component condition failed; preserve candidate as a rejected experiment, not a successful improvement."]})
    render(run, cfg, frozen["tasks"])
    with (run / "实验报告.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n独立复算完成：108条四指标结果、{map_count}张四组异常图重建、操作点计数均一致；另从4张原图重新前向，16张配对图误差为0。固定瓶子图像、各3个独立CPU进程的资源补测，新增工作集峰值中位数差为{resource['extra_peak_working_set_mib']:.3f} MiB，新增提交内存峰值差为{resource['extra_peak_commit_mib']:.3f} MiB；不能推广为整套数据进程内存差。\n目标逐项核对：{'达到预定工程效果要求' if achieved else '仍未达到目标，候选不成立'}。\n")
    (run / "复现说明.txt").write_text(f'''复现内容条件校准实验

实验位置：{run}
在E:\\3k项目目录，现有环境中，OUTPUT替换为新目录：
.venv\\Scripts\\python.exe -u scripts\\run_context_single_pass.py --output-dir runs/quick/OUTPUT
.venv\\Scripts\\python.exe -u scripts\\audit_context_single_pass.py --run runs/quick/OUTPUT

需要原始MVTec AD/VisA数据、本地DINOv2-S权重、完整route1-controls-20261014，以及normal-context-pilot-20261015的正常缓存和审核记录。代码与输入摘要保存在frozen_protocol.json。恢复用--resume，源码和协议必须一致。

27类共享既有PCA模型；四组共享每张测试图的新GPU前向。全局校准在渲染后进行，排序不变；内容条件与消融在32网格评分后使用同一渲染流程。图像分数都来自各自输出图的最大值，没有借用另一组分数。每张新基线对照原CPU缓存，数值差异及指标变化逐类保留。

预测压缩文件保存float32原始残差网格和固定float16内容描述，可重建四组256异常图。评测临时NPY文件和排序临时文件仅在本轮目录生成并清理；原始数据、失败试验及有效交付目录未修改。异常图展示不进入评测。

单遍PCA的完整精度与资源证据在此前同骨干交付中；本轮复用拟合模型，没有把旧PCA收益算作评分组件收益。统计与阈值正常分区不重叠，但原PCA见过两者，不能声称严格独立校准。测试集此前已访问，不能声称新盲测。

DINOv2、SubspaceAD、聚类与正常分位数校准的来源继续保留。实验代码由本项目整理实现，不代表其基础方法原创，效果验收也不等于期刊录用标准。
''', encoding="utf-8")
    for rel, expected in frozen["sha256"].items():
        if rel.startswith(("scripts/", "src/", "configs/")):
            target = run / "code" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
            if sha(target) != expected:
                raise ValueError("Frozen source snapshot changed")
    for rel in ("scripts/audit_context_single_pass.py", "tests/test_context_calibration.py"):
        target = run / "code" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    source_dir = run / "public_sources"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "vendor/SubspaceAD/LICENSE", source_dir / "SubspaceAD-LICENSE")
    (source_dir / "sources.txt").write_text("SubspaceAD https://github.com/CLendering/SubspaceAD\nDINOv2 https://github.com/facebookresearch/dinov2\nMiniBatchKMeans https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.MiniBatchKMeans.html\nNormal score calibration https://www.mdpi.com/2079-9292/15/16/3654\nPrototype calibration research https://www.sciencedirect.com/science/article/abs/pii/S0957417426002447\nExisting techniques and own experimental implementation are distinguished; no originality/SCI publication guarantee.\n", encoding="utf-8")
    verify_seal(frozen["sha256"])
    write(run / "artifact_manifest.json", {"sha256": {p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker", choices=("baseline", "conditional_tail"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "runs/quick").resolve()):
        raise ValueError("Run must be inside project")
    if args.worker:
        worker(run, args.worker, args.output)
    else:
        audit(run)
