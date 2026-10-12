"""Add a completed DINOv2-S/SubspaceAD run to the existing single folder.

No training, downloads, archives, or score-based visual selection are performed.
The default refuses partial experiments or replacement of historical evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_oas_delivery import copy_file, digest, member, read_json, read_jsonl, write_csv, write_json

sys.dont_write_bytecode = True
EXTENSION = "优化实验/DINO小型SubspaceAD验证"
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
NEW_METHODS = ("subspace_max", "subspace_top1p")
METHODS = ("baseline_max", "baseline_top1p", "ml_max", "ml_top1p", "oas_max", "oas_top1p", *NEW_METHODS)
REFERENCE = "https://github.com/CLendering/SubspaceAD/tree/ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
WEIGHT_REFERENCE = "https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056"


def collect_results(package: Path, run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    state = read_json(run / "run_state.json")
    if state.get("status") != "completed":
        raise ValueError("The full SubspaceAD run must be completed before folder mutation")
    config = read_json(run / "run_config.json")
    tasks = [(str(d), str(c)) for d, c in config["expected_matrix"]["categories"]]
    if len(tasks) != 27 or len(set(tasks)) != 27:
        raise ValueError("Expected the frozen complete 27-category matrix")
    control_path = package / "优化实验/OAS验证/category_metrics.csv"
    with control_path.open(encoding="utf-8-sig", newline="") as stream:
        old = list(csv.DictReader(stream))
    controls = {(r["dataset"], r["category"], r["method"]): r for r in old}
    if len(controls) != 162 or {(d, c) for d, c, _ in controls} != set(tasks):
        raise ValueError("Complete original baseline/top1p/ML/OAS evidence is required")
    rows = [{"dataset": r["dataset"], "category": r["category"], "method": r["method"],
             "seed": int(r["seed"]), "test_images": int(r["test_images"]),
             **{k: float(r[k]) for k in METRICS}} for r in old]
    new = read_jsonl(run / "results/category_metrics.jsonl")
    expected = {(d, c, v) for d, c in tasks for v in NEW_METHODS}
    if len(new) != 54 or {(r["dataset"], r["category"], r["variant"]) for r in new} != expected:
        raise ValueError("Expected 54 completed frozen SubspaceAD score rows")
    samples = []
    normal_train_images = 0
    for dataset, category in tasks:
        directory = run / f"categories/{dataset}/{category}"
        completion = read_json(directory / "completion.json")
        if completion.get("status") != "completed":
            raise ValueError(f"Incomplete category seal: {dataset}/{category}")
        for relative, expected_hash in completion["sha256"].items():
            if digest(member(run, relative)) != expected_hash:
                raise ValueError(f"Category evidence mismatch: {relative}")
        train = read_json(directory / "train_manifest.json")
        test = read_json(directory / "test_inventory.json")
        normal_train_images += len(train)
        if train != read_json(member(package, f"models/{dataset}/{category}/train_manifest.json")) or test != read_json(member(package, f"models/{dataset}/{category}/test_inventory.json")):
            raise ValueError("Normal-train/test inventory differs from original delivery")
        records = read_jsonl(directory / "samples.jsonl")
        if len(records) != int(controls[(dataset, category, "baseline_max")]["test_images"]):
            raise ValueError("Baseline and SubspaceAD test inventory lengths differ")
        for record in records:
            if (record["dataset"], record["category"]) != (dataset, category):
                raise ValueError("Sample category identity mismatch")
            if set(record["raw_maps"]) != {"subspace"}:
                raise ValueError("Exactly one SubspaceAD raw map per test sample is required")
            if digest(member(run, record["raw_maps"]["subspace"])) != record["raw_map_sha256"]["subspace"]:
                raise ValueError("SubspaceAD raw-map digest mismatch")
            if digest(member(run, record["mask_path"])) != record["mask_sha256"]:
                raise ValueError("Evaluation-mask digest mismatch")
            samples.append(record)
    if len(samples) != 3887 or len({(r["dataset"], r["category"], r["sample_id"]) for r in samples}) != 3887:
        raise ValueError("Expected 3,887 unique original test samples")
    if normal_train_images != 12288:
        raise ValueError("Full 12,288 normal training images are required")
    for record in new:
        baseline = controls[(record["dataset"], record["category"], "baseline_max")]
        if record.get("status") != "completed" or int(record["test_images"]) != int(baseline["test_images"]):
            raise ValueError("New metric status/sample count mismatch")
        rows.append({"dataset": record["dataset"], "category": record["category"], "method": record["variant"],
                     "seed": int(record["seed"]), "test_images": int(record["test_images"]),
                     **{k: float(record[k]) for k in METRICS}})
    for row in rows:
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
            raise ValueError("Finite unit-interval metrics required")
    index = {(r["dataset"], r["category"], r["method"]): r for r in rows}
    for dataset, category in tasks:
        for stem in ("baseline", "ml", "oas", "subspace"):
            maximum = index[(dataset, category, stem + "_max")]
            top1p = index[(dataset, category, stem + "_top1p")]
            if any(maximum[k] != top1p[k] for k in METRICS[1:]):
                raise ValueError("Image-score aggregation must not change localization metrics")
    rows.sort(key=lambda r: (r["dataset"], r["category"], METHODS.index(r["method"])))
    return rows, samples, state


def make_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    index = {(r["dataset"], r["category"], r["method"]): r for r in rows}
    deltas = []
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
        if row["method"] not in NEW_METHODS:
            continue
        score = "top1p" if row["method"].endswith("top1p") else "max"
        for reference in ("baseline_" + score, "oas_" + score):
            old = index[(row["dataset"], row["category"], reference)]
            deltas.append({"dataset": row["dataset"], "category": row["category"], "method": row["method"], "reference": reference,
                           **{k + "_delta_pp": 100 * (row[k] - old[k]) for k in METRICS}})
    macros = [{"dataset": d, "method": m, "categories": len(group), "test_images": sum(r["test_images"] for r in group),
               **{k: sum(r[k] for r in group) / len(group) for k in METRICS}}
              for (d, m), group in sorted(grouped.items(), key=lambda item: (item[0][0], METHODS.index(item[0][1])))]
    macro_index = {(r["dataset"], r["method"]): r for r in macros}
    macro_deltas = []
    for row in macros:
        if row["method"] not in NEW_METHODS:
            continue
        score = "top1p" if row["method"].endswith("top1p") else "max"
        for reference in ("baseline_" + score, "oas_" + score):
            old = macro_index[(row["dataset"], reference)]
            macro_deltas.append({"dataset": row["dataset"], "method": row["method"], "reference": reference,
                                 **{k + "_delta_pp": 100 * (row[k] - old[k]) for k in METRICS}})
    return macros, deltas, macro_deltas


def measured_resources(run: Path, benchmark: Path) -> dict[str, Any]:
    monitors = sorted(run.parent.glob(run.name + "-monitor-*"))
    observations = [r for p in monitors if (p / "resources.jsonl").is_file() for r in read_jsonl(p / "resources.jsonl")]
    peak = [float(r["peak_working_set_gib"]) for r in observations if r.get("peak_working_set_gib") is not None]
    if not peak or any(not math.isfinite(v) or v < 0 for v in peak):
        raise ValueError("Actual full-run process memory observations are required")
    profile = read_json(benchmark / "benchmark.json")
    if profile.get("test_content_accessed") is not False:
        raise ValueError("Normal-only preflight must explicitly exclude test content")
    gpu_values = []
    shared_gpu_values = []
    for row in observations:
        for key in ("owned_gpu_memory_mib", "gpu_process_memory_mib", "gpu_memory_mib"):
            if row.get(key) is not None:
                gpu_values.append(float(row[key]))
        if row.get("gpu_used_gib") is not None:
            shared_gpu_values.append(float(row["gpu_used_gib"]))
    state = read_json(run / "run_state.json")
    owned_states = [state] + [read_json(path) for path in sorted((run / "execution_history").rglob("run_state.json"))]
    owned_allocated = [int(record["cuda_peak_allocated_bytes"]) for record in owned_states if record.get("cuda_peak_allocated_bytes") is not None]
    owned_reserved = [int(record["cuda_peak_reserved_bytes"]) for record in owned_states if record.get("cuda_peak_reserved_bytes") is not None]
    return {"full_monitor_count": len(monitors), "full_resource_samples": len(observations),
            "full_peak_working_set_gib": max(peak), "normal_train_only_profile": profile,
            "gpu_memory_mib_observed_max": max(gpu_values) if gpu_values else None,
            "cuda_peak_allocated_bytes": max(owned_allocated) if owned_allocated else None, "cuda_peak_reserved_bytes": max(owned_reserved) if owned_reserved else None,
            "cuda_peak_basis": "maximum owned PyTorch allocator counters across final and retained historical run states; driver/context allocations are not included",
            "shared_system_gpu_used_gib_observed_max": max(shared_gpu_values) if shared_gpu_values else None,
            "shared_system_gpu_peak_basis": "whole GPU occupancy including concurrent external jobs; not SubspaceAD process VRAM",
            "gpu_peak_basis": "monitor field associated with owned process, when available; otherwise model telemetry in run state",
            "full_monitor_summaries": [{"monitor": p.name, **read_json(p / "summary.json")} for p in monitors if (p / "summary.json").is_file()]}


def draw_selected(package: Path, run: Path, extension: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps

    original = read_json(package / "report/visualization_manifest.json")["samples"]
    identities = {(r["dataset"], r["category"], r["sample_id"]) for r in original}
    if len(original) != 54 or identities != {(r["dataset"], r["category"], r["sample_id"]) for r in samples if r.get("selected")}:
        raise ValueError("The original 54 visual selections must remain unchanged")
    index = {(r["dataset"], r["category"], r["sample_id"]): r for r in samples}
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_path), 19) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    visual = []
    for position, old in enumerate(original):
        identity = (old["dataset"], old["category"], old["sample_id"])
        new = index[identity]
        image_member = "report/" + old["source_image_copy"]
        baseline_member = "report/" + old["raw_maps"]["baseline"]["path"]
        image = member(package, image_member)
        baseline = member(package, baseline_member)
        if digest(image) != old["source_image_sha256"] or new["image_sha256"] != old["source_image_sha256"]:
            raise ValueError("Selected original image identity/hash changed")
        if digest(baseline) != old["raw_maps"]["baseline"]["sha256"]:
            raise ValueError("Selected historical baseline map changed")
        raw = [np.load(baseline, allow_pickle=False), np.load(member(run, new["raw_maps"]["subspace"]), allow_pickle=False)]
        mask = np.load(member(run, new["mask_path"]), allow_pickle=False)
        if any(v.shape != (256, 256) or v.dtype != np.float32 or not np.isfinite(v).all() for v in raw):
            raise ValueError("Expected finite float32 raw maps at 256 square")
        if mask.dtype != np.bool_ or mask.shape != (256, 256) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Selected GT must be a boolean binary 256-square array")
        if int(new["label"]) == 0 and bool(mask.any()):
            raise ValueError("Normal selected sample has a nonempty evaluation mask")
        scales = []
        for array in raw:
            lower, upper = map(float, np.quantile(array, [0.01, 0.99]))
            if upper <= lower:
                upper = lower + 1
            scales.append({"lower": lower, "upper": upper})
        with Image.open(image) as source:
            input_panel = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        panels = [input_panel, Image.fromarray(np.asarray(mask > 0, dtype=np.uint8) * 255).convert("RGB")]
        panels.extend(Image.fromarray(np.asarray(colormaps["viridis"](np.clip((a - scale["lower"]) / (scale["upper"] - scale["lower"]), 0, 1))[..., :3] * 255, dtype=np.uint8)) for a, scale in zip(raw, scales, strict=True))
        canvas = Image.new("RGB", (1160, 408), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"{identity[0]} / {identity[2]}    label={new['label']}", fill="black", font=font)
        for i, (title, panel) in enumerate(zip(("Input", "Ground truth", "PaDiM baseline", "DINO-S / SubspaceAD"), panels, strict=True)):
            x = 16 + i * 288
            draw.text((x, 47), title, fill="black", font=small)
            canvas.paste(panel, (x, 78))
        for i, scale in enumerate(scales):
            draw.text((16 + (i + 2) * 288, 340), f"Raw 1% / 99%: {scale['lower']:.6g} .. {scale['upper']:.6g}", fill="black", font=small)
        draw.text((16, 365), "Same per-panel 1% / 99% display rule and palette; independent scales, not calibrated confidence.", fill="black", font=small)
        draw.text((16, 387), "Different raw score units: Mahalanobis distance / squared reconstruction residual. Display only.", fill="black", font=small)
        relative = f"visualizations/comparison-{position:03d}.png"
        path = extension / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        visual.append({"dataset": identity[0], "category": identity[1], "sample_id": identity[2], "label": int(new["label"]),
                       "seed": int(new["seed"]), "source_image_copy": image_member, "source_image_sha256": digest(image),
                       "figure": relative, "figure_sha256": digest(path), "baseline_raw_map": {"path": baseline_member, "sha256": digest(baseline)},
                       "raw_maps": {"subspace": {"path": new["raw_maps"]["subspace"], "sha256": new["raw_map_sha256"]["subspace"]}},
                       "mask_path": new["mask_path"], "mask_sha256": new["mask_sha256"],
                       "selection": "same original 54 mechanically selected normal/anomalous samples; no result-based selection",
                       "scale": {"scope": "independent per-method panel, same percentile rule", "quantiles": [0.01, 0.99], "baseline": scales[0], "subspace": scales[1], "colormap": "viridis", "raw_score_units_differ": True, "cross_model_colors_calibrated": False},
                       "display_only_normalization": True, "evaluation_raw_maps_modified": False})
    return visual


def write_report(extension: Path, macro: list[dict[str, Any]], deltas: list[dict[str, Any]], macro_deltas: list[dict[str, Any]], visual: list[dict[str, Any]], state: dict[str, Any], resources: dict[str, Any]) -> None:
    primary = [r for r in macro_deltas if r["method"] == "subspace_max" and r["reference"] == "baseline_max"]
    localization_pass = len(primary) == 2 and all(r["aupro_0_30_delta_pp"] > 0 for r in primary)
    finding = ("本次更强公开基线在两个数据集的宏平均 AUPRO 均超过原 PaDiM 基线。" if localization_pass else "本次公开基线没有在两个数据集同时取得宏平均 AUPRO 超过原 PaDiM 的结果。")
    lines = ["# 小型 DINOv2 + SubspaceAD 验收报告", "", "已完成 MVTec AD 15 类和 VisA 12 类，共 3,887 张原测试图的固定方案验证。正常训练样本只用于 PCA 拟合，没有梯度训练或新的数据集。", "", finding, "",
             "这是公开 SubspaceAD 方法的小型 DINOv2-S / 448 分辨率资源适配及本地实测，定位于更强基线。更换公开基线不等于原创改进，测试集已在前轮访问，本轮属于探索性评测；不能据此保证 SCI 三区录用。", "",
             "## 宏平均实测指标", "", "各类别等权平均，表中百分数仅用于显示。所有方法使用同一批原测试样本、256×256 GT、包含正常图全部背景的像素评测及精确 AUPRO(0–0.30)。", "",
             "|数据集|方法|I-AUROC|P-AUROC|P-AP|AUPRO(0–0.30)|", "|---|---|---:|---:|---:|---:|"]
    for row in macro:
        lines.append("|" + "|".join([row["dataset"], row["method"], *[f"{100 * row[k]:.4f}%" for k in METRICS]]) + "|")
    lines += ["", "max 和 top1p 使用同一张原始分数图；top1p 是共同 256×256 平面最高 655 个值的均值，固定于本轮测试之前。图像聚合改变不能当作像素定位提升。", "", "## 同口径变化及下降项", ""]
    for row in macro_deltas:
        if row["reference"].startswith("baseline_"):
            lines.append(f"- {row['dataset']}，{row['method']} 对 {row['reference']}：I-AUROC {row['image_auroc_delta_pp']:+.4f}、P-AUROC {row['pixel_auroc_delta_pp']:+.4f}、P-AP {row['pixel_ap_delta_pp']:+.4f}、AUPRO {row['aupro_0_30_delta_pp']:+.4f} 个百分点。")
    for dataset in ("mvtec_ad", "visa"):
        failed = [r for r in deltas if r["dataset"] == dataset and r["method"] == "subspace_max" and r["reference"] == "baseline_max" and r["aupro_0_30_delta_pp"] < 0]
        values = "、".join(f"{r['category']} ({r['aupro_0_30_delta_pp']:+.4f})" for r in failed) or "无"
        lines.append(f"- {dataset} 的 AUPRO 下降类别及百分点：{values}。")
    lines += ["", "全部 27 类和下降项在 category_metrics.csv / category_deltas.csv 中保留。原 PaDiM、固定收缩失败、OAS 失败以及 top1% 图像评分结果均保留，未挑选局部类别作为全量结论。", "",
             "## 冻结算法与适配", "", "骨干为无 registers 的 facebook/dinov2-small；RGB 完整画面 Bicubic 到 448，使用固定 HF slow Bit 预处理和 ImageNet 均值/标准差。不裁剪、不做 L2 归一化、不旋转增强、无背景掩膜、无白化或 CLAHE。", "",
             "排除 CLS 后，对 hidden_states[-4] 和 [-5] 的原始中间特征取均值，得到每图 32×32×384 patch 特征。每类别全量正常训练图的全部 patch 参与全局中心化 PCA：float64 均值和散布，协方差分母 Npatch−1，精确特征分解，保留累计解释方差首次达到 0.99 的最小维数。模型参数和特征在评分时使用 float32，分数为平方重建残差之和。没有 Test 数据参与 PCA 或维数选择。", "",
             "后处理按上游 OpenCV INTER_LINEAR 32→448，GaussianBlur(kernel=3×3, sigma=4, BORDER_REFLECT_101)，再 OpenCV INTER_LINEAR 448→共同 256 评测平面。GT 最近邻到 256；评测不做逐图归一化。max 和 top1p 都对这张共同评测分数图聚合，上游 native/448 图像评分与本项目共同口径应区分。", "",
             "小型骨干/448 设置来自上游骨干消融，全量正常训练/无增强来自 full-shot 设置；组合是本项目资源适配，不冒充上游官方完整配置的复现成绩。训练图逐张前向后，立即在 float64 合并均值与中心化散布，最终精确特征分解；不保存完整训练特征银行，不因内存策略丢弃样本或提前截断维数。", "",
             "54 组可视化复用前轮固定样本。两方法采用同一 viridis 色盘和相同的逐面板 1%/99% 显示规则，并各自标注原值上下限；分数单位不同（马氏距离与平方重建残差），颜色没有跨模型置信度校准，只比较空间形态。显示归一化不进入保存分数或指标计算，评测保留原值。", "",
             "## 文件夹与复算", "", "本次只扩展原来的一个交付文件夹，没有创建压缩包。run 内包含 27 个正常训练 PCA 模型、全部 3,887 张新 raw map、3,887 张二值 GT 缓存和哈希、54 条指标及正常训练/测试库存。assets/weights 内包含本地冻结权重、配置及其公开许可证，验证无需下载。原 ResNet18 统计不重复复制。", "",
             "根目录 Verify.ps1 验证整个文件夹。新脚本 code/scripts/verify_subspace_delivery.py 默认校验完整缓存、PCA 模型、权重文件和两张精选原图的 CPU 推理；--metrics-categories 27 可从分数图/GT 缓存独立复算全量定位指标。CPU/GPU 浮点差异以固定、明确的逐图容差核对，指标缓存复算要求绝对误差不超过 1e-12。原始照片仅包含 54 张精选样本，完整重新提取特征和拟合 PCA 仍需要正式完整数据集。", "",
             "随附缓存指标复算和精选原图推理可随交付文件夹迁移，验证脚本使用包内相对路径。完整前向/PCA 运行脚本冻结了原 E:/3k 工程布局，并认证原始数据集绝对路径与输入库存；完整重跑应保留该布局，或者在新的工作区重新生成真实数据对应的认证库存、另行冻结新协议。不能通过改写本轮已封存的库存或结果，冒充原运行的无变更复现。", "",
             "先建立等价 Python 3.11 环境，依赖以 code/pyproject.toml 和 code/uv.lock 为准。原先交付环境声明已留档于本子目录的 历史环境，旧实验代码与结果均保留。验证命令在交付文件夹外写临时排序文件和回执：", "",
             "```powershell", "python -B .\\code\\scripts\\verify_subspace_delivery.py --package-root . --inference-samples 2 --metrics-categories 27 --work-dir E:\\3k\\tmp\\subspace-metric-replay --receipt E:\\3k\\tmp\\subspace-delivery-verification.json", "```", "",
             "原 DRAEM 训练仍保持暂停。本轮没有多种子原创模块消融，因此可交付的是可复核的公开强基线实验与整合材料。", "",
             f"公开算法及原配置来源：[SubspaceAD 固定提交]({REFERENCE})；骨干与许可证：[DINOv2-S 固定 HF revision]({WEIGHT_REFERENCE})。来源、适配范围和许可证均随附。", "", "## 实际资源与执行证据", "",
             f"完整运行峰值工作集 {resources['full_peak_working_set_gib']:.4f} GiB，取 {resources['full_monitor_count']} 个监控目录、{resources['full_resource_samples']} 条采样的实际 owned-process peak_working_set_gib 最大值。正常训练-only 预检单列保存，不将其峰值冒充完整运行峰值。", ""]
    if state.get("started_at_utc") and state.get("finished_at_utc"):
        wall = (datetime.fromisoformat(state["finished_at_utc"]) - datetime.fromisoformat(state["started_at_utc"])).total_seconds()
        lines.append(f"全流程墙钟 {wall:.3f} 秒（{wall / 60:.3f} 分钟），含恢复或等待。CPU/GPU 设置及实际峰值以 run_config / run_state / 完整运行监控为准。")
    lines += ["", "```json", json.dumps(state, ensure_ascii=False, indent=2), "```", ""]
    if resources.get("cuda_peak_allocated_bytes") is not None:
        lines += [f"本运行进程的 PyTorch CUDA 实际峰值 allocated={int(resources['cuda_peak_allocated_bytes']) / 1024**2:.3f} MiB、reserved={int(resources['cuda_peak_reserved_bytes']) / 1024**2:.3f} MiB。这是 owned-process 的 PyTorch 分配器计数，不含 CUDA 驱动或上下文；2 GiB 上限是分配器预算，不能当作实际占用。", ""]
    if resources.get("shared_system_gpu_used_gib_observed_max") is not None:
        lines += [f"共享机器 GPU 总占用的采样最大值为 {resources['shared_system_gpu_used_gib_observed_max']:.4f} GiB，包含并发外部任务，只反映整机资源压力，不是 DINO/SubspaceAD 进程的显存峰值。", ""]
    failed_monitors = [s for s in resources["full_monitor_summaries"] if s.get("status") == "failed"]
    if failed_monitors:
        resource_interruptions = [s for s in failed_monitors if "resource-reserve boundary" in str(s.get("error", ""))]
        bookkeeping_errors = [s for s in failed_monitors if state.get("status") == "completed"
                              and "TotalSeconds" in str(s.get("error", ""))
                              and s.get("finished_at_utc") and state.get("finished_at_utc")
                              and datetime.fromisoformat(s["finished_at_utc"]) >= datetime.fromisoformat(state["finished_at_utc"])]
        lines += [f"保留 {len(failed_monitors)} 份 status=failed 的监控回执及原始原因，其中 {len(resource_interruptions)} 次为真实资源边界中断，{len(bookkeeping_errors)} 次为算法子进程完整结束后的监控记账错误。", ""]
        if resource_interruptions:
            lines += ["真实中断恢复时复核已完成类别哈希，重做未完成类别；中断执行的预测只作历史证据，不计入完成缓存或指标。", ""]
        if bookkeeping_errors:
            lines += ["最终算法子进程的 run_state 已完成全部 27 类、3,887 张测试图及 54 条指标；随后监控读取已结束进程的 TotalSeconds 属性发生错误。这个监控回执的 failed 状态不代表最终算法评测失败，完整性和指标复算另有独立验证。", ""]
        for item in failed_monitors:
            classification = "真实资源边界中断" if item in resource_interruptions else "子进程完成后的监控记账错误" if item in bookkeeping_errors else "其他监控失败记录"
            lines.append(f"- {item['monitor']}（{classification}）：{item.get('error') or '查看随附原始 summary.json'}")
        lines.append("")
    (extension / "DINO_SubspaceAD验收报告.md").write_text("\n".join(lines), encoding="utf-8")
    table = "<table><tr><th>数据集</th><th>方法</th><th>I-AUROC</th><th>P-AUROC</th><th>P-AP</th><th>AUPRO</th></tr>"
    for row in macro:
        table += "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in [row["dataset"], row["method"], *[f"{100 * row[k]:.4f}%" for k in METRICS]]) + "</tr>"
    table += "</table>"
    figures = "".join(f"<figure><figcaption>{html.escape(r['dataset'] + '/' + r['sample_id'])}</figcaption><img loading='lazy' src='{html.escape(r['figure'])}'></figure>" for r in visual)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>小型 DINO / SubspaceAD 验证</title><style>body{font:16px system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:20px;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}figure{margin:32px 0}a{color:#0654a4}</style>"
    page += "<h1>小型 DINOv2 / SubspaceAD 完整验证</h1><p>27 类、3,887 张原测试图；全量正常样本 PCA，无梯度训练。公开强基线的资源适配与本地实测。</p><p>" + html.escape(finding) + "</p>" + table
    page += "<p><a href='DINO_SubspaceAD验收报告.md'>完整报告与复算</a> · <a href='category_metrics.csv'>逐类指标</a> · <a href='category_deltas.csv'>全部差值</a> · <a href='model_summary.csv'>训练 PCA 模型摘要</a></p><p>54 组固定样本使用相同色盘与逐面板1%/99%显示规则，每幅图标注各自原值范围。两方法 raw score 单位不同，颜色不代表跨模型校准置信度；图像着色不代替像素指标评测。</p>" + figures + "</html>"
    (extension / "report.html").write_text(page, encoding="utf-8")


def refresh_manifest(package: Path) -> dict[str, Any]:
    manifest = read_json(package / "DELIVERY-MANIFEST.json")
    files = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path == package / "DELIVERY-MANIFEST.json":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(package) or path.suffix.lower() == ".zip":
            raise ValueError("No escaping file, symlink, or ZIP is allowed in folder delivery")
        files.append({"path": path.relative_to(package).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)})
    manifest.update({"files": files, "delivery_type": "folder", "compressed_archive_included": False,
                     "subspace_extension_completed": True, "subspace_category_count": 27, "subspace_test_images": 3887,
                     "subspace_new_raw_maps": 3887, "subspace_evaluation_masks": 3887, "subspace_metric_rows": 54,
                     "subspace_complete_prediction_cache_included": True, "subspace_figure_count": 54,
                     "scientific_claim_eligible": False, "updated_at": datetime.now(timezone.utc).isoformat()})
    write_json(package / "DELIVERY-MANIFEST.json", manifest)
    return {"member_count": len(files), "total_bytes": sum(r["size_bytes"] for r in files)}


def finish_integration(project: Path, package: Path, run: Path, *, check_only: bool = False, recovery: bool = False) -> dict[str, Any]:
    """Finish a copied extension without rebuilding or deleting measured data."""
    extension = package / EXTENSION
    if (extension / "整合回执.json").exists():
        raise ValueError("Integration already has a receipt; refusing a repeated finish")
    info = read_json(member(extension, "extension_manifest.json"))
    if info.get("status") != "completed" or info.get("test_images") != 3887 or info.get("new_metric_rows") != 54:
        raise ValueError("Only a completed copied 27-category extension can be finished")
    if Path(info["source_run"]).resolve() != run:
        raise ValueError("Copied extension source differs from the requested completed run")
    previous_path = member(extension, "原交付文件清单.json")
    previous_hash = digest(previous_path)
    if digest(member(package, "DELIVERY-MANIFEST.json")) != previous_hash or info["previous_delivery_manifest_sha256"] != previous_hash:
        raise ValueError("The original master manifest changed after extension copying")
    overview = member(package, "项目总览.html")
    saved_overview = member(extension, "DINO之前的项目总览.html")
    if digest(overview) != digest(saved_overview):
        raise ValueError("The current overview differs from the saved pre-integration snapshot")
    previous_files = {r["path"]: r for r in read_json(previous_path)["files"]}
    route_candidates = [relative for relative in ("开源项目与优化路线.txt", "实验说明.txt") if (package / relative).is_file()]
    if not route_candidates:
        raise ValueError("The existing route or experiment-note document is missing")
    route_relative = route_candidates[0]
    for relative in ("先读我.txt", route_relative):
        if relative not in previous_files or digest(member(package, relative)) != previous_files[relative]["sha256"]:
            raise ValueError(f"Historical root document changed before final integration: {relative}")
    old_overview = saved_overview.read_text(encoding="utf-8-sig")
    headings = list(re.finditer(r"<h1\b[^>]*>.*?</h1\s*>", old_overview, flags=re.IGNORECASE | re.DOTALL))
    if len(headings) != 1:
        raise ValueError("Expected exactly one overview h1 element")
    # Verify the copied payload rather than trusting the interrupted builder.
    rows, _, _ = collect_results(package, extension / "run")
    macros, _, _ = make_tables(rows)
    with member(extension, "dataset_macro_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_macros = list(csv.DictReader(stream))
    macro_index = {(r["dataset"], r["method"]): r for r in macros}
    if len(saved_macros) != len(macros) or {(r["dataset"], r["method"]) for r in saved_macros} != set(macro_index):
        raise ValueError("Copied macro table coverage differs from completed results")
    for record in saved_macros:
        expected = macro_index[(record["dataset"], record["method"])]
        if any(float(record[k]) != expected[k] for k in METRICS):
            raise ValueError("Copied macro table differs from the sealed completed metrics")
    config = read_json(extension / "run/run_config.json")
    for relative, expected_hash in config["source_sha256"].items():
        if digest(member(package / "code", relative)) != expected_hash:
            raise ValueError(f"Copied frozen algorithm source differs: {relative}")
    audit_path = project / "artifacts/audits/subspace-dinos-full-20261012.json"
    audit = read_json(audit_path)
    if (audit.get("status") != "passed" or audit.get("categories") != 27 or audit.get("test_images") != 3887
            or audit.get("metric_rows") != 54 or Path(audit["run"]).resolve() != run):
        raise ValueError("Independent complete-run audit must pass before final integration")
    if check_only:
        return {"status": "ready_to_finish_integration", "completed_categories": 27, "test_images": 3887, "delivery_modified": False}
    copy_file(audit_path, extension / "验收记录/独立完整运行审计.json")
    # These are delivery utilities, not the algorithm execution sources. Keep
    # their prior copied versions when repairing the interrupted integration.
    for relative in ("scripts/build_subspace_delivery.py", "scripts/verify_subspace_delivery.py"):
        source = member(project, relative)
        destination = member(package / "code", relative)
        if digest(source) != digest(destination):
            copy_file(destination, extension / "验收记录/整合修复之前的工具" / Path(relative).name)
            shutil.copyfile(source, destination)
    latest_table = "<div class='table-wrap'><table><thead><tr><th>数据集</th><th>原 PaDiM top1% 图像 AUROC</th><th>DINO-S/SubspaceAD top1% 图像 AUROC</th><th>原 PaDiM AUPRO</th><th>DINO-S/SubspaceAD AUPRO</th></tr></thead><tbody>"
    for dataset, label in (("mvtec_ad", "MVTec AD"), ("visa", "VisA")):
        old = macro_index[(dataset, "baseline_top1p")]
        new = macro_index[(dataset, "subspace_top1p")]
        latest_table += "<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in [label, f"{100 * old['image_auroc']:.4f}%", f"{100 * new['image_auroc']:.4f}%", f"{100 * old['aupro_0_30']:.4f}%", f"{100 * new['aupro_0_30']:.4f}%"]) + "</tr>"
    latest_table += "</tbody></table></div>"
    notice = "<section id='latest-dino-subspace'><h2>最新小型 DINOv2 + SubspaceAD 实测</h2><p>已完成 MVTec AD 15 类、VisA 12 类，共 12,288 张正常训练图用于 PCA、3,887 张原测试图用于评测。下表按类别等权平均，图像 AUROC 使用两方法相同的固定 top1% 聚合，AUPRO 使用共同 256×256 GT 和精确 0–0.30 口径。</p>" + latest_table
    notice += "<p>这轮同时提高了两个数据集的宏平均图像检测和像素定位指标；全部下降类别仍保留，MVTec transistor 的 AUPRO 有下降。完整指标、PCA 模型、全量缓存和54组固定样本见 <a href='优化实验/DINO小型SubspaceAD验证/report.html'>DINO / SubspaceAD 总览</a>；<a href='优化实验/DINO小型SubspaceAD验证/DINO_SubspaceAD验收报告.md'>验收报告与复算说明</a>；<a href='code/src/industrial_anomaly_lab/subspace_features.py'>实现代码</a>。</p><p>这是公开强基线的资源适配与本地实测，来源、许可证和探索性研究边界保留；公开基线更换不等于原创创新或三区录用。</p></section><h2>历史 PaDiM / OAS 实验</h2><p>以下内容是此前实验阶段的记录；旧的‘最有效调整’或‘尚未启动’表述不代表本轮最新推荐或运行状态。</p>"
    end = headings[0].end()
    overview.write_text(old_overview[:end] + notice + old_overview[end:], encoding="utf-8")
    readme = member(package, "先读我.txt")
    copy_file(readme, extension / "DINO之前的先读我.txt")
    readme.write_text(readme.read_text(encoding="utf-8-sig") + "\n\n新增小型 DINOv2-S + SubspaceAD 完整验证：优化实验/DINO小型SubspaceAD验证/report.html。code 内含新模块、运行脚本、测试和复算脚本；assets/weights 内含本地 DINO 权重与许可证。全量27类/3887张图/54条指标、PCA模型与完整分数图/掩码均保留。公开来源、资源适配与探索性研究边界见新验收报告；原基线、top1%和OAS历史材料保留。仍是单个文件夹，没有压缩包。\n", encoding="utf-8")
    route = member(package, route_relative)
    copy_file(route, extension / ("DINO之前的" + route_relative))
    route.write_text("DINOv2-S + SubspaceAD 最新验证（2026-10-12）\n已完整验证MVTec AD 15类和VisA12类，指标与复算见 优化实验/DINO小型SubspaceAD验证/report.html。下方路线为历次调查历史，其中SubspaceAD尚未启动的表述不代表当前状态。\n\n" + route.read_text(encoding="utf-8-sig"), encoding="utf-8")
    write_json(extension / "整合回执.json", {"status": "completed", "single_folder_delivery": True, "compressed_archive_created": False, "old_results_and_sources_preserved": True,
               "new_predictions_masks_models_checked_before_copy": True, "original_selected_inputs_reused": True, "recovered_final_document_integration": recovery,
               "completed_payload_not_rebuilt_or_deleted": True, "original_overview_and_manifest_authenticated": True, "independent_full_run_audit_passed": True,
               "created_at": datetime.now(timezone.utc).isoformat()})
    integrity = refresh_manifest(package)
    return {"status": "completed", "extension": str(extension), "new_metric_rows": 54, "figures": 54, "finished_existing_copied_extension": recovery, **integrity}


def build(args: argparse.Namespace) -> dict[str, Any]:
    project, package, run, weights = (p.resolve() for p in (args.project_root, args.package_root, args.run_dir, args.weight_dir))
    if not package.is_relative_to(project / "artifacts/deliveries"):
        raise ValueError("Delivery must remain within this project's explicit deliveries directory")
    if not run.is_relative_to(project / "runs/quick") or not weights.is_relative_to(project / "artifacts/weights"):
        raise ValueError("Run and checkpoint must remain in the explicitly configured project directories")
    extension = package / EXTENSION
    if args.finish_integration:
        return finish_integration(project, package, run, check_only=args.check_only, recovery=True)
    if args.refresh_manifest_only:
        if read_json(extension / "extension_manifest.json").get("status") != "completed":
            raise ValueError("A complete existing extension is required for resealing")
        return refresh_manifest(package)
    if extension.exists():
        raise ValueError("SubspaceAD extension already exists; refusing evidence replacement")
    rows, samples, state = collect_results(package, run)
    config = read_json(run / "run_config.json")
    for relative, expected in config["source_sha256"].items():
        if digest(member(project, relative)) != expected:
            raise ValueError(f"Executed source drift before extension: {relative}")
    benchmark = args.benchmark_dir.resolve() if args.benchmark_dir else run.parent / "subspace-normal-profile-20261012"
    resources = measured_resources(run, benchmark)
    selector = read_json(package / "report/visualization_manifest.json")["samples"]
    if len(selector) != 54 or {(r["dataset"], r["category"], r["sample_id"]) for r in selector} != {(r["dataset"], r["category"], r["sample_id"]) for r in samples if r.get("selected")}:
        raise ValueError("Frozen selection differs before mutation")
    if not weights.is_dir() or not (weights / "model.safetensors").is_file() or not (weights / "LICENSE").is_file():
        raise ValueError("Local complete DINO checkpoint directory and license are required")
    if args.check_only:
        return {"status": "ready_for_extension", "new_metric_rows": 54, "test_images": 3887, "delivery_modified": False}
    previous = (package / "DELIVERY-MANIFEST.json").read_bytes()
    extension.mkdir(parents=True, exist_ok=False)
    (extension / "原交付文件清单.json").write_bytes(previous)
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if relative.parts[0] != "execution_history" and any(p in ("metric_sort", "train_feature_cache", "__pycache__") or p.startswith(".quick-metrics-") for p in relative.parts):
            raise ValueError(f"Temporary cache remains in completed run: {relative}")
        copy_file(path, extension / "run" / relative)
    for path in sorted(benchmark.rglob("*")):
        if path.is_file():
            copy_file(path, extension / "资源记录/正常训练预检" / path.relative_to(benchmark))
    for monitor in sorted(run.parent.glob(run.name + "-monitor-*")):
        for path in sorted(monitor.rglob("*")):
            if path.is_file():
                copy_file(path, extension / "资源记录/完整运行监控" / monitor.name / path.relative_to(monitor))
    source_files = [*config["source_sha256"], "scripts/build_subspace_delivery.py", "scripts/verify_subspace_delivery.py", "scripts/prepare_subspace_reference.py", "docs/subspace_reference.json", "tests/test_subspace_features.py"]
    for relative in ("scripts/Invoke-SubspaceExperiment.ps1", "configs/subspace_dinos_20261012.json", "scripts/check_subspace_feature_equivalence.py", "scripts/audit_subspace_run.py"):
        if (project / relative).is_file():
            source_files.append(relative)
    for relative in dict.fromkeys(source_files):
        copy_file(member(project, relative), package / "code" / relative)
    for name in ("pyproject.toml", "uv.lock"):
        old = member(package / "code", name)
        copy_file(old, extension / "历史环境" / name)
        shutil.copyfile(member(project, name), old)
    weight_records = []
    for path in sorted(weights.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(weights)
        destination = package / "assets/weights" / weights.name / relative
        copy_file(path, destination)
        weight_records.append({"path": destination.relative_to(package).as_posix(), "sha256": digest(destination), "size_bytes": destination.stat().st_size})
    vendor = project / "vendor/SubspaceAD"
    upstream_files = list(config["vendor_source_sha256"])
    for relative in upstream_files:
        if digest(member(vendor, relative)) != config["vendor_source_sha256"][relative]:
            raise ValueError(f"Pinned public reference source drift: {relative}")
        copy_file(member(vendor, relative), extension / "公开来源/SubspaceAD" / relative)
        copy_file(member(vendor, relative), package / "code/vendor/SubspaceAD" / relative)
    copy_file(member(project, "runs/quick/subspace-feature-equivalence.json"), extension / "验收记录/正常训练图特征等价验证.json")
    audit_receipt = project / "artifacts/audits/subspace-dinos-full-20261012.json"
    if audit_receipt.is_file():
        if read_json(audit_receipt).get("status") != "passed":
            raise ValueError("Existing independent full-run audit did not pass")
        copy_file(audit_receipt, extension / "验收记录/独立完整运行审计.json")
    write_json(extension / "公开来源/来源与适配.json", {"subspace_reference": REFERENCE, "subspace_commit": "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7", "weight_reference": WEIGHT_REFERENCE,
               "weight_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056", "weight_files": weight_records,
               "adaptation": "DINOv2-S at 448 with full normal training and exact streaming PCA, evaluated on common 256 plane",
               "claim": "public strong baseline resource adaptation and measured integration; not independent algorithmic novelty", "scientific_claim_eligible": False})
    write_json(extension / "weights_manifest.json", {"schema_version": 1, "directory": "assets/weights/" + weights.name, "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056", "files": weight_records})
    macros, deltas, macro_deltas = make_tables(rows)
    write_csv(extension / "category_metrics.csv", rows, ["dataset", "category", "method", "seed", "test_images", *METRICS])
    write_csv(extension / "dataset_macro_metrics.csv", macros, ["dataset", "method", "categories", "test_images", *METRICS])
    write_csv(extension / "category_deltas.csv", deltas, ["dataset", "category", "method", "reference", *[k + "_delta_pp" for k in METRICS]])
    write_csv(extension / "dataset_macro_deltas.csv", macro_deltas, ["dataset", "method", "reference", *[k + "_delta_pp" for k in METRICS]])
    model_rows = []
    for dataset, category in config["expected_matrix"]["categories"]:
        model = member(run, f"categories/{dataset}/{category}/model.npz")
        import numpy as np
        with np.load(model, allow_pickle=False) as archive:
            model_rows.append({"dataset": dataset, "category": category, "normal_patch_count": int(archive["count"]), "feature_dimensions": int(archive["mean"].size),
                               "retained_dimensions": int(archive["components"].shape[1]), "retained_variance": float(archive["retained_variance"]),
                               "variance_target": float(archive["variance_target"]), "pca_model_sha256": digest(model), "pca_model_bytes": model.stat().st_size,
                               "npz_fields": ";".join(sorted(archive.files)), "metadata": json.dumps(read_json(run / f"categories/{dataset}/{category}/metadata.json"), ensure_ascii=False, separators=(",", ":"))})
    write_csv(extension / "model_summary.csv", model_rows, list(model_rows[0]))
    visual = draw_selected(package, run, extension, samples)
    write_json(extension / "visualization_manifest.json", {"schema_version": 1, "samples": visual, "figure_count": 54, "raw_maps_modified": False})
    write_json(extension / "资源记录/资源摘要.json", resources)
    write_report(extension, macros, deltas, macro_deltas, visual, state, resources)
    write_json(extension / "extension_manifest.json", {"schema_version": 1, "status": "completed", "source_run": str(run), "new_metric_rows": 54, "combined_metric_rows": len(rows), "test_images": 3887,
               "new_raw_maps": 3887, "evaluation_masks": 3887, "figure_count": 54, "pca_models": 27, "scientific_claim_eligible": False,
               "gradient_training_performed": False, "normal_train_pca_fit": True, "previous_delivery_manifest_sha256": hashlib.sha256(previous).hexdigest(), "created_at": datetime.now(timezone.utc).isoformat()})
    copy_file(member(package, "项目总览.html"), extension / "DINO之前的项目总览.html")
    return finish_integration(project, package, run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--refresh-manifest-only", action="store_true")
    parser.add_argument("--finish-integration", action="store_true", help="finish only the authenticated copied extension after document integration was interrupted")
    args = parser.parse_args()
    if args.finish_integration and args.refresh_manifest_only:
        parser.error("--finish-integration and --refresh-manifest-only are mutually exclusive")
    try:
        print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(json.dumps({"status": "extension_failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
