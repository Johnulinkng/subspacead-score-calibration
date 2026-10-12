"""Extend the existing single delivery folder with a completed OAS experiment.

This command never trains, downloads, chooses a winning parameter, or creates
an archive. It consumes completed, sealed results. Existing historical files
are retained; the top-level manifest is refreshed after the extension.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
NEW_METHODS = ("ml_max", "ml_top1p", "oas_max", "oas_top1p")
METHODS = ("baseline_max", "baseline_top1p", *NEW_METHODS)
EXTENSION = "优化实验/OAS验证"
REFERENCE = (
    "https://github.com/scikit-learn/scikit-learn/blob/"
    "102e5daf66759cf896beeee229d1bf7c80ab2ba4/"
    "sklearn/covariance/_shrunk_covariance.py"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def member(root: Path, relative: str) -> Path:
    text = PurePosixPath(relative)
    if not relative or text.is_absolute() or ".." in text.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative member: {relative!r}")
    path = (root / Path(*text.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping file: {relative}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Plain source file required: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(source) != digest(destination):
            raise ValueError(f"Refusing to replace different historical evidence: {destination}")
        return
    shutil.copyfile(source, destination)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_results(source: Path, new_run: Path, top_run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    state = read_json(new_run / "run_state.json")
    if state.get("status") != "completed":
        raise ValueError("OAS run must be fully completed before delivery extension")
    original = read_jsonl(source / "results/category_metrics.jsonl")
    controls = {(r["dataset"], r["category"]): r for r in original if r["variant"] == "baseline"}
    if len(controls) != 27 or sum(int(r["test_images"]) for r in controls.values()) != 3887:
        raise ValueError("Expected the original complete 27-category/3887-test baseline")
    with (top_run / "category_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        top = {(r["dataset"], r["category"]): r for r in csv.DictReader(stream) if r["method"] == "baseline_top1p_mean"}
    if set(top) != set(controls):
        raise ValueError("Top1-percent baseline category set differs")
    new = read_jsonl(new_run / "results/category_metrics.jsonl")
    expected = {(d, c, v) for d, c in controls for v in NEW_METHODS}
    actual = {(r["dataset"], r["category"], r["variant"]) for r in new}
    if len(new) != 108 or actual != expected:
        raise ValueError("Expected all 27 categories and four frozen ML/OAS score variants")
    rows = []
    for (dataset, category), baseline in sorted(controls.items()):
        for method in ("baseline_max", "baseline_top1p"):
            row = {"dataset": dataset, "category": category, "method": method,
                   "seed": int(baseline["seed"]), "test_images": int(baseline["test_images"]),
                   **{key: float(baseline[key]) for key in METRICS}}
            if method == "baseline_top1p":
                row["image_auroc"] = float(top[(dataset, category)]["image_auroc"])
            rows.append(row)
    for record in new:
        baseline = controls[(record["dataset"], record["category"])]
        if record.get("status") != "completed" or int(record["seed"]) != int(baseline["seed"]):
            raise ValueError("New result status or seed mismatch")
        if int(record["test_images"]) != int(baseline["test_images"]):
            raise ValueError("New result sample count differs from baseline")
        rows.append({"dataset": record["dataset"], "category": record["category"], "method": record["variant"],
                     "seed": int(record["seed"]), "test_images": int(record["test_images"]),
                     **{key: float(record[key]) for key in METRICS}})
    for row in rows:
        if any(not math.isfinite(row[key]) or not 0 <= row[key] <= 1 for key in METRICS):
            raise ValueError("Finite unit-interval metrics required")
    # A score aggregation change cannot alter a localization metric.
    index = {(r["dataset"], r["category"], r["method"]): r for r in rows}
    for dataset, category in controls:
        for stem in ("baseline", "ml", "oas"):
            a, b = index[(dataset, category, stem + "_max")], index[(dataset, category, stem + "_top1p")]
            if any(a[key] != b[key] for key in METRICS[1:]):
                raise ValueError("Max and top1p rows must share identical localization metrics")
    samples = []
    for dataset, category in sorted(controls):
        records = read_jsonl(new_run / f"categories/{dataset}/{category}/samples.jsonl")
        if len(records) != int(controls[(dataset, category)]["test_images"]):
            raise ValueError("Category samples inventory count mismatch")
        for record in records:
            if (record["dataset"], record["category"]) != (dataset, category):
                raise ValueError("Sample identity/category mismatch")
            for estimator in ("ml", "oas"):
                if digest(member(new_run, record["raw_maps"][estimator])) != record["raw_map_sha256"][estimator]:
                    raise ValueError("New raw-map hash mismatch")
            if digest(member(new_run, record["mask_path"])) != record["mask_sha256"]:
                raise ValueError("New evaluation-mask hash mismatch")
            samples.append(record)
    if len(samples) != 3887 or len({(r["dataset"], r["category"], r["sample_id"]) for r in samples}) != 3887:
        raise ValueError("Full test sample coverage or uniqueness mismatch")
    rows.sort(key=lambda r: (r["dataset"], r["category"], METHODS.index(r["method"])))
    return rows, samples, state


def tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    index = {(r["dataset"], r["category"], r["method"]): r for r in rows}
    deltas = []
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)
        if row["method"] not in NEW_METHODS:
            continue
        score = "top1p" if row["method"].endswith("top1p") else "max"
        references = ["baseline_" + score]
        if row["method"].startswith("oas"):
            references.append("ml_" + score)
        for reference in references:
            base = index[(row["dataset"], row["category"], reference)]
            deltas.append({"dataset": row["dataset"], "category": row["category"], "method": row["method"], "reference": reference,
                           **{key + "_delta_pp": 100 * (row[key] - base[key]) for key in METRICS}})
    macro = []
    for (dataset, method), group in sorted(groups.items(), key=lambda v: (v[0][0], METHODS.index(v[0][1]))):
        macro.append({"dataset": dataset, "method": method, "categories": len(group),
                      "test_images": sum(r["test_images"] for r in group),
                      **{key: sum(r[key] for r in group) / len(group) for key in METRICS}})
    macro_index = {(r["dataset"], r["method"]): r for r in macro}
    macro_deltas = []
    for row in macro:
        if row["method"] not in NEW_METHODS:
            continue
        score = "top1p" if row["method"].endswith("top1p") else "max"
        refs = ["baseline_" + score] + (["ml_" + score] if row["method"].startswith("oas") else [])
        for reference in refs:
            macro_deltas.append({"dataset": row["dataset"], "method": row["method"], "reference": reference,
                                 **{key + "_delta_pp": 100 * (row[key] - macro_index[(row["dataset"], reference)][key]) for key in METRICS}})
    return macro, deltas, macro_deltas


def resource_summary(new_run: Path) -> dict[str, Any]:
    """Distinguish the normal-only profile from the full-run observed peak."""
    monitors = sorted(new_run.parent.glob(new_run.name + "-monitor-*"))
    observations = []
    for monitor in monitors:
        path = monitor / "resources.jsonl"
        if path.is_file():
            observations.extend(read_jsonl(path))
    peaks = [float(r["peak_working_set_gib"]) for r in observations if r.get("peak_working_set_gib") is not None]
    if not peaks or any(not math.isfinite(v) or v < 0 for v in peaks):
        raise ValueError("Full-run measured monitor peak is required")
    profile = read_json(new_run.parent / "oas-normal-profile-20261012/benchmark.json")
    if profile.get("test_content_accessed") is not False:
        raise ValueError("Resource profile must only access normal training content")
    return {"full_monitor_count": len(monitors), "full_resource_samples": len(observations),
            "full_peak_working_set_gib": max(peaks), "full_peak_basis": "maximum peak_working_set_gib across all full-run monitor records",
            "full_monitor_summaries": [{"monitor": m.name, **read_json(m / "summary.json")} for m in monitors if (m / "summary.json").is_file()],
            "normal_train_only_profile": profile,
            "normal_profile_peak_working_set_gib": int(profile["peak_working_set_bytes"]) / 1024**3,
            "numeric_validation": {"tests_passed": 18, "test_source": "code/tests/test_oas_features.py", "verified_before_full_test_run_by_parent": True}}


def draw_selected(package: Path, new_run: Path, extension: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps

    original = read_json(package / "report/visualization_manifest.json")["samples"]
    if len(original) != 54:
        raise ValueError("Expected the original mechanically preselected 54 samples")
    index = {(r["dataset"], r["category"], r["sample_id"]): r for r in samples}
    selected = {(r["dataset"], r["category"], r["sample_id"]) for r in samples if r.get("selected")}
    if selected != {(r["dataset"], r["category"], r["sample_id"]) for r in original}:
        raise ValueError("New selected flags differ from the original frozen visualization selector")
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_path), 21) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    records = []
    for position, old in enumerate(original):
        identity = (old["dataset"], old["category"], old["sample_id"])
        new = index[identity]
        if new["image_sha256"] != old["source_image_sha256"]:
            raise ValueError("Selected source image differs from the earlier frozen input")
        input_member = "report/" + old["source_image_copy"]
        input_path = member(package, input_member)
        if digest(input_path) != old["source_image_sha256"]:
            raise ValueError("Selected included original source image digest mismatch")
        baseline_member = "report/" + old["raw_maps"]["baseline"]["path"]
        baseline_path = member(package, baseline_member)
        if digest(baseline_path) != old["raw_maps"]["baseline"]["sha256"]:
            raise ValueError("Included selected original baseline raw map digest mismatch")
        raw = [np.load(baseline_path, allow_pickle=False)] + [np.load(member(new_run, new["raw_maps"][k]), allow_pickle=False) for k in ("ml", "oas")]
        mask = np.load(member(new_run, new["mask_path"]), allow_pickle=False)
        if any(a.shape != (256, 256) or not np.isfinite(a).all() for a in raw) or mask.shape != (256, 256):
            raise ValueError("Visual evidence shape/finiteness mismatch")
        if not np.isin(mask, (False, True, 0, 1)).all() or bool(np.any(mask)) != bool(new["label"]):
            raise ValueError("Selected GT binary/label mismatch")
        lower, upper = map(float, np.quantile(np.concatenate([a.ravel() for a in raw]), [0.01, 0.99]))
        if upper <= lower:
            upper = lower + 1.0
        with Image.open(input_path) as source_image:
            displayed_input = source_image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        panels = [displayed_input, Image.fromarray(np.asarray(mask > 0, dtype=np.uint8) * 255).convert("RGB")]
        panels += [Image.fromarray(np.asarray(colormaps["viridis"](np.clip((a - lower) / (upper - lower), 0, 1))[..., :3] * 255, dtype=np.uint8)) for a in raw]
        canvas = Image.new("RGB", (1440, 382), "white")
        drawing = ImageDraw.Draw(canvas)
        drawing.text((16, 9), f"{identity[0]} / {identity[2]}   label={new['label']}", fill="black", font=font)
        for i, (title, panel) in enumerate(zip(("Input", "Ground truth", "Baseline raw map", "ML raw map", "OAS raw map"), panels, strict=True)):
            x = 16 + i * 282
            drawing.text((x, 44), title, fill="black", font=small)
            canvas.paste(panel, (x, 75))
        drawing.text((16, 342), f"Common viridis scale for all three maps: {lower:.6g} .. {upper:.6g} (pooled 1% / 99%; display only)", fill="black", font=small)
        name = f"visualizations/comparison-{position:03d}.png"
        destination = extension / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        records.append({"dataset": identity[0], "category": identity[1], "sample_id": identity[2], "label": int(new["label"]),
                        "seed": int(new.get("seed", old["seed"])), "source_image_copy": input_member,
                        "source_image_sha256": old["source_image_sha256"], "figure": name, "figure_sha256": digest(destination),
                        "baseline_raw_map": {"path": baseline_member, "sha256": old["raw_maps"]["baseline"]["sha256"]},
                        "raw_maps": {k: {"path": new["raw_maps"][k], "sha256": new["raw_map_sha256"][k]} for k in ("ml", "oas")},
                        "mask_path": new["mask_path"], "mask_sha256": new["mask_sha256"],
                        "selection": "same 54 samples as the original explicit selector; no OAS-score-based selection",
                        "scale": {"scope": "pooled three methods in this row", "quantiles": [0.01, 0.99], "lower": lower, "upper": upper, "colormap": "viridis"},
                        "display_only_normalization": True, "evaluation_raw_maps_modified": False})
    return records


def report(extension: Path, macro: list[dict[str, Any]], deltas: list[dict[str, Any]], macro_deltas: list[dict[str, Any]], state: dict[str, Any], visual: list[dict[str, Any]], resources: dict[str, Any]) -> None:
    lines = ["# OAS 自适应协方差验证", "", "本轮已经完成 MVTec AD 15 类、VisA 12 类，共 3,887 张原测试图像的固定候选验证。复用正常训练统计和同一个 ResNet18 权重，没有梯度训练，也没有新增数据集。", "",
             "本轮是测试集已访问后的探索性验证。OAS 是公开的协方差估计方法，本项目实现了它与现有异常检测流水线的衔接、分母对照、完整评测和复算材料；不将公开方法或他人成果声明为新发明。", "",
             "## 宏平均实测指标", "", "以下按类别等权平均，百分数用于显示。I-AUROC 是图像指标；P-AUROC、P-AP、AUPRO(0–0.30) 是定位指标。", "",
             "|数据集|方法|I-AUROC|P-AUROC|P-AP|AUPRO(0–0.30)|", "|---|---|---:|---:|---:|---:|"]
    for row in macro:
        lines.append("|" + "|".join([row["dataset"], row["method"], *[f"{100 * row[k]:.4f}%" for k in METRICS]]) + "|")
    lines += ["", "## OAS 的定位变化", "", "每个候选与相同图像评分规则的原基线比较。ML 对照只把协方差分母 n−1 改为 n，OAS 进一步施加逐位置自适应收缩。", ""]
    for row in macro_deltas:
        if row["method"].startswith("oas_"):
            lines.append(f"- {row['dataset']}：{row['method']} 对 {row['reference']}，I-AUROC {row['image_auroc_delta_pp']:+.4f}、P-AUROC {row['pixel_auroc_delta_pp']:+.4f}、P-AP {row['pixel_ap_delta_pp']:+.4f}、AUPRO {row['aupro_0_30_delta_pp']:+.4f} 个百分点。")
    primary = [r for r in macro_deltas if r["method"] == "oas_max" and r["reference"] == "baseline_max"]
    if all(r["aupro_0_30_delta_pp"] > 0 for r in primary):
        finding = "OAS 在两个数据集的宏平均 AUPRO 均高于原基线，具体指标和下降类别仍须一起审阅。一次探索性改进不等于已达到论文录用或三区研究标准。"
    else:
        finding = "OAS 没有在两个数据集同时取得宏平均 AUPRO 高于原基线的结果，因此不能据此宣布定位目标完成。保留这次实测失败或局部收益，不按已知测试结果继续挑选收缩系数。"
    lines += ["", finding, "", "## 完整负结果与口径", "", "原 alpha=0.1 固定收缩实验仍保留在原 report 内；本轮没有替换该失败记录。完整逐类表和差值表包含所有上升、下降及持平项。", ""]
    for dataset in ("mvtec_ad", "visa"):
        failed = [r for r in deltas if r["dataset"] == dataset and r["method"] == "oas_max" and r["reference"] == "baseline_max" and r["aupro_0_30_delta_pp"] < 0]
        values = "、".join(f"{r['category']} ({r['aupro_0_30_delta_pp']:+.4f})" for r in failed) or "无"
        lines.append(f"- {dataset} 的 OAS AUPRO 下降类别及百分点：{values}。")
    lines += ["", "图像分数提供 max 与固定 top1% 均值（256×256 平面中最大的 655 个值）两套口径。对同一估计器，它们使用完全相同的原始异常图；图像评分变化不冒充像素定位改进。", "",
             "协方差流程：原统计 C=M2/(n−1)，先转为 S=(n−1)/n·C；D=100，μ=trace(S)/D，a=mean(S²)（S 逐元素平方后，对 D×D 项取均值），α=min((a+μ²)/((n+1)(a−μ²/D)),1)。按每个特征位置计算 α，分母为零时取 1。最终协方差为 (1−α)S+αμI+0.01I，0.01 只加一次。中位数或均值仅用于统计展示，不作为整类统一收缩系数。", "",
             "几何口径保持原设置：RGB 双线性到 224、ImageNet 归一化、固定 100 通道、56×56 特征；马氏距离上采样到 224，在此平面做 σ=4 高斯平滑，再双线性到 256；GT 最近邻到 256。评测不做逐图归一化。", "",
             "54 组图使用原先机械选定的同一批正常/异常样本，每一行的 baseline、ML、OAS 使用合并分布 1%/99% 的共同显示尺度。归一化只用于颜色显示，保存和评测的分数图保持原值。", "",
             "## 复算与交付边界", "", "OAS 子目录含当前完成缓存的两种估计器共 7,774 张 raw map、3,887 张二值评测掩码及逐图 SHA256。中断执行的历史文件另存 run/execution_history，只作执行证据；其中不完整预测不计入完成缓存、3,887 张样本、108 条指标或可视化。模型统计复用文件夹现有 models，不重复复制 6.8GB。精选原图复用原 report 的证据材料。", "",
             "先运行根目录 Verify.ps1 验证整个文件夹。再用本地 Python 3.11 运行 code/scripts/verify_oas_delivery.py，可验证 OAS 缓存和精选推理；--metrics-categories 27 可从随附分数图/掩码独立复算所有定位指标。完整模型重新产生全部分数图仍需要正式数据集；此文件夹包含的原始照片仅覆盖 54 个精选样本。", "",
             "在交付文件夹内，用已有 Python 3.11 环境执行下列命令；无需网络下载或重新训练。当前项目可使用 E:\\3k\\.venv\\Scripts\\python.exe，其他电脑可用等价环境。", "",
             "```powershell", "python -B .\\code\\scripts\\verify_oas_delivery.py --package-root . --inference-samples 2 --metrics-categories 27 --work-dir E:\\3k\\tmp\\oas-metric-replay --receipt E:\\3k\\tmp\\oas-delivery-verification.json", "```", "",
             "复算的临时排序文件写入命令指定的 --work-dir（默认系统临时目录），不会加入交付文件夹。未加载完整数据集也能复算随附异常图的指标；这证明缓存指标口径，不代替数据集源文件验证。", "",
             "原 DRAEM 长训练保持用户暂停。仍只有一个固定通道种子；没有多种子误差线、强基线矩阵或独立创新验证。当前属于可复核的实验整合材料，不能据此保证 SCI 三区接收。", "",
             f"OAS 公式和单特征特殊情况对齐 [scikit-learn 1.6.1 固定版本源码]({REFERENCE})。该公开公式的使用与来源在代码和文档中保留；本轮没有复制该项目源码文件。原 PaDiM、torchvision、数据集许可证及署名文件继续保留。", "",
             "## 实际资源与时间", ""]
    if state.get("started_at_utc") and state.get("finished_at_utc"):
        wall = (datetime.fromisoformat(state["finished_at_utc"]) - datetime.fromisoformat(state["started_at_utc"])).total_seconds()
        lines.append(f"全流程墙钟时间 {wall:.3f} 秒（{wall / 60:.3f} 分钟），从首次 started_at_utc 到最终 finished_at_utc 计算，包含中断等待。run_state.elapsed_seconds 是最近执行段的计时，若恢复运行则不能当作整个流程总耗时。")
    lines += ["", f"完整运行的峰值工作集为 {resources['full_peak_working_set_gib']:.4f} GiB，取所有完整运行 monitor/resources.jsonl 内 peak_working_set_gib 的最大值（{resources['full_monitor_count']} 个监控目录、{resources['full_resource_samples']} 条采样）。CPU 线程数固定为 2，不使用 GPU。这是实际进程内存，不是电脑总内存或启动门槛。", "",
              f"正常训练-only 容量预检峰值为 {resources['normal_profile_peak_working_set_gib']:.4f} GiB，耗时 {resources['normal_train_only_profile']['elapsed_seconds']:.3f} 秒，仅前向 16 张正常训练图；使用 209 张正常训练图的既有统计。该预检没有访问测试图，其峰值不能冒充完整 27 类运行峰值。", "",
              "代码冻结和完整测试前，18 项 OAS 数值测试已通过，覆盖 scikit-learn 1.6.1 系数对齐、退化/单通道、分块不变性、正则项只施加一次、ML 分母对照和直接马氏距离。测试源码随附，重新执行需要 pytest。"]
    interrupted = [s for s in resources["full_monitor_summaries"] if s.get("status") == "failed"]
    if interrupted:
        lines += ["", f"完整运行包含 {len(interrupted)} 次失败或资源中断监控记录，全部保留，没有仅统计最后成功段。恢复时复核已完成类别的原 SHA256，保留原结果，重做未完成类别；全流程墙钟包含中断等待。"]
        for item in interrupted:
            lines.append(f"- {item['monitor']}：{item.get('error') or '查看原始监控 summary.json'}")
    lines += ["", "## 原始运行状态", "", "```json", json.dumps(state, ensure_ascii=False, indent=2), "```", ""]
    (extension / "OAS验收报告.md").write_text("\n".join(lines), encoding="utf-8")
    table = "<table><tr><th>数据集</th><th>方法</th><th>I-AUROC</th><th>P-AUROC</th><th>P-AP</th><th>AUPRO</th></tr>"
    for row in macro:
        table += "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in [row["dataset"], row["method"], *[f"{100 * row[k]:.4f}%" for k in METRICS]]) + "</tr>"
    table += "</table>"
    figures = "".join(f"<figure><figcaption>{html.escape(r['dataset'] + '/' + r['sample_id'])}</figcaption><img loading='lazy' src='{html.escape(r['figure'])}'></figure>" for r in visual)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>OAS 验证</title><style>body{font:16px system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:20px;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}figure{margin:32px 0}a{color:#0654a4}</style>"
    page += f"<h1>OAS 自适应协方差验证</h1><p>27 类、3,887 张测试图、两个估计器与两套图像评分；复用训练统计，没有梯度训练。</p><p>{html.escape(finding)}</p>" + table
    page += "<p><a href='OAS验收报告.md'>完整报告及公式</a> · <a href='category_metrics.csv'>逐类指标</a> · <a href='category_deltas.csv'>逐类差值</a> · <a href='dataset_macro_deltas.csv'>宏平均差值</a> · <a href='alpha_summary.csv'>训练统计的 α 分布</a></p><p>以下 54 组原先选定样本共用逐行颜色尺度，仅用于展示；评测保持 raw map 原值。公开 OAS 来源、失败记录和测试已访问边界均保留。</p>" + figures + "</html>"
    (extension / "report.html").write_text(page, encoding="utf-8")


def refresh_manifest(package: Path) -> dict[str, Any]:
    current = read_json(package / "DELIVERY-MANIFEST.json")
    files = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path == package / "DELIVERY-MANIFEST.json":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(package):
            raise ValueError("Delivery contains an escaping file")
        files.append({"path": path.relative_to(package).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)})
    current.update({"files": files, "delivery_type": "folder", "compressed_archive_included": False,
                    "oas_extension_completed": True, "oas_category_count": 27, "oas_test_images": 3887,
                    "oas_new_raw_maps": 7774, "oas_evaluation_masks": 3887, "oas_metric_rows": 108,
                    "oas_complete_prediction_cache_included": True, "oas_figure_count": 54,
                    "oas_execution_history_excluded_from_complete_cache_counts": True,
                    "scientific_claim_eligible": False, "updated_at": datetime.now(timezone.utc).isoformat()})
    write_json(package / "DELIVERY-MANIFEST.json", current)
    return {"member_count": len(files), "total_bytes": sum(f["size_bytes"] for f in files)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    project, package, new_run, source, top_run = [Path(value).resolve() for value in (args.project_root, args.package_root, args.run_dir, args.source_run, args.top1p_run)]
    if not package.is_relative_to(project / "artifacts/deliveries"):
        raise ValueError("Delivery must remain in the explicitly configured project deliveries directory")
    if args.refresh_manifest_only:
        if not (package / EXTENSION / "extension_manifest.json").is_file():
            raise ValueError("An existing completed OAS extension is required")
        return refresh_manifest(package)
    extension = package / EXTENSION
    if extension.exists():
        raise ValueError("OAS extension already exists; refusing an evidence replacement")
    rows, samples, state = collect_results(source, new_run, top_run)
    resources = resource_summary(new_run)
    # Resolve the original preselected inputs before changing the package.
    selector = read_json(package / "report/visualization_manifest.json")["samples"]
    new_keys = {(r["dataset"], r["category"], r["sample_id"]) for r in samples if r.get("selected")}
    if new_keys != {(r["dataset"], r["category"], r["sample_id"]) for r in selector}:
        raise ValueError("Preselected visualization identity mismatch")
    if args.check_only:
        return {"status": "ready_for_extension", "categories": 27, "new_metric_rows": 108, "test_images": len(samples), "delivery_modified": False}
    previous = (package / "DELIVERY-MANIFEST.json").read_bytes()
    extension.mkdir(parents=True, exist_ok=False)
    (extension / "原交付文件清单.json").write_bytes(previous)
    # Include all completed run evidence except process-owned temporary files.
    for path in sorted(new_run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(new_run)
        historical = relative.parts[0] == "execution_history"
        if not historical and any(p.startswith(".quick-metrics-") or p in ("metric_sort", "__pycache__") for p in relative.parts):
            raise ValueError(f"Temporary file remains in completed run: {relative}")
        copy_file(path, extension / "run" / relative)
    benchmark_dir = new_run.parent / "oas-normal-profile-20261012"
    for name in ("benchmark.json", "run_config.json", "run_state.json"):
        if (benchmark_dir / name).is_file():
            copy_file(benchmark_dir / name, extension / "资源记录/正常训练预检" / name)
    for monitor in sorted(new_run.parent.glob(new_run.name + "-monitor-*")):
        if not monitor.is_dir():
            continue
        for path in sorted(monitor.rglob("*")):
            if path.is_file():
                copy_file(path, extension / "资源记录/完整运行监控" / monitor.name / path.relative_to(monitor))
    frozen_sources = read_json(new_run / "run_config.json")["source_sha256"]
    for relative, expected in frozen_sources.items():
        if digest(member(project, relative)) != expected:
            raise ValueError(f"Executed source drift before delivery: {relative}")
    source_files = list(frozen_sources) + ["scripts/build_oas_delivery.py", "scripts/verify_oas_delivery.py", "scripts/audit_oas_run.py", "tests/test_oas_features.py"]
    for optional in ("scripts/Invoke-OASExperiment.ps1", "configs/oas_comparison_20261012.json"):
        if (project / optional).is_file():
            source_files.append(optional)
    for relative in dict.fromkeys(source_files):
        copy_file(project / relative, package / "code" / relative)
    macro, deltas, macro_deltas = tables(rows)
    write_csv(extension / "category_metrics.csv", rows, ["dataset", "category", "method", "seed", "test_images", *METRICS])
    write_csv(extension / "dataset_macro_metrics.csv", macro, ["dataset", "method", "categories", "test_images", *METRICS])
    write_csv(extension / "category_deltas.csv", deltas, ["dataset", "category", "method", "reference", *[k + "_delta_pp" for k in METRICS]])
    write_csv(extension / "dataset_macro_deltas.csv", macro_deltas, ["dataset", "method", "reference", *[k + "_delta_pp" for k in METRICS]])
    import numpy as np
    alphas = []
    for directory in sorted((new_run / "categories").glob("*/*")):
        values = np.load(directory / "oas_alpha.npy", allow_pickle=False)
        if values.shape != (3136,) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Invalid per-position OAS alpha evidence")
        alphas.append({"dataset": directory.parent.name, "category": directory.name, "positions": len(values),
                       "minimum": float(values.min()), "mean": float(values.mean()), "median": float(np.median(values)),
                       "maximum": float(values.max()), "standard_deviation": float(values.std()),
                       "zero_count": int(np.count_nonzero(values == 0)), "one_count": int(np.count_nonzero(values == 1))})
    if len(alphas) != 27:
        raise ValueError("Expected per-position alpha evidence for all 27 categories")
    write_csv(extension / "alpha_summary.csv", alphas, list(alphas[0]))
    visual = draw_selected(package, new_run, extension, samples)
    write_json(extension / "visualization_manifest.json", {"schema_version": 1, "samples": visual, "figure_count": 54, "raw_maps_modified": False})
    write_json(extension / "资源记录/资源与数值验证摘要.json", resources)
    report(extension, macro, deltas, macro_deltas, state, visual, resources)
    write_json(extension / "extension_manifest.json", {"schema_version": 1, "status": "completed", "source_run": str(new_run),
               "original_source_run": str(source), "baseline_top1p_run": str(top_run), "new_metric_rows": 108,
               "combined_metric_rows": len(rows), "test_images": len(samples), "new_raw_maps": len(samples) * 2,
               "evaluation_masks": len(samples), "figure_count": len(visual), "scientific_claim_eligible": False,
               "execution_history_excluded_from_complete_cache_counts": True,
               "execution_history_files_included": sum(1 for p in (new_run / "execution_history").rglob("*") if p.is_file()) if (new_run / "execution_history").is_dir() else 0,
               "oas_reference": REFERENCE, "normal_statistics_reused": True, "gradient_training_performed": False,
               "previous_delivery_manifest_sha256": hashlib.sha256(previous).hexdigest(), "created_at": datetime.now(timezone.utc).isoformat()})
    overview = package / "项目总览.html"
    current_overview = overview.read_text(encoding="utf-8-sig")
    if current_overview.count("<h1>工业异常检测交付总览</h1>") != 1:
        raise ValueError("Expected the exact unique existing overview h1 before extension")
    copy_file(overview, extension / "OAS之前的项目总览.html")
    primary = [r for r in macro_deltas if r["method"] == "oas_max" and r["reference"] == "baseline_max"]
    if len(primary) == 2 and all(r[k + "_delta_pp"] < 0 for r in primary for k in METRICS):
        finding_notice = "OAS 在两个数据集的四项宏平均均低于同口径基线，未达到超过基线的目标。"
    else:
        finding_notice = "完整宏平均及下降类别见报告，一次探索性验证不等于论文目标完成。"
    notice = "<h2>最新 OAS 实测验证</h2><p>本轮已完成两数据集 27 类、3,887 张图的逐位置 OAS 与 ML 分母对照。" + finding_notice + "详见完整结果和54组同尺度可视化。</p><p><a href='优化实验/OAS验证/report.html'>OAS 指标与可视化总览</a> · <a href='优化实验/OAS验证/OAS验收报告.md'>验收及复算报告</a></p>"
    overview.write_text(current_overview.replace("<h1>工业异常检测交付总览</h1>", "<h1>工业异常检测交付总览</h1>" + notice).replace("下一步优先验证训练统计决定的 OAS 定位候选；同时准备", "OAS 定位候选已经完成本地验证，结果见上方最新报告；后续可评估").replace("两条均尚未取得本地定位结果，具体资源门槛与统一协议见路线文件。", "SubspaceAD 路线尚未取得本地结果，资源门槛与统一协议见路线文件。"), encoding="utf-8")
    if "最新 OAS 实测验证" not in overview.read_text(encoding="utf-8"):
        raise ValueError("Latest OAS overview insertion did not persist")
    route = package / "开源项目与优化路线.txt"
    copy_file(route, extension / "OAS之前的开源项目与优化路线.txt")
    prior_route = route.read_text(encoding="utf-8-sig")
    route.write_text("OAS 最新验证（2026-10-12）\n本轮已经完成 MVTec AD 15 类、VisA 12 类、3,887 张图的逐位置 OAS 及 ML 分母对照。实测指标、全部下降项、完整缓存和54组同尺度图见 优化实验/OAS验证/report.html 和 OAS验收报告.md。下方旧路线文字是 OAS 执行前的调查历史，其中‘尚未运行’不代表当前 OAS 状态；SubspaceAD/AnomalyDINO 等后续路线仍未启动本地实验。\n\n" + prior_route, encoding="utf-8")
    root_readme = package / "先读我.txt"
    previous_readme = root_readme.read_text(encoding="utf-8-sig")
    copy_file(root_readme, extension / "OAS之前的先读我.txt")
    root_readme.write_text(previous_readme + "\n\n新增 OAS 完整验证：优化实验/OAS验证/report.html。该目录包含两种新估计器的全量异常图和GT缓存、108条指标、逐类差值、54组原先选定的同尺度图。code/scripts/verify_oas_delivery.py 提供缓存/精选推理/完整定位指标复算；未新增压缩包。OAS公开来源、失败或下降项及探索性研究边界在验收报告中保留。\n", encoding="utf-8")
    write_json(extension / "整合回执.json", {"status": "completed", "single_folder_delivery": True, "compressed_archive_created": False,
               "existing_model_statistics_duplicated": False, "baseline_raw_maps_changed": False, "baseline_fixed_shrinkage_failure_preserved": True,
               "new_prediction_and_mask_hashes_checked_before_copy": True, "selected_source_images_reused": True,
               "created_at": datetime.now(timezone.utc).isoformat()})
    integrity = refresh_manifest(package)
    return {"status": "completed", "extension": str(extension), "new_metric_rows": 108, "figure_count": 54, **integrity}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--top1p-run", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--refresh-manifest-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "extension_failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
