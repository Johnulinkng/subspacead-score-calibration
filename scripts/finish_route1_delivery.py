"""Integrate a built, independently audited route-1 extension without computing.

No process is queried or started. No inference, training, PCA, metric replay,
download, compressed archive, or master-manifest refresh is performed. Root documents
are edited only after their exact archived originals and all gates pass.
The resulting folder still requires the parent's full verification and seals.
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
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
EXTENSION = "优化实验/同骨干Subspace对照-20261014"
METHODS = ("official_cpu_two_pass", "numpy_cpu_two_pass", "current_cpu_one_pass")
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
ROOT_DOCS = ("项目总览.html", "先读我.txt", "实验说明.txt")
UTILITY_SOURCES = ("scripts/finish_route1_delivery.py", "scripts/resume_route1_controls.py", "scripts/checkpoint_resume_route1_controls.py", "scripts/resume_route1_resource_matrix.py", "scripts/await_route1_runs.py", "scripts/assemble_route1_resources.py", "scripts/Set-ProjectEnvironment.ps1")
EXCLUDED_PARTS = {"token_cache", "train_feature_cache", "metric_tmp", "metric_sort", "__pycache__", ".cache", ".venv", ".git"}
EVIDENCE_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".npz"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def member(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if not relative or parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative member: {relative!r}")
    path = (root / Path(*parts.parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping file: {relative}")
    return path


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"Refusing to replace evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def copy_checked(source: Path, destination: Path, expected: str | None = None) -> dict[str, Any]:
    require(source.is_file() and not source.is_symlink(), f"Plain source file required: {source}")
    before = digest(source)
    require(expected is None or before == expected, f"External evidence changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(digest(destination) == before, f"Refusing to replace different copied evidence: {destination}")
    else:
        shutil.copyfile(source, destination)
    require(digest(destination) == before == digest(source), f"Live or changed evidence cannot be sealed: {source}")
    return {"source": str(source), "sha256": before, "size_bytes": destination.stat().st_size}


def validate_extension(package: Path, extension: Path, audit_path: Path) -> dict[str, Any]:
    require(extension.is_dir(), "A completed built extension is required")
    require(not (extension / "整合回执.json").exists(), "Integration already has a receipt; refusing repetition")
    info = read(member(extension, "extension_manifest.json"))
    expected = {"status": "completed", "categories": 27, "normal_train_images": 12288, "test_images": 3887, "metric_rows": 162, "pca_models": 81, "raw_maps": 11661, "evaluation_masks": 3887, "figure_count": 54, "resource_trials": 117, "scientific_claim_eligible": False}
    require(all(info.get(key) == value for key, value in expected.items()), "Complete 27-category/81-model/162-row/117-resource extension required")
    declared = set()
    for row in info["files"]:
        path = member(extension, row["path"])
        require(row["path"] not in declared, "Duplicate extension manifest file")
        declared.add(row["path"])
        require(path.stat().st_size == int(row["size_bytes"]) and digest(path) == row["sha256"], f"Built extension hash mismatch: {row['path']}")
    actual = {p.relative_to(extension).as_posix() for p in extension.rglob("*") if p.is_file()}
    require(actual == declared | {"extension_manifest.json"}, "Extension membership changed since its built seal")
    history = member(extension, "history/DELIVERY-MANIFEST.json")
    require(digest(history) == info["previous_delivery_manifest_sha256"], "Historical manifest snapshot changed")
    require(digest(member(package, "DELIVERY-MANIFEST.json")) == digest(history), "Master manifest changed before this integration")
    old_index = {row["path"]: row for row in read(history)["files"]}
    originals = {}
    for name in ROOT_DOCS:
        current, archived = member(package, name), member(extension, "history/" + name)
        require(name in old_index and digest(current) == digest(archived) == old_index[name]["sha256"], f"Root document differs from its exact archived original: {name}")
        originals[name] = archived.read_text(encoding="utf-8-sig")
    run = extension / "run"
    config, state = read(member(run, "run_config.json")), read(member(run, "run_state.json"))
    require(state.get("status") == "completed" and state.get("completed_categories") == 27 and state.get("metric_rows") == 162, "Copied full precision run incomplete")
    tasks = [(str(d), str(c)) for d, c in config["expected_matrix"]["categories"]]
    require(len(tasks) == len(set(tasks)) == 27 and tuple(config["expected_matrix"]["methods"]) == METHODS, "Frozen category/method matrix changed")
    require({d: sum(td == d for td, _ in tasks) for d, _ in tasks} == {"mvtec_ad": 15, "visa": 12}, "Full MVTec AD/VisA category counts changed")
    protocol = config["protocol"]
    require(protocol["device"] == "cpu" and protocol["resource_repeats"] == 3 and protocol["resource_sample_count"] == 64, "Frozen device/repetition/normal subset protocol changed")
    require(protocol["resource_categories"] == [["mvtec_ad", "bottle"], ["visa", "cashew"]] and set(protocol["resource_modes"]) == {"one_lazy", "two_lazy", "one_eager", "two_eager", "attention_on", "attention_off"} and len(protocol["resource_modes"]) == 6, "The frozen two-category/six-mode resource design changed")
    audit = read(audit_path)
    audited_counts = {"status": "passed", "categories": 27, "metric_rows": 162, "normal_training_images": 12288, "test_images": 3887, "raw_maps": 11661, "bool_masks": 3887, "selected_samples": 54, "scientific_claim_eligible": False, "forward_or_training_executed": False, "pixel_metrics_recomputed": False}
    require(all(audit.get(key) == value for key, value in audited_counts.items()) and tuple(audit["methods"]) == METHODS, "Independent full semantic audit receipt required; its scope must be explicit")
    require(Path(audit["run"]).resolve() == Path(info["source_run"]).resolve(), "Independent audit belongs to another source run")
    require(len(audit["category_audits"]) == 27 and sum(len(row["models"]) for row in audit["category_audits"]) == 81, "Independent audit does not cover all PCA models")
    for filename, key in (("run_config.json", "run_config_sha256"), ("run_state.json", "run_state_sha256"), ("results/category_metrics.jsonl", "category_metrics_sha256"), ("results/samples.jsonl", "samples_sha256")):
        require(digest(member(run, filename)) == audit[key], "Independent audit is not bound to the copied run bytes")
    require(digest(member(extension, "code/scripts/audit_route1_run.py")) == audit["auditor_source_sha256"], "Independent auditor source differs from delivered source")
    categories = csv_rows(member(extension, "category_metrics.csv"))
    category_index = {(r["dataset"], r["category"], r["method"], r["score"]): r for r in categories}
    matrix = {(d, c, m, s) for d, c in tasks for m in METHODS for s in ("max", "top1p")}
    require(len(categories) == 162 and set(category_index) == matrix, "Category report does not cover all paired measurements")
    raw_metrics = lines(member(run, "results/category_metrics.jsonl"))
    raw_index = {(r["dataset"], r["category"], r["variant"]): r for r in raw_metrics}
    require(len(raw_metrics) == 162 and len(raw_index) == 162, "Audit-bound raw metric matrix is incomplete")
    for row in categories:
        raw = raw_index[(row["dataset"], row["category"], row["method"] + "_" + row["score"])]
        require(raw["status"] == "completed" and all(float(row[key]) == float(raw[key]) for key in METRICS), "Reported CSV metrics differ from independently audited raw measurements")
    macros = csv_rows(member(extension, "dataset_macro_metrics.csv"))
    macro_index = {(r["dataset"], r["method"], r["score"]): r for r in macros}
    require(len(macros) == 12 and set(macro_index) == {(d, method, score) for d in ("mvtec_ad", "visa") for method in METHODS for score in ("max", "top1p")}, "Both data sets and both score rules must be retained")
    for row in macros:
        grouped = [r for r in categories if (r["dataset"], r["method"], r["score"]) == (row["dataset"], row["method"], row["score"])]
        require(len(grouped) == int(row["categories"]), "Macro report category count differs")
        for metric in METRICS:
            values = [float(r[metric]) for r in grouped]
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in values) and abs(float(row[metric]) - sum(values) / len(values)) <= 1e-12, "Macro report is not an equal-category mean")
    resource = extension / "资源记录"
    matrix_state = read(member(resource, "matrix_state.json"))
    require(matrix_state["status"] == "completed" and matrix_state["completed_trials"] == 36, "Registered resource matrix must finish")
    trials = lines(member(resource, "resource_trials.jsonl"))
    workers = [r for r in trials if r.get("mode") in protocol["resource_modes"]]
    fits = [r for r in trials if r.get("experiment") == "fit_only"]
    registered = {(d, c, mode, repeat) for d, c in protocol["resource_categories"] for mode in protocol["resource_modes"] for repeat in range(3)}
    actual_workers = {(r["dataset"], r["category"], r["mode"], r["repeat"]) for r in workers}
    require(len(trials) == 117 and len(workers) == 36 and actual_workers == registered and all(r["status"] == "completed" and r["sample_count"] == 64 and r["device"] == "cpu" and r["gpu_used"] is False for r in workers), "All 36 actual CPU/64-normal/three-repeat resource results required")
    require(len(fits) == 81 and {(r["dataset"], r["category"], r["method"]) for r in fits} == {(d, c, m) for d, c in tasks for m in METHODS}, "All 81 actual full-category PCA timings required")
    for row in workers:
        receipt = member(resource, row["receipt_path"])
        parsed = read(receipt)
        require(digest(receipt) == row["receipt_sha256"] and all(row.get(key) == value for key, value in parsed.items()), "Resource result differs from its original worker receipt")
    for row in fits:
        model = read(member(run, f"categories/{row['dataset']}/{row['category']}/diagnostics.json"))["models"][row["method"]]
        require(row["status"] == "completed" and float(row["elapsed_seconds"]) == float(model["elapsed_seconds"]) and row["input_sha256"] == model["input_sha256"] and row["measurement_scope"] == "shared_process_not_per_method", "Fit timing is not bound to completed PCA diagnostics")
    failures = read(member(extension, "failed_resource_attempts.json"))
    failed_paths = list((resource / "failed_attempts").glob("*.json")) if (resource / "failed_attempts").exists() else []
    require(failures["failed_attempt_count"] == len(failed_paths) == matrix_state["failed_attempt_count"], "Retained resource failure attempts were omitted")
    summaries = csv_rows(member(extension, "resource_summary.csv"))
    return {"info": info, "config": config, "audit": audit, "tasks": tasks, "originals": originals, "macro_index": macro_index, "categories": categories, "trials": trials, "summaries": summaries, "failed_retries": len(failed_paths), "audit_sha256": digest(audit_path)}


def measurements(evidence: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    for dataset, category in evidence["config"]["protocol"]["resource_categories"]:
        subset = [row for row in evidence["trials"] if row.get("experiment") == "end_to_end" and row["dataset"] == dataset and row["category"] == category]
        modes = {}
        for mode in ("two_eager", "one_lazy"):
            rows = [r for r in subset if r["mode"] == mode]
            require(len(rows) == 3, "Three per-mode measurements required")
            require(all(math.isfinite(float(row[field])) and float(row[field]) > 0 for row in rows for field in ("fit_seconds", "total_process_seconds", "process_peak_working_set_bytes", "process_peak_commit_bytes")), "Measured resource times and process peaks must be finite and positive")
            summaries = [r for r in evidence["summaries"] if r["experiment"] == "end_to_end" and r["dataset"] == dataset and r["category"] == category and r["method"] == mode]
            require(len(summaries) == 1 and int(summaries[0]["successful_trials"]) == 3, "Actual resource CSV must retain the three successful measurements")
            modes[mode] = {}
            for field in ("fit_seconds", "total_process_seconds", "process_peak_working_set_bytes", "process_peak_commit_bytes"):
                median = float(summaries[0][field + "_median"])
                require(abs(median - statistics.median(float(r[field]) for r in rows)) <= 1e-12, "Resource summary median differs from original measured receipts")
                modes[mode][field] = median
        before, after = modes["two_eager"], modes["one_lazy"]
        comparisons.append({"dataset": dataset, "category": category, "normal_images": 64, "repeats": 3, "reference": "two_eager", "method": "one_lazy", "reference_medians": before, "method_medians": after, "fit_speedup_of_medians": before["fit_seconds"] / after["fit_seconds"], "reductions_percent_of_medians": {key: 100 * (before[key] - after[key]) / before[key] for key in before}})
    primary = []
    for dataset in ("mvtec_ad", "visa"):
        before = evidence["macro_index"][(dataset, METHODS[0], "max")]
        after = evidence["macro_index"][(dataset, METHODS[2], "max")]
        primary.append({"dataset": dataset, "reference": METHODS[0], "method": METHODS[2], "score": "max", "reference_metrics": {k: float(before[k]) for k in METRICS}, "method_metrics": {k: float(after[k]) for k in METRICS}, "delta_pp": {k: 100 * (float(after[k]) - float(before[k])) for k in METRICS}})
    max_macro_delta = max(abs(value) for row in primary for value in row["delta_pp"].values())
    return {"primary_max": primary, "max_primary_macro_delta_pp": max_macro_delta, "paired_metric_max_absolute_error": float(evidence["audit"]["maximum_paired_metric_absolute_error"]), "representative_resource_comparisons": comparisons, "resource_comparison_basis": "ratio/reduction of the three-repeat medians; raw repeat pairs and sample standard deviations are retained separately"}


def prose(measured: dict[str, Any], failures: int) -> tuple[str, str, str, str]:
    primary_text = []
    for row in measured["primary_max"]:
        values = "; ".join(f"{key} {100 * row['reference_metrics'][key]:.6f}% → {100 * row['method_metrics'][key]:.6f}%（{row['delta_pp'][key]:+.8g} 个百分点）" for key in METRICS)
        primary_text.append(f"{row['dataset']}：{values}。")
    resource_text = []
    for row in measured["representative_resource_comparisons"]:
        before, after = row["reference_medians"], row["method_medians"]
        reduction = row["reductions_percent_of_medians"]
        resource_text.append(f"{row['dataset']}/{row['category']} 固定前64张正常图、每模式3次独立进程：两遍eager → 单遍lazy 的特征提取加拟合时间中位数 {before['fit_seconds']:.6g}s → {after['fit_seconds']:.6g}s（{reduction['fit_seconds']:+.4f}% 减少，{row['fit_speedup_of_medians']:.4f}倍速度比）；完整进程时间 {before['total_process_seconds']:.6g}s → {after['total_process_seconds']:.6g}s；进程WS峰值中位数 {before['process_peak_working_set_bytes']/1024**2:.4f}MiB → {after['process_peak_working_set_bytes']/1024**2:.4f}MiB（{reduction['process_peak_working_set_bytes']:+.4f}% 减少）。负的减少比例表示回退。")
    boundary = "官方 SubspaceAD 本来就是两遍流式 PCA。本轮在相同 DINOv2-S/448、隐藏层[-4,-5]、全部正常图、0.99方差保留及共同256×256评测下，实际比较官方Torch CPU两遍、NumPy CPU两遍控制和当前NumPy CPU单遍统计。NumPy两遍控制用于区分后端与遍数影响。此前PaDiM到DINO/SubspaceAD的提升属于公开方法和骨干更换，不能归因于这次执行优化。"
    scope = "主图像评分为保存raw map的max；top1p（最高655像素均值）作为已有公开评分规则辅助报告，像素指标共享同一张raw map。资源2×2和attention开关分别归因；两遍资源worker是项目NumPy CPU控制。三次重复是资源计时，不是训练随机种子。64张正常图子集结果不外推为全类或全数据集加速；81条缓存fit耗时另列，共用进程峰值不按方法分摊。RAM中断后的normal特征提取允许认证checkpoint缓存命中，其normal extraction_seconds包含缓存读取；真实fresh/cached图数和耗时见run/checkpoint_execution_history。测试图始终重新前向，三项PCA仍共享输入并实际重拟合，81条fit_only与36次fresh worker口径保持原样。checkpoint缓存仅用于实现恢复，没有新增异常检测算法组件。"
    audit_scope = f"已完成27类、81个PCA模型、3,887张测试图的11,661张配对raw map和162条指标，另完成36次固定资源试验。独立语义审计通过，配对指标最大绝对误差为{measured['paired_metric_max_absolute_error']:.8g}（比例值），主max宏平均最大变化为{measured['max_primary_macro_delta_pp']:.8g}个百分点。审计核验模型、全部raw map、GT和图像AUROC；本入口整合时，完整像素指标复算和新的CPU原图推理验收尚待执行，最终状态与回执随后保存到验收记录。资源失败重试保留{failures}次，均单列。"
    colleague = "我们继续围绕无监督工业异常检测，选定预训练视觉特征加正常样本子空间建模路线。\n\n" + boundary + "\n\n" + audit_scope + "\n\n" + "\n".join(primary_text) + "\n\n" + "\n".join(resource_text) + "\n\n" + scope + "\n\n具体贡献是单遍统计、按需读取图片及attention输出控制的执行成本优化。每项都有配对结果，全部下降和资源失败保留；是否形成检测精度改进，按同骨干结果实际判断。\n"
    note = "同骨干 SubspaceAD 对照阶段（2026-10-14）\n\n" + boundary + "\n\n" + audit_scope + "\n\n" + "\n".join(primary_text) + "\n\n" + "\n".join(resource_text) + "\n\n" + scope + "\n\n完整配对报告：" + EXTENSION + "/report.html。下方内容保留2026-10-12及之前阶段的原记录，其中‘最新’和后续安排按其历史阶段理解。\n\n"
    paragraphs = [boundary, audit_scope, *primary_text, *resource_text, scope]
    html_section = "<section id='route1-same-dino-20261014'><h2>2026-10-14 同骨干 SubspaceAD 执行对照</h2>" + "".join("<p>" + html.escape(value) + "</p>" for value in paragraphs)
    html_section += f"<p><a href='{EXTENSION}/report.html'>完整同骨干报告与54组配对图</a> · <a href='{EXTENSION}/给同事的说明.txt'>可转发的说明</a> · <a href='{EXTENSION}/复现说明.txt'>命令与验收范围</a> · <a href='{EXTENSION}/验收记录/独立完整运行审计.json'>独立完整运行审计</a></p></section><h2>2026-10-12 及之前的历史实验</h2><p>以下原文及数值保留原阶段记录，涉及PaDiM与DINO公开强基线的对比；其中‘最新’指该历史阶段。</p>"
    readme_append = "\n\n2026-10-14 新增同骨干 SubspaceAD 对照\n" + audit_scope + "\n" + scope + "\n入口：" + EXTENSION + "/report.html。可转发说明和复现命令在同目录。原阶段Verify.ps1与各轮报告继续提供其对应验收；新阶段采用独立脚本：\npython -B \"" + EXTENSION + "/code/scripts/verify_route1_delivery.py\" --package-root . --inference-samples 2 --metrics-categories 27 --work-dir ../同骨干复算临时文件 --receipt ../同骨干复算回执.json --verify-master\n临时文件和回执必须置于交付文件夹外。该命令使用本地权重和缓存，不进行PCA拟合、反向训练或联网下载。\n"
    return html_section, readme_append, note, colleague


def reproduction_text(project: Path) -> str:
    project_command = str(project).replace("'", "''")
    text = "同骨干 SubspaceAD 复现与验收\n\n主表图像评分max，top1p辅助；全27类、12,288张正常训练图、3,887张测试图，三个CPU PCA控制、81模型、162指标。DINOv2-S/448/layers[-4,-5]，共同256GT与raw map评测。已访问测试集，scientific_claim_eligible=false。\n\n交付文件夹内的离线验收（已有Python3.11及冻结依赖；输出在文件夹外）：\npython -B \"" + EXTENSION + "/code/scripts/verify_route1_delivery.py\" --package-root . --inference-samples 2 --metrics-categories 27 --work-dir ../同骨干复算临时文件 --receipt ../同骨干复算回执.json --verify-master\n此命令核验文件、历史证据、模型与raw map，并完整重算四指标和两张原图的三个CPU预测；不进行训练或PCA拟合，不联网。独立完整运行审计只复核保存的像素指标一致性，完整像素重算与新CPU前向需要上述验收回执单独证明。\n\n完整数据再次运行需原项目工作区、全部官方数据集以及配置引用的历史冻结运行/输入缓存；交付文件夹只包含54张固定源图、全部配对预测缓存、模型和权重，不包含全体训练/测试原图。包内单独code目录不能当作已准备全部数据的工作区。以下PowerShell命令从当前项目执行，输出必须是新的目录并重新认证输入：\nSet-Location -LiteralPath '" + project_command + "'\n& .\\.venv\\Scripts\\python.exe -B scripts/run_route1_controls.py --config configs/route1_controls_20261014.json --output-dir runs/quick/route1-reproduction-new\n& .\\.venv\\Scripts\\python.exe -B scripts/run_route1_resource_matrix.py --config configs/route1_controls_20261014.json --output-dir runs/quick/route1-resource-reproduction-new\n资源准入、完整性及resume适配说明见code/docs/route1_controls_protocol.md和code/scripts/resume_route1_controls.py、resume_route1_resource_matrix.py。\n\n冻结运行完成后，再用独立auditor、资源assembler和builder处理已完成证据。原运行首次冻结源码不可热改；资源2类×6模式×3次=36个fresh process，64张normal subset不用于全数据集加速外推；资源重复不是训练种子。共享CPU背景负载与失败重试保留，GPU字段为not_used。\n\n代码和环境在code/，共享DINO权重在原交付assets/weights/；方法/权重/数据许可证与来源保留。旧root入口原文在history/。Verify.ps1提供旧阶段推理验收与master全哈希核对，新阶段推理及四指标采用上述扩展脚本。\n"
    text += "\n缓存的运行环境身份\nPyTorch官方数值说明指出，跨版本、平台及CPU/GPU执行不保证浮点结果逐位相同。因此checkpoint context绑定实际运行库版本、平台、设备与线程等环境身份，环境变化后禁止继续复用旧缓存；这是恢复证据的保护，不是检测算法创新。主来源：PyTorch Numerical accuracy，https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html 。\n"
    return text + "\n实际实现与冻结文档描述的区别\n1. 冻结协议使用‘normal token落盘memmap’的用语。实际runner每张图写一个float32 .npy临时token块，PCA逐块np.load读取，避免大mapping常驻；每类完成后删除token块，保留token_cache_manifest.json的原图和token摘要。它不是全类常驻memmap，也没有同时在RAM保存整个正常特征库。\n2. 冻结协议笼统规定至少0.5秒资源监控。实际完整主run的PowerShell外部监控循环间隔为2秒；独立资源worker为0.5秒（2Hz）采样，并在阶段切换记录样本。主run不能据此声称满足0.5秒监控；阶段采样峰值也不能保证捕获所有短时峰值。Windows全进程OS峰值计数另行保留。\n3. 完整主run复用同一进程和共享特征，峰值只用于运行监控，不能做逐方法内存归因。36个fresh worker的OS进程峰值与采样私有内存分列；缓存PCA时间、未缓存图像提取加拟合时间、模型加载/预热和完整进程时间分别报告。\n4. RAM中断后可用code/scripts/checkpoint_resume_route1_controls.py恢复normal提取。缓存独立于正式结果，逐图认证原文件/像素、token SHA及producer上下文；只对正常训练特征复用，测试图始终fresh forward。认证命中不会改变特征、划分、评分或三PCA输入，每个方法仍实际拟合。恢复类别的normal extraction_seconds含缓存读取，不可当作全新DINO前向速度；fresh/cached数量和独立耗时以run/checkpoint_execution_history中的实际journal为准。只复用helper自生产且即时封存的块，不adopt旧v1裸.npy。checkpoint缓存完成类别后清理，不打包live token/cache数据。81条fit_only和36次fresh资源worker均不从此缓存提取时间推算收益。缓存与任何缓存重建只属于实现恢复，没有新增检测组件。\n冻结协议与算法源码按原样保留，此处记录实际执行口径。\n"


def evidence_files(project: Path, package: Path, source_run: Path, custom: list[Path]) -> list[Path]:
    roots = sorted(source_run.parent.glob(source_run.name + "-monitor-*"))
    roots += sorted(source_run.parent.glob("route1-normal-pilot-*"))
    waiter_parent = project / "runs/waiters"
    if waiter_parent.is_dir():
        roots += sorted(waiter_parent.glob("route1-completion-*"))
    roots += [path.resolve(strict=True) for path in custom]
    found = {}
    for root in roots:
        require(root.resolve().is_relative_to(project) and root.is_dir() and not root.resolve().is_relative_to(package) and not package.is_relative_to(root.resolve()), "Evidence directories must stay outside the delivery and cannot contain it")
        require(not root.name.startswith("route1-checkpoint-cache-"), "Live checkpoint cache cannot be archived as delivery evidence")
        if (root / "state.json").is_file():
            require(read(root / "state.json").get("status") == "completed", "Controller receipt is still live or incomplete")
        if (root / "run_state.json").is_file():
            require(read(root / "run_state.json").get("status") in ("pilot_completed", "completed", "failed"), "Pilot receipt is still live")
        if "-monitor-" in root.name:
            require((root / "summary.json").is_file() and read(root / "summary.json").get("status") in ("completed", "failed"), "Monitor receipt is still live")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS or part.startswith((".quick-metrics-", "route1-checkpoint-cache-")) for part in relative.parts) or "lock" in path.name.casefold() or path.suffix.lower() not in EVIDENCE_SUFFIXES:
                continue
            require(not path.is_symlink() and path.resolve().is_relative_to(project), "Evidence symlink or escaping path rejected")
            found[str(path.resolve())] = path.resolve()
    return sorted(found.values())


def validate_resume_sources(project: Path, extension: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind actually recorded in-memory repair adapters to copied utility code."""
    output = []
    precision_source = member(project, "scripts/resume_route1_controls.py")
    resource_source = member(project, "scripts/resume_route1_resource_matrix.py")
    for path in sorted((extension / "run/execution_history").rglob("resume_entry.json")):
        context = read(path)
        require(context.get("resume_adapter_sha256") == digest(precision_source) and context.get("frozen_runner_sha256") == config["source_sha256"]["scripts/run_route1_controls.py"] and context.get("algorithm_and_saved_results_changed") is False and context.get("frozen_source_hashes_verified") is True, "Recorded precision resume adapter differs from copied utility source")
        output.append({"context": path.relative_to(extension).as_posix(), "adapter": "scripts/resume_route1_controls.py", "sha256": context["resume_adapter_sha256"], "algorithm_and_saved_results_changed": False})
    for path in sorted((extension / "资源记录/resume_context").rglob("resume_entry.json")):
        context = read(path)
        matrix = read(member(extension, "资源记录/matrix_config.json"))
        require(context.get("adapter_sha256") == digest(resource_source) and context.get("original_matrix_source_sha256") == matrix["matrix_source_sha256"] and context.get("worker_sha256") == matrix["worker_sha256"] and context.get("measurement_or_receipt_bytes_changed") is False and context.get("plan_changed") is False, "Recorded resource resume adapter differs from copied utility source")
        output.append({"context": path.relative_to(extension).as_posix(), "adapter": "scripts/resume_route1_resource_matrix.py", "sha256": context["adapter_sha256"], "measurement_or_receipt_bytes_changed": False})
    return output


def validate_checkpoint_history(project: Path, extension: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Audit checkpoint journals as metadata only; never reopen token caches."""
    history = extension / "run/checkpoint_execution_history"
    if not history.is_dir():
        return {"used": False, "sessions": [], "scope": "no checkpoint execution journal present"}
    helper_sha = digest(member(project, "scripts/checkpoint_resume_route1_controls.py"))
    delegate_sha = digest(member(project, "scripts/resume_route1_controls.py"))
    full_run = Path(read(member(extension, "extension_manifest.json"))["source_run"]).resolve()
    run_config_sha = digest(member(extension, "run/run_config.json"))
    categories = {tuple(pair) for pair in config["expected_matrix"]["categories"]}
    category_order = {tuple(pair): index for index, pair in enumerate(config["expected_matrix"]["categories"])}
    timer_fields = ("fresh_forward_seconds", "cache_read_validation_seconds", "source_validation_seconds", "cache_write_seconds", "hook_total_seconds")
    fingerprint_fields = ("metadata_sha256", "token_file_sha256", "token_content_sha256", "source_image_sha256")
    sessions, commits, token_producers, normal_results, cleanup_records, inventories = [], {}, {}, [], [], {}
    context_identities = set()

    def frozen_inventory(dataset: str, category: str, phase: str) -> list[dict[str, Any]]:
        require((dataset, category) in categories and phase in ("normal", "test"), "Checkpoint category/phase is outside the frozen matrix")
        key = (dataset, category, phase)
        if key not in inventories:
            inventories[key] = read(member(extension, f"run/categories/{dataset}/{category}/" + ("train_manifest.json" if phase == "normal" else "test_inventory.json")))
        return inventories[key]

    def token_identity(row: dict[str, Any]) -> tuple[str, ...]:
        return (row["context_sha256"], row["cache_entry"], *(row[field] for field in fingerprint_fields))

    def read_journal(directory: Path, has_summary: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path = directory / "events.jsonl"
        if not path.is_file():
            return [], {"journal_sha256": None, "ignored_truncated_tail_bytes": 0, "ignored_truncated_tail_sha256": None}
        raw = path.read_bytes()
        chunks, records, ignored = raw.splitlines(keepends=True), [], b""
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                record = json.loads(chunk.decode("utf-8-sig"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                require(not has_summary and index == len(chunks) - 1 and not chunk.endswith((b"\n", b"\r")), "Checkpoint journal has a corrupt completed or intermediate event")
                ignored = chunk
                continue
            require(isinstance(record, dict), "Checkpoint journal event must be an object")
            records.append(record)
        return records, {"journal_sha256": hashlib.sha256(raw).hexdigest(), "ignored_truncated_tail_bytes": len(ignored), "ignored_truncated_tail_sha256": hashlib.sha256(ignored).hexdigest() if ignored else None}

    for directory in sorted(path for path in history.iterdir() if path.is_dir()):
        wrapped = read(member(directory, "context.json"))
        context = wrapped["context"]
        encoded = (json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        context_sha = hashlib.sha256(encoded).hexdigest()
        require(wrapped["context_sha256"] == context_sha, "Checkpoint context canonical SHA differs")
        context_identities.add(context_sha)
        require(len(context_identities) == 1, "Checkpoint recovery changed actual runtime or cache context between executions")
        require(context["checkpoint_helper_sha256"] == helper_sha and context["delegate_adapter_sha256"] == delegate_sha and context["run_config_sha256"] == run_config_sha, "Checkpoint producer/helper/delegate differs from the frozen run")
        require(Path(context["full_run"]).resolve() == full_run and Path(context["project_root"]).resolve() == project and Path(context["checkpoint_cache_root"]).absolute() == (project / "runs/quick/route1-checkpoint-cache-20261014").absolute(), "Checkpoint context belongs to another run or cache namespace")
        for key in ("source_sha256", "environment_sha256", "vendor_source_sha256", "input_metadata_sha256"):
            require(context[key] == config[key], "Checkpoint context changed execution dependency hashes")
        require(context["protocol_sha256"] == config["protocol_sha256"] and context["weight_files_sha256"] == config["protocol"]["weight_files_sha256"] and context["checkpoint_sha256"] == config["backbone"]["checkpoint_sha256"], "Checkpoint protocol/weight identity changed")
        require(context["schema_version"] == 1 and context["checkpoint_cache_schema"] == 1 and context["categories"] == config["expected_matrix"]["categories"] and context["device"] == "cpu" and context["cpu_threads"] == 4 and context["feature_dtype"] == "float32" and context["returned_shape"] == [32, 32, 384] and context["stored_shape"] == [1024, 384] and context["image_res"] == 448 and context["layers"] == [-4, -5], "Checkpoint feature geometry/backend differs")
        packages = context["runtime_package_versions"]
        require(set(packages) == {"torch", "torchvision", "numpy", "transformers", "Pillow", "scipy", "scikit-learn", "opencv-python", "threadpoolctl", "safetensors"} and all(isinstance(value, str) and value for value in packages.values()), "Checkpoint actual runtime package versions are missing")
        require(all(isinstance(context[key], str) and context[key] for key in ("python_version", "python_executable", "platform")) and context["thread_environment"] == {key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, "Checkpoint actual Python/platform/thread fingerprint is missing")
        summary_path = directory / "summary.json"
        records, journal_boundary = read_journal(directory, summary_path.is_file())
        counts = {"normal_fresh_count": 0, "normal_cache_hit_count": 0, "test_fresh_count": 0}
        seconds = {key: 0.0 for key in timer_fields}
        consumed = 0
        active_category, phase, next_image_index = None, "normal", 0
        for row in records:
            require(row["context_sha256"] == context_sha, "Checkpoint event belongs to another context")
            kind = row["type"]
            if kind == "cache_commit":
                require(row["producer_execution_id"] == directory.name, "Checkpoint producer execution identity changed")
                dataset, category, normal_index = row["dataset"], row["category"], int(row["normal_index"])
                inventory = frozen_inventory(dataset, category, "normal")
                require(0 <= normal_index < len(inventory) and row["source_image_sha256"] == inventory[normal_index]["sha256"] and row["cache_entry"] == f"{dataset}/{category}/{normal_index:06d}" and all(isinstance(row.get(field), str) and re.fullmatch(r"[0-9a-f]{64}", row[field]) for field in fingerprint_fields), "Checkpoint producer seal is not a frozen normal image")
                key = (context_sha, row["producer_execution_id"], row["cache_entry"])
                require(key not in commits, "Duplicate checkpoint producer commit")
                commits[key] = row
                token_producers.setdefault(token_identity(row), set()).add(row["producer_execution_id"])
                continue
            if kind == "image_failure":
                require(row["execution_id"] == directory.name and row["event_index"] == consumed and isinstance(row.get("error"), str), "Checkpoint failure event queue identity changed")
                continue
            require(kind == "image_result" and row["execution_id"] == directory.name and row["event_index"] == consumed, "Checkpoint image event order/identity changed")
            action = row["action"]
            require(action in ("normal_fresh", "normal_cache_hit", "test_fresh"), "Unknown checkpoint image action")
            dataset, category, image_index = row["dataset"], row["category"], int(row["image_index"])
            inventory = frozen_inventory(dataset, category, row["phase"])
            category_pair = (dataset, category)
            if category_pair != active_category:
                require((active_category is None or (phase == "test" and next_image_index == len(frozen_inventory(*active_category, "test")) and category_order[category_pair] > category_order[active_category])) and row["phase"] == "normal" and image_index == 0, "Checkpoint category order changed or left a partial category")
                active_category, phase, next_image_index = category_pair, "normal", 0
            require(row["phase"] == phase and image_index == next_image_index, "Checkpoint normal/Test image order is repeated or discontinuous")
            require(0 <= image_index < len(inventory), "Checkpoint image index escapes frozen manifest")
            frozen = inventory[image_index]
            expected_image_sha = frozen["sha256"] if row["phase"] == "normal" else frozen["image_sha256"]
            source_path = project / frozen["path"] if row["phase"] == "normal" else Path(frozen["image_path"])
            require(row["source_image_sha256"] == expected_image_sha and Path(row["source_path"]).resolve() == source_path.resolve() and (row["phase"] == "normal") == action.startswith("normal_"), "Checkpoint source/action is not the frozen normal/Test event")
            for key in timer_fields:
                require(math.isfinite(float(row[key])) and float(row[key]) >= 0, "Checkpoint event timing is invalid")
                seconds[key] += float(row[key])
            if action == "test_fresh":
                require(all(row.get(key) is None for key in ("cache_entry", "metadata_sha256", "token_file_sha256", "token_content_sha256")) and float(row["cache_read_validation_seconds"]) == 0, "Test checkpoint was used or claimed")
                counts["test_fresh_count"] += 1
            else:
                require(row["cache_entry"] == f"{dataset}/{category}/{image_index:06d}" and all(isinstance(row.get(field), str) and re.fullmatch(r"[0-9a-f]{64}", row[field]) for field in fingerprint_fields), "Normal checkpoint producer fingerprints are incomplete")
                if action == "normal_cache_hit":
                    require(float(row["fresh_forward_seconds"]) == 0, "Cache hit claims a fresh forward")
                counts[action + "_count"] += 1
                normal_results.append(row)
            consumed += 1
            next_image_index += 1
            if phase == "normal" and next_image_index == len(inventory):
                phase, next_image_index = "test", 0
        summary = read(summary_path) if summary_path.is_file() else None
        if summary is not None:
            require(summary["execution_id"] == directory.name and summary["context_sha256"] == context_sha and summary["checkpoint_helper_sha256"] == helper_sha and summary["delegate_adapter_sha256"] == delegate_sha, "Checkpoint summary producer identity differs")
            require(all(summary[key] == value for key, value in counts.items()) and summary["events_consumed"] == consumed and summary["queue_complete"] == (consumed == summary["expected_events"]), "Checkpoint summary count/queue differs from journal")
            require(summary["expected_events"] == summary["expected_normal_events"] + summary["expected_test_events"] and counts["normal_fresh_count"] + counts["normal_cache_hit_count"] <= summary["expected_normal_events"] and counts["test_fresh_count"] <= summary["expected_test_events"], "Checkpoint normal/Test queue accounting differs")
            require(summary["test_always_fresh"] is True and summary["algorithms_and_protocol_unchanged"] is True and summary["fit_only_and_fresh_resource_workers_changed"] is False and summary["adopted_unsealed_token_count"] == 0, "Checkpoint recovery altered detector/worker boundaries or adopted unsealed tokens")
            require(summary["forward_attempt_count"] == summary["forward_success_count"] + summary["failed_forward_count"] and summary["forward_success_count"] >= counts["normal_fresh_count"] + counts["test_fresh_count"], "Checkpoint actual forward accounting is inconsistent")
            require(summary["status"] in ("completed", "failed") and (summary["status"] != "completed" or summary["queue_complete"]), "Checkpoint completion claims an incomplete queue")
            require(summary["normal_extraction_seconds_scope"] == "includes normal cache read/validation and fresh forwards; not full first-pass DINO time" and all(math.isfinite(float(summary[field])) and float(summary[field]) >= 0 for field in ("elapsed_seconds", "startup_context_and_queue_seconds")) and float(summary["startup_context_and_queue_seconds"]) <= float(summary["elapsed_seconds"]), "Checkpoint extraction/startup timing scope changed")
            for key in timer_fields:
                actual = float(summary[key])
                require(math.isfinite(actual) and actual >= 0, "Checkpoint summary timing is invalid")
                if key == "fresh_forward_seconds":
                    # Failures after/during a forward can lack image_result;
                    # their measured time remains in the original summary.
                    require(actual + 1e-8 >= seconds[key], "Checkpoint summary dropped successful forward time")
                else:
                    require(abs(actual - seconds[key]) <= 1e-8, "Checkpoint summary timing differs from returned-image ledger")
        index_path = directory / "completed_cache_summary.json"
        indexed_summaries = {}
        if index_path.is_file():
            for indexed in read(index_path):
                path = member(directory, indexed["summary_path"])
                require(path.parent == directory.resolve(), "Checkpoint cleanup index points outside its execution directory")
                require(path not in indexed_summaries, "Duplicate checkpoint cleanup summary index")
                require(digest(path) == indexed["summary_sha256"], "Checkpoint cleanup summary index hash differs")
                indexed_summaries[path] = indexed
        # A hard kill can occur after an individual cleanup seal is durable,
        # but before the final list is written. Retain and bind those seals.
        cleanup_paths = set(indexed_summaries) | {path.resolve() for path in directory.glob("cache_summary_*.json")}
        orphan_cleanup_count = 0
        cleanup_evidence = []
        for path in sorted(cleanup_paths):
            require(path.parent == directory.resolve(), "Checkpoint cleanup seal escapes its execution directory")
            cleanup = read(path)
            dataset, category = cleanup["dataset"], cleanup["category"]
            require((dataset, category) in categories and path.name == f"cache_summary_{dataset}_{category}.json", "Checkpoint cleanup category filename differs")
            indexed = indexed_summaries.get(path)
            if indexed is not None:
                require((indexed["dataset"], indexed["category"]) == (dataset, category), "Checkpoint cleanup category index differs")
            else:
                orphan_cleanup_count += 1
            cleanup_evidence.append({"summary_path": path.name, "summary_sha256": digest(path), "indexed": indexed is not None, "physical_removal_evidence": "completed_by_final_cleanup_index" if indexed is not None else "not_proven_intent_only"})
            cat = extension / f"run/categories/{dataset}/{category}"
            train = frozen_inventory(dataset, category, "normal")
            token_manifest = read(member(cat, "token_cache_manifest.json"))["files"]
            require(cleanup["context_sha256"] == context_sha and cleanup["cache_removed_after_completed_category"] is True and cleanup["completion_sha256"] == digest(member(cat, "completion.json")), "Checkpoint cleanup is not bound to a completed category")
            tokens = cleanup["authenticated_token_records"]
            require(len(tokens) == len(train) == len(token_manifest) and (indexed is None or len(tokens) == indexed["normal_images"]), "Checkpoint cleanup token inventory differs")
            for token_index, token in enumerate(tokens):
                require(token["normal_index"] == token_index and token["cache_entry"] == f"{dataset}/{category}/{token_index:06d}" and token["source_image_sha256"] == train[token_index]["sha256"] and token["token_file_sha256"] == token_manifest[token_index]["sha256"], "Checkpoint token differs from the frozen worker's consumed normal block")
                cleanup_records.append({"context_sha256": context_sha, **token})
        sessions.append({"execution_id": directory.name, "context_sha256": context_sha, "helper_sha256": helper_sha, "summary_present": summary is not None, "status": summary["status"] if summary else "interrupted_without_final_summary", **counts, **journal_boundary, "returned_image_timing_sums": seconds, "reported_summary": summary, "cleanup_index_present": index_path.is_file(), "unindexed_cleanup_seals": orphan_cleanup_count, "cleanup_seals": cleanup_evidence, "scope": "missing final summaries remain missing; truncated raw journal tails are retained without invented events; unindexed cleanup seals authenticate tokens but do not prove physical removal; forward time from failed calls is not invented"})
    for row in normal_results:
        producers = token_producers.get(token_identity(row), set())
        require(bool(producers), "Checkpoint cache result lacks an authenticated producer commit")
        if row["action"] == "normal_fresh":
            require(row["execution_id"] in producers, "Fresh normal result belongs to another producer")
    for token in cleanup_records:
        commit = commits.get((token["context_sha256"], token["producer_execution_id"], token["cache_entry"]))
        require(commit is not None and all(commit[key] == token[key] for key in ("metadata_sha256", "token_file_sha256", "token_content_sha256", "source_image_sha256")), "Checkpoint cleanup fingerprint lacks its original producer seal")
    return {"used": bool(sessions), "context_sha256": next(iter(context_identities), None), "sessions": sessions, "producer_commits": len(commits), "returned_normal_results": len(normal_results), "completed_token_bindings": len(cleanup_records), "test_results_always_fresh": True, "normal_extraction_seconds_scope": "contains validated checkpoint reads plus fresh forwards; never used as fresh resource benchmark", "runtime_identity_bound": True, "legacy_unsealed_tokens_adopted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        project, package, audit_path = args.project_root.resolve(strict=True), args.package_root.resolve(strict=True), args.audit_receipt.resolve(strict=True)
        require(audit_path.is_relative_to(project) and not audit_path.is_relative_to(package), "Independent audit receipt must be outside the delivery but inside the project")
        extension = package / EXTENSION
        evidence = validate_extension(package, extension, audit_path)
        measured = measurements(evidence)
        section, readme_append, note, colleague = prose(measured, evidence["failed_retries"])
        old_html = evidence["originals"]["项目总览.html"]
        headings = list(re.finditer(r"<h1\b[^>]*>.*?</h1\s*>", old_html, flags=re.IGNORECASE | re.DOTALL))
        require(len(headings) == 1 and "route1-same-dino-20261014" not in old_html, "Expected one historical h1 and no prior integration section")
        utilities = [(member(project, relative), extension / "code" / relative) for relative in UTILITY_SOURCES]
        resume_bindings = validate_resume_sources(project, extension, evidence["config"])
        checkpoint_bindings = validate_checkpoint_history(project, extension, evidence["config"])
        external = evidence_files(project, package, Path(evidence["info"]["source_run"]).resolve(strict=True), args.evidence_dir)
        prospective = [(audit_path, extension / "验收记录/独立完整运行审计.json"), *utilities, *((source, extension / "验收记录/外部原始记录" / source.relative_to(project)) for source in external)]
        planned = [{"source": str(source), "destination": destination.relative_to(package).as_posix(), "sha256": digest(source), "size_bytes": source.stat().st_size} for source, destination in prospective]
        for source, destination in prospective:
            require(not destination.exists() or digest(destination) == digest(source), "Different integration evidence already exists")
        for name in ("给同事的说明.txt", "复现说明.txt", "验收记录/整合数值摘要.json", "验收记录/外部记录清单.json", "验收记录/checkpoint恢复审计.json"):
            require(not (extension / name).exists(), "Refusing to replace a previous integration artifact")
        if args.check_only:
            print(json.dumps({"status": "ready_to_integrate", "delivery_modified": False, "copied_evidence_files": len(external), "measurements": measured}, ensure_ascii=False, indent=2))
            return 0
        copied = []
        for record, (source, destination) in zip(planned, prospective, strict=True):
            copied_record = copy_checked(source, destination, record["sha256"])
            copied.append({**copied_record, "path": destination.relative_to(extension).as_posix()})
        write_json(extension / "验收记录/整合数值摘要.json", measured)
        write_json(extension / "验收记录/checkpoint恢复审计.json", checkpoint_bindings)
        write_json(extension / "验收记录/外部记录清单.json", {"schema_version": 1, "files": copied, "copy_hash_verified_before_and_after": True, "caches_and_live_locks_excluded": True, "independent_audit_sha256": evidence["audit_sha256"], "recorded_resume_adapter_source_bindings": resume_bindings, "waiter_source_scope": "captured controller utility source; controller state alone is not evidence of measurement completion"})
        (extension / "给同事的说明.txt").write_text(colleague, encoding="utf-8")
        (extension / "复现说明.txt").write_text(reproduction_text(project), encoding="utf-8")
        # Recheck each protected root byte immediately before the first edit.
        old_hashes = {}
        for name in ROOT_DOCS:
            old_hashes[name] = digest(member(extension, "history/" + name))
            require(digest(member(package, name)) == old_hashes[name], "Historical root document changed during evidence copying")
        end = headings[0].end()
        (package / "项目总览.html").write_text(old_html[:end] + section + old_html[end:], encoding="utf-8")
        (package / "先读我.txt").write_text(evidence["originals"]["先读我.txt"] + readme_append, encoding="utf-8")
        (package / "实验说明.txt").write_text(note + evidence["originals"]["实验说明.txt"], encoding="utf-8")
        receipt = {"schema_version": 1, "status": "integrated_pending_main_verification_and_seals", "created_at_utc": datetime.now(timezone.utc).isoformat(), "independent_audit_sha256": evidence["audit_sha256"], "integration_utility_sha256": digest(Path(__file__).resolve()), "root_original_sha256": old_hashes, "root_integrated_sha256": {name: digest(package / name) for name in ROOT_DOCS}, "historical_html_body_preserved": True, "historical_sources_models_and_metrics_replaced": False, "copied_evidence_files": len(copied), "recomputation_or_process_operations_performed": False, "extension_manifest_updated": False, "master_manifest_updated": False, "next_steps": ["refresh extension manifest with builder --seal-only", "full delivered metric and selected CPU inference verification without --verify-master until final sealing", "copy final acceptance receipts then reseal extension and master manifests", "verify final master hashes and all historical/current evidence"]}
        write_json(extension / "整合回执.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "integration_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
