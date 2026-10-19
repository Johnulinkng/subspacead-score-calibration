"""Unchanged four-group acceptance rules for the content-conditioned trial."""
from pathlib import Path
import csv
import html
import numpy as np
from run_score_calibration import write

METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30", "normal_pixel_fpr",
           "background_pixel_fpr", "anomaly_pixel_recall", "normal_image_fpr", "anomaly_image_tpr")


def summarize(run, cfg, rows):
    macros = []
    for dataset, count in (("mvtec_ad", 15), ("visa", 12)):
        for variant in cfg["variants"]:
            subset = [r for r in rows if (r["dataset"], r["variant"]) == (dataset, variant)]
            if len(subset) != count:
                raise ValueError("Incomplete full-category matrix")
            macros.append({"dataset": dataset, "variant": variant, "categories": count,
                **{k: float(np.mean([r[k] for r in subset])) for k in METRICS}})
    checks = []
    acceptance = cfg["acceptance"]
    for d in ("mvtec_ad", "visa"):
        get = lambda v: next(r for r in macros if (r["dataset"], r["variant"]) == (d, v))
        base, candidate, removed = get("baseline"), get("conditional_tail"), get("conditioning_removed")
        pairs = [(r, next(b for b in rows if (b["dataset"], b["category"], b["variant"]) == (d, r["category"], "baseline")))
                 for r in rows if (r["dataset"], r["variant"]) == (d, "conditional_tail")]
        gains = {k: [r[k] - b[k] for r, b in pairs] for k in ("pixel_ap", "aupro_0_30")}
        stable = {k: candidate[k] - base[k] >= acceptance["localization_gain_minimum"] and
            np.mean(np.array(values) > 0) >= acceptance["category_improvement_fraction_minimum"] and
            np.median(values) > acceptance["paired_median_gain_minimum"] for k, values in gains.items()}
        false_gains = [b["normal_pixel_fpr"] - r["normal_pixel_fpr"] for r, b in pairs]
        false_pass = candidate["normal_pixel_fpr"] <= base["normal_pixel_fpr"] * (1 - acceptance["false_positive_relative_reduction_minimum"]) and (
            candidate["anomaly_pixel_recall"] >= base["anomaly_pixel_recall"] - acceptance["pixel_recall_maximum_drop"]) and (
            candidate["anomaly_image_tpr"] >= base["anomaly_image_tpr"] - acceptance["image_tpr_maximum_drop"]) and (
            np.mean(np.array(false_gains) > 0) >= acceptance["category_improvement_fraction_minimum"]) and np.median(false_gains) > 0
        checks.append({"dataset": d, "image_noninferiority": candidate["image_auroc"] >= base["image_auroc"] - acceptance["image_auroc_maximum_drop"],
            "pixel_ap_stable_gain": bool(stable["pixel_ap"]), "aupro_stable_gain": bool(stable["aupro_0_30"]),
            "false_positive_gain_without_recall_loss": bool(false_pass), "delta": {k: candidate[k] - base[k] for k in METRICS},
            "candidate_minus_ablation": {k: candidate[k] - removed[k] for k in METRICS},
            "category_gains": {k: {"positive": int(np.sum(np.array(values) > 0)), "total": len(values), "median": float(np.median(values))} for k, values in gains.items()}})
    common_branch = any(all(row[k] for row in checks) for k in ("pixel_ap_stable_gain", "aupro_stable_gain", "false_positive_gain_without_recall_loss"))
    component_support = all((r["candidate_minus_ablation"]["pixel_ap"] > 0 or r["candidate_minus_ablation"]["aupro_0_30"] > 0) for r in checks)
    candidate_rows = [r for r in rows if r["variant"] == "conditional_tail"]
    overhead = max(r["paired_gpu_extra_fraction"] for r in candidate_rows)
    parameter_bytes = max(r["calibration_parameter_bytes"] for r in candidate_rows)
    resource_pass = overhead <= acceptance["apply_overhead_fraction_of_measured_cpu_forward_maximum"] and parameter_bytes <= acceptance["calibration_parameter_max_mib"] * 1024**2
    success = common_branch and all(r["image_noninferiority"] for r in checks) and component_support and resource_pass
    decision = {"status": "supported" if success else "rejected", "effectiveness_gate_passed": bool(success), "goal_achieved": False,
        "checks": checks, "same_gain_branch_both_datasets": common_branch, "component_support_both_datasets": component_support,
        "resource_timing_and_parameter_gate": bool(resource_pass), "maximum_paired_gpu_extra_fraction": overhead,
        "calibration_parameter_max_bytes": parameter_bytes,
        "notice": "Goal completion still requires independent replay, isolated resource measurements and full requirements audit.",
        "test_notice": cfg["test_notice"], "normal_partition": cfg["normal_partition"], "novelty_notice": cfg["novelty_notice"]}
    write(run / "category_metrics.json", rows)
    write(run / "dataset_macro_metrics.json", macros)
    write(run / "decision.json", decision)
    for name, values in (("category_metrics", rows), ("dataset_macro_metrics", macros)):
        with (run / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in values for k in row)))
            writer.writeheader()
            writer.writerows(values)
    text = ["按内容分组的正常波动校准", "", "当前完整评测结论：" + ("达到预先固定的效果、时间和参数要求，等待独立审核。" if success else "没有达到预先固定的要求，不作为成功改进交付。"), "",
        "四组共享原版SubspaceAD模型和同一次DINOv2-S GPU前向。基线保持原评分机制，每张基线图与此前CPU图核对；普通全局校准为正仿射变换；候选按正常内容原型确定误差上界；消融仅取消内容条件，统一使用全局正常上界。",
        "8个原型及上界只来自既有正常训练统计分区，操作阈值来自另一个正常训练分区。PCA已见过两个分区，因此没有独立校准覆盖率保证。内容描述来自现有PCA前16维，按固定float16精度储存、再用float64距离分配；原重建误差仍是float32。",
        "没有改变骨干、PCA解释方差或评测口径。图像分数始终取本组实际输出异常图的最大值，像素指标使用原精确256平面评测器。单遍PCA的精度和执行收益沿用前面已完成的同骨干对照，不算作本评分组件收益。", ""]
    for row in checks:
        text.append(f"{row['dataset']}：图像AUROC变化{row['delta']['image_auroc']*100:+.4f}个百分点，像素AP变化{row['delta']['pixel_ap']*100:+.4f}个百分点，AUPRO变化{row['delta']['aupro_0_30']*100:+.4f}个百分点；正常图像素误报率变化{row['delta']['normal_pixel_fpr']*100:+.4f}个百分点。")
    text += ["", f"最大类别额外描述计算及校准应用时间约为同一轮GPU基线处理时间的{overhead*100:.3f}%；校准参数最大{parameter_bytes}字节。不是整机进程内存差，独立资源补测另列。",
        cfg["test_notice"], cfg["novelty_notice"], "", "来源：DINOv2 https://github.com/facebookresearch/dinov2；SubspaceAD https://github.com/CLendering/SubspaceAD；内容分组使用正常数据上的既有聚类思路，相关正常校准和原型校准已有研究。本实验不证明方法原创性或期刊录用资格。"]
    report = "\n".join(text) + "\n"
    (run / "实验报告.txt").write_text(report, encoding="utf-8")
    columns = ("dataset", "variant", *METRICS)
    table = "<table><tr>" + "".join(f"<th>{k}</th>" for k in columns) + "</tr>"
    for row in macros:
        table += "<tr>" + "".join(f"<td>{html.escape(row[k]) if isinstance(row[k], str) else format(100*row[k], '.4f')}</td>" for k in columns) + "</tr>"
    table += "</table>"
    (run / "report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>内容条件校准实验</title><style>body{font:16px/1.7 sans-serif;margin:40px}pre{white-space:pre-wrap}table{border-collapse:collapse;font-size:13px}th,td{padding:6px;border:1px solid #ccc}</style><h1>内容条件校准实验</h1><pre>' + html.escape(report) + '</pre><p>下表均为百分比。</p>' + table + '<p><a href="category_metrics.csv">所有类别与下降项</a> · <a href="decision.json">预定要求判断</a></p></html>', encoding="utf-8")
    return decision
