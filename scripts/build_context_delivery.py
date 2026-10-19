"""Create one review folder with code, full offline inputs and honest results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_score_calibration import read, lines, sha, write


def main():
    import hashlib
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from industrial_anomaly_lab.context_calibration import nearest_centers
    from run_context_calibration import load_model
    import importlib.util
    spec = importlib.util.spec_from_file_location("_portable_replay", ROOT / "scripts/portable_context_verify.py")
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not run.is_relative_to((ROOT / "runs/quick").resolve()) or not output.is_relative_to((ROOT / "artifacts/deliveries").resolve()) or output.exists():
        raise ValueError("Use an existing project run and a fresh delivery directory")
    if not read(run / "completion_audit.json")["goal_achieved"]:
        raise ValueError("Final goal audit must pass before delivery")
    for rel, expected in read(run / "artifact_manifest.json")["sha256"].items():
        if sha(run / rel) != expected:
            raise ValueError("Source experiment artifact changed")
    frozen = read(run / "frozen_protocol.json")
    cfg = frozen["protocol"]
    original, normal = ROOT / cfg["source_run"], ROOT / cfg["normal_context_run"]
    output.mkdir(parents=True)
    for filename in ("report.html", "实验报告.txt", "category_metrics.csv", "dataset_macro_metrics.csv", "category_metrics.json", "dataset_macro_metrics.json", "decision.json", "completion_audit.json", "audit_receipt.json", "fresh_image_inference.json", "isolated_resource_summary.json", "frozen_protocol.json"):
        shutil.copy2(run / filename, output / filename)
    shutil.copytree(run / "visualizations", output / "visualizations")
    shutil.copytree(run / "code", output / "code")
    shutil.copytree(run / "public_sources", output / "public_sources")
    index = []
    for d, c in frozen["tasks"]:
        source_cat = run / "categories" / d / c
        target = output / "data" / d / c
        target.mkdir(parents=True)
        shutil.copy2(source_cat / "calibration.npz", target / "calibration.npz")
        shutil.copy2(normal / "categories" / d / c / "normal_context.npz", target / "normal_context.npz")
        shutil.copy2(normal / "categories" / d / c / "normal_records.json", target / "normal_records.json")
        for model in ("current_cpu_one_pass.npz", "official_cpu_two_pass.npz"):
            shutil.copy2(original / "categories" / d / c / "models" / model, target / model)
        samples = lines(original / "categories" / d / c / "samples.jsonl")
        frames = read(source_cat / "frames.json")
        model = load_model(source_cat / "calibration.npz")
        (target / "frames").mkdir()
        (target / "masks").mkdir()
        for j, (sample, frame) in enumerate(zip(samples, frames, strict=True)):
            copied = target / "frames" / f"{j:06d}.npz"
            if sha(source_cat / frame["cache"]) != frame["cache_sha256"]:
                raise ValueError("Original frame mismatch")
            with np.load(source_cat / frame["cache"], allow_pickle=False) as a:
                raw, descriptor = a["residual_grid"], a["descriptor"]
            cluster_labels = nearest_centers(descriptor, model.centers).reshape(32, 32)
            np.savez_compressed(copied, residual_grid=raw, cluster_labels=cluster_labels)
            map_hashes = {}
            for variant in cfg["variants"]:
                actual = model.apply(raw, descriptor, variant)
                packed = replay_module.replay_map(model, raw, cluster_labels, variant)
                np.testing.assert_array_equal(actual, packed)
                map_hashes[variant] = hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()
            original_mask = original / sample["mask_path"]
            mask = np.load(original_mask, allow_pickle=False)
            mask_file = target / "masks" / f"{j:06d}.npz"
            np.savez_compressed(mask_file, mask=mask)
            index.append({"dataset": d, "category": c, "sample_id": sample["sample_id"], "label": sample["label"],
                "frame": copied.relative_to(output).as_posix(), "mask": mask_file.relative_to(output).as_posix(),
                "source_full_frame_sha256": frame["cache_sha256"], "map_content_sha256": map_hashes,
                "original_image_sha256": sample["image_sha256"], "original_mask_file_sha256": sample["mask_sha256"]})
    write(output / "sample_index.json", index)
    write(output / "config.json", cfg)
    shutil.copy2(ROOT / "scripts/portable_context_verify.py", output / "复算指标.py")
    shutil.copy2(ROOT / "pyproject.toml", output / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", output / "uv.lock")
    (output / "requirements-offline.txt").write_text("numpy==1.26.4\nscipy==1.15.2\nscikit-learn==1.6.1\nopencv-python==4.11.0.86\npillow==11.1.0\n", encoding="utf-8")
    metrics = read(run / "dataset_macro_metrics.json")
    text = ["这轮实验做了什么", "", "这次保留原来的DINOv2-S和SubspaceAD框架，主要处理正常结构变化造成的评分差异。前面两种按位置校准的方案没有达到要求，失败结果仍保留在项目里。", "",
        "最终采用正常内容分组：用少量正常图建立8个内容原型，分别估计误差上界；低于正常上界的评分适度压低，明显超过上界的评分保留。实际组合使用已经拟合的单遍PCA模型，并对齐PCA坐标的任意符号，避免影响已有内容原型。", ""]
    for d in ("mvtec_ad", "visa"):
        get = lambda v: next(r for r in metrics if (r["dataset"], r["variant"]) == (d, v))
        base, candidate, removed = get("baseline"), get("conditional_tail"), get("conditioning_removed")
        text.append(f"{d}：像素AP由{base['pixel_ap']*100:.4f}%到{candidate['pixel_ap']*100:.4f}%；图像AUROC由{base['image_auroc']*100:.4f}%到{candidate['image_auroc']*100:.4f}%。去掉内容条件后的像素AP是{removed['pixel_ap']*100:.4f}%。")
    text += ["", "通过的依据是两个数据集像素AP稳定提高，图像检测没有退步，额外计算较小。不能说所有指标都提高，也不能说误报率一致降低：VisA的AUPRO有回退，普通固定阈值下的误报指标也未达到一致收益。", "",
        "消融说明：正常尾部抑制本身贡献了相当一部分收益；内容分组在MVTec AD上增加的收益更明确，在VisA上的平均增益很小。报告没有把全部提升归给内容分组。", "",
        "单遍PCA的拟合时间和内存收益来自此前独立对照，固定64张正常图子集上的提取加拟合时间减少约38%和47%。不能直接当成全数据集或新评分组件的加速比例。", "",
        "本文件夹包含本轮代码快照、两数据集完整指标、54组固定对比图、实际残差与已验证的内容分组缓存、原始标注以及独立复算脚本。为减小文件夹，测试特征描述留在原始实验目录；打包时已逐图验证分组缓存重建的四组原始异常图完全一致，并保存各图内容摘要。无需GPU即可离线核对全部指标；从原始图片重新跑完整模型仍需公开数据和DINO权重。", "",
        "基础方法来自DINOv2、SubspaceAD及已有聚类/分位数校准技术。这里的贡献是评分适配、单遍组合、完整配对实验和复现整理，不把已有方法称为原创。两个测试集此前已访问，且PCA见过校准正常图，所以不声称新盲测或严格独立校准。"]
    (output / "项目说明.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    (output / "复现说明.txt").write_text(f'''如何核对这个文件夹

一、离线复算全部指标，不需要GPU、原图或DINO权重。
使用Python3.11，安装requirements-offline.txt中的依赖，然后运行：
python 复算指标.py
会核对文件摘要、正常校准参数、阈值、操作点与全部27类四组指标，生成复算结果.json。普通全局正仿射校准的指标根据严格排序不变性复用重新算出的基线指标，但每张全局图与操作点仍实际重建核对。预计需要数分钟和临时磁盘空间；复算器保留原实验的8 GiB磁盘余量要求。

在当前机器也可使用：
E:\\3k\\.venv\\Scripts\\python.exe "{output / '复算指标.py'}"

二、从原图完整重跑。
已执行项目在E:\\3k。按公开来源下载MVTec AD、VisA及固定DINOv2-S权重，保持项目目录结构和pyproject.toml/uv.lock环境。代码快照位于code，完整前向脚本是scripts/run_context_single_pass.py，独立审核脚本是scripts/audit_context_single_pass.py。项目配置固定在code/configs/context_calibration_single_pass_20261015.json。它还依赖此前route1-controls及normal-context-pilot输入；这些流程的源码和说明保留在项目里。本文件夹提供对应PCA参数、正常内容缓存和实际测试残差，支持独立离线复算；完整原图重跑不能只复制一个脚本就运行。

本轮执行记录：{run}
此前单遍PCA对照：E:\\3k\\artifacts\\deliveries\\quick-padim-20261012\\交付文件夹\\优化实验\\同骨干Subspace对照-20261014

指标与可视化共享本组实际原始评分。图像AUROC使用本组输出图最大值；像素指标在256平面计算，AUPRO积分上限0.30，按类别等权平均。可视化只换算显示颜色，不进入数值评测。

本文件夹不是新的训练结果：它是已完成真实实验的代码、输入缓存和证据整理。复算指标与原图重新推理是两种不同核查，报告中分别提供记录。
''', encoding="utf-8")
    (output / "public_sources/数据来源.txt").write_text("MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad\nVisA: https://github.com/amazon-science/spot-diff\nData and ground-truth annotations are included only as research experiment evidence. Original dataset terms and attribution apply.\n", encoding="utf-8")
    write(output / "manifest.json", {"tasks": frozen["tasks"], "source_run": str(run),
        "source_artifact_manifest_sha256": sha(run / "artifact_manifest.json"),
        "sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    print(f"Built delivery folder: {output}", flush=True)


if __name__ == "__main__":
    main()
