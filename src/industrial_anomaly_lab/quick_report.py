"""Portable reports for completed quick feature-distribution experiments.

Reports consume existing metrics and anomaly maps.  They do not fit a model,
recalculate metrics, select a best variant, or assert that a modification won.
"""

from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
import math
import os
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
IDENTITY = ("dataset", "category", "variant", "seed")
PAPER = {
    "title": "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization",
    "authors": "Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier",
    "year": 2020,
    "url": "https://arxiv.org/abs/2011.08785",
    "relationship": "Feature-distribution methodological foundation; this quick run is not asserted to be an author-exact PaDiM reproduction.",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL object required: {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _inside(root: Path, value: str) -> Path:
    """Resolve a declared run artifact without accepting traversal or symlinks."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Run artifact must use a safe relative path: {value}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-run artifact: {value}")
    return path


def _source_path(root: Path, value: str, expected_sha: str | None = None) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None and _sha(path) != expected_sha:
        raise ValueError(f"Source SHA-256 mismatch: {value}")
    return path


def _dataset_sources(project: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    """Retain dataset attribution and terms when redistributing selected pixels."""
    declared = config.get("protocol", {}).get("datasets", {})
    records, files = {}, {}
    for dataset in declared:
        if dataset == "mvtec_ad":
            source = _source_path(project, "manifests/mvtec_source_manifest.json")
            metadata = _json(source)
            license_file = _source_path(project, "data/mvtec_ad/license.txt")
            attribution = _source_path(project, "data/mvtec_ad/readme.txt")
            if "Attribution-NonCommercial-ShareAlike 4.0" not in license_file.read_text(encoding="utf-8"):
                raise ValueError("MVTec dataset license differs from the recorded CC-BY-NC-SA-4.0 terms")
            retained = {"data_sources/MVTec-AD-source.json": source, "licenses/MVTec-AD-LICENSE.txt": license_file, "licenses/MVTec-AD-ATTRIBUTION.txt": attribution}
            records[dataset] = {"name": "MVTec AD", "official_url": metadata["official_dataset_page"], "license": "CC-BY-NC-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/", "copyright": "Copyright 2019 MVTec Software GmbH", "attribution": "Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR 2019.", "source_manifest": "data_sources/MVTec-AD-source.json", "license_file": "licenses/MVTec-AD-LICENSE.txt", "attribution_file": "licenses/MVTec-AD-ATTRIBUTION.txt", "commercial_use_allowed": False}
        elif dataset == "visa":
            source = _source_path(project, "manifests/visa_source_manifest.json")
            metadata = _json(source)
            license_file = _source_path(project, "data/visa/raw/VisA/LICENSE-DATASET")
            notice = _source_path(project, "third_party/spot-diff/NOTICE", metadata["licenses"]["notice"]["sha256"])
            code_license = _source_path(project, "third_party/spot-diff/LICENSE", metadata["licenses"]["code"]["sha256"])
            readme = _source_path(project, "third_party/spot-diff/README.md")
            split = _source_path(project, metadata["official_one_class_protocol"]["split_file"], metadata["official_one_class_protocol"]["split_sha256"])
            preparation = _source_path(project, metadata["official_one_class_protocol"]["preparation_script"], metadata["official_one_class_protocol"]["preparation_script_sha256"])
            if "Attribution 4.0 International" not in license_file.read_text(encoding="utf-8"):
                raise ValueError("VisA dataset license differs from the recorded CC-BY-4.0 terms")
            retained = {"data_sources/VisA-source.json": source, "assets/data-protocols/visa_1cls.csv": split, "licenses/VisA-DATASET-LICENSE.txt": license_file, "licenses/VisA-NOTICE.txt": notice, "licenses/VisA-preparation-Apache2.txt": code_license, "data_sources/VisA-official-README.md": readme, "assets/data-protocols/visa_prepare_data.py": preparation}
            records[dataset] = {"name": "VisA", "official_url": metadata["official_repository"], "dataset_download_url": metadata["dataset_download_url"], "source_commit": metadata["commit"], "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "copyright": "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.", "attribution": "Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer. SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation. ECCV 2022. https://arxiv.org/abs/2207.14315", "source_manifest": "data_sources/VisA-source.json", "license_file": "licenses/VisA-DATASET-LICENSE.txt", "notice_file": "licenses/VisA-NOTICE.txt", "split_file": "assets/data-protocols/visa_1cls.csv", "split_sha256": _sha(split), "preparation_script": "assets/data-protocols/visa_prepare_data.py", "preparation_script_sha256": _sha(preparation)}
        else:
            raise ValueError(f"Dataset provenance is not configured: {dataset}")
        files.update(retained)
        records[dataset]["original_selected_images"] = "Exact source file bytes copied without modification"
        records[dataset]["derived_visualizations"] = "RGB/bilinear-resized input display, nearest-resized binary evaluation masks, heatmap overlays and comparison panels; dataset attribution and terms retained for these derived image contents"
        records[dataset]["retained_files"] = {name: {"sha256": _sha(path), "size_bytes": path.stat().st_size} for name, path in retained.items()}
    return records, files


def _expected_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    expected = config.get("expected_results")
    if isinstance(expected, list) and expected:
        result = {tuple(item[field] for field in IDENTITY) for item in expected}
        if len(result) != len(expected):
            raise ValueError("Duplicate expected result identities")
        return result
    matrix = config.get("expected_matrix")
    if isinstance(matrix, dict):
        datasets = matrix.get("datasets")
        variants, seeds = matrix.get("variants"), matrix.get("seeds")
        if isinstance(datasets, dict) and datasets and variants and seeds:
            return {
                (dataset, category, variant, seed)
                for dataset, categories in datasets.items()
                for category, variant, seed in itertools.product(categories, variants, seeds)
            }
    raise ValueError("Expected result matrix is required to verify completion")


def _completed_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = _json(root / "run_config.json")
    state = _json(root / "run_state.json")
    if state.get("status") not in {"complete", "completed"}:
        raise ValueError(f"Run is incomplete: {state.get('status', 'missing status')}")
    rows = _jsonl(root / "results" / "category_metrics.jsonl")
    if not rows:
        raise ValueError("No completed category results")
    seen = set()
    for row in rows:
        if row.get("status") not in {"complete", "completed"}:
            raise ValueError("Incomplete category metric row")
        key = tuple(row[field] for field in IDENTITY)
        if key in seen:
            raise ValueError(f"Duplicate category result: {key}")
        if not all(isinstance(row[field], str) and row[field] for field in IDENTITY[:3]):
            raise ValueError(f"Invalid category identity: {key}")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError("Integer seed required")
        seen.add(key)
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"Numeric completed metric required: {metric}")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Metric must be finite in [0,1]: {metric}")
    expected = _expected_keys(config)
    if seen != expected:
        raise ValueError(f"Incomplete result coverage: missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}")
    return config, state, sorted(rows, key=lambda row: tuple(row[field] for field in IDENTITY))


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tables(rows: list[dict[str, Any]], report: Path) -> list[dict[str, Any]]:
    _csv(report / "category_metrics.csv", rows, list(IDENTITY) + ["status"] + list(METRICS))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["variant"], row["seed"])].append(row)
    macro = [
        {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "category_count": len(values),
            **{metric: sum(row[metric] for row in values) / len(values) for metric in METRICS},
        }
        for (dataset, variant, seed), values in sorted(groups.items())
    ]
    _csv(report / "macro_metrics.csv", macro, ["dataset", "variant", "seed", "category_count"] + list(METRICS))
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[(row["dataset"], row["category"], row["seed"])][row["variant"]] = row
    comparison = []
    for (dataset, category, seed), variants in sorted(pairs.items()):
        if "baseline" not in variants:
            raise ValueError("A baseline row is required for each comparison category")
        for variant, candidate in sorted(variants.items()):
            if variant == "baseline":
                continue
            base = variants["baseline"]
            comparison.append({
                "dataset": dataset, "category": category, "seed": seed, "variant": variant,
                **{f"baseline_{metric}": base[metric] for metric in METRICS},
                **{f"{metric}": candidate[metric] for metric in METRICS},
                **{f"delta_{metric}": candidate[metric] - base[metric] for metric in METRICS},
            })
    fields = ["dataset", "category", "seed", "variant"]
    fields += [f"baseline_{metric}" for metric in METRICS] + list(METRICS) + [f"delta_{metric}" for metric in METRICS]
    _csv(report / "metric_comparison.csv", comparison, fields)
    _write_json(report / "macro_metrics.json", {"aggregation": "equal_category_weight_per_dataset_variant_seed", "rows": macro})
    return macro


def _selected_samples(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    samples = _jsonl(_inside(root, str(config.get("samples_index", "results/samples.jsonl"))))
    if not samples:
        raise ValueError("No prediction samples available for visual evidence")
    selected = [sample for sample in samples if sample.get("selected") is True]
    if not selected:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            groups[(sample["dataset"], sample["category"], int(sample["label"]))].append(sample)
        selected = [sorted(group, key=lambda sample: str(sample["sample_id"]))[0] for _, group in sorted(groups.items())]
    keys = [(sample["dataset"], sample["category"], sample["sample_id"], sample.get("seed")) for sample in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate selected prediction samples")
    for sample in selected:
        if int(sample["label"]) not in {0, 1}:
            raise ValueError("Binary sample label required")
        maps = sample.get("raw_maps")
        if not isinstance(maps, dict) or "baseline" not in maps or len(maps) < 2:
            raise ValueError("Selected sample must contain baseline and comparison raw maps")
    return sorted(selected, key=lambda sample: (sample["dataset"], sample["category"], str(sample["sample_id"])))


def _render(root: Path, project: Path, config: Mapping[str, Any], report: Path) -> list[dict[str, Any]]:
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image, ImageDraw, ImageFont

    selected = _selected_samples(root, config)
    arrays: dict[tuple[int, str], Any] = {}
    pooled: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for index, sample in enumerate(selected):
        shape = None
        for variant, path_text in sample["raw_maps"].items():
            path = _inside(root, path_text)
            expected = sample.get("raw_map_sha256", {}).get(variant)
            if expected is not None and _sha(path) != expected:
                raise ValueError(f"Raw map SHA-256 mismatch: {path_text}")
            array = np.load(path, allow_pickle=False)
            if not isinstance(array, np.ndarray) or array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError(f"Finite 2D raw map required: {path_text}")
            if shape is not None and array.shape != shape:
                raise ValueError("Comparison raw map shapes differ")
            shape = array.shape
            arrays[(index, variant)] = array
            pooled[(sample["dataset"], sample["category"])].append(array.ravel())
    scales = {}
    for key, values in pooled.items():
        lower, upper = np.quantile(np.concatenate(values), [0.01, 0.99]).tolist()
        if upper <= lower:
            upper = lower + max(abs(lower) * 1e-6, 1e-6)
        scales[key] = (lower, upper)
    evidence = report / "visual_evidence"
    evidence.mkdir()
    records = []
    font = ImageFont.load_default(size=15)
    for index, sample in enumerate(selected):
        variants = sorted(sample["raw_maps"], key=lambda item: (item != "baseline", item))
        height, width = arrays[(index, variants[0])].shape
        image_path = _source_path(project, sample["image_path"], sample.get("image_sha256"))
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        mask_text = sample.get("mask_path")
        if mask_text:
            mask_path = _source_path(root, mask_text, sample.get("mask_sha256"))
            if mask_path.suffix.lower() == ".npy":
                mask_values = np.load(mask_path, allow_pickle=False)
                if mask_values.ndim != 2 or not np.isfinite(mask_values).all():
                    raise ValueError("Finite 2D evaluation mask required")
                mask = Image.fromarray(((mask_values > 0) * 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
            else:
                with Image.open(mask_path) as source:
                    mask = source.convert("L").resize((width, height), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask) > 0
            mask = Image.fromarray((mask_array * 255).astype(np.uint8)).convert("RGB")
        else:
            if int(sample["label"]) == 1:
                raise ValueError("Anomaly visualization requires a ground-truth mask")
            mask_path = None
            mask = Image.new("RGB", (width, height), "black")
        slug = f"{index:03d}-" + hashlib.sha256(f"{sample['dataset']}/{sample['category']}/{sample['sample_id']}".encode()).hexdigest()[:12]
        folder = evidence / slug
        folder.mkdir()
        image.save(folder / "input.png")
        mask.save(folder / "ground_truth.png")
        shutil.copyfile(image_path, folder / ("source_image" + image_path.suffix.lower()))
        if mask_path:
            shutil.copyfile(mask_path, folder / ("evaluation_mask" + mask_path.suffix.lower()))
        lower, upper = scales[(sample["dataset"], sample["category"])]
        panels = [("Input", image), ("Ground truth", mask)]
        copied_maps = {}
        for variant in variants:
            array = arrays[(index, variant)]
            normalized = np.clip((array - lower) / (upper - lower), 0, 1)
            heat = Image.fromarray((colormaps["viridis"](normalized)[..., :3] * 255).astype(np.uint8))
            overlay = Image.blend(image, heat, 0.55)
            panels.append((variant, overlay))
            map_name = hashlib.sha256(variant.encode()).hexdigest()[:10] + ".npy"
            shutil.copyfile(_inside(root, sample["raw_maps"][variant]), folder / map_name)
            copied_maps[variant] = {"path": (folder / map_name).relative_to(report).as_posix(), "sha256": _sha(folder / map_name)}
        title_height, footer_height, gap = 48, 30, 8
        canvas = Image.new("RGB", ((width + gap) * len(panels) + gap, height + title_height + footer_height), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (title, panel) in enumerate(panels):
            left = gap + column * (width + gap)
            draw.text((left, 8), title, font=font, fill="black")
            canvas.paste(panel, (left, title_height))
        draw.text((gap, height + title_height + 6), f"Shared category scale: {lower:.6g} to {upper:.6g} (pooled p1/p99)", font=font, fill="black")
        figure = report / f"comparison-{slug}.png"
        canvas.save(figure)
        records.append({
            "dataset": sample["dataset"], "category": sample["category"], "sample_id": sample["sample_id"],
            "label": int(sample["label"]), "seed": sample.get("seed"),
            "figure": figure.relative_to(report).as_posix(), "figure_sha256": _sha(figure),
            "input": (folder / "input.png").relative_to(report).as_posix(), "source_image_path": str(image_path), "source_image_sha256": _sha(image_path),
            "source_image_copy": (folder / ("source_image" + image_path.suffix.lower())).relative_to(report).as_posix(),
            "ground_truth": (folder / "ground_truth.png").relative_to(report).as_posix(), "source_mask_path": str(mask_path) if mask_path else None, "source_mask_sha256": _sha(mask_path) if mask_path else None,
            "evaluation_mask_copy": (folder / ("evaluation_mask" + mask_path.suffix.lower())).relative_to(report).as_posix() if mask_path else None,
            "raw_maps": copied_maps,
            "scale": {"lower": lower, "upper": upper, "quantiles": [0.01, 0.99], "scope": "pooled_selected_variants_per_category", "colormap": "viridis", "overlay_alpha": 0.55},
            "sample_selection": "explicit_manifest" if sample.get("selected") is True else "lexicographic_first_per_category_label",
            "display_only_normalization": True,
        })
    _write_json(report / "visualization_manifest.json", {"schema_version": 1, "samples": records, "metrics_recomputed": False, "raw_maps_modified": False})
    return records


def _html_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    cells = []
    for row in rows:
        values = [f"{row[field]:.6f}" if isinstance(row.get(field), float) else str(row.get(field, "")) for field in fields]
        cells.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in fields) + "</tr></thead><tbody>" + "".join(cells) + "</tbody></table>"


def build_report(run_root: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Build a new report after verifying complete registered result coverage."""
    root = Path(run_root).resolve()
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    config, state, rows = _completed_inputs(root)
    dataset_sources, _ = _dataset_sources(project, config)
    report = root / "report"
    if report.exists():
        raise FileExistsError(f"Report already exists; preserve it and choose a new run: {report}")
    report.mkdir()
    macro = _tables(rows, report)
    figures = _render(root, project, config, report)
    macro_pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in macro:
        macro_pairs[(row["dataset"], row["seed"])][row["variant"]] = row
    effects = []
    for (dataset, seed), variants in sorted(macro_pairs.items()):
        if "baseline" in variants and "shrinkage" in variants:
            deltas = {metric: variants["shrinkage"][metric] - variants["baseline"][metric] for metric in METRICS}
            effects.append({"dataset": dataset, "seed": seed, "delta": deltas, "all_primary_metrics_increased": all(value > 0 for value in deltas.values()), "confirmation_status": "single_seed_observation_requires_confirmation"})
    primary_deltas = [delta for effect in effects for delta in effect["delta"].values()]
    if primary_deltas and all(delta < 0 for delta in primary_deltas):
        observed_outcome = "all_primary_dataset_macro_metrics_decreased"
        outcome_text = "本轮固定协方差收缩变体在两个数据集的四项宏平均指标均低于基线，没有实现整体提升。逐类结果和下降项均保留。"
    elif primary_deltas and all(delta > 0 for delta in primary_deltas):
        observed_outcome = "all_primary_dataset_macro_metrics_increased_in_this_run"
        outcome_text = "本轮实测中收缩变体的主指标宏平均上升；重复实验稳定性与统计显著性尚未验证。"
    else:
        observed_outcome = "mixed_or_unchanged_primary_dataset_macro_metrics"
        outcome_text = "本轮各指标存在提升、下降或持平，不能作一致提升结论。全部差值保存在对比表中。"
    seed_count = len({row["seed"] for row in rows})
    seed_text = f"本轮完成 {seed_count} 个种子。" + ("单个种子结果不能用于确认重复实验稳定性或统计显著性。" if seed_count == 1 else "科研结论还需独立确认。")
    _write_json(report / "effect_summary.json", {"comparison": "shrinkage_minus_baseline", "dataset_effects": effects, "observed_outcome": observed_outcome, "seed_count": seed_count, "scientific_improvement_confirmed": False, "selection_of_best_variant_performed": False})
    provenance = {
        "schema_version": 1,
        "project_generated_components": ["quick experiment integration", "local implementation", "local results", "comparison report and delivery packaging"],
        "original_method_claimed": False,
        "paper": PAPER,
        "torchvision": {"version": "0.22.0", "license": "BSD-3-Clause", "source": "https://github.com/pytorch/vision/tree/v0.22.0"},
        "backbone": config.get("backbone", {}),
        "run_sources": config.get("source_provenance", config.get("sources", {})),
        "execution_source_sha256": config.get("source_sha256", {}),
        "initial_execution": config.get("initial_execution"),
        "resumed_executions": config.get("resumed_executions", []),
        "protocol_sha256": config.get("protocol_sha256"),
        "protocol": config.get("protocol", {}),
        "datasets": dataset_sources,
        "run_config_sha256": _sha(root / "run_config.json"),
        "category_metrics_sha256": _sha(root / "results" / "category_metrics.jsonl"),
        "scope": config.get("claim_scope", "completed quick experiment; no publication acceptance or author-exact reproduction claim"),
    }
    _write_json(report / "source_provenance.json", provenance)
    resources = {"recorded_run_elapsed_seconds": state.get("elapsed_seconds"), "configured_resources": config.get("protocol", {}).get("resources", {}), "measured_summary": None}
    if (root / "resource_summary.json").is_file():
        resources["measured_summary"] = _json(root / "resource_summary.json")
        resources["measured_summary_sha256"] = _sha(root / "resource_summary.json")
    if (root / "resource_samples.jsonl").is_file():
        resources["resource_samples_sha256"] = _sha(root / "resource_samples.jsonl")
    _write_json(report / "resource_evidence.json", resources)
    command = config.get("reproduction_command", "python scripts/run_quick_experiment.py --config configs/quick_reproduction_20261012.json --output-dir runs/quick/reproduction")
    instructions = """# 快速实验复现与核验\n\n本包保存本轮实际完成的代码、模型统计、指标表和可视化证据。它采用公开的预训练骨干和 PaDiM 特征分布思路；项目集成与本轮运行结果由本项目产生，原始方法、骨干权重和数据来源保留在 source_provenance.json。\n\n## 环境与输入\n\n1. 使用 Python 3.11，根据 code/pyproject.toml 与 code/uv.lock 安装固定版本依赖。\n2. 将 MVTec AD、VisA 官方数据置于配置指定目录，相对路径以 code/ 为根；完整数据集不随交付包分发。\n3. 将 assets/weights/ 中的骨干权重复制到 code/ 下配置指定的 weights 路径，并核对 SHA-256；拟合模型统计保存在 models/。\n4. 从包内 code/ 目录运行下面的命令。原机器绝对路径需要改成复现机器路径，其余配置和种子保留。输出目录必须是新的空目录。\n\n```text\n""" + str(command) + "\n```\n\n## 结果口径\n\ncategory_metrics.csv 为已有实际结果的逐类原值；macro_metrics.csv 按同一数据集、变体、种子内各类别等权平均。metric_comparison.csv 的差值按变体减 baseline 计算，不筛除下降项。指标均在 [0,1] 范围。\n\n可视化只改变显示色阶，所有对照变体共享同类别合并原始异常图的 1% 与 99% 分位色阶。选样按显式清单，或每类别每标签字典序第一项；原始异常图与其哈希随精选证据保存。显示归一化不参与指标计算。\n\n## 核验\n\n打开 report.html 查看表格和图片。ZIP 内 DELIVERY-MANIFEST.json 列出每一项的字节数与 SHA-256；外部 ZIP 回执保存压缩包哈希和逐文件验证结果。\n\n该快速实验不保证论文录用；创新性、重复种子确认、更多对照与完整消融需要根据实际结果继续验证。预训练权重与数据使用条款以各来源条款为准。\n"
    instructions += "\n## 独立抽样推理验收\n\n解压后进入包内 code/ 目录执行：\n\n```text\npython scripts/verify_quick_delivery.py --package-root .. --inference-samples 2\n```\n\n先逐文件流式核对 DELIVERY-MANIFEST.json 中的字节数和 SHA-256，再用随包骨干权重与已拟合统计重算前两个精选样本的 baseline/shrinkage 原始异常图，按 atol=1e-5、rtol=1e-6 与保存值逐像素比较。该步骤无需完整数据集、联网或重新拟合。将 --inference-samples 改为 0 可仅执行文件完整性检查。\n"
    instructions += "\nWindows 也可从解压根目录运行 Verify.ps1。该入口寻找已有 Python 3.11，不安装软件；如果没有可用环境，会给出可复用的 -PythonPath 参数。原机器示例：\n\n```powershell\npowershell -NoProfile -ExecutionPolicy Bypass -File .\\Verify.ps1 -PythonPath E:\\3k\\.venv\\Scripts\\python.exe\n```\n\n所有初始运行、资源中断和恢复记录均保留在 run_config.json 的 initial_execution/resumed_executions、execution_history/ 与 results/resume_audit.json 中。resource_evidence.json 展示实测资源摘要；历史中断不作为完成结果。\n"
    variants = config.get("protocol", {}).get("variants", {})
    variant_rows = [{"variant": name, "shrinkage_alpha": settings["shrinkage_alpha"], "sigma": settings["sigma"]} for name, settings in variants.items()]
    variant_md = "\n## 四个变体与消融\n\n各变体共享正常训练数据、骨干权重、抽取通道与已拟合均值/协方差；仅改变固定收缩系数与平滑参数。\n\n| 变体 | 收缩系数 alpha | 224 像素平面平滑 sigma |\n|---|---:|---:|\n" + "".join(f"| {item['variant']} | {item['shrinkage_alpha']} | {item['sigma']} |\n" for item in variant_rows)
    instructions = instructions.replace("## 环境与输入", "## 本轮实际结论\n\n" + outcome_text + "\n\n" + seed_text + "\n\n## 环境与输入")
    instructions += variant_md
    if dataset_sources:
        instructions += "\n## 数据来源与精选图片许可\n\nMVTec AD：MVTec Software GmbH，CC BY-NC-SA 4.0；VisA：Amazon.com, Inc. 或其关联公司，CC BY 4.0。许可证、版权与论文信息位于包根目录 licenses/、data_sources/ 以及本报告 source_provenance.json。随包 source_image 文件逐字节保留原图；input.png、二值 GT 显示、异常热图叠加和对比图是明确标记的派生显示内容，并保留对应数据许可。\n\nMVTec AD 完整数据按官方页面获取并解压到 code/data/mvtec_ad。VisA 完整原始数据按 data_sources/VisA-source.json 的官方 URL 获取，解压到 code/data/visa/raw/VisA；从 code/ 目录使用随包官方脚本和固定 1cls 划分准备：\n\n```text\npython ../assets/data-protocols/visa_prepare_data.py --split-type 1cls --data-folder data/visa/raw/VisA --save-folder data/visa --split-file ../assets/data-protocols/visa_1cls.csv\n```\n\nVisA CSV 的 SHA-256 固定为 " + str(dataset_sources.get("visa", {}).get("split_sha256", "not applicable")) + "。该命令生成配置使用的 data/visa/1cls 目录，保留官方正常训练/测试分配；不会使用测试异常图拟合模型。\n"
    (report / "复现说明.md").write_text(instructions, encoding="utf-8")
    backbone_name = str(config.get("backbone", {}).get("architecture", "configuration-defined backbone"))
    source_md = "# 来源与产出边界\n\n本项目产生本轮集成代码、运行结果、报告和打包流程。\n\n方法基础：" + PAPER["authors"] + ". " + PAPER["title"] + f" ({PAPER['year']})，{PAPER['url']}。\n\n骨干与预训练权重：torchvision " + backbone_name + " ImageNet，https://github.com/pytorch/vision/tree/v0.22.0 。BSD 3-Clause 许可证保存在 licenses/torchvision-LICENSE.txt。\n\n本轮具体模型配置、源码哈希和权重信息见 source_provenance.json 与 run_config.json。本报告没有把公开方法或公开权重声明为本项目原创。\n"
    for dataset, record in dataset_sources.items():
        source_md += "\n## " + record["name"] + "\n\n" + record["copyright"] + "\n\n" + record["attribution"] + "\n\n官方来源：" + record["official_url"] + "\n\n数据许可：" + record["license"] + "，" + record["license_url"] + "。完整原文：包根目录 " + record["license_file"] + "。\n\n原始精选图片复制未作修改；RGB 缩放显示、二值掩码显示、热图叠加和对比图是派生显示内容，保留本数据许可。来源清单及其哈希见 source_provenance.json 的 datasets 字段。\n"
    (report / "来源说明.md").write_text(source_md, encoding="utf-8")
    sections = "".join(f"<figure><figcaption>{html.escape(str(item['dataset']) + '/' + str(item['category']) + '/' + str(item['sample_id']))}</figcaption><img src='{html.escape(item['figure'], quote=True)}' alt='Anomaly map comparison'></figure>" for item in figures)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>工业异常检测快速实验</title><style>body{font:16px system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;background:#fafafa;color:#222}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccc;padding:8px;text-align:right}th{background:#eee}img{max-width:100%;height:auto}figure{margin:30px 0}a{color:#0754a0}pre{white-space:pre-wrap}</style><h1>工业异常检测快速实验</h1><p>登记结果矩阵已完整完成。表格保存实际指标；差值和下降项均保留。可视化使用同类别共享色阶。</p>"
    page += "<p>" + html.escape(outcome_text) + "</p><p>" + html.escape(seed_text) + "</p><h2>变体与消融定义</h2>" + _html_table(variant_rows, ["variant", "shrinkage_alpha", "sigma"])
    page += "<p><a href='category_metrics.csv'>逐类指标</a> · <a href='metric_comparison.csv'>对比表</a> · <a href='macro_metrics.csv'>宏平均表</a> · <a href='复现说明.md'>复现说明</a> · <a href='来源说明.md'>来源说明</a></p><h2>数据集宏平均</h2>" + _html_table(macro, ["dataset", "variant", "seed", "category_count"] + list(METRICS))
    page += "<h2>逐类结果</h2>" + _html_table(rows, list(IDENTITY) + list(METRICS)) + "<h2>可视化对比</h2>" + sections + "<p>这是一轮快速实验交付。科研创新性和论文录用结论需进一步验证。</p></html>"
    page = page.replace("</html>", "<details><summary>运行时间与资源记录</summary><pre>" + html.escape(json.dumps(resources, ensure_ascii=False, indent=2)) + "</pre></details></html>")
    (report / "report.html").write_text(page, encoding="utf-8")
    result = {"schema_version": 1, "status": "completed", "report_root": str(report), "result_count": len(rows), "figure_count": len(figures), "macro_rows": macro, "run_state": state, "project_root": str(project)}
    _write_json(report / "report_summary.json", result)
    return result


def _project_files(project: Path, config: Mapping[str, Any]) -> list[Path]:
    declared = config.get("source_files", config.get("source_sha256"))
    if isinstance(declared, dict):
        files = [_source_path(project, str(path), str(sha)) for path, sha in declared.items()]
    elif isinstance(declared, list) and declared:
        files = [_source_path(project, str(item["path"]), item.get("sha256")) if isinstance(item, dict) else _source_path(project, str(item)) for item in declared]
    else:
        files = [*project.glob("src/industrial_anomaly_lab/quick*.py"), *project.glob("scripts/*quick*.py")]
        files += [project / "src/industrial_anomaly_lab" / name for name in ("__init__.py", "metrics.py", "run_artifacts.py")]
    # The runner's dataset adapter and metric evaluator have these local imports.
    if config.get("source_sha256"):
        files += [project / "src/industrial_anomaly_lab" / name for name in ("__init__.py", "metrics.py", "data_protocols.py", "draem_runtime.py")]
    if config.get("protocol_sha256"):
        protocols = [path for path in project.glob("configs/quick*.json") if _sha(path) == config["protocol_sha256"]]
        if len(protocols) != 1:
            raise ValueError("Frozen quick protocol file is missing or changed")
        files += protocols
    for helper in ("Invoke-QuickExperiment.ps1", "Set-ProjectEnvironment.ps1", "audit_quick_run.py"):
        if (project / "scripts" / helper).is_file():
            files.append(project / "scripts" / helper)
    for test_name in ("test_quick_features.py", "test_quick_metrics.py", "test_quick_report.py"):
        if (project / "tests" / test_name).is_file():
            files.append(project / "tests" / test_name)
    files += [project / "pyproject.toml", project / "uv.lock", project / "scripts" / "package_quick_delivery.py", project / "scripts" / "verify_quick_delivery.py", project / "src" / "industrial_anomaly_lab" / "quick_report.py"]
    unique = sorted(set(path.resolve() for path in files))
    for path in unique:
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError(f"Missing or external source file: {path}")
    return unique


def package_delivery(run_root: str | Path, output_zip: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Archive completed output and verify all archived bytes after writing."""
    root = Path(run_root).resolve()
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    config, _, _ = _completed_inputs(root)
    output = Path(output_zip).resolve()
    if output.exists():
        raise FileExistsError(f"Archive exists; refusing overwrite: {output}")
    if not (root / "report" / "report_summary.json").is_file():
        build_report(root, project)
    report = root / "report"
    summary = _json(report / "report_summary.json")
    provenance = _json(report / "source_provenance.json")
    dataset_sources, dataset_files = _dataset_sources(project, config)
    if summary.get("status") != "completed" or provenance.get("run_config_sha256") != _sha(root / "run_config.json") or provenance.get("category_metrics_sha256") != _sha(root / "results" / "category_metrics.jsonl"):
        raise ValueError("Report no longer matches completed run inputs")
    if provenance.get("datasets", {}) != dataset_sources:
        raise ValueError("Report dataset attribution is missing or stale; regenerate the report")
    payload: dict[str, Path] = {"run_config.json": root / "run_config.json", "run_state.json": root / "run_state.json", "results/category_metrics.jsonl": root / "results" / "category_metrics.jsonl"}
    payload.update(dataset_files)
    if (project / "scripts" / "Verify-QuickDelivery.ps1").is_file():
        payload["Verify.ps1"] = project / "scripts" / "Verify-QuickDelivery.ps1"
    for path in sorted((root / "execution_history").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Execution history escapes run root")
            payload[path.relative_to(root).as_posix()] = path
    for path in sorted((root / "monitoring").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".log"}:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Monitoring evidence escapes run root")
            payload[path.relative_to(root).as_posix()] = path
    for path in sorted((root / "results").glob("*.json*")):
        if path.is_file():
            payload["results/" + path.name] = path
    for filename in ("resource_samples.jsonl", "stdout.log", "stderr.log", "launch.json", "resource_summary.json"):
        if (root / filename).is_file():
            payload["execution/" + filename] = root / filename
    for path in sorted(report.rglob("*")):
        if path.is_file():
            payload["report/" + path.relative_to(report).as_posix()] = path
    for path in _project_files(project, config):
        payload["code/" + path.relative_to(project).as_posix()] = path
    for path in sorted((root / "models").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".npz", ".npy", ".json", ".jsonl"}:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Model artifact escapes run root")
            payload["models/" + path.relative_to(root / "models").as_posix()] = path
    if not any(name.startswith("models/") and name.endswith((".npz", ".npy")) for name in payload):
        raise ValueError("No fitted model statistics available for delivery")
    expected_models = {f"models/{dataset}/{category}/statistics.npz" for dataset, category, _, _ in _expected_keys(config)}
    if not expected_models.issubset(payload):
        raise ValueError(f"Missing fitted models for registered categories: {sorted(expected_models - set(payload))}")
    sealed_models = {}
    for name in expected_models:
        seal = payload[name].with_name("statistics_integrity.json")
        if seal.is_file():
            sealed_models[name] = _json(seal)["sha256"]
    backbone = config.get("backbone", {})
    if backbone.get("weight_path"):
        weight = _source_path(project, backbone["weight_path"], backbone.get("sha256"))
        payload["assets/weights/" + weight.name] = weight
    license_paths = list((project / ".venv" / "Lib" / "site-packages").glob("torchvision-*.dist-info/LICENSE"))
    license_paths += list((project / ".venv" / "lib").glob("python*/site-packages/torchvision-*.dist-info/LICENSE"))
    if not license_paths:
        explicit = config.get("torchvision_license_path")
        if explicit:
            license_paths = [_source_path(project, explicit)]
    if len(license_paths) != 1:
        raise ValueError("Exactly one torchvision distribution license is required")
    payload["licenses/torchvision-LICENSE.txt"] = license_paths[0]
    torch_licenses = list((project / ".venv" / "Lib" / "site-packages" / "torch").glob("LICENSE*"))
    torch_licenses += list((project / ".venv" / "lib").glob("python*/site-packages/torch/LICENSE*"))
    torch_licenses += list((project / ".venv" / "Lib" / "site-packages").glob("torch-*.dist-info/LICENSE"))
    torch_licenses += list((project / ".venv" / "Lib" / "site-packages").glob("torch-*.dist-info/NOTICE"))
    torch_licenses += list((project / ".venv" / "lib").glob("python*/site-packages/torch-*.dist-info/LICENSE"))
    torch_licenses += list((project / ".venv" / "lib").glob("python*/site-packages/torch-*.dist-info/NOTICE"))
    for path in torch_licenses:
        if path.is_file():
            payload["licenses/torch-" + path.name] = path
    for item in config.get("additional_licenses", []):
        path = _source_path(project, item["path"], item.get("sha256"))
        name = str(item["name"])
        if Path(name).name != name or ":" in name:
            raise ValueError("Invalid additional license name")
        payload["licenses/" + name] = path
    manifest_files = [{"path": name, "size_bytes": path.stat().st_size, "sha256": _sha(path)} for name, path in sorted(payload.items())]
    for entry in manifest_files:
        if entry["path"] in sealed_models and entry["sha256"] != sealed_models[entry["path"]]:
            raise ValueError(f"Fitted model differs from its execution seal: {entry['path']}")
    manifest = {
        "schema_version": 1, "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(root), "result_count": summary["result_count"], "figure_count": summary["figure_count"],
        "full_dataset_included": False, "selected_dataset_samples_included": True, "all_prediction_maps_included": False, "selected_raw_evidence_included": True,
        "files": manifest_files,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(payload.items()):
            archive.write(path, name)
        archive.writestr("DELIVERY-MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(output) as archive:
        expected_names = set(payload) | {"DELIVERY-MANIFEST.json"}
        if set(archive.namelist()) != expected_names or len(archive.namelist()) != len(expected_names):
            raise ValueError("ZIP member coverage mismatch")
        for entry in manifest["files"]:
            digest = hashlib.sha256()
            total = 0
            with archive.open(entry["path"]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    total += len(chunk)
                    digest.update(chunk)
            if total != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP member verification failed: {entry['path']}")
    receipt = {"schema_version": 1, "status": "verified", "output_zip": str(output), "size_bytes": output.stat().st_size, "sha256": _sha(output), "member_count": len(payload) + 1, "result_count": summary["result_count"], "figure_count": summary["figure_count"], "all_member_hashes_verified": True}
    _write_json(output.with_suffix(output.suffix + ".receipt.json"), receipt)
    return receipt
