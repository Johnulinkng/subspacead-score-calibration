"""Append the complete same-backbone SubspaceAD control experiment.

Only the new extension is written. Historical experiment files, sources,
weights, licenses, and root documents are never replaced by this command.
The caller refreshes the master delivery manifest after final integration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
EXTENSION = "优化实验/同骨干Subspace对照-20261014"
METHODS = ("official_cpu_two_pass", "numpy_cpu_two_pass", "current_cpu_one_pass")
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
LABELS = {"official_cpu_two_pass": "官方两遍流式 PCA / CPU", "numpy_cpu_two_pass": "NumPy 两遍控制 / CPU", "current_cpu_one_pass": "当前单遍统计 / CPU"}
REFERENCE = "https://github.com/CLendering/SubspaceAD/tree/ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    path = (root / Path(*parts.parts)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping member: {relative}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Plain source file required: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(source) != digest(destination):
            raise ValueError(f"Refusing to replace different evidence: {destination}")
        return
    shutil.copyfile(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(p == "__pycache__" or p.startswith(".quick-metrics-") for p in relative.parts):
            raise ValueError(f"Temporary cache must not enter the delivery: {path}")
        if path.suffix.lower() == ".zip":
            raise ValueError("Folder delivery cannot contain ZIP archives")
        copy_file(path, destination / relative)


def table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def authenticate_historical(package: Path) -> dict[str, Any]:
    """Verify every old member before the first delivery mutation."""
    manifest = read(member(package, "DELIVERY-MANIFEST.json"))
    if manifest.get("status") != "completed" or manifest.get("schema_version") != 1:
        raise ValueError("A completed historical delivery manifest is required")
    expected = set()
    for record in manifest["files"]:
        relative = record["path"]
        if relative in expected:
            raise ValueError(f"Duplicate historical manifest member: {relative}")
        expected.add(relative)
        path = member(package, relative)
        if path.stat().st_size != int(record["size_bytes"]) or digest(path) != record["sha256"]:
            raise ValueError(f"Historical delivery has unsealed changes: {relative}")
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual != expected | {"DELIVERY-MANIFEST.json"}:
        raise ValueError("Historical delivery membership differs from its current manifest")
    return {"members": len(expected), "manifest_sha256": digest(package / "DELIVERY-MANIFEST.json")}


def collect_results(package: Path, run: Path) -> dict[str, Any]:
    state = read(member(run, "run_state.json"))
    config = read(member(run, "run_config.json"))
    if state.get("status") != "completed":
        raise ValueError("All 27 categories must finish before delivery mutation")
    protocol = config["protocol"]
    if any(protocol.get(key) != value for key, value in {"image_res": 448, "output_size": 256, "layers": [-4, -5], "feature_dim": 384, "explained_variance": 0.99, "batch_size": 1, "cpu_threads": 4, "device": "cpu"}.items()):
        raise ValueError("Frozen same-backbone/CPU control geometry changed")
    tasks = [(str(d), str(c)) for d, c in config["expected_matrix"]["categories"]]
    methods = tuple(config["expected_matrix"]["methods"])
    if len(tasks) != 27 or len(set(tasks)) != 27 or methods != METHODS:
        raise ValueError("The fixed full 27-category/three-method matrix is required")
    dataset_counts = {d: sum(task[0] == d for task in tasks) for d, _ in tasks}
    if dataset_counts != {"mvtec_ad": 15, "visa": 12}:
        raise ValueError("MVTec AD 15 plus VisA 12 categories are required")
    rows = read_lines(member(run, "results/category_metrics.jsonl"))
    expected = {(d, c, m + "_" + score) for d, c in tasks for m in methods for score in ("max", "top1p")}
    keys = [(r["dataset"], r["category"], r.get("variant", r.get("method"))) for r in rows]
    if len(rows) != 162 or set(keys) != expected:
        raise ValueError("Expected 162 paired completed metric rows")
    normalized = []
    for row, (_, _, variant) in zip(rows, keys, strict=True):
        if row.get("status") != "completed" or int(row["seed"]) != int(protocol["seed"]) or any(not math.isfinite(float(row[k])) or not 0 <= float(row[k]) <= 1 for k in METRICS):
            raise ValueError("Finite completed unit-interval measurements are required")
        normalized.append({"dataset": row["dataset"], "category": row["category"], "method": variant.rsplit("_", 1)[0], "score": variant.rsplit("_", 1)[1], "variant": variant, "seed": int(row["seed"]), "test_images": int(row["test_images"]), **{k: float(row[k]) for k in METRICS}})
    index = {(r["dataset"], r["category"], r["variant"]): r for r in normalized}
    total_train, total_test = 0, 0
    diagnostics, selected = [], []
    for dataset, category in tasks:
        cat = run / f"categories/{dataset}/{category}"
        seal = read(member(cat, "completion.json"))
        if seal.get("status") != "completed":
            raise ValueError(f"Incomplete category seal: {dataset}/{category}")
        for relative, expected_hash in seal["sha256"].items():
            if digest(member(run, relative)) != expected_hash:
                raise ValueError(f"Completed category evidence drift: {relative}")
        train, inventory = read(member(cat, "train_manifest.json")), read(member(cat, "test_inventory.json"))
        for filename in ("train_manifest.json", "test_inventory.json"):
            if digest(member(cat, filename)) != digest(member(package, f"models/{dataset}/{category}/{filename}")):
                raise ValueError("Control inventories must match the frozen original delivery byte for byte")
        samples = read_lines(member(cat, "samples.jsonl"))
        if len(samples) != len(inventory) or len({s["sample_id"] for s in samples}) != len(samples):
            raise ValueError("Category test sample count/uniqueness mismatch")
        for method in methods:
            member(cat, f"models/{method}.npz")
            a, b = index[(dataset, category, method + "_max")], index[(dataset, category, method + "_top1p")]
            if a["test_images"] != len(samples) or b["test_images"] != len(samples) or any(a[k] != b[k] for k in METRICS[1:]):
                raise ValueError("Paired sample counts/localization metrics differ across image aggregation")
        for sample in samples:
            if set(sample["raw_maps"]) != set(methods) or set(sample["raw_map_sha256"]) != set(methods):
                raise ValueError("Each test sample requires all three unmodified raw maps")
            if (sample["dataset"], sample["category"]) != (dataset, category):
                raise ValueError("Misfiled test sample")
            for method in methods:
                if digest(member(run, sample["raw_maps"][method])) != sample["raw_map_sha256"][method]:
                    raise ValueError("Test map differs from its immediate seal")
            if digest(member(run, sample["mask_path"])) != sample["mask_sha256"]:
                raise ValueError("GT differs from its immediate seal")
            if sample.get("selected"):
                selected.append(sample)
        diagnostic = read(member(cat, "diagnostics.json"))
        diagnostics.append({"dataset": dataset, "category": category, "diagnostics": diagnostic})
        total_train += len(train)
        total_test += len(samples)
    if total_train != 12288 or total_test != 3887 or len(selected) != 54:
        raise ValueError("Full 12,288 normal training / 3,887 test / 54 selected inputs required")
    normalized.sort(key=lambda r: (r["dataset"], r["category"], methods.index(r["method"]), r["score"]))
    return {"config": config, "state": state, "tasks": tasks, "rows": normalized, "diagnostics": diagnostics, "selected": selected, "train_images": total_train, "test_images": total_test}


def make_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    index = {(r["dataset"], r["category"], r["method"], r["score"]): r for r in rows}
    deltas = []
    for row in rows:
        grouped[(row["dataset"], row["method"], row["score"])].append(row)
        comparisons = () if row["method"] == METHODS[0] else (METHODS[0],) if row["method"] == METHODS[1] else METHODS[:2]
        for reference in comparisons:
            old = index[(row["dataset"], row["category"], reference, row["score"])]
            deltas.append({"dataset": row["dataset"], "category": row["category"], "method": row["method"], "reference": reference, "score": row["score"], **{k + "_delta_pp": 100 * (row[k] - old[k]) for k in METRICS}})
    macros = [{"dataset": d, "method": m, "score": s, "categories": len(group), "test_images": sum(r["test_images"] for r in group), **{k: sum(r[k] for r in group) / len(group) for k in METRICS}} for (d, m, s), group in sorted(grouped.items())]
    macro_index = {(r["dataset"], r["method"], r["score"]): r for r in macros}
    macro_deltas = []
    for row in macros:
        comparisons = () if row["method"] == METHODS[0] else (METHODS[0],) if row["method"] == METHODS[1] else METHODS[:2]
        for reference in comparisons:
            old = macro_index[(row["dataset"], reference, row["score"])]
            macro_deltas.append({"dataset": row["dataset"], "method": row["method"], "reference": reference, "score": row["score"], **{k + "_delta_pp": 100 * (row[k] - old[k]) for k in METRICS}})
    return macros, deltas, macro_deltas


def resource_tables(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trials = read_lines(path)
    if not trials:
        raise ValueError("Actual resource trials, including retained failures, are required")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if row.get("status") in (None, "running", "pending", "initializing"):
            raise ValueError("Resource trials must be terminal before packaging")
        family = row.get("experiment", row.get("family", row.get("scope", "unclassified")))
        groups[(family, row.get("dataset", ""), row.get("category", ""), row.get("method", row.get("variant", "")), row.get("loader", row.get("input_loading", "")), row.get("attention", row.get("output_attentions", row.get("attention_outputs", ""))), row.get("train_images", row.get("normal_train_images", row.get("sample_count", row.get("subset_count", "")))))].append(row)
    summaries = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        good = [r for r in rows if r["status"] in ("completed", "passed", "ok", "verified")]
        def measured(*keys: str) -> list[float]:
            result = []
            for trial in good:
                value = next((trial[k] for k in keys if trial.get(k) is not None), None)
                if value is not None:
                    if not math.isfinite(float(value)) or float(value) < 0:
                        raise ValueError("Resource measurement must be nonnegative and finite")
                    result.append(float(value))
            return result
        result = dict(zip(("experiment", "dataset", "category", "method", "loader", "attention", "train_images"), key, strict=True))
        result.update(attempts=len(rows), successful_trials=len(good), failed_or_blocked_trials=len(rows) - len(good))
        for name, aliases in (("elapsed_seconds", ("elapsed_seconds", "fit_seconds")), ("fit_seconds", ("fit_seconds",)), ("total_process_seconds", ("total_process_seconds",)), ("feature_forward_seconds", ("feature_forward_seconds",)), ("image_decode_seconds", ("image_decode_seconds",)), ("pca_total_kernel_seconds", ("pca_total_kernel_seconds",)), ("process_peak_working_set_bytes", ("process_peak_working_set_bytes", "peak_working_set_bytes")), ("process_peak_commit_bytes", ("process_peak_commit_bytes", "process_peak_private_commit_bytes", "peak_private_commit_bytes")), ("sampled_peak_private_bytes", ("sampled_peak_private_bytes",)), ("cuda_peak_allocated_bytes", ("cuda_peak_allocated_bytes",)), ("cuda_peak_reserved_bytes", ("cuda_peak_reserved_bytes",))):
            values = measured(*aliases)
            result[name + "_measured_trials"] = len(values)
            result[name + "_median"] = statistics.median(values) if values else None
            result[name + "_mean"] = statistics.mean(values) if values else None
            result[name + "_sample_stddev"] = statistics.stdev(values) if len(values) > 1 else None
            result[name + "_min"] = min(values) if values else None
            result[name + "_max"] = max(values) if values else None
        result["measurement_scopes"] = sorted({str(r.get("measurement_scope", r.get("memory_scope", "unspecified"))) for r in rows})
        result["failure_reasons"] = [r.get("error", r.get("reason", r.get("exception", "unspecified failure"))) for r in rows if r not in good]
        summaries.append(result)
    return trials, summaries


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], percent: bool = False) -> str:
    content = "<table><thead><tr>" + "".join("<th>" + html.escape(label) + "</th>" for _, label in columns) + "</tr></thead><tbody>"
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                shown = "未测 / 不适用"
            elif isinstance(value, float):
                shown = f"{value * 100:.6f}%" if percent and key in METRICS else f"{value:.6g}"
            else:
                shown = str(value)
            cells.append("<td>" + html.escape(shown) + "</td>")
        content += "<tr>" + "".join(cells) + "</tr>"
    return content + "</tbody></table>"


def resource_pairs(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair registered repetitions; shared-process memory is deliberately blank."""
    index = {}
    for trial in trials:
        experiment = trial.get("experiment", trial.get("family", "unclassified"))
        mode = trial.get("mode", trial.get("method", trial.get("variant", "")))
        repeat = trial.get("repeat", trial.get("trial_index", 0))
        key = (experiment, trial.get("dataset", ""), trial.get("category", ""), mode, repeat)
        if key in index:
            raise ValueError(f"Duplicate resource trial identity: {key}")
        index[key] = trial
    comparisons = (("one_lazy", "two_lazy", "one_vs_two_pass_lazy"), ("one_eager", "two_eager", "one_vs_two_pass_eager"), ("one_lazy", "one_eager", "lazy_vs_eager_one_pass"), ("two_lazy", "two_eager", "lazy_vs_eager_two_pass"), ("one_lazy", "two_eager", "combined_one_lazy_vs_two_eager"), ("attention_off", "attention_on", "attention_output_off_vs_on"), (METHODS[2], METHODS[1], "numpy_one_vs_two_pass_fit_only"), (METHODS[2], METHODS[0], "current_vs_official_fit_includes_backend_difference"))
    output = []
    for (experiment, dataset, category, mode, repeat), after in index.items():
        for current, reference, contrast in comparisons:
            if mode != current:
                continue
            before = index.get((experiment, dataset, category, reference, repeat))
            if before is None:
                continue
            success = before["status"] in ("completed", "passed", "ok", "verified") and after["status"] in ("completed", "passed", "ok", "verified")
            scope = str(before.get("measurement_scope", before.get("memory_scope", "unspecified"))) + " | " + str(after.get("measurement_scope", after.get("memory_scope", "unspecified")))
            result = {"experiment": experiment, "dataset": dataset, "category": category, "contrast": contrast, "method": current, "reference": reference, "repeat": repeat, "status": "paired_completed" if success else "paired_failure_or_blocked", "method_status": after["status"], "reference_status": before["status"], "measurement_scope": scope, "memory_comparison_applicable": "shared" not in scope.lower(), "method_error": after.get("error", after.get("reason")), "reference_error": before.get("error", before.get("reason"))}
            for field, aliases in (("fit_seconds", ("fit_seconds", "elapsed_seconds")), ("total_process_seconds", ("total_process_seconds",)), ("process_peak_working_set_bytes", ("process_peak_working_set_bytes", "peak_working_set_bytes")), ("process_peak_commit_bytes", ("process_peak_commit_bytes", "process_peak_private_commit_bytes", "peak_private_commit_bytes")), ("sampled_peak_private_bytes", ("sampled_peak_private_bytes",))):
                a = next((after[k] for k in aliases if after.get(k) is not None), None)
                b = next((before[k] for k in aliases if before.get(k) is not None), None)
                allowed = success and ("bytes" not in field or result["memory_comparison_applicable"])
                result[field + "_method"] = a if allowed else None
                result[field + "_reference"] = b if allowed else None
                result[field + "_reduction_percent"] = 100 * (float(b) - float(a)) / float(b) if allowed and a is not None and b is not None and float(b) > 0 else None
            a, b = result["fit_seconds_method"], result["fit_seconds_reference"]
            result["fit_speedup_ratio"] = float(b) / float(a) if success and a is not None and b is not None and float(a) > 0 else None
            output.append(result)
    return sorted(output, key=lambda r: (r["experiment"], r["dataset"], r["category"], r["contrast"], r["repeat"]))


def resource_interactions(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    modes = ("two_eager", "one_eager", "two_lazy", "one_lazy")
    for trial in trials:
        mode = trial.get("mode", trial.get("method"))
        if mode in modes:
            groups[(trial.get("experiment", "end_to_end"), trial["dataset"], trial["category"], trial.get("repeat", trial.get("trial_index", 0)))][mode] = trial
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        if set(group) != set(modes):
            continue
        success = all(row["status"] in ("completed", "passed", "ok", "verified") for row in group.values())
        result = dict(zip(("experiment", "dataset", "category", "repeat"), key, strict=True))
        result.update(status="completed" if success else "failed_or_blocked_component", component_statuses={mode: group[mode]["status"] for mode in modes}, formula="(two_eager-one_lazy)-(two_eager-one_eager)-(two_eager-two_lazy) = one_eager+two_lazy-two_eager-one_lazy", meaning="positive interaction means additional observed time saving beyond the sum of the two single-component savings; runtime fluctuation is not removed")
        for field, aliases in (("fit_seconds", ("fit_seconds", "elapsed_seconds")), ("total_process_seconds", ("total_process_seconds",))):
            values = {mode: next((group[mode][k] for k in aliases if group[mode].get(k) is not None), None) for mode in modes}
            valid = success and all(value is not None for value in values.values())
            interaction = float(values["one_eager"]) + float(values["two_lazy"]) - float(values["two_eager"]) - float(values["one_lazy"]) if valid else None
            result[field + "_interaction_saving"] = interaction
            result[field + "_interaction_percent_of_two_eager"] = 100 * interaction / float(values["two_eager"]) if valid and float(values["two_eager"]) > 0 else None
        rows.append(result)
    return rows


def collect_failure_history(resource: Path) -> list[dict[str, Any]]:
    paths = sorted((resource / "failed_attempts").rglob("*.json")) if (resource / "failed_attempts").is_dir() else []
    state_path = resource / "matrix_state.json"
    if state_path.is_file():
        state = read(state_path)
        if state.get("status") != "completed" or state.get("completed_trials") != 36 or state.get("failed_attempt_count") != len(paths):
            raise ValueError("Resource matrix completion/failure-history count differs from retained attempt files")
    rows = []
    for path in paths:
        receipt = read(path)
        if receipt.get("status") not in ("failed", "oom", "blocked", "resource_gate_blocked"):
            raise ValueError("An archived failed-attempt receipt has a nonfailure status")
        rows.append({"evidence_path": "资源记录/" + path.relative_to(resource).as_posix(), "sha256": digest(path), "dataset": receipt.get("dataset"), "category": receipt.get("category"), "mode": receipt.get("mode"), "repeat": receipt.get("repeat"), "status": receipt["status"], "error": receipt.get("error", receipt.get("reason")), "receipt": receipt})
    return rows


def draw_selected(package: Path, extension: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    output = []
    for position, record in enumerate(samples):
        image_path = member(package, record["source_image_copy"])
        raw_maps = [np.load(member(extension / "run", record["raw_maps"][method]), allow_pickle=False) for method in METHODS]
        mask = np.load(member(extension / "run", record["mask_path"]), allow_pickle=False)
        if any(raw.dtype != np.float32 or raw.shape != (256, 256) or not np.isfinite(raw).all() for raw in raw_maps) or mask.dtype != np.bool_ or mask.shape != (256, 256):
            raise ValueError("Selected display requires the unmodified common raw map/GT geometry")
        lower, upper = map(float, np.quantile(raw_maps[0], [0.01, 0.99]))
        if upper <= lower:
            upper = lower + 1
        with Image.open(image_path) as source:
            original = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        panels = [original, Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")]
        panels.extend(Image.fromarray(np.asarray(colormaps["viridis"](np.clip((raw - lower) / (upper - lower), 0, 1))[..., :3] * 255, dtype=np.uint8)) for raw in raw_maps)
        canvas = Image.new("RGB", (1472, 388), "white")
        drawing = ImageDraw.Draw(canvas)
        drawing.text((16, 10), f"{record['dataset']} / {record['sample_id']}    label={record['label']}", fill="black", font=font)
        for index, (label, panel) in enumerate(zip(("Input", "Ground truth", "Official 2-pass / CPU", "NumPy 2-pass / CPU", "Current 1-pass / CPU"), panels, strict=True)):
            drawing.text((16 + index * 288, 47), label, fill="black", font=small)
            canvas.paste(panel, (16 + index * 288, 76))
        drawing.text((16, 342), f"All three PCA panels share the official raw-map 1% / 99% scale: {lower:.8g} .. {upper:.8g}; viridis.", fill="black", font=small)
        drawing.text((16, 365), "Display clipping only; identical squared-residual units. Saved maps and evaluation scores remain unmodified.", fill="black", font=small)
        relative = f"visualizations/comparison-{position:03d}.png"
        path = extension / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        output.append({**record, "figure": relative, "figure_sha256": digest(path), "display_scale": {"reference": METHODS[0], "quantiles": [0.01, 0.99], "lower": lower, "upper": upper, "colormap": "viridis", "scope": "shared across the three same-unit PCA maps for this sample"}, "display_only_normalization": True, "evaluation_raw_maps_modified": False})
    return output


def write_report(extension: Path, evidence: dict[str, Any], macros: list[dict[str, Any]], macro_deltas: list[dict[str, Any]], deltas: list[dict[str, Any]], resources: list[dict[str, Any]], trials: list[dict[str, Any]]) -> None:
    precision = [r for r in macro_deltas if r["method"] == METHODS[2] and r["reference"] == METHODS[0]]
    max_delta = max(abs(r[k + "_delta_pp"]) for r in precision for k in METRICS)
    failures = [r for r in trials if r["status"] not in ("completed", "passed", "ok", "verified")]
    shared = [r for r in trials if "shared" in str(r.get("measurement_scope", r.get("memory_scope", ""))).lower()]
    failure_history = read(member(extension, "failed_resource_attempts.json"))["attempts"]
    paragraphs = [
        "本轮完成相同 DINOv2-S 骨干下的官方两遍流式 PCA、NumPy 两遍控制及当前单遍统计三个实现对照。MVTec AD 15 类、VisA 12 类；全部 12,288 张正常训练图和 3,887 张测试图保持原冻结清单，得到 162 条配对指标。",
        "官方 SubspaceAD 已采用两遍流式 PCA。当前实现使用单遍均值与中心二阶统计量合并，保留完整协方差，再做特征分解。按需加载图片是单独的执行优化；两项不能合称新增流式 PCA 算法。",
        "三个 PCA 控制均在 CPU 上进行 float64 统计，使用相同 float32 DINO patch 特征、N−1 协方差及 0.99 方差保留目标。官方 Torch 与当前 NumPy 实现的差别包含统计后端；NumPy 两遍控制用于拆开后端与遍数影响。DINO 前向设备与线程数以冻结 run_config.json 为准。",
        "共同骨干为 DINOv2-S，448 全帧输入，隐藏层 [-4,-5] 均值，384 维，32×32 patch；相同重建残差评分、448 平面插值及 3×3 sigma=4 平滑，再缩放至共同 256×256 评测平面。GT 使用最近邻缩放；正常图背景进入像素评测；AUPRO 使用完整阈值、0–0.30 范围及等区域权重。",
        f"当前单遍相对官方两遍的两数据集宏平均指标最大绝对变化为 {max_delta:.8g} 个百分点。此处精度变化来自同骨干实现对照；此前 PaDiM 到 DINO/SubspaceAD 的提升属于公开方法与特征骨干更换，不能归因于单遍统计。",
        "max 与 top1p 共享同一张未经归一化的原始分数图；top1p 是 256×256 平面最高 655 个像素的 float64 均值。两者像素定位指标完全相同，其图像 AUROC 差值仅表征聚合方式。",
        "完整数据与性能基准的再次运行需要在原项目工作区准备原数据集；交付文件夹包含冻结配置、代码、全部模型及预测缓存、54 组输入和本地权重，可以离线复算指标与选定样本推理。文件夹未打包全部训练与测试原图。",
        "资源证据分为拟合统计本身、完整特征提取加拟合、图片加载三类，以原始试验字段为准。完整 27 类运行内共用进程的峰值仅用于监控；不能按方法分配，也不能用来计算内存下降。独立进程受控试验的类别、正常图子集、重复次数和并发状态全部保留，时间包含项见每条试验。",
        "fit_only 是全类共享 token 缓存上的真实统计与特征分解时间，包含输入读取及内容校验，排除 DINO 前向。64 张图的资源 worker 两遍组为项目 NumPy CPU 控制，实现原版相同的均值/协方差目标；其对照不能冒充作者默认 GPU 整套运行。该 worker 的 fit_seconds 包含正常图解码、DINO 前向、PCA 和特征分解，排除权重加载与预热；total_process_seconds 另包含完整进程初始化、加载、预热及结束监控。",
        f"最终资源试验和全类 fit_only 共保留 {len(trials)} 条记录，其中最终失败或资源门槛阻断 {len(failures)} 条；另保留 {len(failure_history)} 次资源 matrix 的失败重试尝试，单列于 failed_resource_attempts.json，不与同编号的成功重复合并。带 shared 标记的共享进程观察 {len(shared)} 条。缺测字段留空，OOM 和阻断记录保留，不从成功均值中抹去。资源子集结果不代表两个数据集全类的同等降幅。",
        "本轮没有增加异常检测算法组件，因此算法组件消融不适用；两遍到单遍、eager 到 lazy、attention 输出开关属于执行组件对照。预训练特征保持冻结，无反向梯度训练。固定种子与已访问测试集的探索性比较，不构成多随机种子稳定性或独立未访问测试集证明。",
        "来源、原版代码、权重与许可证保留。工作定位是公开 SubspaceAD 方法的执行优化与可复现实测，scientific_claim_eligible=false；未将公开方法改写成自主原创算法。",
        "离线验收：使用扩展内 code/scripts/verify_route1_delivery.py，提供 --package-root 和文件夹路径。默认重算所有缓存图像 AUROC、验证全量原始图/GT/模型和配对模型差异；--metrics-categories 27 复算所有四项指标；--inference-samples 2 使用本地 DINO 权重进行新 CPU 前向。临时文件与回执必须在交付文件夹外。"
    ]
    text = "同骨干 SubspaceAD 实现与资源对照（2026-10-14）\n\n" + "\n\n".join(paragraphs)
    text += "\n\n宏平均指标（各类别等权；原值单位为 0..1）\n"
    for row in macros:
        text += f"{row['dataset']} | {LABELS[row['method']]} | {row['score']} | " + " | ".join(f"{k}={row[k]:.10f}" for k in METRICS) + "\n"
    text += "\n每类全部差值、下降项及资源重复测量见 CSV；原始证据见 run 和 资源记录。\n"
    (extension / "同骨干对照报告.txt").write_text(text, encoding="utf-8")
    regressions = [r for r in deltas if r["method"] == METHODS[2] and r["reference"] == METHODS[0] and r["score"] == "max" and any(r[k + "_delta_pp"] < 0 for k in METRICS)]
    body = "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs)
    body += "<h2>数据集宏平均</h2>" + html_table(macros, [("dataset", "数据集"), ("method", "实现"), ("score", "图像聚合"), *((k, k) for k in METRICS)], percent=True)
    body += "<h2>当前单遍相对官方两遍：宏平均差值（百分点）</h2>" + html_table(precision, [("dataset", "数据集"), ("score", "图像聚合"), *((k + "_delta_pp", k) for k in METRICS)])
    body += "<h2>保留全部下降项</h2>" + (html_table(regressions, [("dataset", "数据集"), ("category", "类别"), *((k + "_delta_pp", k) for k in METRICS)]) if regressions else "<p>当前单遍相对官方 max 的四项指标没有负差值类别。</p>")
    body += "<h2>资源受控试验摘要</h2>" + html_table(resources, [("experiment", "试验范围"), ("dataset", "数据集"), ("category", "类别"), ("method", "实现"), ("loader", "图片加载"), ("attention", "attention 输出"), ("train_images", "正常图数"), ("successful_trials", "成功次数"), ("failed_or_blocked_trials", "失败/阻断"), ("elapsed_seconds_median", "试验时间中位数(s)"), ("total_process_seconds_median", "完整进程时间(s)"), ("process_peak_working_set_bytes_median", "进程 WS 峰值(bytes)"), ("process_peak_commit_bytes_median", "进程提交峰值(bytes)"), ("sampled_peak_private_bytes_median", "采样私有内存峰值(bytes)"), ("measurement_scopes", "测量范围")])
    pairs = resource_pairs(trials)
    body += "<h2>逐次配对的速度与内存变化</h2><p>减少比例=(参考值−当前值)/参考值；负值表示变差。共享进程的内存对照留空。完整进程时间与加载权重、预热后的特征提取加拟合时间分列；重复测量是运行波动测量，不是随机种子稳定性实验。</p>" + html_table(pairs, [("dataset", "数据集"), ("category", "类别"), ("contrast", "对照组件"), ("repeat", "重复"), ("status", "状态"), ("fit_speedup_ratio", "记录范围内速度比"), ("fit_seconds_reduction_percent", "试验时间减少(%)"), ("total_process_seconds_reduction_percent", "完整进程减少(%)"), ("process_peak_working_set_bytes_reduction_percent", "WS 峰值减少(%)"), ("process_peak_commit_bytes_reduction_percent", "提交峰值减少(%)")])
    body += "<h2>单遍与图片加载的 2×2 时间交互</h2><p>交互节省=(两遍eager−单遍lazy)−(两遍eager−单遍eager)−(两遍eager−两遍lazy)。正值表示合并条件观测到的节省超过两个单项节省之和；负值表示不足。共享机器运行波动也会进入该值，三次重复不能据此证明统计显著性。</p>" + html_table(resource_interactions(trials), [("dataset", "数据集"), ("category", "类别"), ("repeat", "重复"), ("status", "状态"), ("fit_seconds_interaction_saving", "正式拟合交互节省(s)"), ("total_process_seconds_interaction_saving", "完整进程交互节省(s)")])
    body += "<h2>资源门槛失败与重试记录</h2>" + (html_table(failure_history, [("dataset", "数据集"), ("category", "类别"), ("mode", "模式"), ("repeat", "最终重复编号"), ("status", "失败状态"), ("error", "原因"), ("evidence_path", "原 receipt")]) if failure_history else "<p>该资源 matrix 没有归档的失败重试记录。</p>")
    body += "<h2>可复核材料</h2><ul>"
    for target, label in (("category_metrics.csv", "162 条每类配对指标"), ("dataset_macro_metrics.csv", "两数据集宏平均"), ("category_deltas.csv", "完整逐类差值（含下降）"), ("dataset_macro_deltas.csv", "宏平均差值"), ("model_diagnostics.json", "PCA 模型及原始图数值差异"), ("resource_trials.csv", "资源试验逐条记录"), ("resource_summary.csv", "资源重复测量汇总"), ("resource_paired_comparisons.csv", "单遍、加载、attention 的逐次配对变化"), ("resource_interactions.json", "2×2 逐次时间交互"), ("failed_resource_attempts.json", "资源失败重试历史"), ("同骨干对照报告.txt", "纯文本报告"), ("code/scripts/verify_route1_delivery.py", "离线复算脚本"), ("公开来源/来源与贡献.json", "来源与贡献边界"), ("history/DELIVERY-MANIFEST.json", "进入本阶段前的文件清单"), ("../../项目总览.html", "项目总览")):
        body += f"<li><a href='{html.escape(target, quote=True)}'>{html.escape(label)}</a></li>"
    body += f"<li><a href='{REFERENCE}'>官方 SubspaceAD 固定版本</a></li></ul>"
    selected = read(member(extension, "selected_samples.json"))["samples"]
    body += "<h2>54 组原冻结样本可视化</h2><p>每类第一张正常图与第一张异常图；三个同单位 PCA 分数图共享该样本官方图的 1%/99% 显示范围。此项仅用于显示，原始分数和评测保持不变。</p>"
    for sample in selected:
        body += "<figure><figcaption>" + html.escape(f"{sample['dataset']} / {sample['sample_id']}") + "</figcaption><a href='" + html.escape(sample["figure"], quote=True) + "'><img loading='lazy' style='max-width:100%;height:auto' src='" + html.escape(sample["figure"], quote=True) + "' alt='paired same-backbone PCA maps'></a></figure>"
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>同骨干 SubspaceAD 对照</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;font-family:system-ui,sans-serif;line-height:1.7;color:#17202a}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto;margin:20px 0}th,td{border:1px solid #d8dde3;padding:7px 10px;white-space:nowrap}th{background:#edf3f8}a{color:#125ca6}</style><h1>同骨干 SubspaceAD 实现与资源对照</h1>" + body + "</html>"
    (extension / "report.html").write_text(page, encoding="utf-8")


def seal_extension(extension: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    path = extension / "extension_manifest.json"
    information = metadata if metadata is not None else read(path)
    files = []
    for source in sorted(extension.rglob("*")):
        if not source.is_file() or source == path:
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(extension.resolve()) or source.suffix.lower() == ".zip":
            raise ValueError("No symlinks, escaping evidence, or ZIP allowed")
        files.append({"path": source.relative_to(extension).as_posix(), "size_bytes": source.stat().st_size, "sha256": digest(source)})
    information["files"] = files
    information["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(path, information)
    return {"status": "extension_sealed", "members": len(files), "bytes": sum(r["size_bytes"] for r in files), "master_manifest_updated": False}


def build(args: argparse.Namespace) -> dict[str, Any]:
    project, package = args.project_root.resolve(), args.package_root.resolve()
    extension = package / EXTENSION
    if args.seal_only:
        return seal_extension(extension)
    run = args.run_dir.resolve()
    resource = args.resource_dir.resolve() if args.resource_dir else run
    evidence = collect_results(package, run)
    resource_path = member(resource, "resource_trials.jsonl")
    trials, summaries = resource_tables(resource_path)
    failure_history = collect_failure_history(resource)
    pairs = resource_pairs(trials)
    if not pairs:
        raise ValueError("Resource comparisons require paired component conditions")
    config = evidence["config"]
    protocol = config.get("protocol", {})
    registered = {(str(d), str(c), str(mode), repeat) for d, c in protocol.get("resource_categories", []) for mode in protocol.get("resource_modes", []) for repeat in range(int(protocol.get("resource_repeats", 0)))}
    observed = [(str(row.get("dataset", "")), str(row.get("category", "")), str(row.get("mode", "")), int(row.get("repeat", row.get("trial_index", -1)))) for row in trials if row.get("mode") in protocol.get("resource_modes", [])]
    if registered and (set(observed) != registered or len(observed) != len(registered)):
        raise ValueError("Every registered resource mode/repeat must be retained exactly once, including failures")
    fit_rows = [r for r in trials if r.get("experiment") == "fit_only"]
    expected_fit = {(d, c, method) for d, c in evidence["tasks"] for method in METHODS}
    if len(fit_rows) != 81 or {(r["dataset"], r["category"], r["method"]) for r in fit_rows} != expected_fit:
        raise ValueError("All 81 actual full-category PCA fitting measurements are required")
    for relative, expected in config["source_sha256"].items():
        if digest(member(project, relative)) != expected:
            raise ValueError(f"Executed source differs from source to be delivered: {relative}")
    protocol_source = member(project, "configs/route1_controls_20261014.json")
    if digest(protocol_source) != config["protocol_sha256"] or read(protocol_source) != protocol:
        raise ValueError("Frozen protocol differs from the copied reproduction configuration")
    if extension.exists():
        raise ValueError("Refusing to rebuild or replace an existing experiment extension")
    if args.check_only:
        return {"status": "ready_to_build", "delivery_modified": False, "categories": 27, "metric_rows": 162, "test_images": 3887, "resource_trials": len(trials)}
    authenticated = authenticate_historical(package)
    extension.mkdir(parents=True)
    copy_file(package / "DELIVERY-MANIFEST.json", extension / "history/DELIVERY-MANIFEST.json")
    if args.archive_root_docs:
        for name in ("项目总览.html", "先读我.txt", "实验说明.txt", "Verify.ps1"):
            if (package / name).is_file():
                copy_file(package / name, extension / "history" / name)
    copy_tree(run, extension / "run")
    if resource != run:
        copy_tree(resource, extension / "资源记录")
    else:
        copy_file(resource_path, extension / "资源记录/resource_trials.jsonl")
    source_files = list(config["source_sha256"])
    source_files += ["scripts/build_route1_delivery.py", "scripts/verify_route1_delivery.py", "src/industrial_anomaly_lab/subspace_features.py", "src/industrial_anomaly_lab/quick_metrics.py", "src/industrial_anomaly_lab/metrics.py", "src/industrial_anomaly_lab/__init__.py", "pyproject.toml", "uv.lock"]
    source_files.extend(relative for relative in ("configs/route1_controls_20261014.json", "scripts/Invoke-Route1Controls.ps1", "scripts/run_route1_resource_matrix.py", "scripts/audit_route1_run.py", "docs/route1_controls_protocol.md", "tests/test_subspace_controls.py", "tests/test_subspace_features.py") if (project / relative).is_file())
    for relative in args.include_source:
        source_files.append(relative)
    for key in ("config_path", "protocol_path"):
        value = config.get(key)
        if value and not Path(value).is_absolute() and (project / value).is_file():
            source_files.append(value)
    for relative in dict.fromkeys(source_files):
        copy_file(member(project, relative), extension / "code" / relative)
    for relative, expected in config.get("environment_sha256", {}).items():
        if digest(member(extension / "code", relative)) != expected:
            raise ValueError(f"Executed dependency declaration differs: {relative}")
    vendor_records = config.get("vendor_source_sha256", {})
    for relative, expected in vendor_records.items():
        source = member(project / "vendor/SubspaceAD", relative)
        if digest(source) != expected:
            raise ValueError(f"Frozen public source differs: {relative}")
        copy_file(source, extension / "code/vendor/SubspaceAD" / relative)
    old_weights = read(member(package, "优化实验/DINO小型SubspaceAD验证/weights_manifest.json"))
    for record in old_weights["files"]:
        source = member(package, record["path"])
        if digest(source) != record["sha256"] or source.stat().st_size != int(record["size_bytes"]):
            raise ValueError("Shared historical DINO assets changed")
    write(extension / "weights_manifest.json", old_weights)
    write(extension / "公开来源/来源与贡献.json", {"official_subspacead": REFERENCE, "official_pca_is_already_streaming": True, "official_passes": 2, "current_passes": 1, "same_backbone": "DINOv2-S 448 layers[-4,-5]", "pca_controls": list(METHODS), "algorithm_component_added": False, "execution_components": ["one-pass centered sufficient statistics", "lazy image loading", "optional attention output retention"], "scientific_claim_eligible": False, "licenses_preserved": True, "historical_attribution_retained": True, "protocol": protocol})
    macros, deltas, macro_deltas = make_tables(evidence["rows"])
    table(extension / "category_metrics.csv", evidence["rows"], ["dataset", "category", "method", "score", "variant", "seed", "test_images", *METRICS])
    table(extension / "dataset_macro_metrics.csv", macros, ["dataset", "method", "score", "categories", "test_images", *METRICS])
    table(extension / "category_deltas.csv", deltas, ["dataset", "category", "method", "reference", "score", *[k + "_delta_pp" for k in METRICS]])
    table(extension / "dataset_macro_deltas.csv", macro_deltas, ["dataset", "method", "reference", "score", *[k + "_delta_pp" for k in METRICS]])
    write(extension / "model_diagnostics.json", {"schema_version": 1, "categories": evidence["diagnostics"], "model_comparison_uses_projection_matrices": True})
    table(extension / "resource_trials.csv", trials, list(dict.fromkeys(key for row in trials for key in row)))
    table(extension / "resource_summary.csv", summaries, list(summaries[0]))
    table(extension / "resource_paired_comparisons.csv", pairs, list(pairs[0]))
    write(extension / "resource_interactions.json", {"schema_version": 1, "comparisons": resource_interactions(trials)})
    write(extension / "failed_resource_attempts.json", {"schema_version": 1, "attempts": failure_history, "failed_attempt_count": len(failure_history), "not_merged_into_final_36_repetitions": True})
    selected_original = read(member(package, "report/visualization_manifest.json"))["samples"]
    original_index = {(r["dataset"], r["category"], r["sample_id"]): r for r in selected_original}
    selected = []
    for row in evidence["selected"]:
        identity = (row["dataset"], row["category"], row["sample_id"])
        old = original_index[identity]
        image_relative = "report/" + old["source_image_copy"]
        if digest(member(package, image_relative)) != row["image_sha256"] or row["image_sha256"] != old["source_image_sha256"]:
            raise ValueError("Mechanically selected original input changed")
        selected.append({**row, "source_image_copy": image_relative, "source_image_sha256": row["image_sha256"]})
    selected = draw_selected(package, extension, selected)
    write(extension / "selected_samples.json", {"samples": selected, "selection": "exactly the original first normal and first anomaly per category; no metric-based selection", "count": 54, "figure_count": 54, "raw_maps_modified": False})
    write_report(extension, evidence, macros, macro_deltas, deltas, summaries, trials)
    manifest = {"schema_version": 1, "status": "completed", "source_run": str(run), "resource_source": str(resource), "methods": list(METHODS), "categories": 27, "normal_train_images": 12288, "test_images": 3887, "metric_rows": 162, "pca_models": 81, "raw_maps": 11661, "evaluation_masks": 3887, "selected_inputs": 54, "figure_count": 54, "resource_trials": len(trials), "resource_failed_or_blocked_trials": sum(r["status"] not in ("completed", "passed", "ok", "verified") for r in trials), "resource_failed_retry_attempts_retained": len(failure_history), "previous_delivery_manifest_sha256": authenticated["manifest_sha256"], "scientific_claim_eligible": False, "gradient_training_performed": False, "historical_files_replaced_by_builder": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    sealed = seal_extension(extension, manifest)
    return {**sealed, "status": "extension_built", "path": str(extension), "metric_rows": 162, "categories": 27, "master_manifest_updated": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("--include-source", action="append", default=[])
    parser.add_argument("--archive-root-docs", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    if not args.seal_only and args.run_dir is None:
        parser.error("--run-dir is required unless --seal-only")
    if args.seal_only and args.check_only:
        parser.error("--seal-only and --check-only are mutually exclusive")
    try:
        print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError, TypeError) as error:
        print(json.dumps({"status": "extension_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
