"""Verify the same-DINO controls and replay frozen cache or local CPU inference.

This verifier performs no fitting, gradient training, download, or network call.
Temporary metric sort files and receipts must remain outside the delivery.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
EXTENSION = "优化实验/同骨干Subspace对照-20261014"
METHODS = ("official_cpu_two_pass", "numpy_cpu_two_pass", "current_cpu_one_pass")
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


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


def verify_integrity(package: Path, extension: Path, verify_master: bool) -> dict[str, Any]:
    info = read(member(extension, "extension_manifest.json"))
    if info.get("status") != "completed" or info.get("categories") != 27 or info.get("metric_rows") != 162 or info.get("scientific_claim_eligible") is not False:
        raise ValueError("Complete frozen 27-category/162-row extension is required")
    declared = set()
    for row in info["files"]:
        relative = row["path"]
        if relative in declared:
            raise ValueError("Duplicate extension manifest member")
        declared.add(relative)
        path = member(extension, relative)
        if path.stat().st_size != int(row["size_bytes"]) or digest(path) != row["sha256"]:
            raise ValueError(f"Extension hash mismatch: {relative}")
    actual = {p.relative_to(extension).as_posix() for p in extension.rglob("*") if p.is_file()}
    if actual != declared | {"extension_manifest.json"}:
        raise ValueError("Extension membership changed after its seal")
    historical = member(extension, "history/DELIVERY-MANIFEST.json")
    if digest(historical) != info["previous_delivery_manifest_sha256"]:
        raise ValueError("Historical manifest snapshot differs from authenticated original")
    old_manifest = read(historical)
    archived = []
    allowed_doc_changes = {"项目总览.html", "先读我.txt", "实验说明.txt", "Verify.ps1"}
    for row in old_manifest["files"]:
        path = member(package, row["path"])
        if path.stat().st_size == int(row["size_bytes"]) and digest(path) == row["sha256"]:
            continue
        if row["path"] not in allowed_doc_changes:
            raise ValueError(f"Historical experiment evidence/source/license was replaced: {row['path']}")
        archive = member(extension, "history/" + row["path"])
        if archive.stat().st_size != int(row["size_bytes"]) or digest(archive) != row["sha256"]:
            raise ValueError(f"Changed root document lacks exact archived bytes: {row['path']}")
        archived.append(row["path"])
    master_members = None
    if verify_master:
        master = read(member(package, "DELIVERY-MANIFEST.json"))
        master_expected = set()
        for row in master["files"]:
            if row["path"] in master_expected:
                raise ValueError("Duplicate master manifest member")
            master_expected.add(row["path"])
            path = member(package, row["path"])
            if path.stat().st_size != int(row["size_bytes"]) or digest(path) != row["sha256"]:
                raise ValueError(f"Master manifest hash mismatch: {row['path']}")
        all_members = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
        if all_members != master_expected | {"DELIVERY-MANIFEST.json"}:
            raise ValueError("Master manifest membership mismatch")
        master_members = len(master_expected)
    return {"status": "verified", "extension_members": len(declared), "historical_members_preserved": len(old_manifest["files"]), "root_documents_archived_before_change": archived, "master_manifest_verified": verify_master, "master_members": master_members}


def verify_sources(package: Path, extension: Path, run: Path) -> dict[str, Any]:
    config = read(member(run, "run_config.json"))
    code = extension / "code"
    protocol = config["protocol"]
    if any(protocol.get(key) != value for key, value in {"image_res": 448, "output_size": 256, "layers": [-4, -5], "feature_dim": 384, "explained_variance": 0.99, "batch_size": 1, "cpu_threads": 4, "device": "cpu"}.items()):
        raise ValueError("Frozen same-backbone control geometry changed")
    protocol_source = member(code, "configs/route1_controls_20261014.json")
    if digest(protocol_source) != config["protocol_sha256"] or read(protocol_source) != protocol:
        raise ValueError("Delivered reproduction protocol differs from executed protocol")
    for relative, expected in config["source_sha256"].items():
        if digest(member(code, relative)) != expected:
            raise ValueError(f"Executed source differs from delivered source: {relative}")
    for relative, expected in config.get("environment_sha256", {}).items():
        if digest(member(code, relative)) != expected:
            raise ValueError(f"Frozen dependency declaration differs: {relative}")
    for relative, expected in config.get("vendor_source_sha256", {}).items():
        if digest(member(code / "vendor/SubspaceAD", relative)) != expected:
            raise ValueError(f"Official source snapshot differs: {relative}")
    weights = read(member(extension, "weights_manifest.json"))
    for row in weights["files"]:
        path = member(package, row["path"])
        if path.stat().st_size != int(row["size_bytes"]) or digest(path) != row["sha256"]:
            raise ValueError("DINO weights, configuration, or license changed")
    backbone = config["backbone"]
    weight_directory = (package / weights["directory"]).resolve(strict=True)
    if not weight_directory.is_relative_to(package):
        raise ValueError("Weight directory escapes package")
    if digest(member(weight_directory, "model.safetensors")) != backbone["checkpoint_sha256"]:
        raise ValueError("Run and delivered DINO checkpoint differ")
    for relative, expected in backbone["files_sha256"].items():
        if digest(member(weight_directory, relative)) != expected:
            raise ValueError("Run and delivered DINO artifacts differ")
    old_config = read(member(package, "优化实验/DINO小型SubspaceAD验证/run/run_config.json"))
    if config["backbone"]["checkpoint_sha256"] != old_config["backbone"]["checkpoint_sha256"]:
        raise ValueError("Control and historical optimized run use different weights")
    return {"status": "verified", "source_files": len(config["source_sha256"]), "vendor_files": len(config.get("vendor_source_sha256", {})), "same_historical_dino_checkpoint": True, "weight_files": len(weights["files"])}


def compare_models(models: dict[str, Any], dataset: str, category: str) -> list[dict[str, Any]]:
    import numpy as np
    comparisons = []
    for left, right in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
        a, b = models[left], models[right]
        projector_a, projector_b = a.components @ a.components.T, b.components @ b.components.T
        common = min(a.k, b.k)
        comparisons.append({"dataset": dataset, "category": category, "reference": left, "method": right, "reference_rank": a.k, "method_rank": b.k, "rank_equal": a.k == b.k, "mean_max_absolute_error": float(np.max(np.abs(a.mean - b.mean))), "eigenvalues_common_rank_max_absolute_error": float(np.max(np.abs(a.eigenvalues[:common] - b.eigenvalues[:common]))), "projection_matrix_max_absolute_error": float(np.max(np.abs(projector_a - projector_b))), "projection_matrix_frobenius_error": float(np.linalg.norm(projector_a - projector_b)), "comparison_ignores_eigenvector_sign": True})
    return comparisons


def verify_cache(package: Path, extension: Path, run: Path) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from industrial_anomaly_lab.subspace_features import image_aggregates, load_subspace_model

    config = read(member(run, "run_config.json"))
    if read(member(run, "run_state.json")).get("status") != "completed":
        raise ValueError("Delivered full run did not complete")
    tasks = [(str(d), str(c)) for d, c in config["expected_matrix"]["categories"]]
    if len(tasks) != 27 or len(set(tasks)) != 27 or tuple(config["expected_matrix"]["methods"]) != METHODS:
        raise ValueError("Unexpected category/method matrix")
    metric_rows = lines(member(run, "results/category_metrics.jsonl"))
    metric_index = {(r["dataset"], r["category"], r.get("variant", r.get("method"))): r for r in metric_rows}
    expected = {(d, c, m + "_" + score) for d, c in tasks for m in METHODS for score in ("max", "top1p")}
    if len(metric_rows) != 162 or set(metric_index) != expected:
        raise ValueError("Incomplete or duplicated 162-row metric matrix")
    total, train_total, maps_total, seal_count = 0, 0, 0, 0
    identities = set()
    historical_run = package / "优化实验/DINO小型SubspaceAD验证/run"
    historical_samples = { (r["dataset"], r["category"], r["sample_id"]): r for r in lines(member(historical_run, "results/samples.jsonl")) }
    if len(historical_samples) != 3887:
        raise ValueError("Historical authenticated GT inventory is incomplete")
    pair_model_rows, map_rows, image_rows = [], [], []
    for dataset, category in tasks:
        cat = run / f"categories/{dataset}/{category}"
        completion = read(member(cat, "completion.json"))
        if completion.get("status") != "completed":
            raise ValueError("Incomplete category completion seal")
        for relative, expected_hash in completion["sha256"].items():
            if digest(member(run, relative)) != expected_hash:
                raise ValueError(f"Immediate category evidence hash differs: {relative}")
            seal_count += 1
        train, inventory = read(member(cat, "train_manifest.json")), read(member(cat, "test_inventory.json"))
        for filename in ("train_manifest.json", "test_inventory.json"):
            if digest(member(cat, filename)) != digest(member(package, f"models/{dataset}/{category}/{filename}")):
                raise ValueError("Train/test inventories differ from original delivery")
        train_total += len(train)
        models = {method: load_subspace_model(member(cat, f"models/{method}.npz")) for method in METHODS}
        for method_index, (method, model) in enumerate(models.items()):
            if model.feature_dim != 384 or model.count != len(train) * 1024 or model.variance_target != 0.99 or model.retained_variance + 1e-12 < 0.99:
                raise ValueError(f"Frozen normal-training geometry/PCA target differs: {method}")
            metadata = model.metadata
            if metadata.get("fit_device") != "cpu" or metadata.get("fit_dtype") != "float64" or metadata.get("input_dtype") != "float32" or metadata.get("pass_count") != (2, 2, 1)[method_index] or metadata.get("token_count") != model.count or metadata.get("batch_count") != len(train) or metadata.get("control_variant") != method:
                raise ValueError("Saved PCA model does not prove the frozen CPU input/pass contract")
        if len({model.metadata.get("input_sha256") for model in models.values()}) != 1 or not models[METHODS[0]].metadata.get("input_sha256") or len({model.metadata.get("batch_partition_sha256") for model in models.values()}) != 1:
            raise ValueError("Three PCA controls did not consume identical token bytes/partitions")
        official_sha = config["vendor_source_sha256"]["src/subspacead/core/pca.py"]
        if models[METHODS[0]].metadata.get("official_pca_source_sha256") != official_sha:
            raise ValueError("Official PCA model metadata lacks the pinned author source identity")
        pair_model_rows.extend(compare_models(models, dataset, category))
        samples = lines(member(cat, "samples.jsonl"))
        inventory_index = {r["sample_id"]: r for r in inventory}
        if len(samples) != len(inventory) or len(inventory_index) != len(inventory):
            raise ValueError("Test inventory has missing or duplicated samples")
        labels = []
        scores: dict[str, dict[str, list[float]]] = {method: {score: [] for score in ("max", "top1p")} for method in METHODS}
        pair_maps = {(a, b): {"maximum_absolute_error": 0.0, "sum_absolute_error": 0.0, "pixels": 0, "maps": 0} for a, b in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2]))}
        for sample in samples:
            identity = (sample["dataset"], sample["category"], sample["sample_id"])
            if identity in identities or identity[:2] != (dataset, category):
                raise ValueError("Duplicate or misfiled test sample")
            identities.add(identity)
            frozen = inventory_index.get(sample["sample_id"])
            if frozen is None or frozen["image_sha256"] != sample["image_sha256"] or int(frozen["label"]) != int(sample["label"]):
                raise ValueError("Test image hash/label differs from frozen inventory")
            if set(sample["raw_maps"]) != set(METHODS) or set(sample["raw_map_sha256"]) != set(METHODS):
                raise ValueError("A sample lacks the three paired maps")
            mask_path = member(run, sample["mask_path"])
            if digest(mask_path) != sample["mask_sha256"]:
                raise ValueError("GT seal mismatch")
            mask = np.load(mask_path, allow_pickle=False)
            if mask.dtype != np.bool_ or mask.shape != (256, 256) or int(sample["label"]) == 0 and bool(mask.any()):
                raise ValueError("Invalid common evaluation GT")
            historical = historical_samples.get(identity)
            if historical is None or int(historical["label"]) != int(sample["label"]) or historical["image_sha256"] != sample["image_sha256"]:
                raise ValueError("New sample does not match the frozen historical image/GT identity")
            historical_mask = member(historical_run, historical["mask_path"])
            if digest(historical_mask) != historical["mask_sha256"] or not np.array_equal(mask, np.load(historical_mask, allow_pickle=False)):
                raise ValueError("New 256-square GT differs from the authenticated historical evaluation mask")
            raw_maps = {}
            for method in METHODS:
                path = member(run, sample["raw_maps"][method])
                if digest(path) != sample["raw_map_sha256"][method]:
                    raise ValueError("Raw-map seal mismatch")
                raw = np.load(path, allow_pickle=False)
                if raw.dtype != np.float32 or raw.shape != (256, 256) or not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError("Expected finite nonnegative raw float32 256-square map")
                values = image_aggregates(raw)
                if values["top_count"] != 655:
                    raise ValueError("Fixed image score aggregation changed")
                for score in ("max", "top1p"):
                    if values[score] != float(sample["image_scores"][method][score]):
                        raise ValueError("Cached image score differs from raw map")
                    scores[method][score].append(values[score])
                raw_maps[method] = raw
                maps_total += 1
            for (left, right), result in pair_maps.items():
                delta = np.abs(raw_maps[left].astype(np.float64) - raw_maps[right].astype(np.float64))
                result["maximum_absolute_error"] = max(result["maximum_absolute_error"], float(delta.max()))
                result["sum_absolute_error"] += float(delta.sum())
                result["pixels"] += delta.size
                result["maps"] += 1
            labels.append(int(sample["label"]))
        for method in METHODS:
            for score in ("max", "top1p"):
                row = metric_index[(dataset, category, method + "_" + score)]
                value = float(roc_auc_score(labels, scores[method][score]))
                error = abs(value - float(row["image_auroc"]))
                if error > 1e-12 or int(row["test_images"]) != len(samples) or row.get("status") != "completed" or any(not math.isfinite(float(row[k])) or not 0 <= float(row[k]) <= 1 for k in METRICS):
                    raise ValueError("Cached image AUROC/sample-count/metric validity mismatch")
                image_rows.append({"dataset": dataset, "category": category, "variant": method + "_" + score, "absolute_error": error})
            a, b = metric_index[(dataset, category, method + "_max")], metric_index[(dataset, category, method + "_top1p")]
            if any(float(a[k]) != float(b[k]) for k in METRICS[1:]):
                raise ValueError("Image aggregation changed localization metrics")
        for (left, right), result in pair_maps.items():
            map_rows.append({"dataset": dataset, "category": category, "reference": left, "method": right, "maps": result["maps"], "maximum_absolute_error": result["maximum_absolute_error"], "mean_absolute_error": result["sum_absolute_error"] / result["pixels"]})
        total += len(samples)
        print(json.dumps({"cache_completed_category": category, "dataset": dataset, "test_images_verified": total, "raw_maps_verified": maps_total}), flush=True)
    if total != 3887 or train_total != 12288 or maps_total != 11661:
        raise ValueError("Complete normal/test/raw-map coverage differs")
    selected = read(member(extension, "selected_samples.json"))
    old_selected = read(member(package, "report/visualization_manifest.json"))["samples"]
    identity = lambda r: (r["dataset"], r["category"], r["sample_id"])
    if selected.get("figure_count") != 54 or len(selected["samples"]) != 54 or {identity(r) for r in selected["samples"]} != {identity(r) for r in old_selected}:
        raise ValueError("New paired visuals changed the original mechanical sample selection")
    for record in selected["samples"]:
        if digest(member(extension, record["figure"])) != record["figure_sha256"] or digest(member(package, record["source_image_copy"])) != record["source_image_sha256"] or record.get("evaluation_raw_maps_modified") is not False:
            raise ValueError("Selected paired figure/source identity or evaluation boundary changed")
    verify_metric_tables(extension, metric_rows)
    return {"status": "verified", "categories": 27, "normal_train_images": train_total, "test_images": total, "raw_maps": maps_total, "evaluation_masks": total, "pca_models": 81, "metric_rows": 162, "category_evidence_hashes": seal_count, "cached_image_auroc_replays": image_rows, "pca_model_comparisons": pair_model_rows, "raw_map_comparisons": map_rows, "all_maps_masks_and_models_authenticated": True}, tasks


def verify_metric_tables(extension: Path, raw_rows: list[dict[str, Any]]) -> None:
    index = {(r["dataset"], r["category"], r.get("variant", r.get("method"))): r for r in raw_rows}
    with member(extension, "category_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    if len(table) != 162 or {(r["dataset"], r["category"], r["variant"]) for r in table} != set(index):
        raise ValueError("Report category table coverage differs from run")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        expected = index[(row["dataset"], row["category"], row["variant"])]
        if any(float(row[k]) != float(expected[k]) for k in METRICS):
            raise ValueError("Report category metrics differ from raw evidence")
        grouped[(row["dataset"], row["method"], row["score"])].append(row)
    with member(extension, "dataset_macro_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        macros = list(csv.DictReader(stream))
    if len(macros) != 12 or {(r["dataset"], r["method"], r["score"]) for r in macros} != set(grouped):
        raise ValueError("Report macro table coverage differs")
    for row in macros:
        group = grouped[(row["dataset"], row["method"], row["score"])]
        if int(row["categories"]) != len(group) or any(abs(float(row[k]) - sum(float(r[k]) for r in group) / len(group)) > 1e-15 for k in METRICS):
            raise ValueError("Report macro table is not an equal-category mean")
    category_index = {(r["dataset"], r["category"], r["method"], r["score"]): r for r in table}
    macro_index = {(r["dataset"], r["method"], r["score"]): r for r in macros}
    for filename, source_index, identity_fields in (("category_deltas.csv", category_index, ("dataset", "category")), ("dataset_macro_deltas.csv", macro_index, ("dataset",))):
        with member(extension, filename).open(encoding="utf-8-sig", newline="") as stream:
            deltas = list(csv.DictReader(stream))
        expected_keys = set()
        source_rows = table if filename.startswith("category") else macros
        for row in source_rows:
            references = () if row["method"] == METHODS[0] else (METHODS[0],) if row["method"] == METHODS[1] else METHODS[:2]
            for reference in references:
                expected_keys.add((*[row[key] for key in identity_fields], row["method"], reference, row["score"]))
        actual_keys = [(*[r[key] for key in identity_fields], r["method"], r["reference"], r["score"]) for r in deltas]
        if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
            raise ValueError("Paired delta table dropped or duplicated comparisons")
        for row in deltas:
            before = source_index[(*[row[k] for k in identity_fields], row["reference"], row["score"])]
            after = source_index[(*[row[k] for k in identity_fields], row["method"], row["score"])]
            if any(abs(float(row[k + "_delta_pp"]) - 100 * (float(after[k]) - float(before[k]))) > 1e-12 for k in METRICS):
                raise ValueError("Paired delta table differs from measured metrics")


def verify_metrics(run: Path, tasks: list[tuple[str, str]], count: int, work: Path) -> list[dict[str, Any]]:
    if not 0 <= count <= 27:
        raise ValueError("Metric category count must be in 0..27")
    if count == 0:
        return []
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from industrial_anomaly_lab.subspace_features import image_aggregates
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    output = []
    for dataset, category in tasks[:count]:
        cat = run / f"categories/{dataset}/{category}"
        samples = lines(member(cat, "samples.jsonl"))
        stored = {r.get("variant", r.get("method")): r for r in read(member(cat, "metrics.json"))}
        masks = [member(run, r["mask_path"]) for r in samples]
        labels = [int(r["label"]) for r in samples]
        for method in METHODS:
            paths = [member(run, r["raw_maps"][method]) for r in samples]
            values = evaluate_from_files(paths, masks, labels, temp_root=work)
            top1p = [image_aggregates(np.load(p, allow_pickle=False))["top1p"] for p in paths]
            image_top1p = float(roc_auc_score(labels, top1p))
            for score in ("max", "top1p"):
                rebuilt = {**values, "image_auroc": values["image_auroc"] if score == "max" else image_top1p}
                errors = {k: abs(float(rebuilt[k]) - float(stored[method + "_" + score][k])) for k in METRICS}
                if any(value > 1e-12 for value in errors.values()):
                    raise ValueError(f"Independent metric replay mismatch: {dataset}/{category}/{method}/{score}: {errors}")
                output.append({"dataset": dataset, "category": category, "variant": method + "_" + score, "status": "verified", "absolute_errors": errors})
        print(json.dumps({"metrics_completed_category": category, "dataset": dataset, "metric_rows_verified": len(output)}), flush=True)
    return output


def verify_inference(package: Path, extension: Path, run: Path, count: int, atol: float, rtol: float) -> list[dict[str, Any]]:
    if not 0 <= count <= 54:
        raise ValueError("Inference sample count must be in 0..54")
    if count == 0:
        return []
    import cv2
    import numpy as np
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model, render_subspace_map
    cv2.setNumThreads(4)
    config = read(member(run, "run_config.json"))
    backbone = config["backbone"]
    weights = read(member(extension, "weights_manifest.json"))
    extractor = LocalDinoExtractor(package / weights["directory"], expected_checkpoint_sha256=backbone["checkpoint_sha256"], expected_artifact_sha256=backbone["files_sha256"], image_res=448, layers=(-4, -5), device="cpu", cpu_threads=4)
    selected = read(member(extension, "selected_samples.json"))["samples"][:count]
    if len(selected) != count:
        raise ValueError("Missing selected source inputs")
    models, output = {}, []
    for sample in selected:
        image = member(package, sample["source_image_copy"])
        if digest(image) != sample["source_image_sha256"]:
            raise ValueError("Selected source image differs from original mechanical selection")
        features = extractor(image)
        for method in METHODS:
            key = (sample["dataset"], sample["category"], method)
            if key not in models:
                models[key] = load_subspace_model(member(run, f"categories/{key[0]}/{key[1]}/models/{method}.npz"))
            model = models[key]
            predicted = render_subspace_map(model.reconstruction_scores(features.reshape(-1, 384)).reshape(32, 32), model_size=448, output_size=256)
            stored = np.load(member(run, sample["raw_maps"][method]), allow_pickle=False)
            absolute = np.abs(predicted.astype(np.float64) - stored.astype(np.float64))
            passed = bool(np.allclose(predicted, stored, atol=atol, rtol=rtol))
            result = {"dataset": key[0], "category": key[1], "method": method, "sample_id": sample["sample_id"], "status": "verified" if passed else "failed", "device": "cpu", "maximum_absolute_error": float(absolute.max()), "mean_absolute_error": float(absolute.mean()), "atol": atol, "rtol": rtol, "tolerance_basis": "CPU/GPU float32 inference execution tolerance; stored raw maps and metrics remain exact"}
            if not passed:
                raise ValueError(f"Fresh CPU inference differs from frozen cache: {result}")
            output.append(result)
        print(json.dumps({"inference_completed_sample": sample["sample_id"], "method_maps_verified": len(output)}), flush=True)
    del extractor, models
    gc.collect()
    return output


def verify_resources(extension: Path) -> dict[str, Any]:
    trials = lines(member(extension, "资源记录/resource_trials.jsonl"))
    if not trials or any(r.get("status") in (None, "pending", "running", "initializing") for r in trials):
        raise ValueError("Actual terminal resource trials required")
    with member(extension, "resource_trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    if len(exported) != len(trials):
        raise ValueError("Resource report dropped trials, including failures")
    for original, row in zip(trials, exported, strict=True):
        for key, value in original.items():
            expected = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else "" if value is None else str(value)
            if row[key] != expected:
                raise ValueError("Resource CSV differs from retained raw trial")
    protocol = read(member(extension, "run/run_config.json"))["protocol"]
    config = read(member(extension, "run/run_config.json"))
    allowed_protocol_hashes = {config["protocol_sha256"], digest(member(extension, "run/run_config.json"))}
    successful_registered = 0
    incomplete_failed_records = []
    fit_rows = []
    for trial in trials:
        if trial.get("experiment") == "fit_only":
            fit_rows.append(trial)
        mode = trial.get("mode")
        if mode not in protocol.get("resource_modes", []):
            continue
        if trial.get("device") != "cpu" or trial.get("gpu_used") is not False or trial.get("test_content_accessed") is not False:
            raise ValueError("Registered resource receipt changed the frozen CPU/normal-only execution boundary")
        completed = trial["status"] in ("completed", "passed", "ok", "verified")
        if completed:
            raw_receipt_path = member(extension / "资源记录", trial["receipt_path"])
            raw_receipt = read(raw_receipt_path)
            if digest(raw_receipt_path) != trial["receipt_sha256"] or any(trial.get(key) != value for key, value in raw_receipt.items()) or set(trial) - set(raw_receipt) != {"receipt_path", "receipt_sha256", "measurement_scope"}:
                raise ValueError("Final resource JSONL does not match the authenticated original worker receipt")
        required_fields = ("protocol_sha256", "source_sha256", "samples", "operation_counts", "pass_feature_sha256", "checkpoint_sha256", "sample_count", "train_manifest_sha256")
        if not completed and any(key not in trial for key in required_fields):
            incomplete_failed_records.append({"dataset": trial.get("dataset"), "category": trial.get("category"), "mode": mode, "repeat": trial.get("repeat"), "status": trial["status"], "missing_fields": [key for key in required_fields if key not in trial]})
            continue
        if trial["protocol_sha256"] not in allowed_protocol_hashes or trial.get("protocol_id") != protocol["protocol_id"] or trial.get("cpu_threads") != 4 or trial["checkpoint_sha256"] != config["backbone"]["checkpoint_sha256"]:
            raise ValueError("Resource receipt protocol/backbone/threads differ from frozen execution")
        required_sources = ("scripts/run_route1_resource_trial.py", "src/industrial_anomaly_lab/subspace_features.py")
        for relative in required_sources:
            if trial["source_sha256"].get(relative) != config["source_sha256"][relative]:
                raise ValueError("Resource receipt uses another extractor or worker source")
        dataset, category = str(trial["dataset"]), str(trial["category"])
        train_path = member(extension, f"run/categories/{dataset}/{category}/train_manifest.json")
        train = read(train_path)
        expected_subset = [{"path": r["path"], "sha256": r["sha256"]} for r in train[:64]]
        if trial["train_manifest_sha256"] != digest(train_path) or trial["sample_count"] != 64 or trial["samples"] != expected_subset:
            raise ValueError("Resource receipt changed the registered first 64 normal images")
        if trial.get("input_loading") != ("eager" if "eager" in mode else "lazy") or trial.get("attention_outputs") != (mode == "attention_on"):
            raise ValueError("Resource receipt changed its registered loader/attention component")
        passes = 2 if mode.startswith("two") else 1
        counts = {"generator_calls": passes, "image_decodes": 64 * passes, "feature_forwards": 64 * passes, "normal_patch_tokens_consumed": 64 * passes * 1024}
        hashes = trial["pass_feature_sha256"]
        if trial["operation_counts"] != counts or len(hashes) != passes or len(set(hashes)) != 1 or any(not isinstance(value, str) or len(value) != 64 for value in hashes) or trial.get("warmup_attention_reference_exact_equal") is not True:
            raise ValueError("Resource receipt does not prove the registered identical-feature passes")
        successful_registered += int(completed)
    # The source image subset, extractor, and weights are identical across modes.
    # Their feature-content digest must therefore agree across all successful repeats.
    per_category_hashes = defaultdict(set)
    for trial in trials:
        if trial.get("mode") in protocol.get("resource_modes", []) and trial["status"] in ("completed", "passed", "ok", "verified"):
            per_category_hashes[(trial["dataset"], trial["category"])].add(trial["pass_feature_sha256"][0])
    if any(len(values) != 1 for values in per_category_hashes.values()):
        raise ValueError("Registered resource modes/repeats consumed different normal feature bytes")
    expected_fit = {(str(d), str(c), method) for d, c in config["expected_matrix"]["categories"] for method in METHODS}
    if len(fit_rows) != 81 or {(r["dataset"], r["category"], r["method"]) for r in fit_rows} != expected_fit:
        raise ValueError("Actual full-category PCA timing matrix has missing or duplicate rows")
    for row in fit_rows:
        dataset, category, method = row["dataset"], row["category"], row["method"]
        metadata = read(member(extension, f"run/categories/{dataset}/{category}/diagnostics.json"))["models"][method]
        train = read(member(extension, f"run/categories/{dataset}/{category}/train_manifest.json"))
        if row["status"] != "completed" or row.get("device") != "cpu" or row.get("gpu_used") is not False or row.get("repeat") != 0 or row.get("train_images") != len(train) or float(row["elapsed_seconds"]) != float(metadata["elapsed_seconds"]) or row.get("input_sha256") != metadata["input_sha256"]:
            raise ValueError("Full-category fit timing row is not derived from its completed PCA fit")
        if row.get("measurement_scope") != "shared_process_not_per_method" or any(row.get(key) is not None for key in ("process_peak_working_set_bytes", "peak_working_set_bytes", "process_peak_commit_bytes", "process_peak_private_commit_bytes", "sampled_peak_private_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")):
            raise ValueError("A shared-process fit timing row claims attributable per-method memory")
    failed_files = sorted((extension / "资源记录/failed_attempts").rglob("*.json")) if (extension / "资源记录/failed_attempts").is_dir() else []
    failure_history = read(member(extension, "failed_resource_attempts.json"))
    matrix_state = read(member(extension, "资源记录/matrix_state.json"))
    matrix_config = read(member(extension, "资源记录/matrix_config.json"))
    if matrix_state.get("status") != "completed" or matrix_state.get("completed_trials") != 36 or matrix_state.get("failed_attempt_count") != len(failed_files) or failure_history["failed_attempt_count"] != len(failed_files) or len(failure_history["attempts"]) != len(failed_files):
        raise ValueError("Resource matrix/report omitted retained failed retry attempts")
    if matrix_config["protocol_sha256"] not in allowed_protocol_hashes or matrix_config["worker_sha256"] != config["source_sha256"]["scripts/run_route1_resource_trial.py"] or matrix_config["matrix_source_sha256"] != digest(member(extension, "code/scripts/run_route1_resource_matrix.py")):
        raise ValueError("Resource matrix was produced by another protocol/worker/orchestrator")
    failure_index = {row["evidence_path"]: row for row in failure_history["attempts"]}
    if set(failure_index) != {path.relative_to(extension).as_posix() for path in failed_files}:
        raise ValueError("Failed resource attempt report differs from its retained receipt files")
    for path in failed_files:
        row = failure_index[path.relative_to(extension).as_posix()]
        receipt = read(path)
        if digest(path) != row["sha256"] or receipt != row["receipt"] or receipt.get("status") not in ("failed", "oom", "blocked", "resource_gate_blocked"):
            raise ValueError("Failed resource attempt receipt was not retained faithfully")
    registered = {(str(d), str(c), str(mode), repeat) for d, c in protocol.get("resource_categories", []) for mode in protocol.get("resource_modes", []) for repeat in range(int(protocol.get("resource_repeats", 0)))}
    observed = [(str(row.get("dataset", "")), str(row.get("category", "")), str(row.get("mode", "")), int(row.get("repeat", row.get("trial_index", -1)))) for row in trials if row.get("mode") in protocol.get("resource_modes", [])]
    if registered and (set(observed) != registered or len(observed) != len(registered)):
        raise ValueError("Resource result matrix omitted registered attempts")
    def grouping(row: dict[str, Any]) -> tuple[str, ...]:
        values = (row.get("experiment", row.get("family", row.get("scope", "unclassified"))), row.get("dataset", ""), row.get("category", ""), row.get("method", row.get("variant", "")), row.get("loader", row.get("input_loading", "")), row.get("attention", row.get("output_attentions", row.get("attention_outputs", ""))), row.get("train_images", row.get("normal_train_images", row.get("sample_count", row.get("subset_count", "")))))
        return tuple(str(value) for value in values)
    groups = defaultdict(list)
    for trial in trials:
        groups[grouping(trial)].append(trial)
    with member(extension, "resource_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    key_fields = ("experiment", "dataset", "category", "method", "loader", "attention", "train_images")
    if len(summaries) != len(groups) or {tuple(r[k] for k in key_fields) for r in summaries} != set(groups):
        raise ValueError("Resource summary grouping differs from raw trials")
    alias_fields = {"elapsed_seconds": ("elapsed_seconds", "fit_seconds"), "fit_seconds": ("fit_seconds",), "total_process_seconds": ("total_process_seconds",), "feature_forward_seconds": ("feature_forward_seconds",), "image_decode_seconds": ("image_decode_seconds",), "pca_total_kernel_seconds": ("pca_total_kernel_seconds",), "process_peak_working_set_bytes": ("process_peak_working_set_bytes", "peak_working_set_bytes"), "process_peak_commit_bytes": ("process_peak_commit_bytes", "process_peak_private_commit_bytes", "peak_private_commit_bytes"), "sampled_peak_private_bytes": ("sampled_peak_private_bytes",), "cuda_peak_allocated_bytes": ("cuda_peak_allocated_bytes",), "cuda_peak_reserved_bytes": ("cuda_peak_reserved_bytes",)}
    for summary in summaries:
        group = groups[tuple(summary[k] for k in key_fields)]
        successes = [r for r in group if r["status"] in ("completed", "passed", "ok", "verified")]
        if int(summary["attempts"]) != len(group) or int(summary["successful_trials"]) != len(successes) or int(summary["failed_or_blocked_trials"]) != len(group) - len(successes):
            raise ValueError("Resource summary suppressed failures")
        for field, aliases in alias_fields.items():
            values = [next((r[key] for key in aliases if r.get(key) is not None), None) for r in successes]
            values = [float(value) for value in values if value is not None]
            if int(summary[field + "_measured_trials"]) != len(values):
                raise ValueError("Resource summary invented or dropped measurements")
            for suffix, expected in (("median", statistics.median(values) if values else None), ("mean", statistics.mean(values) if values else None), ("sample_stddev", statistics.stdev(values) if len(values) > 1 else None), ("min", min(values) if values else None), ("max", max(values) if values else None)):
                exported_value = summary[field + "_" + suffix]
                if expected is None and exported_value != "" or expected is not None and (not math.isfinite(float(exported_value)) or abs(float(exported_value) - expected) > 1e-12):
                    raise ValueError("Resource summary statistic differs from raw measurements")
    with member(extension, "resource_paired_comparisons.csv").open(encoding="utf-8-sig", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    index = {}
    for row in trials:
        key = (str(row.get("experiment", row.get("family", "unclassified"))), str(row.get("dataset", "")), str(row.get("category", "")), str(row.get("mode", row.get("method", row.get("variant", "")))), str(row.get("repeat", row.get("trial_index", 0))))
        if key in index:
            raise ValueError("Duplicate resource comparison trial identity")
        index[key] = row
    component_pairs = (("one_lazy", "two_lazy"), ("one_eager", "two_eager"), ("one_lazy", "one_eager"), ("two_lazy", "two_eager"), ("one_lazy", "two_eager"), ("attention_off", "attention_on"), (METHODS[2], METHODS[1]), (METHODS[2], METHODS[0]))
    expected_pairs = {(experiment, dataset, category, current, reference, repeat) for experiment, dataset, category, current, repeat in index for method, reference in component_pairs if current == method and (experiment, dataset, category, reference, repeat) in index}
    actual_pairs = [(r["experiment"], r["dataset"], r["category"], r["method"], r["reference"], r["repeat"]) for r in pairs]
    if set(actual_pairs) != expected_pairs or len(actual_pairs) != len(expected_pairs):
        raise ValueError("Resource paired table dropped or duplicated component comparisons")
    for pair in pairs:
        base = (pair["experiment"], pair["dataset"], pair["category"])
        after = index[(*base, pair["method"], pair["repeat"])]
        before = index[(*base, pair["reference"], pair["repeat"])]
        success = after["status"] in ("completed", "passed", "ok", "verified") and before["status"] in ("completed", "passed", "ok", "verified")
        expected_status = "paired_completed" if success else "paired_failure_or_blocked"
        if pair["status"] != expected_status:
            raise ValueError("Resource paired table treats a failed trial as success")
        scope = str(before.get("measurement_scope", before.get("memory_scope", "unspecified"))) + " | " + str(after.get("measurement_scope", after.get("memory_scope", "unspecified")))
        for field, aliases in (("fit_seconds", ("fit_seconds", "elapsed_seconds")), ("total_process_seconds", ("total_process_seconds",)), ("process_peak_working_set_bytes", ("process_peak_working_set_bytes", "peak_working_set_bytes")), ("process_peak_commit_bytes", ("process_peak_commit_bytes", "process_peak_private_commit_bytes", "peak_private_commit_bytes")), ("sampled_peak_private_bytes", ("sampled_peak_private_bytes",))):
            a = next((after[k] for k in aliases if after.get(k) is not None), None)
            b = next((before[k] for k in aliases if before.get(k) is not None), None)
            allowed = success and ("bytes" not in field or "shared" not in scope.lower())
            if not allowed:
                if any(pair[field + suffix] != "" for suffix in ("_method", "_reference", "_reduction_percent")):
                    raise ValueError("Resource paired table compares failed/shared-process memory")
                continue
            for suffix, value in (("_method", a), ("_reference", b)):
                if value is None and pair[field + suffix] != "" or value is not None and float(pair[field + suffix]) != float(value):
                    raise ValueError("Resource paired values differ from trial measurements")
            expected_reduction = 100 * (float(b) - float(a)) / float(b) if a is not None and b is not None and float(b) > 0 else None
            value = pair[field + "_reduction_percent"]
            if expected_reduction is None and value != "" or expected_reduction is not None and abs(float(value) - expected_reduction) > 1e-12:
                raise ValueError("Resource paired reduction differs from measured values")
        a = next((after[k] for k in ("fit_seconds", "elapsed_seconds") if after.get(k) is not None), None)
        b = next((before[k] for k in ("fit_seconds", "elapsed_seconds") if before.get(k) is not None), None)
        expected_ratio = float(b) / float(a) if success and a is not None and b is not None and float(a) > 0 else None
        value = pair["fit_speedup_ratio"]
        if expected_ratio is None and value != "" or expected_ratio is not None and abs(float(value) - expected_ratio) > 1e-12:
            raise ValueError("Resource paired speed ratio differs from measured durations")
    interactions = read(member(extension, "resource_interactions.json"))["comparisons"]
    interaction_index = {(str(r["experiment"]), str(r["dataset"]), str(r["category"]), str(r["repeat"])): r for r in interactions}
    expected_interactions = {(experiment, dataset, category, repeat) for experiment, dataset, category, mode, repeat in index if mode == "two_eager" and all((experiment, dataset, category, component, repeat) in index for component in ("one_eager", "two_lazy", "one_lazy"))}
    if set(interaction_index) != expected_interactions or len(interactions) != len(interaction_index):
        raise ValueError("The 2x2 interaction table dropped or duplicated registered groups")
    for key, row in interaction_index.items():
        experiment, dataset, category, repeat = key
        components = {mode: index[(experiment, dataset, category, mode, repeat)] for mode in ("two_eager", "one_eager", "two_lazy", "one_lazy")}
        success = all(record["status"] in ("completed", "passed", "ok", "verified") for record in components.values())
        if row["status"] != ("completed" if success else "failed_or_blocked_component"):
            raise ValueError("The 2x2 interaction table treats a failed component as successful")
        for field, aliases in (("fit_seconds", ("fit_seconds", "elapsed_seconds")), ("total_process_seconds", ("total_process_seconds",))):
            values = {mode: next((record[k] for k in aliases if record.get(k) is not None), None) for mode, record in components.items()}
            valid = success and all(value is not None for value in values.values())
            expected = float(values["one_eager"]) + float(values["two_lazy"]) - float(values["two_eager"]) - float(values["one_lazy"]) if valid else None
            value = row[field + "_interaction_saving"]
            if expected is None and value is not None or expected is not None and (value is None or abs(float(value) - expected) > 1e-12):
                raise ValueError("The 2x2 interaction does not match its four measured runtime values")
            expected_percent = 100 * expected / float(values["two_eager"]) if valid and float(values["two_eager"]) > 0 else None
            value = row[field + "_interaction_percent_of_two_eager"]
            if expected_percent is None and value is not None or expected_percent is not None and (value is None or abs(float(value) - expected_percent) > 1e-12):
                raise ValueError("The 2x2 interaction percentage does not match its baseline runtime")
    info = read(member(extension, "extension_manifest.json"))
    if info["resource_trials"] != len(trials) or info["resource_failed_retry_attempts_retained"] != len(failed_files):
        raise ValueError("Extension manifest resource/failure counts differ from retained evidence")
    return {"status": "verified", "trials": len(trials), "failed_or_blocked_final_trials": sum(r["status"] not in ("completed", "passed", "ok", "verified") for r in trials), "failed_retry_attempts_retained": len(failed_files), "full_category_fit_timings_authenticated": len(fit_rows), "successful_registered_trials_protocol_authenticated": successful_registered, "incomplete_failed_receipts_retained_without_comparison": incomplete_failed_records, "all_resource_matrix_failures_preserved": True, "source_measurement_scopes_retained": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--inference-samples", type=int, default=2)
    parser.add_argument("--metrics-categories", type=int, default=0)
    parser.add_argument("--inference-atol", type=float, default=1e-4)
    parser.add_argument("--inference-rtol", type=float, default=1e-4)
    parser.add_argument("--verify-master", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    package = args.package_root.resolve()
    extension, run = package / EXTENSION, package / EXTENSION / "run"
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.gettempdir()) / "route1-delivery-metric-replay"
    if work.is_relative_to(package) or args.receipt and args.receipt.resolve().is_relative_to(package):
        parser.error("Temporary work and receipt must remain outside the delivery")
    if any(not math.isfinite(v) or not 0 <= v <= 1e-3 for v in (args.inference_atol, args.inference_rtol)):
        parser.error("Inference tolerances must be finite within 0..1e-3")
    work.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(extension / "code/src"), str(extension / "code/scripts")]
    started = time.monotonic()
    try:
        integrity = verify_integrity(package, extension, args.verify_master)
        sources = verify_sources(package, extension, run)
        cache, tasks = verify_cache(package, extension, run)
        resources = verify_resources(extension)
        inference = verify_inference(package, extension, run, args.inference_samples, args.inference_atol, args.inference_rtol)
        metrics = verify_metrics(run, tasks, args.metrics_categories, work)
        result = {"status": "verified", "integrity": integrity, "sources": sources, "cache": cache, "resources": resources, "fresh_cpu_inference": inference, "metric_replay": metrics, "elapsed_seconds": time.monotonic() - started, "network_access_required": False, "gradient_training_performed": False, "normal_training_fit_performed": False, "gpu_used": False, "cpu_threads": 4}
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError, TypeError) as error:
        print(json.dumps({"status": "verification_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
