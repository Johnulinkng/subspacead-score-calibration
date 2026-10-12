"""Replay the delivered DINO/SubspaceAD cache and selected local CPU inference.

No training, model downloads, or network access are needed. Metric temporary
files and the verification receipt are always outside the delivered folder.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
EXTENSION = "优化实验/DINO小型SubspaceAD验证"
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path: Path) -> list[dict[str, Any]]:
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
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping delivery member: {relative}")
    return path


def verify_cache(package: Path) -> tuple[dict[str, Any], Path, list[tuple[str, str]]]:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from industrial_anomaly_lab.subspace_features import image_aggregates, load_subspace_model

    extension = package / EXTENSION
    information = read(extension / "extension_manifest.json")
    if information.get("status") != "completed" or information.get("test_images") != 3887 or information.get("new_metric_rows") != 54:
        raise ValueError("A completed full 27-category SubspaceAD delivery is required")
    run = extension / "run"
    config = read(run / "run_config.json")
    if read(run / "run_state.json").get("status") != "completed":
        raise ValueError("Delivered run is incomplete")
    for relative, expected in config["source_sha256"].items():
        if digest(member(package / "code", relative)) != expected:
            raise ValueError(f"Executed source and delivered source differ: {relative}")
    for relative, expected in config["environment_sha256"].items():
        if digest(member(package / "code", relative)) != expected:
            raise ValueError(f"Frozen dependency declaration differs: {relative}")
    for relative, expected in config["vendor_source_sha256"].items():
        if digest(member(package / "code/vendor/SubspaceAD", relative)) != expected:
            raise ValueError(f"Pinned public source differs: {relative}")
    weights = read(extension / "weights_manifest.json")
    weight_directory = (package / weights["directory"]).resolve(strict=True)
    if not weight_directory.is_relative_to(package):
        raise ValueError("Weight directory escapes delivery")
    for record in weights["files"]:
        path = member(package, record["path"])
        if path.stat().st_size != int(record["size_bytes"]) or digest(path) != record["sha256"]:
            raise ValueError("DINO checkpoint/configuration/license digest mismatch")
    backbone = config["backbone"]
    if digest(member(weight_directory, "model.safetensors")) != backbone["checkpoint_sha256"]:
        raise ValueError("Delivered checkpoint differs from frozen executed checkpoint")
    for relative, expected in backbone["files_sha256"].items():
        if digest(member(weight_directory, relative)) != expected:
            raise ValueError("Frozen DINO model artifact digest mismatch")
    tasks = [(str(d), str(c)) for d, c in config["expected_matrix"]["categories"]]
    if len(tasks) != 27 or len(set(tasks)) != 27:
        raise ValueError("Invalid frozen category matrix")
    all_rows = read_lines(run / "results/category_metrics.jsonl")
    row_index = {(r["dataset"], r["category"], r["variant"]): r for r in all_rows}
    expected = {(d, c, variant) for d, c in tasks for variant in ("subspace_max", "subspace_top1p")}
    if len(all_rows) != 54 or set(row_index) != expected:
        raise ValueError("Invalid full metric matrix")
    seals = 0
    total = 0
    identities = set()
    summaries = []
    for dataset, category in tasks:
        directory = run / f"categories/{dataset}/{category}"
        seal = read(directory / "completion.json")
        if seal.get("status") != "completed":
            raise ValueError("Incomplete category completion seal")
        for relative, expected_hash in seal["sha256"].items():
            if digest(member(run, relative)) != expected_hash:
                raise ValueError(f"Category evidence hash mismatch: {relative}")
            seals += 1
        fitted = load_subspace_model(member(run, f"categories/{dataset}/{category}/model.npz"))
        model_path = member(run, f"categories/{dataset}/{category}/model.npz")
        model_seal = read(directory / "model_integrity.json")
        if digest(model_path) != model_seal["sha256"] or model_path.stat().st_size != int(model_seal["size_bytes"]):
            raise ValueError("PCA model's immediate integrity seal differs")
        train = read(directory / "train_manifest.json")
        test = read(directory / "test_inventory.json")
        if fitted.feature_dim != 384 or fitted.count != len(train) * 1024 or fitted.variance_target != 0.99:
            raise ValueError("PCA feature dimension, normal-training count, or frozen target mismatch")
        if fitted.retained_variance + 1e-12 < fitted.variance_target:
            raise ValueError("PCA model did not reach the frozen variance target")
        if fitted.metadata.get("train_manifest_sha256") != digest(directory / "train_manifest.json") or fitted.metadata.get("checkpoint_sha256") != backbone["checkpoint_sha256"]:
            raise ValueError("PCA model metadata differs from authenticated normal training and backbone")
        original_train = member(package, f"models/{dataset}/{category}/train_manifest.json")
        original_test = member(package, f"models/{dataset}/{category}/test_inventory.json")
        if digest(original_train) != digest(directory / "train_manifest.json") or digest(original_test) != digest(directory / "test_inventory.json"):
            raise ValueError("New train/test manifests differ from original frozen inputs")
        samples = read_lines(directory / "samples.jsonl")
        frozen_samples = {r["sample_id"]: r for r in test}
        if len(frozen_samples) != len(test):
            raise ValueError("Duplicate sample in frozen test inventory")
        if len(samples) != len(test):
            raise ValueError("Sample count differs from category frozen test inventory")
        labels, maximum, top1p = [], [], []
        for sample in samples:
            identity = (sample["dataset"], sample["category"], sample["sample_id"])
            if identity in identities or identity[:2] != (dataset, category):
                raise ValueError("Duplicate or misfiled test sample")
            frozen = frozen_samples.get(sample["sample_id"])
            if frozen is None or int(frozen["label"]) != int(sample["label"]) or frozen["image_sha256"] != sample["image_sha256"]:
                raise ValueError("Test sample label/source-image hash differs from original input inventory")
            identities.add(identity)
            path = member(run, sample["raw_maps"]["subspace"])
            mask_path = member(run, sample["mask_path"])
            if digest(path) != sample["raw_map_sha256"]["subspace"] or digest(mask_path) != sample["mask_sha256"]:
                raise ValueError("Sample map/mask digest mismatch")
            raw = np.load(path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)
            if raw.dtype != np.float32 or raw.shape != (256, 256) or not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError("Expected finite nonnegative float32 256-square raw map")
            if mask.dtype != np.bool_ or mask.shape != raw.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError("Evaluation mask must be a boolean binary 256-square array")
            # The source image label is authoritative. A tiny anomalous source
            # region may vanish under the frozen nearest-neighbor GT resize;
            # that does not turn the image into a normal sample.
            if int(sample["label"]) == 0 and bool(mask.any()):
                raise ValueError("Normal-labeled sample has a nonempty evaluation mask")
            score = image_aggregates(raw)
            if score["top_count"] != 655:
                raise ValueError("Fixed top1-percent count differs from 655")
            saved = sample["image_scores"]["subspace"]
            if float(saved["max"]) != score["max"] or float(saved["top1p"]) != score["top1p"]:
                raise ValueError("Cached image score differs from unmodified raw map")
            labels.append(int(sample["label"]))
            maximum.append(score["max"])
            top1p.append(score["top1p"])
        total += len(samples)
        image_errors = {}
        for suffix, scores in (("max", maximum), ("top1p", top1p)):
            row = row_index[(dataset, category, "subspace_" + suffix)]
            value = float(roc_auc_score(labels, scores))
            error = abs(value - float(row["image_auroc"]))
            if error > 1e-12 or int(row["test_images"]) != len(samples):
                raise ValueError("Independent cached image AUROC replay mismatch")
            image_errors[suffix] = error
        a = row_index[(dataset, category, "subspace_max")]
        b = row_index[(dataset, category, "subspace_top1p")]
        if any(float(a[k]) != float(b[k]) for k in METRICS[1:]):
            raise ValueError("Top1-percent aggregation changed a localization metric")
        summaries.append({"dataset": dataset, "category": category, "normal_patch_count": fitted.count, "pca_dimensions": fitted.k,
                          "retained_variance": fitted.retained_variance, "image_auroc_replay_errors": image_errors})
        print(json.dumps({"cache_completed_category": category, "dataset": dataset, "test_samples_verified": total}), flush=True)
    if total != 3887:
        raise ValueError("Full sample coverage differs from 3,887")
    return {"status": "verified", "categories": 27, "test_images": total, "metric_rows": 54, "category_evidence_hashes": seals,
            "pca_models": 27, "weight_files": len(weights["files"]), "all_raw_maps_and_masks_verified": True,
            "all_cached_image_auroc_independently_replayed": True, "category_summaries": summaries}, run, tasks


def verify_inference(package: Path, run: Path, sample_count: int, atol: float, rtol: float) -> list[dict[str, Any]]:
    if not 0 <= sample_count <= 54:
        raise ValueError("Inference sample count must be in 0..54")
    if not sample_count:
        return []
    import numpy as np
    import cv2
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model, render_subspace_map

    cv2.setNumThreads(4)

    config = read(run / "run_config.json")
    weights = read(package / EXTENSION / "weights_manifest.json")
    backbone = config["backbone"]
    extractor = LocalDinoExtractor(package / weights["directory"], expected_checkpoint_sha256=backbone["checkpoint_sha256"],
                                   expected_artifact_sha256=backbone["files_sha256"], image_res=448, layers=(-4, -5), device="cpu", cpu_threads=4)
    selected = read(package / EXTENSION / "visualization_manifest.json")["samples"][:sample_count]
    if len(selected) != sample_count:
        raise ValueError("Requested selected inputs missing")
    models = {}
    output = []
    for sample in selected:
        identity = (sample["dataset"], sample["category"])
        if identity not in models:
            models[identity] = load_subspace_model(member(run, f"categories/{identity[0]}/{identity[1]}/model.npz"))
        image = member(package, sample["source_image_copy"])
        if digest(image) != sample["source_image_sha256"]:
            raise ValueError("Selected original source-image digest mismatch")
        features = extractor(image)
        fitted = models[identity]
        scores = fitted.reconstruction_scores(features.reshape(-1, 384))
        predicted = render_subspace_map(scores.reshape(32, 32), model_size=448, output_size=256)
        raw_record = sample["raw_maps"]["subspace"]
        old = np.load(member(run, raw_record["path"]), allow_pickle=False)
        if digest(member(run, raw_record["path"])) != raw_record["sha256"]:
            raise ValueError("Selected source map digest mismatch")
        absolute = np.abs(predicted.astype(np.float64) - old.astype(np.float64))
        passed = bool(np.allclose(predicted, old, atol=atol, rtol=rtol))
        item = {"dataset": identity[0], "category": identity[1], "sample_id": sample["sample_id"], "status": "verified" if passed else "failed",
                "device": "cpu", "maximum_absolute_error": float(absolute.max()), "mean_absolute_error": float(absolute.mean()), "atol": atol, "rtol": rtol,
                "tolerance_basis": "explicit float32 CPU/GPU execution tolerance; stored raw maps and cache metrics are unchanged"}
        if not passed:
            raise ValueError(f"Selected CPU inference replay mismatch: {item}")
        output.append(item)
        print(json.dumps({"inference_completed": len(output), "sample_id": sample["sample_id"], "maximum_absolute_error": item["maximum_absolute_error"]}), flush=True)
    del extractor, models
    gc.collect()
    return output


def verify_metrics(run: Path, tasks: list[tuple[str, str]], category_count: int, work: Path) -> list[dict[str, Any]]:
    if not 0 <= category_count <= 27:
        raise ValueError("Metrics category count must be in 0..27")
    if not category_count:
        return []
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    from industrial_anomaly_lab.subspace_features import image_aggregates

    output = []
    for dataset, category in tasks[:category_count]:
        directory = run / f"categories/{dataset}/{category}"
        samples = read_lines(directory / "samples.jsonl")
        rows = {r["variant"]: r for r in read(directory / "metrics.json")}
        maps = [member(run, r["raw_maps"]["subspace"]) for r in samples]
        masks = [member(run, r["mask_path"]) for r in samples]
        labels = [int(r["label"]) for r in samples]
        measured = evaluate_from_files(maps, masks, labels, temp_root=work)
        top1p = [image_aggregates(np.load(p, allow_pickle=False))["top1p"] for p in maps]
        top1p_auroc = float(roc_auc_score(labels, top1p))
        for suffix in ("max", "top1p"):
            regenerated = dict(measured)
            if suffix == "top1p":
                regenerated["image_auroc"] = top1p_auroc
            variant = "subspace_" + suffix
            errors = {k: abs(float(regenerated[k]) - float(rows[variant][k])) for k in METRICS}
            if any(error > 1e-12 for error in errors.values()):
                raise ValueError(f"Independent full metric replay mismatch: {dataset}/{category}/{variant}: {errors}")
            output.append({"dataset": dataset, "category": category, "variant": variant, "status": "verified", "absolute_errors": errors})
        print(json.dumps({"metrics_completed_category": category, "dataset": dataset, "metric_rows_verified": len(output)}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--inference-samples", type=int, default=2)
    parser.add_argument("--metrics-categories", type=int, default=0)
    parser.add_argument("--inference-atol", type=float, default=1e-4)
    parser.add_argument("--inference-rtol", type=float, default=1e-4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    package = args.package_root.resolve()
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.gettempdir()) / "subspace-delivery-metric-replay"
    if work.is_relative_to(package) or args.receipt and args.receipt.resolve().is_relative_to(package):
        parser.error("Work directory and receipt must remain outside the immutable delivery folder")
    if not 0 <= args.inference_atol <= 1e-3 or not 0 <= args.inference_rtol <= 1e-3:
        parser.error("Inference tolerances must be finite and within 0..1e-3")
    work.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(package / "code/src"), str(package / "code/scripts")]
    started = time.monotonic()
    try:
        integrity, run, tasks = verify_cache(package)
        inference = verify_inference(package, run, args.inference_samples, args.inference_atol, args.inference_rtol)
        metrics = verify_metrics(run, tasks, args.metrics_categories, work)
        result = {"status": "verified", "cache_integrity": integrity, "inference": inference, "metric_replay": metrics,
                  "elapsed_seconds": time.monotonic() - started, "gradient_training_performed": False, "network_access_required": False, "gpu_used": False, "cpu_threads": 4}
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(json.dumps({"status": "verification_failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
