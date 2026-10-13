"""Read-only, NumPy-only semantic audit of a completed route-1 control run.

Authenticate all category payloads and prediction coverage, compare paired
PCA state and every raw map, independently replay max/top-655 image scores
and tie-aware image AUROC. Pixel AUROC/AP/AUPRO are checked for paired
consistency here; their full replay belongs to the delivery verifier.
No backbone forward pass, PCA fitting, Torch import, or network access occurs.
The audit receipt is exclusively created outside all audited input trees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
sys.dont_write_bytecode = True

import numpy as np


METHODS = ["official_cpu_two_pass", "numpy_cpu_two_pass", "current_cpu_one_pass"]
VARIANTS = [method + "_" + score for method in METHODS for score in ("max", "top1p")]
MODEL_COMMIT = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
PCA_SHA = "cb81701be6ba2f32d629dca821862a5d83f7ffedd071522ed683ddf990ff1fcd"
CHECKPOINT_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
WEIGHT_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
METRICS = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")
MODEL_FIELDS = {
    "schema_version", "mean", "components", "eigenvalues", "count", "variance_target",
    "retained_variance", "total_variance", "eps", "metadata_json",
}
TOLERANCES = {
    "mean_atol": 1e-9, "mean_rtol": 1e-10,
    "eigenvalue_atol": 1e-9, "eigenvalue_rtol": 1e-9,
    "projector_atol": 1e-9, "projector_rtol": 1e-9,
    "score_atol": 1e-4, "score_rtol": 3e-5,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def bounded(root: Path, relative: str) -> Path:
    entry = Path(relative)
    require(not entry.is_absolute(), f"Expected relative sealed path: {relative}")
    path = (root / entry).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), f"Sealed path escapes root or is not a file: {path}")
    return path


def same_float(actual: Any, expected: float, name: str, tolerance: float = 1e-12) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"Invalid numeric value: {name}")
    require(np.isfinite(actual) and abs(float(actual) - expected) <= tolerance, f"Numeric mismatch: {name}")


def rank_auroc(labels: list[int], scores: list[float]) -> float:
    """Independent Mann-Whitney AUROC with averaged ranks for score ties."""
    require(len(labels) == len(scores) > 0, "Invalid score/label lengths")
    require(all(value in (0, 1) for value in labels), "Invalid binary labels")
    require(all(np.isfinite(value) for value in scores), "Nonfinite image scores")
    positives = sum(labels)
    negatives = len(labels) - positives
    require(positives > 0 and negatives > 0, "AUROC requires both labels")
    order = sorted(range(len(scores)), key=scores.__getitem__)
    start = 0
    rank_sum = 0.0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2
        rank_sum += average_rank * sum(labels[order[index]] for index in range(start, stop))
        start = stop
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def array_check(reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    require(reference.shape == candidate.shape, "Paired model array shape differs")
    require(bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)), "Paired model array tolerance exceeded")
    difference = np.abs(reference - candidate)
    return {
        "max_absolute_error": float(difference.max(initial=0)),
        "max_relative_error": float((difference / np.maximum(np.abs(reference), 1e-12)).max(initial=0)),
    }


def validate_model(path: Path, method: str, train_count: int, fit: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == MODEL_FIELDS, "Invalid model NPZ field set")
        for field in ("schema_version", "count"):
            require(archive[field].shape == () and archive[field].dtype == np.int64, "Invalid model integer scalar")
        require(int(archive["schema_version"]) == 1, "Unsupported model schema")
        count = int(archive["count"])
        require(count == train_count * 1024, "PCA training token count mismatch")
        for field in ("variance_target", "retained_variance", "total_variance", "eps"):
            require(archive[field].shape == () and archive[field].dtype == np.float64 and bool(np.isfinite(archive[field])), "Invalid model float scalar")
        require(float(archive["variance_target"]) == .99 and float(archive["eps"]) == 1e-6, "Frozen PCA variance/epsilon changed")
        mean, components, eigenvalues = (archive[field].copy() for field in ("mean", "components", "eigenvalues"))
        require(mean.shape == (384,), "PCA mean shape mismatch")
        require(components.ndim == 2 and components.shape[0] == 384 and 1 <= components.shape[1] <= 384, "PCA component shape mismatch")
        rank = components.shape[1]
        require(eigenvalues.shape == (rank,), "PCA eigenvalue shape mismatch")
        for values in (mean, components, eigenvalues):
            require(values.dtype == np.float64 and bool(np.isfinite(values).all()), "Invalid model float64 array")
        require(bool(np.all(eigenvalues >= 0)) and bool(np.all(np.diff(eigenvalues) <= 0)), "Invalid descending PCA eigenvalues")
        orthogonality = float(np.max(np.abs(components.T @ components - np.eye(rank))))
        require(orthogonality <= 1e-9, "PCA basis is not orthonormal")
        retained = float(archive["retained_variance"])
        total = float(archive["total_variance"])
        require(total > 0 and .99 - 1e-12 <= retained <= 1 + 1e-12, "PCA retained variance invalid")
        same_float(retained, float(eigenvalues.sum() / total), "retained variance", 1e-11)
        if rank > 1:
            require(float(eigenvalues[:-1].sum() / total) < .99 + 1e-12, "PCA rank exceeds fixed variance requirement")
        metadata = json.loads(str(archive["metadata_json"].item()))
    require(isinstance(metadata, dict), "Invalid fitted-model metadata")
    expected = {
        "control_variant": method, "variant": method, "paired_control": True,
        "fit_device": "cpu", "fit_dtype": "float64", "input_dtype": "float32",
        "feature_dim": 384, "rank": rank, "token_count": count,
        "batch_count": train_count, "variance_target": .99,
        "fit_epochs": None, "backpropagation": False,
        "subspacead_commit": MODEL_COMMIT, "token_normalization": "none",
        "aggregation": "mean_before_pca", "scoring": "squared_reconstruction_residual",
        "whiten": False, "drop_k": 0,
        "covariance_denominator": "normal_patch_token_count-1",
    }
    require(all(metadata.get(field) == value for field, value in expected.items()), "Fitted-model identity/estimation metadata mismatch")
    require(all(metadata.get(field) == value for field, value in fit.items()), "Fitted-model metadata differs from recorded fit diagnostics")
    same_float(fit["retained_variance"], retained, "fit retained variance")
    same_float(fit["total_variance"], total, "fit total variance")
    pass_count = 1 if method == METHODS[2] else 2
    backend = "torch" if method == METHODS[0] else "numpy"
    expected_source = PCA_SHA if method == METHODS[0] else None
    require(fit["pass_count"] == pass_count and fit["linear_algebra_backend"] == backend and fit["source_sha256"] == expected_source, "Fit method attribution mismatch")
    require(metadata["official_pca_source_sha256"] == expected_source, "Official-source model attribution mismatch")
    require(valid_sha(fit["input_sha256"]) and valid_sha(fit["batch_partition_sha256"]), "Invalid token content/partition SHA256")
    require(np.isfinite(fit["elapsed_seconds"]) and fit["elapsed_seconds"] > 0, "Invalid fit duration")
    return {
        "mean": mean, "components": components, "eigenvalues": eigenvalues,
        "rank": rank, "count": count, "total_variance": total,
        "summary": {"rank": rank, "normal_token_count": count, "retained_variance": retained,
                    "max_component_orthogonality_error": orthogonality, "pass_count": pass_count,
                    "linear_algebra_backend": backend, "input_sha256": fit["input_sha256"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    project = args.project_root.resolve(strict=True)
    run = args.run.resolve(strict=True)
    receipt = args.receipt.resolve()
    require(not receipt.exists(), "Refusing to overwrite an audit receipt")
    config, state = read(run / "run_config.json"), read(run / "run_state.json")
    protocol = config["protocol"]
    source = (project / protocol["source_run"]).resolve(strict=True)
    baseline = (project / protocol["baseline_run"]).resolve(strict=True)
    vendor = (project / protocol["vendor_root"]).resolve(strict=True)
    weights = (project / protocol["weights_dir"]).resolve(strict=True)
    for protected in (run, source, baseline, vendor, weights):
        require(not receipt.is_relative_to(protected), "Receipt must remain outside immutable input trees")
    protocol_path = args.protocol or project / "configs/route1_controls_20261014.json"
    require(read(protocol_path) == protocol and digest(protocol_path) == config["protocol_sha256"], "Frozen route-1 protocol drift")
    require(config["pilot"] is False, "Pilot results are not a full control experiment")
    require(state["status"] == "completed" and state["current_task"] is None, "Full control run is incomplete")
    expected_state = {"completed_categories": 27, "training_images": 12288, "test_images": 3887, "metric_rows": 162}
    require(all(state.get(field) == value for field, value in expected_state.items()), "Whole-run completion counters mismatch")
    require(state["test_content_accessed"] is True and state["gradient_training_performed"] is False, "Run computation boundary mismatch")
    require(state["scientific_claim_eligible"] is False and config["scientific_claim_eligible"] is False and protocol["scientific_claim_eligible"] is False, "Exploratory claim boundary drift")
    expected_protocol = {
        "device": "cpu", "image_res": 448, "output_size": 256, "layers": [-4, -5],
        "feature_dim": 384, "explained_variance": .99, "eps": 1e-6,
        "batch_size": 1, "workers": 0, "cpu_threads": 4, "methods": METHODS,
        "vendor_commit": MODEL_COMMIT, "checkpoint_sha256": CHECKPOINT_SHA,
        "weight_revision": WEIGHT_REVISION, "map_atol": 1e-4, "map_rtol": 3e-5,
        "metric_atol": 1e-5,
    }
    require(all(protocol.get(field) == value for field, value in expected_protocol.items()), "Frozen route-1 method geometry/identity/tolerance drift")
    require(Path(config["source_run"]).resolve() == source and Path(config["project_root"]).resolve() == project, "Run root identity mismatch")
    checked: dict[str, str] = {}

    def check(path: Path, expected: str) -> str:
        resolved = path.resolve(strict=True)
        require(resolved.is_file() and valid_sha(expected), "Invalid sealed file/hash")
        name = str(resolved)
        if name not in checked:
            checked[name] = digest(resolved)
        require(checked[name] == expected, f"SHA256 mismatch: {resolved}")
        return checked[name]

    for relative, expected in config["source_sha256"].items():
        check(bounded(project, relative), expected)
    for relative, expected in config["environment_sha256"].items():
        check(bounded(project, relative), expected)
    for relative, expected in config["input_metadata_sha256"].items():
        check(bounded(source, relative), expected)
    for relative, expected in config["vendor_source_sha256"].items():
        check(bounded(vendor, relative), expected)
    require(config["vendor_source_sha256"]["src/subspacead/core/pca.py"] == PCA_SHA, "Author PCA source pin changed")
    for relative, expected in protocol["weight_files_sha256"].items():
        check(bounded(weights, relative), expected)
    require(config["backbone"]["files_sha256"] == protocol["weight_files_sha256"] and config["backbone"]["checkpoint_sha256"] == CHECKPOINT_SHA, "Frozen backbone seal mismatch")
    old_config = read(source / "run_config.json")
    require(read(source / "run_state.json")["status"] == "completed", "Historical source run incomplete")
    check(baseline / "run_config.json", old_config["input_metadata_sha256"]["run_config.json"])
    baseline_config = read(baseline / "run_config.json")
    tasks = [tuple(task) for task in old_config["expected_matrix"]["categories"]]
    baseline_tasks = [(dataset, category) for dataset, entry in baseline_config["protocol"]["datasets"].items() for category in entry["categories"]]
    require(tasks == baseline_tasks and len(tasks) == len(set(tasks)) == 27, "Full category matrix differs from frozen controls")
    require(config["expected_matrix"] == {"categories": [list(task) for task in tasks], "methods": METHODS, "variants": VARIANTS}, "Frozen route-1 result matrix mismatch")
    all_rows, all_samples, category_audits = [], [], []
    totals = {"normal_training_images": 0, "test_images": 0, "raw_maps": 0, "bool_masks": 0, "selected_samples": 0}
    score_error = 0.0
    maximum_raw_error = {method: 0.0 for method in METHODS[1:]}
    maximum_metric_error = 0.0
    for dataset, category in tasks:
        cat, pred = run / "categories" / dataset / category, run / "predictions" / dataset / category
        old_cat = source / "categories" / dataset / category
        seal = read(cat / "completion.json")
        require(seal["status"] == "completed", "Category is incomplete")
        actual_payload = {path.relative_to(run).as_posix() for directory in (cat, pred) for path in directory.rglob("*") if path.is_file() and path != cat / "completion.json"}
        require(set(seal["sha256"]) == actual_payload, "Category seal does not cover its complete payload")
        for relative, expected in seal["sha256"].items():
            require(relative.startswith(f"categories/{dataset}/{category}/") or relative.startswith(f"predictions/{dataset}/{category}/"), "Category seal escapes its payload")
            check(bounded(run, relative), expected)
        train, inventory = read(cat / "train_manifest.json"), read(cat / "test_inventory.json")
        require(train == read(old_cat / "train_manifest.json") and inventory == read(old_cat / "test_inventory.json"), "Train/Test inventories changed")
        require(len(train) > 0 and len({entry["path"] for entry in train}) == len(train), "Invalid normal-training inventory")
        normal_root = (project / baseline_config["protocol"]["datasets"][dataset]["root"] / category / "train" / "good").resolve(strict=True)
        normal_paths = {(project / entry["path"]).resolve(strict=True) for entry in train}
        require(all(path.is_relative_to(normal_root) for path in normal_paths), "Non-normal-training image used for PCA")
        test_paths = {Path(entry["image_path"]).resolve(strict=True) for entry in inventory}
        require(len(test_paths) == len(inventory) and not normal_paths.intersection(test_paths), "Duplicate Test paths or Train/Test overlap")
        cache_manifest = read(cat / "token_cache_manifest.json")
        require(cache_manifest["shape_per_image"] == [1024, 384] and cache_manifest["dtype"] == "float32" and cache_manifest["temporary_cache_deleted_after_completion"] is True, "Normal token-cache manifest geometry mismatch")
        require(len(cache_manifest["files"]) == len(train), "Incomplete normal token-cache manifest")
        for index, (token, image) in enumerate(zip(cache_manifest["files"], train, strict=True)):
            require(token["index"] == index and token["source_path"] == image["path"] and token["source_sha256"] == image["sha256"] and valid_sha(token["sha256"]) and token["size_bytes"] > 1024 * 384 * 4, "Token-cache source identity mismatch")
        require(not (run / "token_cache" / dataset / category).exists(), "Per-category normal token cache was not removed")
        diagnostics, metadata = read(cat / "diagnostics.json"), read(cat / "metadata.json")
        require(set(diagnostics["models"]) == set(METHODS) and metadata["models"] == diagnostics["models"], "Category fit attribution mismatch")
        require(metadata["dataset"] == dataset and metadata["category"] == category and metadata["train_images"] == len(train) and metadata["test_images"] == len(inventory) and metadata["device"] == "cpu" and metadata["shared_feature_extraction"] is True, "Category metadata identity mismatch")
        models = {method: validate_model(cat / "models" / (method + ".npz"), method, len(train), diagnostics["models"][method]) for method in METHODS}
        fits = diagnostics["models"]
        require(len({fit["input_sha256"] for fit in fits.values()}) == 1 and len({fit["batch_partition_sha256"] for fit in fits.values()}) == 1, "Three PCA fits used different input tokens or partitions")
        partition = hashlib.sha256()
        for _ in train:
            partition.update((1024).to_bytes(8, "little"))
        require(fits[METHODS[0]]["batch_partition_sha256"] == partition.hexdigest(), "Normal-token batch partition differs from one-image batches")
        model_comparisons = {}
        reference = models[METHODS[0]]
        require(set(diagnostics["comparisons"]) == set(METHODS[1:]), "Missing paired PCA diagnostics")
        query_digests = set()
        for method in METHODS[1:]:
            candidate = models[method]
            require(candidate["rank"] == reference["rank"] and candidate["count"] == reference["count"], "Paired PCA rank/count mismatch")
            independent = {
                "mean": array_check(reference["mean"], candidate["mean"], atol=TOLERANCES["mean_atol"], rtol=TOLERANCES["mean_rtol"]),
                "eigenvalues": array_check(reference["eigenvalues"], candidate["eigenvalues"], atol=TOLERANCES["eigenvalue_atol"], rtol=TOLERANCES["eigenvalue_rtol"]),
                "projector": array_check(reference["components"] @ reference["components"].T, candidate["components"] @ candidate["components"].T, atol=TOLERANCES["projector_atol"], rtol=TOLERANCES["projector_rtol"]),
                "total_variance": array_check(np.asarray(reference["total_variance"]), np.asarray(candidate["total_variance"]), atol=TOLERANCES["eigenvalue_atol"], rtol=TOLERANCES["eigenvalue_rtol"]),
            }
            recorded = diagnostics["comparisons"][method]
            require(recorded["passed"] is True and recorded["tolerances"] == TOLERANCES and recorded["detection_gain_claim"] is False, "Paired PCA diagnostics/claim boundary mismatch")
            require(all(check_record["passed"] is True for check_record in recorded["checks"].values()), "Recorded PCA check failed")
            for field, measured in independent.items():
                same_float(recorded["checks"][field]["max_absolute_error"], measured["max_absolute_error"], f"independent model {field}/max_absolute_error", 1e-10)
                # BLAS reduction order can move roundoff between near-zero
                # projector entries. Independently enforce allclose above;
                # the descriptive relative-error peak need not be bitwise
                # identical across the runner/auditor thread counts.
                relative_peak = recorded["checks"][field]["max_relative_error"]
                require(isinstance(relative_peak, (int, float)) and np.isfinite(relative_peak) and relative_peak >= 0, "Invalid recorded relative PCA error")
            identities = recorded["checks"]["identical_fit_input"]
            require(identities["reference_sha256"] == fits[METHODS[0]]["input_sha256"] and identities["candidate_sha256"] == fits[method]["input_sha256"], "Recorded paired input digest identity mismatch")
            query = recorded["checks"]["reconstruction_scores"]
            require(query["token_count"] == min(4, len(train)) * 1024 and query["batch_count"] == min(4, len(train)) and query["atol"] == 1e-4 and query["rtol"] == 3e-5 and valid_sha(query["query_input_sha256"]), "Recorded normal-query diagnostics mismatch")
            query_digests.add(query["query_input_sha256"])
            model_comparisons[method] = independent
        require(len(query_digests) == 1, "Paired diagnostics used different normal query tokens")
        samples = lines(cat / "samples.jsonl")
        old_samples = lines(old_cat / "samples.jsonl")
        require(len(samples) == len(inventory) == len(old_samples), "Incomplete category predictions")
        require([sample["sample_id"] for sample in samples] == [entry["sample_id"] for entry in inventory] and len({sample["sample_id"] for sample in samples}) == len(samples), "Sample order/identity mismatch")
        expected_payload = {f"categories/{dataset}/{category}/{name}" for name in (
            "train_manifest.json", "test_inventory.json", "token_cache_manifest.json", "diagnostics.json",
            "metadata.json", "metrics.json", "samples.jsonl",
        )}
        expected_payload.update(f"categories/{dataset}/{category}/models/{method}.npz" for method in METHODS)
        labels = []
        scores = {method: {"max": [], "top1p": []} for method in METHODS}
        raw_errors = {method: {"maximum_absolute_error": 0.0, "mean_absolute_error_sum": 0.0} for method in METHODS[1:]}
        selected = []
        for index, (sample, entry, old) in enumerate(zip(samples, inventory, old_samples, strict=True)):
            require(sample["dataset"] == dataset and sample["category"] == category and sample["seed"] == protocol["seed"], "Sample method identity mismatch")
            require(sample["sample_id"] == old["sample_id"] and sample["label"] == entry["label"] == old["label"] and sample["label"] in (0, 1), "Sample label/identity mismatch")
            require(sample["image_sha256"] == entry["image_sha256"] == old["image_sha256"] and Path(sample["image_path"]).resolve() == Path(entry["image_path"]).resolve(), "Test image provenance mismatch")
            require(sample["selected"] is old["selected"], "Frozen visual selection changed")
            if sample["selected"]:
                selected.append(sample["sample_id"])
            labels.append(sample["label"])
            paths = {method: f"predictions/{dataset}/{category}/{method}/{index:06d}.npy" for method in METHODS}
            mask_relative = f"predictions/{dataset}/{category}/masks/{index:06d}.npy"
            require(sample["raw_maps"] == paths and set(sample["raw_map_sha256"]) == set(METHODS) and set(sample["image_scores"]) == set(METHODS) and sample["mask_path"] == mask_relative, "Prediction path/method coverage mismatch")
            expected_payload.update(paths.values())
            expected_payload.add(mask_relative)
            mask = np.load(bounded(run, mask_relative), allow_pickle=False)
            check(bounded(run, mask_relative), sample["mask_sha256"])
            require(mask.shape == (256, 256) and mask.dtype == np.bool_, "Invalid bool256 GT mask")
            old_mask_path = bounded(source, old["mask_path"])
            check(old_mask_path, old["mask_sha256"])
            require(np.array_equal(mask, np.load(old_mask_path, allow_pickle=False)), "GT pixels changed from frozen source")
            if sample["label"] == 0:
                require(not mask.any(), "Normal Test GT mask is not empty")
            raw_reference = None
            for method in METHODS:
                path = bounded(run, paths[method])
                check(path, sample["raw_map_sha256"][method])
                raw = np.load(path, allow_pickle=False)
                require(raw.shape == (256, 256) and raw.dtype == np.float32 and bool(np.isfinite(raw).all()) and bool(np.all(raw >= 0)), "Invalid saved float32 raw anomaly map")
                # Sorting is independent of the runner's partition aggregation.
                maximum = float(raw.max())
                top = float(np.sort(raw.ravel().astype(np.float64))[-655:].mean(dtype=np.float64))
                stored = sample["image_scores"][method]
                same_float(stored["max"], maximum, "raw map max")
                same_float(stored["top1p"], top, "independent sorted top655")
                score_error = max(score_error, abs(stored["top1p"] - top))
                scores[method]["max"].append(maximum)
                scores[method]["top1p"].append(top)
                if raw_reference is None:
                    raw_reference = raw
                else:
                    require(bool(np.allclose(raw_reference, raw, atol=protocol["map_atol"], rtol=protocol["map_rtol"])), "Full paired raw-map tolerance exceeded")
                    delta = np.abs(raw.astype(np.float64) - raw_reference.astype(np.float64))
                    raw_errors[method]["maximum_absolute_error"] = max(raw_errors[method]["maximum_absolute_error"], float(delta.max()))
                    raw_errors[method]["mean_absolute_error_sum"] += float(delta.mean())
                totals["raw_maps"] += 1
            totals["bool_masks"] += 1
        require(set(seal["sha256"]) == expected_payload, "Unexpected/missing completed category payload")
        require(len(selected) == 2, "Expected two frozen visualization samples per category")
        for method, error in raw_errors.items():
            error["mean_absolute_error"] = error.pop("mean_absolute_error_sum") / len(samples)
            for field, value in error.items():
                same_float(diagnostics["raw_map_comparisons"][method][field], value, "full raw comparison " + field)
            maximum_raw_error[method] = max(maximum_raw_error[method], error["maximum_absolute_error"])
        rows = read(cat / "metrics.json")
        require(len(rows) == 6 and [row["variant"] for row in rows] == VARIANTS, "Category metric variant coverage mismatch")
        reference_rows = {row["variant"].rsplit("_", 1)[1]: row for row in rows[:2]}
        metric_errors = {}
        for row in rows:
            method, aggregation = row["variant"].rsplit("_", 1)
            require(row["dataset"] == dataset and row["category"] == category and row["seed"] == protocol["seed"] and row["status"] == "completed" and row["method"] == row["variant"] and row["train_images"] == len(train) and row["test_images"] == len(samples) and row["pca_rank"] == models[method]["rank"], "Metric identity/count/rank mismatch")
            same_float(row["image_auroc"], rank_auroc(labels, scores[method][aggregation]), "independent image rank AUROC")
            for field in METRICS:
                require(isinstance(row[field], (int, float)) and np.isfinite(row[field]) and 0 <= row[field] <= 1, "Invalid saved metric")
            paired = {field: abs(row[field] - reference_rows[aggregation][field]) for field in METRICS}
            require(max(paired.values()) <= protocol["metric_atol"], "Paired metric tolerance exceeded")
            require(diagnostics["metric_absolute_errors"][row["variant"]] == paired, "Recorded metric differences mismatch")
            maximum_metric_error = max(maximum_metric_error, max(paired.values()))
            metric_errors[row["variant"]] = paired
        for index in range(0, 6, 2):
            require(all(rows[index][field] == rows[index + 1][field] for field in METRICS[1:]), "Pixel metrics changed under image-score aggregation")
        totals["normal_training_images"] += len(train)
        totals["test_images"] += len(samples)
        totals["selected_samples"] += len(selected)
        all_rows.extend(rows)
        all_samples.extend(samples)
        category_audits.append({
            "dataset": dataset, "category": category, "normal_training_images": len(train),
            "test_images": len(samples), "completion_sha256": digest(cat / "completion.json"),
            "models": {method: model["summary"] for method, model in models.items()},
            "model_comparisons": model_comparisons, "raw_map_comparisons": raw_errors,
            "metric_absolute_errors": metric_errors, "selected_sample_ids": selected,
        })
        print(json.dumps({"audited_categories": len(category_audits), "dataset": dataset, "category": category}), flush=True)
    require(totals == {"normal_training_images": 12288, "test_images": 3887, "raw_maps": 11661, "bool_masks": 3887, "selected_samples": 54}, "Whole-run input/output totals mismatch")
    require(len(all_rows) == 162 and len(all_samples) == 3887, "Global row/sample counts mismatch")
    require(lines(run / "results/category_metrics.jsonl") == all_rows and lines(run / "results/samples.jsonl") == all_samples, "Global aggregates differ from sealed category records")
    audit = {
        "schema_version": 1, "status": "passed", "audit_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "source_run": str(source), "scope": "independent NumPy semantic/hash/PCA/raw-map/image-score audit",
        "categories": 27, "methods": METHODS, "metric_rows": 162, **totals,
        "scientific_claim_eligible": False, "detection_gain_claim": False,
        "torch_imported": "torch" in sys.modules, "forward_or_training_executed": False,
        "network_access_used": False, "pixel_metrics_recomputed": False,
        "pixel_metric_scope": "paired saved values verified; full AUROC/AP/AUPRO replay is required in delivery verification",
        "normal_token_scope": "all three fit content/partition SHA256 records authenticated and identical; normal token cache was deleted, so backbone features were not re-extracted",
        "normal_query_scope": "authenticated recorded normal-query reconstruction diagnostics; no token replay after cache deletion",
        "source_data_scope": "frozen normal/Test inventory identities and split paths verified; every saved GT compared with authenticated historical pixels; original image content was authenticated by runner",
        "authenticated_unique_files": len(checked), "independent_top1p_max_abs_error": score_error,
        "maximum_raw_map_absolute_error": maximum_raw_error, "maximum_paired_metric_absolute_error": maximum_metric_error,
        "category_audits": category_audits, "auditor_source_sha256": digest(Path(__file__).resolve()),
        "run_config_sha256": digest(run / "run_config.json"), "run_state_sha256": digest(run / "run_state.json"),
        "category_metrics_sha256": digest(run / "results/category_metrics.jsonl"),
        "samples_sha256": digest(run / "results/samples.jsonl"), "elapsed_seconds": time.monotonic() - started,
    }
    require(audit["torch_imported"] is False, "Lightweight auditor unexpectedly imported Torch")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "receipt": str(receipt), "elapsed_seconds": audit["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
