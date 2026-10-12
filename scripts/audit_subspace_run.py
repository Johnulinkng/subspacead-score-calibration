"""Independently audit a completed DINOv2-S/SubspaceAD experiment.

The default audit authenticates sealed files, normal-train/test inventories,
PCA state, every raw map and mask, image scores, and tie-aware rank AUROC.
It does not import Torch, fit PCA, run the backbone, or download anything.
Optional pixel metric recomputation uses the source-sealed common evaluator.
The receipt must be a new file outside the audited input trees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
sys.dont_write_bytecode = True

import numpy as np


MODEL_FIELDS = {
    "schema_version", "mean", "components", "eigenvalues", "count", "variance_target",
    "retained_variance", "total_variance", "eps", "metadata_json",
}
PIXEL_FIELDS = ("pixel_auroc", "pixel_ap", "aupro_0_30")
MODEL_COMMIT = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
WEIGHT_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
CHECKPOINT_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bounded(root: Path, relative: str) -> Path:
    entry = Path(relative)
    require(not entry.is_absolute(), f"Expected relative sealed path: {relative}")
    path = (root / entry).resolve(strict=True)
    require(path.is_relative_to(root), f"Path escapes audited root: {path}")
    require(path.is_file(), f"Expected sealed file: {path}")
    return path


def rank_auroc(labels: list[int], scores: list[float]) -> float:
    """Mann-Whitney AUROC using averaged tie ranks, independently of sklearn."""
    require(len(labels) == len(scores) and len(labels) > 0, "Invalid score/label lengths")
    require(all(label in (0, 1) for label in labels), "Invalid binary labels")
    require(all(np.isfinite(score) for score in scores), "Nonfinite image scores")
    positives = sum(labels)
    negatives = len(labels) - positives
    require(positives > 0 and negatives > 0, "AUROC requires both labels")
    ordered = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(labels[ordered[index]] for index in range(start, end))
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def same_float(actual: Any, expected: float, name: str, tolerance: float = 1e-12) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"Invalid numeric {name}")
    require(np.isfinite(actual) and abs(float(actual) - expected) <= tolerance, f"Numeric mismatch: {name}")


def validate_model(path: Path, *, train_count: int, dataset: str, category: str,
                   protocol_sha: str, train_sha: str, seed: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as state:
        require(set(state.files) == MODEL_FIELDS, f"Invalid model NPZ fields: {dataset}/{category}")
        for field in ("schema_version", "count"):
            require(state[field].shape == () and state[field].dtype == np.int64, f"Invalid integer model scalar: {field}")
        require(int(state["schema_version"]) == 1, "Unsupported PCA schema")
        require(int(state["count"]) == train_count * 1024, "PCA patch count mismatch")
        for field in ("variance_target", "retained_variance", "total_variance", "eps"):
            require(state[field].shape == () and state[field].dtype == np.float64, f"Invalid float model scalar: {field}")
            require(bool(np.isfinite(state[field])), f"Nonfinite model scalar: {field}")
        require(float(state["variance_target"]) == 0.99, "PCA variance target changed")
        require(float(state["eps"]) == 1e-6, "PCA epsilon changed")
        mean, components, eigenvalues = (state[name] for name in ("mean", "components", "eigenvalues"))
        require(mean.shape == (384,), "PCA mean dimension mismatch")
        require(components.ndim == 2 and components.shape[0] == 384 and 1 <= components.shape[1] <= 384, "PCA component shape mismatch")
        k = components.shape[1]
        require(eigenvalues.shape == (k,), "PCA eigenvalue shape mismatch")
        for field, values in (("mean", mean), ("components", components), ("eigenvalues", eigenvalues)):
            require(values.dtype == np.float64 and bool(np.isfinite(values).all()), f"Invalid float64 model array: {field}")
        require(bool(np.all(eigenvalues >= 0)) and bool(np.all(np.diff(eigenvalues) <= 0)), "Invalid descending eigenvalues")
        orthogonality = float(np.max(np.abs(components.T @ components - np.eye(k))))
        require(orthogonality < 1e-9, "PCA components are not orthonormal")
        retained = float(state["retained_variance"])
        total = float(state["total_variance"])
        require(total > 0 and 0.99 <= retained <= 1 + 1e-12, "Invalid full-training PCA variance coverage")
        same_float(retained, float(eigenvalues.sum() / total), "retained variance", 1e-11)
        if k > 1:
            require(float(eigenvalues[:-1].sum() / total) < 0.99 + 1e-12, "PCA rank exceeds fixed variance requirement")
        metadata = json.loads(str(state["metadata_json"].item()))
        expected = {
            "dataset": dataset, "category": category, "seed": seed, "train_images": train_count,
            "train_manifest_sha256": train_sha, "checkpoint_sha256": CHECKPOINT_SHA,
            "protocol_sha256": protocol_sha, "image_res": 448, "feature_shape": [32, 32, 384],
            "layers": [-4, -5], "official_commit": MODEL_COMMIT, "whiten": False, "drop_k": 0,
            "aggregation": "mean_before_pca", "token_normalization": "none",
            "fit_algorithm": "float64_chan_mean_scatter_no_intermediate_rank_truncation",
            "covariance_denominator": "normal_patch_token_count-1",
            "scoring": "squared_reconstruction_residual", "subspacead_commit": MODEL_COMMIT,
        }
        require(isinstance(metadata, dict), "Invalid model metadata")
        require(all(metadata.get(key) == value for key, value in expected.items()), "PCA model training/provenance metadata mismatch")
        return {"k": k, "patch_count": int(state["count"]), "retained_variance": retained,
                "max_component_orthogonality_error": orthogonality}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--recompute-pixel-metrics", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    project = args.project_root.resolve(strict=True)
    run = args.run.resolve(strict=True)
    receipt = args.receipt.resolve()
    require(not receipt.exists(), "Refusing to overwrite an audit receipt")
    config = read(run / "run_config.json")
    state = read(run / "run_state.json")
    protocol = config["protocol"]
    baseline = (project / protocol["baseline_run"]).resolve(strict=True)
    vendor = (project / protocol["vendor_root"]).resolve(strict=True)
    weights = (project / protocol["weights_dir"]).resolve(strict=True)
    for protected in (run, baseline, vendor, weights):
        require(not receipt.is_relative_to(protected), "Receipt must stay outside audited input trees")
    protocol_path = args.protocol or project / "configs/subspace_dinos_20261012.json"
    require(read(protocol_path) == protocol and digest(protocol_path) == config["protocol_sha256"], "Frozen protocol drift")
    require(config["benchmark_train_only"] is False, "A normal-only benchmark is not a full experiment")
    require(state["status"] == "completed" and state["current_task"] is None, "Experiment is incomplete")
    require(state["scientific_claim_eligible"] is False and protocol["scientific_claim_eligible"] is False, "Exploratory claim boundary drift")
    require(state["test_content_accessed"] is True, "Full evaluation did not record Test access")
    required_state = {"completed_categories": 27, "training_images": 12288, "test_images": 3887, "metric_rows": 54}
    require(all(state.get(key) == value for key, value in required_state.items()), "Full result counters mismatch")
    require(protocol["image_res"] == 448 and protocol["output_size"] == 256 and protocol["layers"] == [-4, -5]
            and protocol["feature_dim"] == 384 and protocol["batch_size"] == 1 and protocol["workers"] == 0
            and protocol["explained_variance"] == 0.99 and protocol["eps"] == 1e-6, "Frozen method geometry changed")
    require(protocol["vendor_commit"] == MODEL_COMMIT and protocol["weight_revision"] == WEIGHT_REVISION
            and protocol["checkpoint_sha256"] == CHECKPOINT_SHA, "Pinned public method identity changed")
    checked_hashes: dict[str, str] = {}

    def check(path: Path, expected: str) -> str:
        resolved = path.resolve(strict=True)
        require(resolved.is_file(), f"Missing sealed file: {resolved}")
        name = str(resolved)
        if name not in checked_hashes:
            checked_hashes[name] = digest(resolved)
        require(checked_hashes[name] == expected, f"SHA256 mismatch: {resolved}")
        return checked_hashes[name]

    for relative, expected in config["source_sha256"].items():
        check(bounded(project, relative), expected)
    for relative, expected in config["environment_sha256"].items():
        check(bounded(project, relative), expected)
    for relative, expected in config["input_metadata_sha256"].items():
        check(bounded(baseline, relative), expected)
    for relative, expected in config["vendor_source_sha256"].items():
        check(bounded(vendor, relative), expected)
    for relative, expected in config["backbone"]["files_sha256"].items():
        check(bounded(weights, relative), expected)
    require(config["backbone"]["files_sha256"] == protocol["weight_files_sha256"], "Weight manifest differs from protocol")
    require(config["backbone"]["checkpoint_sha256"] == CHECKPOINT_SHA, "Backbone seal drift")
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    require(commit == MODEL_COMMIT, "Vendor commit drift")
    require(not subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True).strip(), "Vendor tree is modified")
    original = read(baseline / "run_config.json")
    require(read(baseline / "run_state.json")["status"] == "completed", "Original baseline is incomplete")
    require(original["protocol"]["seeds"] == [protocol["seed"]], "Baseline seed changed")
    tasks = [(dataset, category) for dataset, entry in original["protocol"]["datasets"].items() for category in entry["categories"]]
    require(len(tasks) == 27 and len(set(tasks)) == 27, "Unexpected category matrix")
    require(config["expected_matrix"] == {"categories": [list(task) for task in tasks], "variants": ["subspace_max", "subspace_top1p"]}, "Frozen result matrix mismatch")
    old_samples_list = read_lines(baseline / "results/samples.jsonl")
    old_samples = {(sample["dataset"], sample["category"], sample["sample_id"]): sample for sample in old_samples_list}
    require(len(old_samples) == len(old_samples_list) == 3887, "Baseline sample identity duplicates or missing records")
    extractor = config["extractor"]
    extractor_expected = {"checkpoint_sha256": CHECKPOINT_SHA, "subspacead_commit": MODEL_COMMIT,
        "model_resolution": 448, "hidden_state_layers": [-4, -5], "hidden_state_count": 13,
        "resolved_hidden_state_indices": [9, 8], "aggregation": "mean_before_pca", "feature_dim": 384,
        "feature_grid": [32, 32], "drop_front_tokens": 1, "token_normalization": "none", "center_crop": False,
        "clahe": False, "dtype": "float32", "attention": "eager_without_retained_attention_outputs"}
    require(all(extractor.get(key) == value for key, value in extractor_expected.items()), "Extractor metadata drift")
    loading = extractor["checkpoint_loading"]
    require(loading["strict"] is True and loading["state_dict_key_count"] == 223
            and all(loading[field] == 0 for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "Checkpoint strict loading did not pass")
    all_rows: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    category_audits: list[dict[str, Any]] = []
    selected_ids = []
    train_total = test_total = raw_total = mask_total = 0
    maximum_score_error = 0.0
    if args.recompute_pixel_metrics:
        sys.path.insert(0, str(project / "src"))
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files

    for dataset, category in tasks:
        cat = run / "categories" / dataset / category
        pred = run / "predictions" / dataset / category
        old_model = baseline / "models" / dataset / category
        completion = read(cat / "completion.json")
        require(completion["status"] == "completed", f"Category incomplete: {dataset}/{category}")
        actual_files = {path.relative_to(run).as_posix() for directory in (cat, pred)
                        for path in directory.rglob("*") if path.is_file() and path != cat / "completion.json"}
        require(set(completion["sha256"]) == actual_files, f"Completion file coverage mismatch: {dataset}/{category}")
        for relative, expected in completion["sha256"].items():
            require(relative.startswith(f"categories/{dataset}/{category}/") or relative.startswith(f"predictions/{dataset}/{category}/"), "Category seal escapes its payload")
            check(bounded(run, relative), expected)
        train_sha = check(cat / "train_manifest.json", config["input_metadata_sha256"][f"models/{dataset}/{category}/train_manifest.json"])
        check(cat / "test_inventory.json", config["input_metadata_sha256"][f"models/{dataset}/{category}/test_inventory.json"])
        train = read(cat / "train_manifest.json")
        inventory = read(cat / "test_inventory.json")
        require(train == read(old_model / "train_manifest.json") and inventory == read(old_model / "test_inventory.json"), "Full train/test inventories differ from controls")
        require(len(train) > 0 and len({entry["path"] for entry in train}) == len(train), "Invalid normal training inventory")
        data_root = (project / original["protocol"]["datasets"][dataset]["root"]).resolve(strict=True)
        normal_root = (data_root / category / "train" / "good").resolve(strict=True)
        for entry in train:
            image_path = (project / entry["path"]).resolve(strict=True)
            require(image_path.is_relative_to(normal_root), "PCA training inventory contains a non-normal-training path")
        test_paths = {str(Path(sample["image_path"]).resolve(strict=True)) for sample in inventory}
        require(not test_paths.intersection(str((project / item["path"]).resolve()) for item in train), "Normal training overlaps Test")
        require(len(test_paths) == len(inventory), "Duplicate Test image paths")
        model_seal = read(cat / "model_integrity.json")
        check(cat / "model.npz", model_seal["sha256"])
        require((cat / "model.npz").stat().st_size == model_seal["size_bytes"], "Model size seal differs")
        model = validate_model(cat / "model.npz", train_count=len(train), dataset=dataset, category=category,
                               protocol_sha=config["protocol_sha256"], train_sha=train_sha, seed=protocol["seed"])
        metadata = read(cat / "metadata.json")
        require(metadata["dataset"] == dataset and metadata["category"] == category and metadata["train_images"] == len(train)
                and metadata["patch_count"] == len(train) * 1024 and metadata["k"] == model["k"]
                and metadata["variance_target"] == 0.99 and metadata["train_manifest_sha256"] == train_sha
                and metadata["protocol_sha256"] == config["protocol_sha256"] and metadata["extractor"] == extractor,
                "Category metadata differs from fitted state")
        same_float(metadata["retained_variance"], model["retained_variance"], "category retained variance")
        samples = read_lines(cat / "samples.jsonl")
        require(len(samples) == len(inventory), "Prediction inventory is incomplete")
        sample_ids = [sample["sample_id"] for sample in samples]
        require(sample_ids == [sample["sample_id"] for sample in inventory] and len(set(sample_ids)) == len(samples), "Prediction order/identity mismatch")
        expected_payload = {f"categories/{dataset}/{category}/{name}" for name in
                            ("metadata.json", "metrics.json", "model.npz", "model_integrity.json", "samples.jsonl", "train_manifest.json", "test_inventory.json")}
        map_paths, mask_paths, labels, scores_max, scores_top = [], [], [], [], []
        category_selected = []
        for index, (sample, expected) in enumerate(zip(samples, inventory, strict=True)):
            key = (dataset, category, sample["sample_id"])
            old = old_samples[key]
            require(sample["dataset"] == dataset and sample["category"] == category and sample["seed"] == protocol["seed"], "Sample method identity mismatch")
            require(sample["label"] == expected["label"] == old["label"] and sample["label"] in (0, 1), "Sample label mismatch")
            require(sample["image_sha256"] == expected["image_sha256"] == old["image_sha256"], "Test image content identity mismatch")
            require(Path(sample["image_path"]).resolve() == Path(expected["image_path"]).resolve(), "Sample image path mismatch")
            require(sample["selected"] is old["selected"], "Visual sample selection differs from frozen controls")
            if sample["selected"]:
                category_selected.append(sample["sample_id"])
                selected_ids.append(key)
            map_relative = f"predictions/{dataset}/{category}/subspace/{index:06d}.npy"
            mask_relative = f"predictions/{dataset}/{category}/masks/{index:06d}.npy"
            require(sample["raw_maps"] == {"subspace": map_relative} and sample["mask_path"] == mask_relative, "Raw map/mask path ordering mismatch")
            expected_payload.update((map_relative, mask_relative))
            map_path = bounded(run, map_relative)
            mask_path = bounded(run, mask_relative)
            check(map_path, sample["raw_map_sha256"]["subspace"])
            check(mask_path, sample["mask_sha256"])
            raw = np.load(map_path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)
            require(raw.shape == (256, 256) and raw.dtype == np.float32 and bool(np.isfinite(raw).all()) and bool(np.all(raw >= 0)), "Invalid raw anomaly map")
            require(mask.shape == (256, 256) and mask.dtype == np.bool_, "Invalid bool256 ground-truth mask")
            old_mask = np.load(bounded(baseline, old["mask_path"]), allow_pickle=False)
            require(np.array_equal(mask, old_mask), "Saved GT pixels differ from old baseline")
            if sample["label"] == 0:
                require(not mask.any(), "Normal Test image contains a defect mask")
            # Sorting is deliberately independent of the experiment's partition helper.
            maximum = float(np.max(raw))
            top_mean = float(np.sort(raw.ravel().astype(np.float64))[-655:].mean(dtype=np.float64))
            stored = sample["image_scores"]["subspace"]
            same_float(stored["max"], maximum, "sample max")
            same_float(stored["top1p"], top_mean, "sample sorted top655")
            maximum_score_error = max(maximum_score_error, abs(top_mean - stored["top1p"]))
            scores_max.append(maximum); scores_top.append(top_mean); labels.append(sample["label"])
            map_paths.append(map_path); mask_paths.append(mask_path)
            raw_total += 1; mask_total += 1
        require(set(completion["sha256"]) == expected_payload, "Unexpected or missing completed category payload")
        require(len(category_selected) == 2, "Expected two frozen visualization samples per category")
        rows = read(cat / "metrics.json")
        require(len(rows) == 2 and [row["variant"] for row in rows] == ["subspace_max", "subspace_top1p"], "Category metric variants mismatch")
        aucs = [rank_auroc(labels, scores_max), rank_auroc(labels, scores_top)]
        for index, row in enumerate(rows):
            require(row["dataset"] == dataset and row["category"] == category and row["seed"] == protocol["seed"]
                    and row["status"] == "completed" and row["train_images"] == len(train) and row["test_images"] == len(inventory)
                    and row["patch_count"] == len(train) * 1024 and row["pca_rank"] == model["k"], "Metric identity/count mismatch")
            same_float(row["image_auroc"], aucs[index], "independent rank AUROC")
            same_float(row["image_auroc_raw_max"], aucs[0], "stored max AUROC trace")
            same_float(row["image_auroc_top1p_mean"], aucs[1], "stored top1p AUROC trace")
            require(row["score_definition"] == protocol["score_definitions"][("max", "top1p")[index]], "Score definition mismatch")
            for field in PIXEL_FIELDS:
                require(isinstance(row[field], (int, float)) and np.isfinite(row[field]) and 0 <= row[field] <= 1, f"Invalid pixel metric: {field}")
                require(row[field] == rows[0][field], f"Pixel metric changed with only image aggregation: {field}")
        pixel_check = None
        if args.recompute_pixel_metrics:
            recomputed = evaluate_from_files(map_paths, mask_paths, labels)
            for field in PIXEL_FIELDS:
                same_float(rows[0][field], float(recomputed[field]), f"recomputed {field}")
            pixel_check = "passed"
        category_audits.append({"dataset": dataset, "category": category, "train_images": len(train),
            "test_images": len(inventory), **model, "image_auroc_max_independent": aucs[0],
            "image_auroc_top1p_independent": aucs[1], "selected_sample_ids": category_selected,
            "completion_sha256": digest(cat / "completion.json"), "pixel_metric_recompute": pixel_check})
        train_total += len(train); test_total += len(inventory)
        all_rows.extend(rows); all_samples.extend(samples)
        print(json.dumps({"audited_categories": len(category_audits), "dataset": dataset, "category": category}, ensure_ascii=False), flush=True)
    require(train_total == 12288 and test_total == raw_total == mask_total == 3887, "Whole-run input/output count mismatch")
    require(len(all_rows) == 54 and len(selected_ids) == len(set(selected_ids)) == 54, "Whole-run metrics/selection mismatch")
    original_selected = {(sample["dataset"], sample["category"], sample["sample_id"]) for sample in old_samples_list if sample["selected"]}
    require(set(selected_ids) == original_selected, "Global frozen selected samples differ")
    require(read_lines(run / "results/category_metrics.jsonl") == all_rows, "Global category rows differ from sealed category rows")
    require(read_lines(run / "results/samples.jsonl") == all_samples, "Global sample rows differ from sealed category rows")
    audit = {
        "schema_version": 1, "status": "passed", "audit_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "baseline": str(baseline), "scope": "independent semantic, hash, PCA, raw-map/mask and image-score audit",
        "source_data_boundary": "Normal/Test source images were authenticated by the sealed experiment; this audit verifies their frozen inventory identities and split paths. Original stored GT mask pixels are compared for all samples.",
        "scientific_claim_eligible": False, "torch_imported": "torch" in sys.modules,
        "forward_or_training_executed": False, "network_access_used": False,
        "categories": 27, "metric_rows": 54, "normal_training_images": train_total, "test_images": test_total,
        "raw_maps": raw_total, "bool_masks": mask_total, "frozen_selected_samples": len(selected_ids),
        "authenticated_unique_files": len(checked_hashes), "independent_top1p_max_abs_error": maximum_score_error,
        "pixel_metrics_recomputed": bool(args.recompute_pixel_metrics), "category_audits": category_audits,
        "auditor_source_sha256": digest(Path(__file__).resolve()),
        "run_config_sha256": digest(run / "run_config.json"), "run_state_sha256": digest(run / "run_state.json"),
        "category_metrics_sha256": digest(run / "results/category_metrics.jsonl"),
        "samples_sha256": digest(run / "results/samples.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "Verification of measured public-method reproduction does not establish an original improvement or journal acceptance.",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "receipt": str(receipt), "elapsed_seconds": audit["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
