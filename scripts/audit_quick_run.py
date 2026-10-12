"""Independently check complete quick-run coverage and saved array contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    config = read(run / "run_config.json")
    state = read(run / "run_state.json")
    if state["status"] != "completed" or state["completed_categories"] != 27:
        raise ValueError("The complete 27-category experiment is required")
    import numpy as np
    from PIL import Image
    rows = [json.loads(line) for line in (run / "results/category_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [json.loads(line) for line in (run / "results/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    protocol = config["protocol"]
    expected = {(dataset, category, variant, seed) for dataset, spec in protocol["datasets"].items()
                for category in spec["categories"] for variant in protocol["variants"] for seed in protocol["seeds"]}
    observed = {(r["dataset"],r["category"],r["variant"],r["seed"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Metric coverage is incomplete or duplicated")
    total_train = total_test = total_maps = 0
    categories = []
    for dataset, spec in protocol["datasets"].items():
        for category in spec["categories"]:
            model_root = run / "models" / dataset / category
            seal = read(model_root / "statistics_integrity.json")
            if digest(model_root / "statistics.npz") != seal["sha256"]:
                raise ValueError("Fitted statistics differ from their content seal")
            train = read(model_root / "train_manifest.json")
            inventory = read(model_root / "test_inventory.json")
            selected = [s for s in samples if s["dataset"] == dataset and s["category"] == category]
            metric_rows = [r for r in rows if r["dataset"] == dataset and r["category"] == category]
            if len(selected) != len(inventory) or len({s["sample_id"] for s in selected}) != len(inventory):
                raise ValueError("Sample coverage is incomplete or duplicated")
            if {s["sample_id"] for s in selected} != {s["sample_id"] for s in inventory}:
                raise ValueError("Predicted sample IDs do not match the complete test inventory")
            if any(r["train_images"] != len(train) or r["test_images"] != len(inventory) for r in metric_rows):
                raise ValueError("Training or test counts disagree with the registered results")
            labels = {s["sample_id"]:s["label"] for s in inventory}
            inventory_by_id = {s["sample_id"]:s for s in inventory}
            checksums = []
            for sample in selected:
                if sample["label"] != labels[sample["sample_id"]]:
                    raise ValueError("Stored sample label differs from the inventory")
                mask_path = run / sample["mask_path"]
                mask = np.load(mask_path, allow_pickle=False)
                if mask.shape != (256,256) or mask.dtype != np.bool_:
                    raise ValueError("Evaluation mask violates the fixed contract")
                if sample["label"] == 0 and mask.any():
                    raise ValueError("Normal test image has abnormal evaluation pixels")
                original_mask_path = inventory_by_id[sample["sample_id"]]["mask_path"]
                if original_mask_path:
                    with Image.open(original_mask_path) as image:
                        expected_mask = np.asarray(image.convert("L").resize((256,256), Image.Resampling.NEAREST)) > 0
                    if not np.array_equal(mask,expected_mask):
                        raise ValueError("Stored evaluation mask differs from the nearest-resized original GT")
                if set(sample["raw_maps"]) != set(protocol["variants"]):
                    raise ValueError("A sample is missing one or more registered variants")
                for variant, relative in sample["raw_maps"].items():
                    path = (run / relative).resolve()
                    if not path.is_relative_to(run):
                        raise ValueError("Prediction path escapes the quick run")
                    raw = np.load(path, allow_pickle=False)
                    if raw.shape != (256,256) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                        raise ValueError("Raw anomaly map violates the fixed contract")
                    checksums.append({"sample_id":sample["sample_id"],"variant":variant,"path":relative,"sha256":digest(path),"raw_map_max":float(raw.max())})
                    total_maps += 1
            manifest_path = model_root / "prediction_integrity.jsonl"
            with manifest_path.open("x", encoding="utf-8") as handle:
                for row in checksums:
                    handle.write(json.dumps(row) + "\n")
            total_train += len(train)
            total_test += len(inventory)
            categories.append({"dataset":dataset,"category":category,"train_images":len(train),"test_images":len(inventory),"map_count":len(checksums),"status":"passed"})
    result = {"status":"passed","category_count":len(categories),"metric_rows":len(rows),"normal_training_images":total_train,
              "test_images":total_test,"raw_map_count":total_maps,"categories":categories,
              "checks":{"all_fitted_model_seals_match":True,"complete_test_inventory_used":True,"all_four_variants_use_identical_samples":True,
                        "all_saved_raw_maps_float32_finite_256":True,"all_masks_binary_nearest_contract":True},
              "scientific_improvement_confirmed":False}
    (run / "results/integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key != "categories"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
