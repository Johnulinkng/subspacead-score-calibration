"""Verify an extracted delivery and repeat selected inference without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True


def _digest(path: Path) -> tuple[int, str]:
    total, hasher = 0, hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(chunk)
            hasher.update(chunk)
    return total, hasher.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _within(root: Path, relative: str) -> Path:
    text = PurePosixPath(relative)
    if not relative or text.is_absolute() or ".." in text.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid manifest member path: {relative!r}")
    path = (root / Path(*text.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping package member: {relative}")
    return path


def verify_manifest(package_root: str | Path) -> dict[str, Any]:
    """Stream every declared member using only the Python standard library."""
    root = Path(package_root).resolve()
    manifest = _read(root / "DELIVERY-MANIFEST.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Completed delivery manifest schema 1 is required")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("No declared delivery files")
    expected = set()
    total = 0
    for entry in entries:
        relative = str(entry["path"])
        if relative in expected:
            raise ValueError(f"Duplicate manifest member: {relative}")
        expected.add(relative)
        size, digest = _digest(_within(root, relative))
        if size != entry["size_bytes"] or digest != entry["sha256"]:
            raise ValueError(f"File integrity mismatch: {relative}")
        total += size
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected | {"DELIVERY-MANIFEST.json"}:
        raise ValueError(f"Unexpected or missing extracted files: {sorted(actual - expected - {'DELIVERY-MANIFEST.json'})}")
    return {"status": "verified", "member_count": len(entries), "total_bytes": total, "all_member_hashes_verified": True}


def verify_inference(package_root: str | Path, sample_count: int = 2) -> list[dict[str, Any]]:
    """Recompute saved baseline/shrinkage raw maps using included local inputs."""
    if isinstance(sample_count, bool) or sample_count < 0:
        raise ValueError("Inference sample count must be nonnegative")
    if sample_count == 0:
        return []
    root = Path(package_root).resolve()
    config = _read(root / "run_config.json")
    visual = _read(root / "report" / "visualization_manifest.json")
    selected = visual.get("samples", [])[:sample_count]
    if len(selected) != sample_count:
        raise ValueError("Requested more inference samples than the delivery contains")
    protocol = config.get("protocol", {})
    model = protocol.get("model", {})
    backbone = config.get("backbone", {})
    weight_name = Path(backbone["weight_path"]).name
    weight = _within(root, "assets/weights/" + weight_name)
    sys.path.insert(0, str(root / "code" / "src"))
    import numpy as np
    from PIL import Image
    from industrial_anomaly_lab.quick_features import FeatureExtractor, load_statistics, prepare_scorer, render_maps

    seed = int(selected[0]["seed"])
    extractor = FeatureExtractor(weight, expected_sha256=backbone["sha256"], seed=seed, dimensions=int(model.get("dimensions", 100)), device="cpu", cpu_threads=2)
    active_category = None
    stats = None
    scorers: dict[float, Any] = {}
    outputs = []
    for sample in selected:
        if int(sample["seed"]) != seed:
            raise ValueError("One verification call requires a single fixed seed")
        category_key = (sample["dataset"], sample["category"])
        if category_key != active_category:
            scorers.clear()
            stats = load_statistics(_within(root, f"models/{sample['dataset']}/{sample['category']}/statistics.npz"))
            if not np.array_equal(stats.channel_indices, extractor.channel_indices):
                raise ValueError("Saved feature channel indices differ from the configured seed")
            for variant in ("baseline", "shrinkage"):
                settings = protocol["variants"][variant]
                alpha = float(settings["shrinkage_alpha"])
                scorers[alpha] = prepare_scorer(stats, shrinkage_alpha=alpha)
            active_category = category_key
        input_path = _within(root, "report/" + sample["source_image_copy"])
        if _digest(input_path)[1] != sample["source_image_sha256"]:
            raise ValueError("Selected source input digest differs from its recorded original")
        features = extractor(input_path)
        distances = {alpha: scorer.distance_maps(features) for alpha, scorer in scorers.items()}
        record: dict[str, Any] = {"dataset": sample["dataset"], "category": sample["category"], "sample_id": sample["sample_id"], "variants": {}, "atol": 1e-5, "rtol": 1e-6}
        for variant in ("baseline", "shrinkage"):
            settings = protocol["variants"][variant]
            raw_record = sample["raw_maps"][variant]
            raw_path = _within(root, "report/" + raw_record["path"])
            stored = np.load(raw_path, allow_pickle=False)
            if stored.ndim != 2 or not np.isfinite(stored).all():
                raise ValueError("Stored raw map is invalid")
            regenerated = render_maps(distances[float(settings["shrinkage_alpha"])], sigma=float(settings["sigma"]), output_size=int(stored.shape[0]))
            regenerated = np.asarray(regenerated, dtype=np.float32).reshape(stored.shape)
            equal = bool(np.allclose(regenerated, stored, atol=1e-5, rtol=1e-6))
            record["variants"][variant] = {"passed": equal, "stored_raw_max": float(stored.max()), "regenerated_raw_max": float(regenerated.max()), "maximum_absolute_error": float(np.max(np.abs(regenerated - stored)))}
            if not equal:
                raise ValueError(f"Inference mismatch: {sample['sample_id']}/{variant}")
        mask_path = sample.get("evaluation_mask_copy")
        if mask_path:
            persisted = _within(root, "report/" + mask_path)
            if _digest(persisted)[1] != sample["source_mask_sha256"]:
                raise ValueError("Selected evaluation mask digest mismatch")
            if persisted.suffix == ".npy":
                mask = np.load(persisted, allow_pickle=False) > 0
            else:
                with Image.open(persisted) as image:
                    mask = np.asarray(image.convert("L").resize(stored.shape[::-1], Image.Resampling.NEAREST)) > 0
            with Image.open(_within(root, "report/" + sample["ground_truth"])) as image:
                shown = np.asarray(image.convert("L")) > 0
            if not np.array_equal(mask, shown):
                raise ValueError("Rendered ground truth differs from persisted evaluation mask")
        record["evaluation_mask_verified"] = True
        record["status"] = "verified"
        outputs.append(record)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--inference-samples", type=int, default=2)
    args = parser.parse_args()
    try:
        integrity = verify_manifest(args.package_root)
        print(json.dumps({"file_integrity": integrity}, ensure_ascii=False), flush=True)
        inference = verify_inference(args.package_root, args.inference_samples)
        print(json.dumps({"status": "verified", "inference_samples": inference, "training_performed": False, "network_access_required": False}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(json.dumps({"status": "verification_failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
