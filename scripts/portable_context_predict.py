"""Predict one original image with the delivered single-pass model and calibration."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "code/src"))
import numpy as np
from threadpoolctl import threadpool_limits
from industrial_anomaly_lab.context_calibration import ContextCalibration, quantized_descriptor
from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, load_subspace_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mvtec_ad", "visa"), required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = p.parse_args()
    threadpool_limits(4)
    model_dir = PACKAGE / "data" / args.dataset / args.category
    reference = load_subspace_model(model_dir / "official_cpu_two_pass.npz")
    current = load_subspace_model(model_dir / "current_cpu_one_pass.npz")
    if current.k != reference.k:
        raise ValueError("PCA ranks differ")
    dots = np.sum(current.components * reference.components, axis=0)
    if np.min(np.abs(dots)) < 1 - 1e-8:
        raise ValueError("Content coordinate alignment failed")
    current = replace(current, components=current.components * np.where(dots < 0, -1., 1.))
    with np.load(model_dir / "calibration.npz", allow_pickle=False) as a:
        calibration = ContextCalibration(a["centers"].copy(), a["tails"].copy(), float(a["global_tail"]), float(a["global_center"]), float(a["global_scale"]))
    weights = json.loads((PACKAGE / "backbone.json").read_text(encoding="utf-8"))
    extractor = LocalDinoExtractor(args.weights, expected_checkpoint_sha256=weights["checkpoint_sha256"],
        expected_artifact_sha256=weights["weight_files_sha256"], device=args.device, cpu_threads=4, max_cuda_memory_gib=2.)
    tokens = extractor(args.image).reshape(-1, 384)
    raw = current.reconstruction_scores(tokens).reshape(32, 32)
    z = quantized_descriptor(tokens, current, 16)
    variants = ("baseline", "global", "conditional_tail", "conditioning_removed")
    scores = {v: calibration.apply(raw, z, v) for v in variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **scores)
    print(json.dumps({"image": str(args.image), "actual_pca": "current_cpu_one_pass", "device": args.device,
        "image_scores": {v: float(value.max()) for v, value in scores.items()},
        "map_content_sha256": {v: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for v, value in scores.items()}}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
