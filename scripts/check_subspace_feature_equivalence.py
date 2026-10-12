"""Normal-train-only check of omitted attention outputs against pinned author code."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "SubspaceAD" / "src"))
sys.dont_write_bytecode = True


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/subspace_dinos_20261012.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    baseline = ROOT / protocol["baseline_run"]
    manifest_path = baseline / "models" / "mvtec_ad" / "bottle" / "train_manifest.json"
    item = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
    image_path = (ROOT / item["path"]).resolve(strict=True)
    normal_root = (ROOT / "data" / "mvtec_ad" / "bottle" / "train" / "good").resolve(strict=True)
    if not image_path.is_relative_to(normal_root) or sha(image_path) != item["sha256"]:
        raise ValueError("Crosscheck image must be the authenticated first normal-train sample")
    import numpy as np
    import torch
    from PIL import Image
    from threadpoolctl import threadpool_limits
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor
    from subspacead.core.extractor import FeatureExtractor as OfficialFeatureExtractor

    threadpool_limits(limits=protocol["cpu_threads"])
    extractor = LocalDinoExtractor(
        ROOT / protocol["weights_dir"],
        expected_checkpoint_sha256=protocol["checkpoint_sha256"],
        expected_artifact_sha256=protocol["weight_files_sha256"],
        image_res=448, layers=(-4, -5), device=protocol["device"],
        cpu_threads=protocol["cpu_threads"], max_cuda_memory_gib=protocol["max_cuda_memory_gib"],
    )
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    without_attentions = extractor.extract(image)
    inputs = extractor.processor(
        images=[image], return_tensors="pt", do_resize=True,
        size={"height": 448, "width": 448}, do_center_crop=False,
        crop_size={"height": 448, "width": 448},
    ).to(extractor.device)
    reference = object.__new__(OfficialFeatureExtractor)
    with torch.inference_mode():
        outputs = extractor.model(**inputs, output_hidden_states=True, output_attentions=True)
        if outputs.attentions is None or len(outputs.attentions) != 12:
            raise ValueError("Reference forward did not return all requested attentions")
        official = reference._aggregate_layers(
            outputs.hidden_states, [-4, -5], [], "mean", 1, 1024, 32, 32,
        )[0]
    maximum = float(np.max(np.abs(without_attentions - official)))
    result = {
        "status": "passed" if np.array_equal(without_attentions, official) else "failed",
        "scope": "normal_train_only; no Test image or mask content accessed",
        "image_path": str(image_path), "image_sha256": item["sha256"],
        "train_manifest_sha256": sha(manifest_path),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "protocol_sha256": sha(args.config),
        "module_sha256": sha(ROOT / "src/industrial_anomaly_lab/subspace_features.py"),
        "author_extractor_sha256": sha(ROOT / "vendor/SubspaceAD/src/subspacead/core/extractor.py"),
        "adapter_shape": list(without_attentions.shape), "official_shape": list(official.shape),
        "adapter_dtype": str(without_attentions.dtype), "official_dtype": str(official.dtype),
        "max_abs_error": maximum, "exact_equal": bool(np.array_equal(without_attentions, official)),
        "reference_attention_layers": len(outputs.attentions),
        "elapsed_seconds": time.monotonic() - started,
        "device": str(extractor.device),
    }
    if extractor.device.type == "cuda":
        result.update(
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(extractor.device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(extractor.device),
        )
    del outputs, inputs, extractor
    gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
