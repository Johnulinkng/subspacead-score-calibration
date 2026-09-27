"""Audit every extracted author checkpoint without executing serialized code."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.draem_runtime import build_models  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checkpoint_root = PROJECT_ROOT / "artifacts" / "weights" / "DRAEM_checkpoints"
    output = PROJECT_ROOT / "manifests" / "draem_pretrained_checkpoints.json"
    paths = sorted(checkpoint_root.glob("*.pckl"))
    if len(paths) != 30:
        raise ValueError(f"expected 30 checkpoints, found {len(paths)}")

    reconstructive, discriminative = build_models("cpu")
    rows: list[dict[str, object]] = []
    categories: set[str] = set()
    for path in paths:
        role = "discriminative" if path.stem.endswith("_seg") else "reconstructive"
        model = discriminative if role == "discriminative" else reconstructive
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"invalid state dict: {path}")
        if not all(isinstance(key, str) for key in state):
            raise ValueError(f"non-string checkpoint key: {path}")
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError(f"non-tensor checkpoint value: {path}")
        model.load_state_dict(state, strict=True)
        marker = "_bs8_"
        category = path.name.split(marker, maxsplit=1)[1]
        category = category.removesuffix("_seg.pckl").removesuffix(".pckl").rstrip("_")
        categories.add(category)
        rows.append(
            {
                "file": path.relative_to(PROJECT_ROOT).as_posix(),
                "category": category,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "key_count": len(state),
                "tensor_element_count": sum(value.numel() for value in state.values()),
                "strict_model_load": "passed",
                "torch_load": "weights_only=True",
            }
        )
        del state
        gc.collect()

    expected_categories = {
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper",
    }
    if categories != expected_categories:
        raise ValueError(f"category mismatch: {sorted(categories)}")
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        if {row["role"] for row in category_rows} != {"reconstructive", "discriminative"}:
            raise ValueError(f"checkpoint pair missing for {category}")

    payload = {
        "schema_version": 1,
        "source_archive_manifest": "manifests/draem_pretrained_weights.json",
        "checkpoint_count": len(rows),
        "category_count": len(categories),
        "pairing_check": "passed",
        "safe_load_check": "passed",
        "strict_model_compatibility": "passed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "checkpoints"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
