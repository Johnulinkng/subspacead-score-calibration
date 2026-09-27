"""CLI for deterministic DRAEM inference and raw-map export."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.draem_infer import (  # noqa: E402
    InferenceSettings,
    run_inference,
)


def _project_config() -> dict[str, object]:
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer one complete MVTec AD or VisA category with DRAEM."
    )
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _project_config()
    categories = config["datasets"][args.dataset]["categories"]
    if args.category not in categories:
        raise ValueError(
            f"category '{args.category}' is not registered for {args.dataset}"
        )
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else PROJECT_ROOT / config["paths"]["mvtec" if args.dataset == "mvtec_ad" else "visa"]
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / config["paths"]["runs"]
        / "draem"
        / args.dataset
        / args.category
        / f"infer_seed{args.seed}_{timestamp}"
    )
    settings = InferenceSettings(
        dataset=args.dataset,
        category=args.category,
        dataset_root=dataset_root,
        reconstructive_checkpoint=args.checkpoint_dir / "reconstructive.pt",
        discriminative_checkpoint=args.checkpoint_dir / "discriminative.pt",
        output_dir=output_dir,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    result = run_inference(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
