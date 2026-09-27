"""Fit one PatchCore memory bank from train/good through the safe adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.patchcore_runner import (  # noqa: E402
    DEFAULT_BACKBONE_CHECKPOINT,
    WRN50_IMAGENET1K_V1_SHA256,
    PatchCoreFitSettings,
    inspect_fit_inputs,
    run_patchcore_fit,
)


def _project_config() -> dict[str, object]:
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PatchCore from one category's train/good split only."
    )
    parser.add_argument("--dataset", required=True, choices=("mvtec_ad", "visa"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE_CHECKPOINT
    )
    parser.add_argument(
        "--backbone-sha256", default=WRN50_IMAGENET1K_V1_SHA256
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--execution-scope",
        required=True,
        choices=("full", "debug_limited"),
    )
    parser.add_argument(
        "--run-purpose",
        required=True,
        choices=(
            "debug_limited",
            "development_full",
            "author_reproduction",
            "formal_full",
        ),
        help="Explicit claim boundary; must agree with execution-scope/max-steps.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coreset-device", default="cpu")
    parser.add_argument("--faiss-workers", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional for debug_limited and forbidden for full.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Hash/audit train/good and the local checkpoint; do not fit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _project_config()
    dataset_config = config["datasets"][args.dataset]
    if args.category not in dataset_config["categories"]:
        raise ValueError(
            f"category '{args.category}' is not registered for {args.dataset}"
        )
    path_key = "mvtec" if args.dataset == "mvtec_ad" else "visa"
    dataset_root = args.dataset_root or PROJECT_ROOT / config["paths"][path_key]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / config["paths"]["runs"]
        / "patchcore"
        / args.dataset
        / args.category
        / f"fit_seed{args.seed}_{timestamp}"
    )
    settings = PatchCoreFitSettings(
        dataset=args.dataset,
        category=args.category,
        dataset_root=dataset_root,
        output_dir=output_dir,
        backbone_checkpoint=args.backbone_checkpoint,
        backbone_sha256=args.backbone_sha256,
        seed=args.seed,
        execution_scope=args.execution_scope,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        coreset_device=args.coreset_device,
        faiss_workers=args.faiss_workers,
        max_steps=args.max_steps,
        run_purpose=args.run_purpose,
    )
    if args.validate_only:
        result = inspect_fit_inputs(settings)
        result.pop("train_manifest_rows", None)
    else:
        result = run_patchcore_fit(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
