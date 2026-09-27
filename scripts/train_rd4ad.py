"""Train one RD4AD category through the audited project adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.rd4ad_train import (  # noqa: E402
    AUTHOR_BATCH_SIZE,
    AUTHOR_EPOCHS,
    AUTHOR_IMAGE_SIZE,
    AUTHOR_LEARNING_RATE,
    Rd4adTrainSettings,
    inspect_training_inputs,
    run_training,
)


def _project_config() -> dict[str, object]:
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one RD4AD category without reading formal test images."
    )
    parser.add_argument("--dataset", required=True, choices=("mvtec_ad", "visa"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--image-size", type=int, default=AUTHOR_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=AUTHOR_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=AUTHOR_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=AUTHOR_LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--execution-scope",
        choices=("full", "debug_limited"),
        help="Optional consistency check; --run-purpose is authoritative.",
    )
    parser.add_argument(
        "--run-purpose",
        choices=("debug_limited", "formal_full", "author_reproduction"),
        help="Required for training; RD4AD development_full is not implemented.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Required for debug_limited and forbidden for full runs.",
    )
    parser.add_argument(
        "--formal-input-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "manifests"
            / "formal_training_inputs"
            / "v1"
            / "manifest.json"
        ),
        help="Train-only content-addressed bundle required by formal_full.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate train-only inventory, source and teacher; do not train.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.validate_only and args.run_purpose is None:
        raise ValueError("--run-purpose is required for every training launch")
    resolved_purpose = args.run_purpose or "author_reproduction"
    expected_scope = "debug_limited" if resolved_purpose == "debug_limited" else "full"
    if args.execution_scope is not None and args.execution_scope != expected_scope:
        raise ValueError("--execution-scope disagrees with --run-purpose")
    config = _project_config()
    categories = config["datasets"][args.dataset]["categories"]
    if args.category not in categories:
        raise ValueError(
            f"category '{args.category}' is not registered for {args.dataset}"
        )
    path_key = "mvtec" if args.dataset == "mvtec_ad" else "visa"
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else PROJECT_ROOT / config["paths"][path_key]
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT
        / config["paths"]["runs"]
        / "rd4ad"
        / args.dataset
        / args.category
        / f"train_seed{args.seed}_{timestamp}"
    )
    settings = Rd4adTrainSettings(
        dataset=args.dataset,
        category=args.category,
        dataset_root=dataset_root,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_sha256=args.teacher_sha256,
        output_dir=output_dir,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        workers=args.workers,
        device=args.device,
        execution_scope=expected_scope,
        max_steps=args.max_steps,
        run_purpose=resolved_purpose,
        formal_input_manifest=args.formal_input_manifest,
    )
    result = inspect_training_inputs(settings) if args.validate_only else run_training(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
