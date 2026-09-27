"""CLI for one project-layer DRAEM category training run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.draem_train import (  # noqa: E402
    TrainSettings,
    inspect_training_inputs,
    run_training,
)
from industrial_anomaly_lab.training_inputs import (  # noqa: E402
    AUTHOR_FULL_TRAIN,
    VISA_DEVELOPMENT_PROTOCOL,
)


def _project_config() -> dict[str, object]:
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    config = _project_config()
    baseline = config["baseline"]
    example = baseline["official_readme_example"]
    parser = argparse.ArgumentParser(
        description="Train one DRAEM category without modifying the author repository."
    )
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--anomaly-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=example["image_size"])
    parser.add_argument("--batch-size", type=int, default=example["batch_size"])
    parser.add_argument("--epochs", type=int, default=example["epochs"])
    parser.add_argument("--learning-rate", type=float, default=example["learning_rate"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override DataLoader pinned host memory. Omit to use the existing "
            "automatic policy (enabled for CUDA); use --no-pin-memory for a "
            "resource-constrained smoke."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--run-purpose",
        choices=("debug_limited", "development_full", "formal_full", "author_reproduction"),
        help="Required for training; validate-only defaults to author_reproduction.",
    )
    parser.add_argument(
        "--input-protocol",
        choices=(AUTHOR_FULL_TRAIN, VISA_DEVELOPMENT_PROTOCOL),
        default=AUTHOR_FULL_TRAIN,
        help=(
            "Keep author_full_train for an author-style reproduction. "
            "A development run must explicitly select visa_train_dev_dtd_v1."
        ),
    )
    parser.add_argument(
        "--development-protocol-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "development_protocol.json",
        help="Pinned config used only with visa_train_dev_dtd_v1.",
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
        "--max-steps",
        type=int,
        help="Debug-only step cap; outputs are marked ineligible for claims.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths and counts without allocating a model or training.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.validate_only and args.run_purpose is None:
        raise ValueError("--run-purpose is required for every training launch")
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
    anomaly_source_root = (
        args.anomaly_source_root
        if args.anomaly_source_root is not None
        else PROJECT_ROOT / config["paths"]["dtd"]
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
        / f"train_seed{args.seed}_{timestamp}"
    )
    settings = TrainSettings(
        dataset=args.dataset,
        category=args.category,
        dataset_root=dataset_root,
        anomaly_source_root=anomaly_source_root,
        output_dir=output_dir,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        workers=args.workers,
        pin_memory=args.pin_memory,
        device=args.device,
        max_steps=args.max_steps,
        input_protocol=args.input_protocol,
        development_protocol_config=(
            args.development_protocol_config
            if args.input_protocol == VISA_DEVELOPMENT_PROTOCOL
            else None
        ),
        formal_input_manifest=args.formal_input_manifest,
        run_purpose=args.run_purpose or "author_reproduction",
    )
    result = inspect_training_inputs(settings) if args.validate_only else run_training(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
