"""Read-only PatchCore source/dependency and training-layout audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.data_protocols import validate_train_layout  # noqa: E402
from industrial_anomaly_lab.patchcore_adapter import (  # noqa: E402
    PatchCoreDataset,
    PatchCoreGeometry,
    audit_patchcore_runtime,
)


def _project_config() -> dict[str, object]:
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen PatchCore source and a normal-only training layout; "
            "does not import PatchCore, train, or inspect the test split."
        )
    )
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
    train_images = validate_train_layout(dataset_root, args.category)
    geometry = PatchCoreGeometry()
    dataset = PatchCoreDataset(
        args.dataset,
        dataset_root,
        args.category,
        split="train",
        geometry=geometry,
    )
    payload = {
        **audit_patchcore_runtime(),
        "dataset": args.dataset,
        "category": args.category,
        "dataset_root": str(Path(dataset_root).resolve()),
        "normal_train_image_count": len(train_images),
        "adapter_image_shape": list(dataset.imagesize),
        "evaluation_size": geometry.evaluation_size,
        "test_split_inspected": False,
        "training_or_test_evaluation_run": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
