from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.rd4ad_infer import (
    Rd4adInferenceSettings,
    run_rd4ad_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one RD4AD category and persist raw anomaly maps."
    )
    parser.add_argument("--dataset", required=True, choices=("mvtec_ad", "visa"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--formal-test-inventory-dir",
        type=Path,
        help="Required by formal_full; validated F0 inventory bundle for this dataset.",
    )
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--execution-scope",
        choices=("full", "debug_limited"),
        help="Optional consistency check; --run-purpose is authoritative.",
    )
    parser.add_argument(
        "--run-purpose",
        required=True,
        choices=("debug_limited", "formal_full", "author_reproduction"),
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_scope = "debug_limited" if args.run_purpose == "debug_limited" else "full"
    if args.execution_scope is not None and args.execution_scope != expected_scope:
        raise ValueError("--execution-scope disagrees with --run-purpose")
    result = run_rd4ad_inference(
        Rd4adInferenceSettings(
            dataset=args.dataset,
            category=args.category,
            dataset_root=args.dataset_root,
            teacher_checkpoint=args.teacher_checkpoint,
            student_checkpoint=args.student_checkpoint,
            output_dir=args.output_dir,
            execution_scope=expected_scope,
            seed=args.seed,
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            gaussian_sigma=args.gaussian_sigma,
            run_purpose=args.run_purpose,
            checkpoint_manifest=args.student_checkpoint.parent / "manifest.json",
            formal_test_inventory_dir=args.formal_test_inventory_dir,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
