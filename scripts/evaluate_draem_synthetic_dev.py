"""Validate, audit or evaluate DRAEM on the frozen synthetic development set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.draem_synthetic_dev import (  # noqa: E402
    SyntheticDevEvaluationSettings,
    SyntheticDevSettings,
    inspect_synthetic_development,
    run_render_integrity_audit,
    run_synthetic_development_evaluation,
)
from industrial_anomaly_lab.run_artifacts import write_json  # noqa: E402


def _project_config() -> dict[str, object]:
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use only pinned VisA-dev and DTD-dev manifests. This command never "
            "discovers or reads formal test/ground_truth paths."
        )
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--anomaly-source-root", type=Path)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "development_protocol.json",
    )
    parser.add_argument("--seed", type=int, default=1887537248)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument(
        "--render-audit-count",
        type=int,
        metavar="N",
        help="Render the deterministic prefix twice; computes no model metric.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the complete frozen normal+synthetic development set.",
    )
    parser.add_argument("--reconstructive-checkpoint", type=Path)
    parser.add_argument("--discriminative-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project = _project_config()
    registered_categories = project["datasets"]["visa"]["categories"]
    if args.category not in registered_categories:
        raise ValueError(f"unregistered VisA category: {args.category}")
    dataset_root = args.dataset_root or PROJECT_ROOT / project["paths"]["visa"]
    dtd_root = args.anomaly_source_root or PROJECT_ROOT / project["paths"]["dtd"]
    basic = SyntheticDevSettings(
        category=args.category,
        dataset_root=dataset_root,
        anomaly_source_root=dtd_root,
        protocol_config_path=args.protocol_config,
        seed=args.seed,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.validate_only:
        result = inspect_synthetic_development(basic)
        output_dir = args.output_dir
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            evidence_path = output_dir / "validation.json"
            if evidence_path.exists():
                raise FileExistsError(evidence_path)
            write_json(evidence_path, result)
    elif args.render_audit_count is not None:
        output_dir = args.output_dir or (
            PROJECT_ROOT
            / "artifacts"
            / "validation"
            / f"synthetic_dev_render_{args.category}_{timestamp}"
        )
        result = run_render_integrity_audit(
            basic, output_dir=output_dir, count=args.render_audit_count
        )
    else:
        if args.reconstructive_checkpoint is None or args.discriminative_checkpoint is None:
            raise ValueError("--evaluate requires both DRAEM checkpoints")
        output_dir = args.output_dir or (
            PROJECT_ROOT
            / "runs"
            / "draem"
            / "synthetic_dev"
            / args.category
            / f"seed{args.seed}_{timestamp}"
        )
        result = run_synthetic_development_evaluation(
            SyntheticDevEvaluationSettings(
                **basic.__dict__,
                reconstructive_checkpoint=args.reconstructive_checkpoint,
                discriminative_checkpoint=args.discriminative_checkpoint,
                output_dir=output_dir,
                batch_size=args.batch_size,
                workers=args.workers,
                device=args.device,
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
