"""Render SHA-traceable method comparisons from completed formal runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.visualization_delivery import (  # noqa: E402
    render_visualization_comparisons,
)


def _project_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("output must stay inside the E:\\3k project") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed and optional machine-selected failure cohorts from "
            "the exact SHA-verified raw maps used by metrics."
        )
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--failure-resolution", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render_visualization_comparisons(
        selection_manifest_path=args.selection_manifest,
        dataset_root=args.dataset_root,
        prediction_directories=args.prediction_dir,
        failure_resolution_path=args.failure_resolution,
        output_dir=_project_output(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
