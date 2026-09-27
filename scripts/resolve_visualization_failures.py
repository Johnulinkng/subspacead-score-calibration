"""Resolve failure-case IDs using only the rule frozen at F0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.visualization_delivery import (  # noqa: E402
    resolve_visualization_failures,
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
        description="Apply the frozen failure ranking; manual sample IDs are not accepted."
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resolve_visualization_failures(
        selection_manifest_path=args.selection_manifest,
        prediction_directories=args.prediction_dir,
        output_path=_project_output(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
