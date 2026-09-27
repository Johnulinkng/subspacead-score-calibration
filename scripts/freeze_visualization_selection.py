"""Freeze F0 visualization samples without consulting model predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.visualization_delivery import (  # noqa: E402
    freeze_visualization_selection,
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
            "Freeze result-independent visualization IDs and a machine-only "
            "failure ranking rule at F0. This command must run before inference."
        )
    )
    parser.add_argument("--test-inventory", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--method-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_visualization_selection(
        test_inventory_path=args.test_inventory,
        dataset_root=args.dataset_root,
        selection_config_path=args.selection_config,
        method_registry_path=args.method_registry,
        output_path=_project_output(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
