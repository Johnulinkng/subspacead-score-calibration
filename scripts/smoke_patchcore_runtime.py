"""Run a no-download, no-dataset PatchCore author-runtime CPU smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.patchcore_adapter import (  # noqa: E402
    run_patchcore_synthetic_smoke,
)
from industrial_anomaly_lab.run_artifacts import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the frozen PatchCore runtime with random tensors and a "
            "tiny local backbone; never downloads weights or reads a dataset."
        )
    )
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/validation/patchcore_runtime_smoke.json",
    )
    args = parser.parse_args()
    result = run_patchcore_synthetic_smoke(args.seed)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
