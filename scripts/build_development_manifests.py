"""Build frozen VisA train/dev and DTD source manifests without model execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.development_protocol import build_protocol_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hash only VisA train/good and DTD files, then write deterministic "
            "development manifests. This command never opens formal test images."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "development_protocol.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "development" / "visa_v1",
    )
    args = parser.parse_args()
    summary = build_protocol_artifacts(PROJECT_ROOT, args.config, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
