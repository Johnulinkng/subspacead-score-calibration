"""Validate or build the final deterministic experiment delivery ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.delivery_package import (  # noqa: E402
    build_delivery_archive,
    validate_delivery_spec,
    verify_delivery_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless all four formal deliverables are present and "
            "hash-consistent; validation never reads dataset images."
        )
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=PROJECT_ROOT / "configs" / "delivery_spec.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and formal evidence without creating any package.",
    )
    mode.add_argument(
        "--output",
        type=Path,
        help="Build a new ZIP at this project-local path, then reopen and verify it.",
    )
    mode.add_argument(
        "--verify-archive",
        type=Path,
        help="Verify an existing archive instead of evaluating the delivery spec.",
    )
    args = parser.parse_args()
    try:
        if args.verify_archive is not None:
            result = verify_delivery_archive(args.verify_archive)
        elif args.validate_only:
            result = validate_delivery_spec(PROJECT_ROOT, args.spec)
            result.pop("_payload_paths", None)
            result = {
                "status": "valid",
                "package_name": result["package_name"],
                "delivery_status": result["delivery_status"],
                "payload_tree_sha256": result["payload_tree_sha256"],
                "file_count": len(result["files"]),
                "formal_run_count": result["formal_evidence"]["run_count"],
            }
        else:
            result = build_delivery_archive(PROJECT_ROOT, args.spec, args.output)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "archive_created": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
