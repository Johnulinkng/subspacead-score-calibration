"""Audit a local torchvision WideResNet50-2 ImageNet V1 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.run_artifacts import utc_now, write_json  # noqa: E402
from industrial_anomaly_lab.wrn50_weights import (  # noqa: E402
    WRN50_IMAGENET1K_V1_FILENAME,
    inspect_wrn50_checkpoint,
    official_wrn50_source_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hash and strict-load a local WideResNet50-2 ImageNet V1 state "
            "dict. This command never downloads a checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "weights" / WRN50_IMAGENET1K_V1_FILENAME,
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "wide_resnet50_2_imagenet1k_v1.json",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record a pending manifest and exit successfully when the file is absent.",
    )
    parser.add_argument(
        "--record-official-acquisition",
        action="store_true",
        help=(
            "Record acquisition from the frozen official torchvision HTTPS URL "
            "through scripts/Download-Resumable.ps1."
        ),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    previous_acquisition = None
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(previous, dict):
            previous_acquisition = previous.get("acquisition")
    if checkpoint.is_file():
        audit = inspect_wrn50_checkpoint(
            checkpoint, expected_sha256=args.expected_sha256
        )
        payload = {
            "schema_version": 1,
            "status": (
                "strict_load_and_sha_verified"
                if args.expected_sha256 is not None
                else "strict_load_verified_sha_recorded"
            ),
            "checked_at_utc": utc_now(),
            "checkpoint_available": True,
            "full_sha256": audit["sha256"],
            "formal_use_allowed": (
                args.expected_sha256 is not None
                and bool(audit["url_hash_prefix_matched"])
            ),
            "consumers": ["PatchCore", "RD4AD"],
            "audit_network_accessed": False,
            "audit": audit,
        }
        if args.record_official_acquisition:
            payload["acquisition"] = {
                "source_url": audit["official_url"],
                "transport": "HTTPS via scripts/Download-Resumable.ps1",
                "network_accessed": True,
                "expected_bytes_from_http": 138223492,
                "observed_bytes": audit["size_bytes"],
                "observed_sha256": audit["sha256"],
                "url_hash_prefix_matched": audit["url_hash_prefix_matched"],
                "local_file_mtime_utc": datetime.fromtimestamp(
                    checkpoint.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        elif isinstance(previous_acquisition, dict) and (
            previous_acquisition.get("observed_sha256") == audit["sha256"]
        ):
            payload["acquisition"] = previous_acquisition
    else:
        payload = {
            "schema_version": 1,
            "status": "checkpoint_missing",
            "checked_at_utc": utc_now(),
            "checkpoint_available": False,
            "checkpoint_path": str(checkpoint),
            "full_sha256": None,
            "formal_use_allowed": False,
            "consumers": ["PatchCore", "RD4AD"],
            "audit_network_accessed": False,
            "strict_load_verified": False,
            "source": official_wrn50_source_metadata(),
        }
        write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if args.allow_missing:
            return 0
        raise FileNotFoundError(checkpoint)

    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
