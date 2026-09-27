"""Audit the official VisA split and optional raw/prepared trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.visa import (  # noqa: E402
    audit_visa_prepared_layout,
    audit_visa_raw_layout,
    audit_visa_split,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "spot-diff" / "split_csv" / "1cls.csv",
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--skip-mask-values", action="store_true")
    args = parser.parse_args()

    split_file = args.split_file.resolve()
    result: dict[str, object] = {
        "schema_version": 1,
        "dataset": "VisA",
        "protocol": "official_1cls",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_file": str(split_file),
        "split_sha256": _sha256(split_file),
        "split": audit_visa_split(split_file),
    }
    if args.raw_root:
        result["raw"] = audit_visa_raw_layout(args.raw_root.resolve(), split_file)
    if args.prepared_root:
        result["prepared"] = audit_visa_prepared_layout(
            args.prepared_root.resolve(),
            split_file,
            check_mask_values=not args.skip_mask_values,
        )

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
