from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from industrial_anomaly_lab.formal_test_inventory import (
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_PROTOCOL,
    DEFAULT_REGISTRATION,
    PROJECT_ROOT,
    build_formal_test_inventory_bundle,
    validate_formal_test_inventory_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or revalidate a formal F0 test inventory after method freeze."
    )
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        if args.dataset is not None or args.dataset_root is not None:
            raise SystemExit(
                "--validate-only reads dataset identity/root from the receipt; "
                "do not pass --dataset or --dataset-root"
            )
        receipt = validate_formal_test_inventory_bundle(
            output_dir=args.output_dir,
            project_root=args.project_root,
        )
    else:
        if args.dataset is None or args.dataset_root is None:
            raise SystemExit("build mode requires --dataset and --dataset-root")
        receipt = build_formal_test_inventory_bundle(
            dataset=args.dataset,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            project_root=args.project_root,
            project_config_path=args.project_config,
            registration_path=args.registration,
            protocol_path=args.protocol,
            prepared_manifest_path=args.prepared_manifest,
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
