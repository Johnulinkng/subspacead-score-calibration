"""Build the train-only formal input bundle without touching test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.formal_training_inputs import (  # noqa: E402
    DEFAULT_FORMAL_INPUT_MANIFEST,
    build_formal_training_input_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hash MVTec AD/VisA train/good and DTD images into a formal, "
            "test-isolated input bundle."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FORMAL_INPUT_MANIFEST.parent,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        project = json.load(handle)
    datasets = project["datasets"]
    paths = project["paths"]
    result = build_formal_training_input_bundle(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        dataset_roots={
            "mvtec_ad": PROJECT_ROOT / paths["mvtec"],
            "visa": PROJECT_ROOT / paths["visa"],
        },
        dtd_root=PROJECT_ROOT / paths["dtd"],
        categories={
            "mvtec_ad": datasets["mvtec_ad"]["categories"],
            "visa": datasets["visa"]["categories"],
        },
        source_manifests={
            "mvtec_ad": PROJECT_ROOT / datasets["mvtec_ad"]["source_manifest"],
            "visa": PROJECT_ROOT / "manifests" / "visa_source_manifest.json",
            "dtd": PROJECT_ROOT / "manifests" / "dtd_r1.0.1.json",
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
