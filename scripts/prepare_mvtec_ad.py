"""Verify and safely extract MVTec AD without decoding formal test images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.mvtec import (  # noqa: E402
    audit_mvtec_layout,
    extract_and_audit_mvtec,
    sha256_file,
)


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if os.path.commonpath((str(PROJECT_ROOT), str(resolved))) != str(PROJECT_ROOT):
        raise ValueError(f"path must stay inside project root: {resolved}")
    return resolved


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the official MVTec AD archive, safely extract through an "
            "isolated staging directory, and write a machine-readable manifest."
        )
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "mvtec_source_manifest.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data" / "archives" / "mvtec_anomaly_detection.tar.xz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "mvtec_ad",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "mvtec_prepared.json",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit an already extracted output tree without opening image contents.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_manifest_path = _inside_project(args.source_manifest)
    archive_path = _inside_project(args.archive)
    output_root = _inside_project(args.output_root)
    json_output = _inside_project(args.json_output)
    if json_output in {archive_path, source_manifest_path}:
        raise ValueError("json output must not replace the archive or source manifest")
    if output_root == PROJECT_ROOT:
        raise ValueError("output root must not be the project root")
    if json_output == output_root or output_root in json_output.parents:
        raise ValueError("json output must remain outside the extracted dataset root")
    if output_root == archive_path or output_root in archive_path.parents:
        raise ValueError("archive must remain outside the extracted dataset root")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("dataset") != "MVTec AD" or source.get("schema_version") != 1:
        raise ValueError("source manifest is not the registered MVTec AD schema v1")
    archive_record = source["archive"]
    expected_bytes = int(archive_record["expected_bytes_from_http"])
    expected_sha256 = str(archive_record["expected_sha256_from_anomalib"])
    if expected_bytes <= 0 or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        raise ValueError("source manifest has an invalid archive size or SHA-256")
    provenance = {
        "source_manifest": source_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "preparation_script": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "preparation_script_sha256": sha256_file(Path(__file__).resolve()),
        "audit_module": "src/industrial_anomaly_lab/mvtec.py",
        "audit_module_sha256": sha256_file(
            PROJECT_ROOT / "src" / "industrial_anomaly_lab" / "mvtec.py"
        ),
    }

    if args.audit_only:
        actual_bytes = archive_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"archive size mismatch: expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha256 = sha256_file(archive_path)
        if actual_sha256.casefold() != expected_sha256.casefold():
            raise ValueError(
                f"archive SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        layout = audit_mvtec_layout(output_root)
        payload: dict[str, object] = {
            "schema_version": 1,
            "dataset": "MVTec AD",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "safe_extraction_completed_then_existing_tree_reaudit",
            "provenance": provenance,
            "official_dataset_page": source["official_dataset_page"],
            "dataset_license": source["dataset_license"],
            "commercial_use_allowed": source["commercial_use_allowed"],
            "archive": {
                "path": archive_path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "expected_bytes": expected_bytes,
                "expected_sha256": expected_sha256,
                "expected_bytes_match": True,
                "expected_sha256_match": True,
                "member_count": int(archive_record["archive_member_count"]),
                "path_safety_passed": archive_record["path_safety_passed"] is True,
            },
            "extraction": {
                "destination": output_root.relative_to(PROJECT_ROOT).as_posix(),
                "safe_extraction_completed": archive_record["path_safety_passed"] is True,
                "initial_verified_at_utc": archive_record["verified_at_utc"],
                "existing_destination_overwritten": False,
            },
            "layout": layout,
            "content_access_boundary": {
                "image_decode_performed": False,
                "test_image_content_viewed": False,
                "training_started": False,
                "model_inference_started": False,
                "metrics_computed": False,
            },
        }
        payload["layout"]["root"] = output_root.relative_to(PROJECT_ROOT).as_posix()
    else:
        if json_output.exists():
            raise FileExistsError(f"manifest already exists: {json_output}")
        payload = extract_and_audit_mvtec(
            archive_path,
            output_root,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        payload["provenance"] = provenance
        payload["official_dataset_page"] = source["official_dataset_page"]
        payload["dataset_license"] = source["dataset_license"]
        payload["commercial_use_allowed"] = source["commercial_use_allowed"]
        payload["archive"]["path"] = archive_path.relative_to(PROJECT_ROOT).as_posix()
        payload["extraction"]["destination"] = output_root.relative_to(PROJECT_ROOT).as_posix()
        payload["layout"]["root"] = output_root.relative_to(PROJECT_ROOT).as_posix()

    _write_json_atomic(json_output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
