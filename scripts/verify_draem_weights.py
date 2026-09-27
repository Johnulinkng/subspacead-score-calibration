"""Verify and optionally extract the author-provided DRAEM weight archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "artifacts" / "weights" / "DRAEM_checkpoints.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "weights" / "DRAEM_checkpoints"
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "draem_pretrained_weights.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {member.filename}")
    return members


def assert_direct_child(path: Path, parent: Path) -> None:
    if path.resolve().parent != parent.resolve():
        raise ValueError(f"unexpected staging location: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    archive_sha256 = sha256_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive)
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"CRC failure: {corrupt_member}")
        file_members = [member for member in members if not member.is_dir()]
        top_levels = sorted({PurePosixPath(m.filename).parts[0] for m in members})
        uncompressed_bytes = sum(member.file_size for member in file_members)

        if args.extract:
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=".draem-weights-", dir=output_path.parent)
            )
            assert_direct_child(staging, output_path.parent)
            try:
                archive.extractall(staging)
                extracted_root = staging
                if len(top_levels) == 1 and (staging / top_levels[0]).is_dir():
                    extracted_root = staging / top_levels[0]
                shutil.move(str(extracted_root), str(output_path))
            finally:
                if staging.exists():
                    assert_direct_child(staging, output_path.parent)
                    shutil.rmtree(staging)

    extracted_files = []
    if args.extract:
        extracted_files = sorted(
            path.relative_to(output_path).as_posix()
            for path in output_path.rglob("*")
            if path.is_file()
        )
    payload = {
        "schema_version": 1,
        "source": "DRAEM author Google Drive release",
        "source_file_id": "1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1",
        "archive_path": archive_path.relative_to(PROJECT_ROOT).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "zip_crc_test": "passed",
        "unsafe_path_count": 0,
        "member_count": len(members),
        "file_count": len(file_members),
        "uncompressed_bytes": uncompressed_bytes,
        "top_levels": top_levels,
        "extracted": args.extract,
        "output_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "extracted_file_count": len(extracted_files),
        "extracted_files": extracted_files,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
