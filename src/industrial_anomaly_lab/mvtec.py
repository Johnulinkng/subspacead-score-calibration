"""Mechanical, non-decoding preparation checks for the official MVTec AD archive."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .data_protocols import build_test_manifest


MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_COPY_BUFFER_BYTES = 1024 * 1024
DEFAULT_MAX_MEMBERS = 100_000
DEFAULT_MAX_EXTRACTED_BYTES = 20 * 1024**3


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_COPY_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member_parts(name: str) -> tuple[str, ...]:
    """Convert a POSIX tar name to safe Windows path components."""

    if not name or "\x00" in name:
        raise ValueError(f"unsafe empty/NUL archive member name: {name!r}")
    if "\\" in name:
        raise ValueError(f"archive member uses a backslash: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith("/"):
        raise ValueError(f"archive member is absolute: {name!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise ValueError(f"archive member traverses a parent: {name!r}")
    for part in parts:
        if ":" in part:
            raise ValueError(f"archive member contains a Windows stream/drive: {name!r}")
        if part.endswith((" ", ".")):
            raise ValueError(f"archive member has a Windows-ambiguous suffix: {name!r}")
        base_name = part.split(".", 1)[0].upper()
        if base_name in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"archive member uses a reserved Windows name: {name!r}")
    return parts


def _assert_target_below(root: Path, parts: tuple[str, ...]) -> Path:
    target = root.joinpath(*parts)
    root_resolved = root.resolve()
    target_resolved = target.resolve(strict=False)
    if os.path.commonpath((str(root_resolved), str(target_resolved))) != str(
        root_resolved
    ):
        raise ValueError(f"archive member escapes extraction root: {'/'.join(parts)!r}")
    return target


def _copy_exact(source: BinaryIO, destination: BinaryIO, expected_bytes: int) -> int:
    copied = 0
    while True:
        block = source.read(_COPY_BUFFER_BYTES)
        if not block:
            break
        destination.write(block)
        copied += len(block)
    if copied != expected_bytes:
        raise IOError(
            f"archive member length mismatch: expected {expected_bytes}, copied {copied}"
        )
    return copied


def _extract_tar_xz_to_staging(
    archive_path: Path,
    staging_root: Path,
    *,
    max_members: int,
    max_extracted_bytes: int,
) -> dict[str, object]:
    seen: dict[str, str] = {}
    canonical_paths: dict[str, str] = {}
    member_count = 0
    file_count = 0
    directory_count = 0
    extracted_bytes = 0

    with tarfile.open(archive_path, mode="r:xz") as archive:
        for member in archive:
            member_count += 1
            if member_count > max_members:
                raise ValueError(f"archive exceeds member limit {max_members}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(
                    f"archive contains a link or special member: {member.name!r}"
                )
            parts = _safe_member_parts(member.name)
            if not parts:
                if member.isdir():
                    continue
                raise ValueError("archive contains a root-level nameless file")

            key = "/".join(parts).casefold()
            kind = "directory" if member.isdir() else "file"
            for prefix_length in range(1, len(parts) + 1):
                display_prefix = "/".join(parts[:prefix_length])
                folded_prefix = display_prefix.casefold()
                previous_spelling = canonical_paths.get(folded_prefix)
                if previous_spelling is not None and previous_spelling != display_prefix:
                    raise ValueError(
                        "archive has a case-colliding Windows path: "
                        f"{previous_spelling!r} and {display_prefix!r}"
                    )
                canonical_paths[folded_prefix] = display_prefix
            if key in seen:
                raise ValueError(
                    f"archive has duplicate/case-colliding members: {member.name!r}"
                )
            for prefix_length in range(1, len(parts)):
                prefix = "/".join(parts[:prefix_length]).casefold()
                if seen.get(prefix) == "file":
                    raise ValueError(
                        f"archive member descends through a file: {member.name!r}"
                    )
            if kind == "file" and seen.get(key) == "directory":
                raise ValueError(f"archive file replaces a directory: {member.name!r}")
            seen[key] = kind

            target = _assert_target_below(staging_root, parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directory_count += 1
                continue

            if member.size < 0:
                raise ValueError(f"archive member has negative size: {member.name!r}")
            if extracted_bytes + member.size > max_extracted_bytes:
                raise ValueError(
                    f"archive exceeds extracted-byte limit {max_extracted_bytes}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise IOError(f"could not read regular archive member: {member.name!r}")
            with source, target.open("xb") as destination:
                _copy_exact(source, destination, member.size)
            file_count += 1
            extracted_bytes += member.size

    return {
        "path_safety_passed": True,
        "links_and_special_members_rejected": True,
        "case_collision_check_passed": True,
        "member_count": member_count,
        "regular_file_count": file_count,
        "directory_count": directory_count,
        "extracted_file_bytes": extracted_bytes,
        "max_members": max_members,
        "max_extracted_bytes": max_extracted_bytes,
    }


def _locate_dataset_root(extraction_root: Path) -> tuple[Path, str]:
    expected = set(MVTEC_CATEGORIES)

    def directory_names(path: Path) -> set[str]:
        return {entry.name for entry in path.iterdir() if entry.is_dir()}

    if directory_names(extraction_root) == expected:
        return extraction_root, "."
    candidates = [
        child
        for child in extraction_root.iterdir()
        if child.is_dir() and directory_names(child) == expected
    ]
    if len(candidates) != 1:
        raise ValueError(
            "archive must contain the exact 15 MVTec categories either at its root "
            "or below one wrapper directory"
        )
    other_directories = [
        child.name
        for child in extraction_root.iterdir()
        if child.is_dir() and child != candidates[0]
    ]
    if other_directories:
        raise ValueError(f"unexpected top-level directories: {other_directories}")
    return candidates[0], candidates[0].name


def _images_only(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required image directory is missing: {directory}")
    files: list[Path] = []
    for entry in directory.iterdir():
        if entry.is_dir():
            raise ValueError(f"nested directory is not allowed in image folder: {entry}")
        if not entry.is_file() or entry.suffix.lower() != ".png":
            raise ValueError(f"unexpected non-PNG entry in image folder: {entry}")
        if entry.stat().st_size <= 0:
            raise ValueError(f"empty image/mask file: {entry}")
        files.append(entry.resolve())
    if not files:
        raise ValueError(f"image directory is empty: {directory}")
    return sorted(files, key=lambda path: path.name.casefold())


def _metadata_tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    return digest.hexdigest(), file_count, total_bytes


def audit_mvtec_layout(dataset_root: str | Path) -> dict[str, object]:
    """Audit names, sizes and mask pairing without decoding an image."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec AD root is missing: {root}")
    root_directories = {entry.name for entry in root.iterdir() if entry.is_dir()}
    if root_directories != set(MVTEC_CATEGORIES):
        raise ValueError(
            f"MVTec category set mismatch: expected {list(MVTEC_CATEGORIES)}, "
            f"found {sorted(root_directories)}"
        )

    category_reports: dict[str, dict[str, object]] = {}
    total_train_good = 0
    total_test_good = 0
    total_test_anomaly = 0
    total_masks = 0
    for category in MVTEC_CATEGORIES:
        category_root = root / category
        category_directories = {
            entry.name for entry in category_root.iterdir() if entry.is_dir()
        }
        if category_directories != {"train", "test", "ground_truth"}:
            raise ValueError(
                f"{category}: expected train/test/ground_truth directories, "
                f"found {sorted(category_directories)}"
            )

        train_root = category_root / "train"
        for structural_root in (
            train_root,
            category_root / "test",
            category_root / "ground_truth",
        ):
            unexpected_files = [
                entry.name for entry in structural_root.iterdir() if not entry.is_dir()
            ]
            if unexpected_files:
                raise ValueError(
                    f"{category}: unexpected entries in {structural_root.name}: "
                    f"{unexpected_files}"
                )
        train_types = {entry.name for entry in train_root.iterdir() if entry.is_dir()}
        if train_types != {"good"}:
            raise ValueError(f"{category}: train must contain only good, found {train_types}")
        train_images = _images_only(train_root / "good")

        test_root = category_root / "test"
        test_types = {entry.name for entry in test_root.iterdir() if entry.is_dir()}
        if "good" not in test_types or len(test_types) < 2:
            raise ValueError(f"{category}: test must contain good and anomaly types")
        anomaly_types = test_types - {"good"}
        ground_truth_root = category_root / "ground_truth"
        ground_truth_types = {
            entry.name for entry in ground_truth_root.iterdir() if entry.is_dir()
        }
        if ground_truth_types != anomaly_types:
            raise ValueError(
                f"{category}: ground_truth types {sorted(ground_truth_types)} do not "
                f"match anomalous test types {sorted(anomaly_types)}"
            )

        test_counts = {
            defect_type: len(_images_only(test_root / defect_type))
            for defect_type in sorted(test_types)
        }
        mask_counts = {
            defect_type: len(_images_only(ground_truth_root / defect_type))
            for defect_type in sorted(ground_truth_types)
        }
        samples = build_test_manifest("mvtec_ad", root, category)
        expected_masks = {
            sample.mask_path.resolve()
            for sample in samples
            if sample.mask_path is not None
        }
        actual_masks = {
            path
            for defect_type in ground_truth_types
            for path in _images_only(ground_truth_root / defect_type)
        }
        if actual_masks != expected_masks:
            extra = sorted(str(path) for path in actual_masks - expected_masks)
            missing = sorted(str(path) for path in expected_masks - actual_masks)
            raise ValueError(
                f"{category}: mask inventory mismatch, extra={extra[:3]}, "
                f"missing={missing[:3]}"
            )

        test_good = test_counts["good"]
        test_anomaly = sum(
            count for name, count in test_counts.items() if name != "good"
        )
        metadata_sha256, category_file_count, category_bytes = _metadata_tree_digest(
            category_root
        )
        category_reports[category] = {
            "train_good_images": len(train_images),
            "test_good_images": test_good,
            "test_anomaly_images": test_anomaly,
            "test_images_total": test_good + test_anomaly,
            "ground_truth_masks": len(actual_masks),
            "test_by_type": test_counts,
            "ground_truth_by_type": mask_counts,
            "mask_pairing": "exact_by_<test_stem>_mask.png",
            "metadata_tree_sha256": metadata_sha256,
            "all_category_files": category_file_count,
            "all_category_file_bytes": category_bytes,
        }
        total_train_good += len(train_images)
        total_test_good += test_good
        total_test_anomaly += test_anomaly
        total_masks += len(actual_masks)

    tree_sha256, all_file_count, all_file_bytes = _metadata_tree_digest(root)
    return {
        "root": str(root),
        "category_count": len(category_reports),
        "categories": category_reports,
        "totals": {
            "train_good_images": total_train_good,
            "test_good_images": total_test_good,
            "test_anomaly_images": total_test_anomaly,
            "test_images_total": total_test_good + total_test_anomaly,
            "ground_truth_masks": total_masks,
            "all_files": all_file_count,
            "all_file_bytes": all_file_bytes,
        },
        "mask_pairing_complete": True,
        "all_required_images_and_masks_nonempty": True,
        "metadata_tree_sha256": tree_sha256,
        "image_decode_performed": False,
        "test_image_content_viewed": False,
    }


def extract_and_audit_mvtec(
    archive_path: str | Path,
    destination_root: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
) -> dict[str, object]:
    """Verify, safely stage, structurally audit and atomically install MVTec AD."""

    archive = Path(archive_path).resolve()
    destination = Path(destination_root).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    actual_bytes = archive.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"archive size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(archive)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ValueError(
            f"archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination.parent).free
    minimum_free = min(max_extracted_bytes, max(actual_bytes * 2, 64 * 1024**2))
    if free_bytes < minimum_free:
        raise OSError(
            f"insufficient free space: need at least {minimum_free}, found {free_bytes}"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=".mvtec-extract-", dir=destination.parent)
    ).resolve()
    installed = False
    try:
        archive_audit = _extract_tar_xz_to_staging(
            archive,
            staging,
            max_members=max_members,
            max_extracted_bytes=max_extracted_bytes,
        )
        dataset_root, archive_prefix = _locate_dataset_root(staging)
        audit_mvtec_layout(dataset_root)
        dataset_root.replace(destination)
        installed = True
        if staging.exists():
            shutil.rmtree(staging)
        layout = audit_mvtec_layout(destination)
        return {
            "schema_version": 1,
            "dataset": "MVTec AD",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive": {
                "path": str(archive),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "expected_bytes_match": True,
                "expected_sha256_match": True,
                "archive_root_prefix": archive_prefix,
                **archive_audit,
            },
            "extraction": {
                "destination": str(destination),
                "atomic_staging_then_move": True,
                "existing_destination_overwritten": False,
            },
            "layout": layout,
            "content_access_boundary": {
                "operations": [
                    "compressed archive SHA-256",
                    "tar member metadata validation",
                    "stream copy to isolated staging",
                    "filesystem path and byte-size inventory",
                    "filename-only image/mask pairing",
                ],
                "image_decode_performed": False,
                "test_image_content_viewed": False,
                "training_started": False,
                "model_inference_started": False,
                "metrics_computed": False,
            },
        }
    finally:
        if not installed and staging.exists():
            expected_parent = destination.parent.resolve()
            if (
                staging.parent == expected_parent
                and staging.name.startswith(".mvtec-extract-")
            ):
                shutil.rmtree(staging)
            else:
                raise RuntimeError(f"refusing to clean unexpected staging path: {staging}")
