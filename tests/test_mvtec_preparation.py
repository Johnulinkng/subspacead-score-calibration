from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from industrial_anomaly_lab.mvtec import (
    MVTEC_CATEGORIES,
    audit_mvtec_layout,
    extract_and_audit_mvtec,
)


def _write_nonempty(path: Path, payload: bytes = b"not-decoded-image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _build_synthetic_tree(root: Path) -> None:
    for category in MVTEC_CATEGORIES:
        _write_nonempty(root / category / "train/good/000.png")
        _write_nonempty(root / category / "test/good/000.png")
        _write_nonempty(root / category / "test/defect/001.png")
        _write_nonempty(root / category / "ground_truth/defect/001_mask.png")
        _write_nonempty(root / category / "readme.txt", b"metadata")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safe_extract_audits_all_categories_without_decoding_images(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source/mvtec_anomaly_detection"
    _build_synthetic_tree(source)
    archive = tmp_path / "mvtec.tar.xz"
    with tarfile.open(archive, "w:xz") as handle:
        handle.add(source, arcname="mvtec_anomaly_detection")

    destination = tmp_path / "installed"
    report = extract_and_audit_mvtec(
        archive,
        destination,
        expected_bytes=archive.stat().st_size,
        expected_sha256=_sha256(archive),
    )

    assert report["archive"]["path_safety_passed"] is True
    assert report["archive"]["archive_root_prefix"] == "mvtec_anomaly_detection"
    assert report["layout"]["category_count"] == 15
    assert report["layout"]["totals"]["train_good_images"] == 15
    assert report["layout"]["totals"]["test_good_images"] == 15
    assert report["layout"]["totals"]["test_anomaly_images"] == 15
    assert report["layout"]["totals"]["ground_truth_masks"] == 15
    assert report["layout"]["image_decode_performed"] is False
    assert report["content_access_boundary"]["metrics_computed"] is False
    assert (destination / "bottle/test/defect/001.png").read_bytes() == b"not-decoded-image"


def test_safe_extract_rejects_parent_traversal_and_removes_staging(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "malicious.tar.xz"
    payload = b"escape"
    with tarfile.open(archive, "w:xz") as handle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="traverses a parent"):
        extract_and_audit_mvtec(
            archive,
            destination,
            expected_bytes=archive.stat().st_size,
            expected_sha256=_sha256(archive),
        )

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not list(tmp_path.glob(".mvtec-extract-*"))


def test_layout_audit_rejects_orphan_ground_truth_mask(tmp_path: Path) -> None:
    _build_synthetic_tree(tmp_path)
    _write_nonempty(tmp_path / "bottle/ground_truth/defect/999_mask.png")

    with pytest.raises(ValueError, match="mask inventory mismatch"):
        audit_mvtec_layout(tmp_path)


def test_layout_audit_rejects_missing_category(tmp_path: Path) -> None:
    _build_synthetic_tree(tmp_path)
    missing = tmp_path / "zipper"
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()

    with pytest.raises(ValueError, match="category set mismatch"):
        audit_mvtec_layout(tmp_path)
