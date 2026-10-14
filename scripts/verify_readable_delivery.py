"""Verify edited reports while retaining the original experiment checks.

Only the documented author text may differ from the authenticated delivery.
The original verifier still checks sources, models, maps, resources and metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
EXTENSION = "优化实验/同骨干Subspace对照-20261014"
RECORDS = "文稿整理记录/20261014"
BASE_MASTER_SHA256 = "bbf37c926d0354b576d39f0b4d972592233e84844a4b57ffb9cc22092f17ad1f"
ORIGINAL_VERIFIER_SHA256 = "d019fa0b9b73c02d9c0cae3c1a914e1e7c93ed054cb8f34bdd865b485db306b4"
ROOT_DOCS = {"项目总览.html", "先读我.txt", "实验说明.txt", "Verify.ps1"}
AUTHOR_ROOT_DOCS = ROOT_DOCS - {"Verify.ps1"}
PROTECTED_PARTS = {
    "code", "vendor", "run", "history", "licenses", "assets", "models", "results",
    "execution", "execution_history", "monitoring", "data_sources", "data", "weights",
    "公开来源", "验收记录", "资源记录", "输入缓存", "文稿整理记录",
}
PROTECTED_NAMES = ("license", "notice", "copyright", "provenance", "source", "来源", "许可", "之前的")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_relative(relative: str) -> PurePosixPath:
    require(isinstance(relative, str) and bool(relative), "A relative path is required")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and ".." not in parts.parts and "\\" not in relative and ":" not in relative,
            f"Invalid relative path: {relative!r}")
    require(parts.as_posix() == relative and "." not in parts.parts, f"Noncanonical relative path: {relative!r}")
    return parts


def member(root: Path, relative: str) -> Path:
    parts = canonical_relative(relative)
    root = root.resolve()
    path = root.joinpath(*parts.parts)
    require(not path.is_symlink(), f"Symlink is not an evidence member: {relative}")
    resolved = path.resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(), f"Missing or escaping member: {relative}")
    cursor = path.parent
    while cursor != root:
        require(not cursor.is_symlink() and not (hasattr(cursor, "is_junction") and cursor.is_junction()),
                f"Linked parent is not allowed: {relative}")
        cursor = cursor.parent
    return resolved


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def author_document(relative: str) -> bool:
    """Accept report prose, never executable, provenance or historical content."""
    parts = canonical_relative(relative)
    if parts.suffix.lower() not in {".html", ".md", ".txt"}:
        return False
    if any(part.casefold() in PROTECTED_PARTS for part in parts.parts):
        return False
    if any(token in parts.name.casefold() for token in PROTECTED_NAMES):
        return False
    if len(parts.parts) == 1:
        return relative in AUTHOR_ROOT_DOCS
    if len(parts.parts) == 2 and parts.parts[0] == "report":
        return parts.name in {"report.html", "复现说明.md"}
    return (len(parts.parts) == 3 and parts.parts[0] == "优化实验"
            and parts.parts[1] in {"DINO小型SubspaceAD验证", "OAS验证", "同骨干Subspace对照-20261014"}
            and parts.name in {"report.html", "DINO_SubspaceAD验收报告.md", "OAS验收报告.md",
                               "复现说明.txt", "给同事的说明.txt", "同骨干对照报告.txt"})


def manifest_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(isinstance(manifest.get("files"), list), "Manifest needs a files list")
    result: dict[str, dict[str, Any]] = {}
    folded = set()
    for row in manifest["files"]:
        relative = row["path"]
        canonical_relative(relative)
        require(relative.casefold() not in folded, f"Duplicate manifest member: {relative}")
        require(valid_sha(row.get("sha256")) and isinstance(row.get("size_bytes"), int)
                and not isinstance(row["size_bytes"], bool) and row["size_bytes"] >= 0,
                f"Invalid manifest hash or size: {relative}")
        result[relative] = row
        folded.add(relative.casefold())
    return result


def matches(path: Path, row: dict[str, Any]) -> bool:
    return path.stat().st_size == row["size_bytes"] and digest(path) == row["sha256"]


def check_seal(root: Path, filename: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read(member(root, filename))
    index = manifest_index(manifest)
    for relative, row in index.items():
        require(matches(member(root, relative), row), f"Current seal hash differs: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(index) | {filename}, f"Current seal membership differs: {root}")
    return manifest, index


def check_header(current: dict[str, Any], before: dict[str, Any], ignored: set[str]) -> None:
    left = {key: value for key, value in current.items() if key not in ignored}
    right = {key: value for key, value in before.items() if key not in ignored}
    require(left == right, "Document changes cannot alter experiment manifest metadata")


def check_master_header(current: dict[str, Any], before: dict[str, Any], old_extension_sha: str, current_extension_sha: str) -> None:
    """Only the pointer to the newly sealed report extension may advance."""
    previous = before.get("route1_extension")
    latest = current.get("route1_extension")
    require(isinstance(previous, dict) and isinstance(latest, dict), "Both master headers need the route1 extension metadata")
    require(previous.get("extension_manifest_sha256") == old_extension_sha,
            "Base master extension pointer differs from its authenticated original extension seal")
    require(latest.get("extension_manifest_sha256") == current_extension_sha,
            "Current master extension pointer differs from the actual extension seal")
    normalized = copy.deepcopy(current)
    normalized["route1_extension"]["extension_manifest_sha256"] = previous["extension_manifest_sha256"]
    check_header(normalized, before, {"files", "updated_at", "sealed_at_utc"})


def verify_editorial_ledger(package: Path, baseline: dict[str, dict[str, Any]], base_sha: str) -> dict[str, Any]:
    receipt = member(package, RECORDS + "/文稿整理回执.json")
    ledger = read(receipt)
    require(ledger.get("schema_version") == 1 and ledger.get("status") == "completed"
            and ledger.get("base_master_sha256") == base_sha and isinstance(ledger.get("files"), list),
            "Completed editorial ledger must name its authenticated base manifest")
    index: dict[str, dict[str, Any]] = {}
    folded = set()
    for row in ledger["files"]:
        relative = row["path"]
        require(author_document(relative), f"Only authored report prose may be edited: {relative}")
        require(relative in baseline and relative.casefold() not in folded, f"Unknown or duplicate edited document: {relative}")
        require(valid_sha(row.get("before_sha256")) and valid_sha(row.get("after_sha256"))
                and row["before_sha256"] != row["after_sha256"], f"Invalid changed-document hashes: {relative}")
        require(row["before_sha256"] == baseline[relative]["sha256"], f"Ledger before hash differs from authenticated base: {relative}")
        expected_copy = RECORDS + "/原文/" + relative
        require(row.get("original_copy") == expected_copy, f"Original copy must preserve the relative document path: {relative}")
        require(matches(member(package, expected_copy), baseline[relative]), f"Original document archive differs: {relative}")
        require(digest(member(package, relative)) == row["after_sha256"], f"Edited document differs from ledger: {relative}")
        index[relative] = row
        folded.add(relative.casefold())
    return {"path": RECORDS + "/文稿整理回执.json", "sha256": digest(receipt), "files": index}


def verify_integrity(package: Path, extension: Path, verify_master: bool = True) -> dict[str, Any]:
    """Extend only the historic-document comparison; all current seals are strict."""
    require(verify_master is True, "The edited-report verifier requires the complete master seal")
    package, extension = package.resolve(), extension.resolve()
    require(extension == package / EXTENSION, "Unexpected extension directory")
    base_path = member(package, RECORDS + "/整理前DELIVERY-MANIFEST.json")
    require(digest(base_path) == BASE_MASTER_SHA256, "Archived base master differs from the authenticated delivery")
    base = read(base_path)
    baseline = manifest_index(base)
    before_extension = member(package, RECORDS + "/整理前extension_manifest.json")
    require(matches(before_extension, baseline[EXTENSION + "/extension_manifest.json"]), "Archived extension seal differs from the base master")
    old_extension = read(before_extension)
    ledger = verify_editorial_ledger(package, baseline, BASE_MASTER_SHA256)
    info, declared = check_seal(extension, "extension_manifest.json")
    require(info.get("status") == "completed" and info.get("categories") == 27
            and info.get("metric_rows") == 162 and info.get("scientific_claim_eligible") is False,
            "Complete frozen 27-category/162-row extension is required")
    check_header(info, old_extension, {"files", "sealed_at_utc"})
    master, master_index = check_seal(package, "DELIVERY-MANIFEST.json")
    check_master_header(master, base, digest(before_extension), digest(member(extension, "extension_manifest.json")))
    require(set(baseline) <= set(master_index), "A member of the original delivery was removed")
    changed = set()
    for relative, row in baseline.items():
        if matches(member(package, relative), row):
            continue
        if relative == EXTENSION + "/extension_manifest.json":
            continue
        require(relative in ledger["files"], f"Historic source, result or unregistered document changed: {relative}")
        changed.add(relative)
    require(changed == set(ledger["files"]), "Ledger must list exactly the changed authored documents")
    historical = member(extension, "history/DELIVERY-MANIFEST.json")
    require(digest(historical) == info["previous_delivery_manifest_sha256"], "Historical pre-control manifest differs")
    original_index = manifest_index(read(historical))
    archived = []
    editorial_historical = []
    for relative, row in original_index.items():
        if matches(member(package, relative), row):
            continue
        if relative in ROOT_DOCS:
            require(matches(member(extension, "history/" + relative), row), f"Changed root document lacks its original historical bytes: {relative}")
            archived.append(relative)
            continue
        edit = ledger["files"].get(relative)
        require(edit is not None and edit["before_sha256"] == row["sha256"]
                and matches(member(package, edit["original_copy"]), row),
                f"Historical experiment evidence/source/license was replaced: {relative}")
        editorial_historical.append(relative)
    return {"status": "verified", "extension_members": len(declared),
            "historical_members_preserved": len(original_index), "root_documents_archived_before_change": archived,
            "master_manifest_verified": True, "master_members": len(master_index),
            "editorial_ledger": {key: value for key, value in ledger.items() if key != "files"},
            "authored_documents_changed": sorted(changed), "historical_authored_documents_archived": editorial_historical,
            "base_delivery_members_preserved_or_documented": len(baseline),
            "executed_sources_models_maps_and_numeric_results_changed": False}


def load_original(package: Path) -> Any:
    source = member(package, EXTENSION + "/code/scripts/verify_route1_delivery.py")
    require(digest(source) == ORIGINAL_VERIFIER_SHA256, "Original verifier source was changed")
    spec = importlib.util.spec_from_file_location("original_route1_delivery_verifier", source)
    require(spec is not None and spec.loader is not None, "Cannot load original verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_integrity = verify_integrity
    return module


def wrapper_identity(package: Path, original: Any, args: argparse.Namespace) -> dict[str, Any]:
    extension = package / EXTENSION
    identity = original.verification_identity(package, extension, extension / "run", args)
    ledger_path = member(package, RECORDS + "/文稿整理回执.json")
    ledger = read(ledger_path)
    document_identity = []
    for row in ledger["files"]:
        require(author_document(row["path"]), "Identity contains a protected edited member")
        document_identity.append({"path": row["path"], "current_sha256": digest(member(package, row["path"])),
                                  "original_sha256": digest(member(package, row["original_copy"]))})
    return {"original_verification_identity": identity,
            "original_verifier_sha256": digest(Path(original.__file__).resolve()),
            "wrapper_sha256": digest(Path(__file__).resolve()), "editorial_ledger_sha256": digest(ledger_path),
            "base_master_archive_sha256": digest(member(package, RECORDS + "/整理前DELIVERY-MANIFEST.json")),
            "base_extension_archive_sha256": digest(member(package, RECORDS + "/整理前extension_manifest.json")),
            "current_master_sha256": digest(member(package, "DELIVERY-MANIFEST.json")),
            "authored_document_identity": document_identity,
            "requested_metrics_categories": args.metrics_categories, "requested_inference_samples": args.inference_samples,
            "requested_inference_atol": args.inference_atol, "requested_inference_rtol": args.inference_rtol,
            "requested_verify_master": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--inference-samples", type=int, default=2)
    parser.add_argument("--metrics-categories", type=int, default=0)
    parser.add_argument("--inference-atol", type=float, default=1e-4)
    parser.add_argument("--inference-rtol", type=float, default=1e-4)
    parser.add_argument("--verify-master", action="store_true", default=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    package = args.package_root.resolve()
    extension, run = package / EXTENSION, package / EXTENSION / "run"
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.gettempdir()) / "readable-delivery-metric-replay"
    if work.is_relative_to(package) or args.receipt and args.receipt.resolve().is_relative_to(package):
        parser.error("Temporary work and the new receipt must remain outside the delivery")
    if args.receipt and args.receipt.exists():
        parser.error("A new receipt path is required; prior acceptance receipts are never overwritten")
    if not 0 <= args.metrics_categories <= 27 or not 0 <= args.inference_samples <= 54:
        parser.error("Metric categories must be 0..27 and inference samples 0..54")
    if any(not math.isfinite(v) or not 0 <= v <= 1e-3 for v in (args.inference_atol, args.inference_rtol)):
        parser.error("Inference tolerances must be finite within 0..1e-3")
    started = time.monotonic()
    try:
        original = load_original(package)
        identity = wrapper_identity(package, original, args)
        integrity = original.verify_integrity(package, extension, True)
        work.mkdir(parents=True, exist_ok=True)
        sys.path[:0] = [str(extension / "code/src"), str(extension / "code/scripts")]
        sources = original.verify_sources(package, extension, run)
        cache, tasks = original.verify_cache(package, extension, run)
        resources = original.verify_resources(extension)
        inference = original.verify_inference(package, extension, run, args.inference_samples, args.inference_atol, args.inference_rtol)
        metrics = original.verify_metrics(run, tasks, args.metrics_categories, work)
        after_identity = wrapper_identity(package, original, args)
        require(after_identity == identity, "Package, ledger, verifier or requested scope changed during acceptance")
        result = {"status": "verified", "verification_identity": identity["original_verification_identity"],
                  "editorial_verification_identity": identity, "identity_before": identity, "identity_after": after_identity,
                  "integrity": integrity, "sources": sources, "cache": cache, "resources": resources,
                  "fresh_cpu_inference": inference, "metric_replay": metrics,
                  "elapsed_seconds": time.monotonic() - started, "network_access_required": False,
                  "gradient_training_performed": False, "normal_training_fit_performed": False,
                  "gpu_used": False, "cpu_threads": 4, "original_algorithm_checks_reused_unchanged": True}
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, ImportError, TypeError, AttributeError) as error:
        print(json.dumps({"status": "verification_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
