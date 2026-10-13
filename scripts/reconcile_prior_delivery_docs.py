"""Reseal exactly two known pre-experiment document edits, preserving evidence.

Stdlib only. No document/result/source is edited and no model is executed.
All historical members are authenticated before the master-only replacement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


KNOWN_DOCS = {
    "先读我.txt": {"old_size": 2527, "old_sha": "6b6d484fa0057db739a938f9615651265677a7ce3da174fdba0bfca52584e79f", "current_size": 2474, "current_sha": "0e1cafe1be4dcd0a06914ae7427bec51491d117db232e141c7f9421909117af4"},
    "实验说明.txt": {"old_size": 2045, "old_sha": "363adc0aac3a9a1b5acc0529352469eaea019ea27fced492faef78c06b332fe2", "current_size": 2004, "current_sha": "358bda7490f5d09daa830d8b7cb1191ecc98842625aedbcd5831c11ef56a878c"},
}
PRIOR_RECEIPT_SHA = "b1771978eaafbaffadd9e3d368c60e7d66e32da40622aa81250adf562f01f92a"
ORIGINAL_MASTER_SHA = "a2d7862686dbcd742d9ec161f2b570982e30a91d5339d01df1a1ff91329b0d75"
PACKAGE_RELATIVE = "artifacts/deliveries/quick-padim-20261012/交付文件夹"
PRIOR_RECEIPT_RELATIVE = "artifacts/deliveries/quick-padim-20261012/内部整理记录-20261012/DINO内容整理/检查回执.json"
EVIDENCE_RELATIVE = "artifacts/audits/route1-prior-doc-reseal-20261014"
MAIN_STATE_RELATIVE = "runs/quick/route1-controls-20261014/run_state.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse(raw: bytes):
    return json.loads(raw.decode("utf-8-sig"))


def member(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    require(bool(relative) and not parts.is_absolute() and ".." not in parts.parts and "\\" not in relative and ":" not in relative, "Invalid historical member path")
    target = root.joinpath(*parts.parts).absolute()
    require(target.is_file() and not target.is_symlink() and target.resolve(strict=True) == target and target.is_relative_to(root), "Historical member is missing, redirected, or escaping")
    return target


def check(project: Path, package: Path) -> dict:
    require(package == (project / PACKAGE_RELATIVE).resolve(strict=True), "Only the known historical delivery can be reconciled")
    evidence = project / EVIDENCE_RELATIVE
    require(evidence.parent.resolve(strict=True).is_relative_to(project) and not evidence.exists(), "An exclusive fresh external evidence directory is required")
    master = member(package, "DELIVERY-MANIFEST.json")
    original_bytes = master.read_bytes()
    require(sha_bytes(original_bytes) == ORIGINAL_MASTER_SHA, "Original master is not the known DINO maintenance seal")
    original = parse(original_bytes)
    require(original.get("schema_version") == 1 and original.get("status") == "completed" and original.get("delivery_type") == "folder" and len(original["files"]) == 24946, "Original complete 24946-member manifest required")
    prior = member(project, PRIOR_RECEIPT_RELATIVE)
    prior_bytes = prior.read_bytes()
    require(sha_bytes(prior_bytes) == PRIOR_RECEIPT_SHA, "Original DINO documentation check receipt changed")
    prior_record = parse(prior_bytes)
    require(prior_record["status"] == "verified" and prior_record["manifest_sha256"] == ORIGINAL_MASTER_SHA and prior_record["full_file_integrity"]["member_count"] == 24946 and prior_record["full_file_integrity"]["all_member_hashes_verified"] is True and prior_record["result_values_changed"] is False, "Original DINO maintenance receipt identity/scope differs")
    main_state = member(project, MAIN_STATE_RELATIVE)
    main_bytes = main_state.read_bytes()
    main_started = datetime.fromisoformat(parse(main_bytes)["started_at_utc"].replace("Z", "+00:00"))
    require(main_started.utcoffset() is not None, "Frozen main start must include timezone")
    rows = {row["path"]: row for row in original["files"]}
    require(len(rows) == 24946 and "DELIVERY-MANIFEST.json" not in rows, "Duplicate or self-referential original member")
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file() and path != master}
    require(actual == set(rows), "Historical delivery file membership changed")
    doc_bytes, changes = {}, []
    for relative, row in rows.items():
        path = member(package, relative)
        if relative in KNOWN_DOCS:
            known = KNOWN_DOCS[relative]
            raw, stat = path.read_bytes(), path.stat()
            require(row["size_bytes"] == known["old_size"] and row["sha256"] == known["old_sha"] and len(raw) == known["current_size"] and sha_bytes(raw) == known["current_sha"], "A document differs from the exact four known SHA/size values")
            require(stat.st_mtime < main_started.timestamp(), "Document edit is not older than the frozen main start")
            doc_bytes[relative] = raw
            changes.append({"path": relative, **known, "current_mtime_ns": stat.st_mtime_ns, "current_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        else:
            require(path.stat().st_size == row["size_bytes"] and sha_file(path) == row["sha256"], "Non-allowlisted historical file drift: " + relative)
    require(set(doc_bytes) == set(KNOWN_DOCS), "Both known document edits are required")
    updated = copy.deepcopy(original)
    for row in updated["files"]:
        if row["path"] in KNOWN_DOCS:
            known = KNOWN_DOCS[row["path"]]
            row["size_bytes"], row["sha256"] = known["current_size"], known["current_sha"]
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    reversible = copy.deepcopy(updated)
    reversible["updated_at"] = original["updated_at"]
    for row in reversible["files"]:
        if row["path"] in KNOWN_DOCS:
            known = KNOWN_DOCS[row["path"]]
            row["size_bytes"], row["sha256"] = known["old_size"], known["old_sha"]
    require(reversible == original, "Reseal changed values beyond two entries and updated_at")
    updated_bytes = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return {"master": master, "original_bytes": original_bytes, "updated_bytes": updated_bytes, "doc_bytes": doc_bytes, "prior_bytes": prior_bytes, "main_bytes_sha256": sha_bytes(main_bytes), "main_started_at_utc": main_started.astimezone(timezone.utc).isoformat(), "evidence": evidence, "changes": changes, "members": len(rows)}


def write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
    require(path.read_bytes() == raw, "Evidence copy byte verification failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        project, package = args.project_root.resolve(strict=True), args.package_root.resolve(strict=True)
        checked = check(project, package)
        report = {"schema_version": 1, "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(), "package_root": str(package), "historical_members_authenticated": checked["members"], "exact_file_membership_authenticated": True, "non_document_member_hashes_unchanged": 24944, "only_known_two_document_drifts": checked["changes"], "document_mtimes_before_frozen_main_start": True, "frozen_main_started_at_utc": checked["main_started_at_utc"], "frozen_main_state_sha256": checked["main_bytes_sha256"], "original_master_sha256": sha_bytes(checked["original_bytes"]), "planned_master_sha256": sha_bytes(checked["updated_bytes"]), "original_dino_maintenance_receipt_sha256": PRIOR_RECEIPT_SHA, "documents_restored_or_modified": False, "result_or_computation_source_changed": False, "model_or_metric_computation_performed": False, "utility_sha256": sha_file(Path(__file__).resolve())}
        if args.check_only:
            print(json.dumps({**report, "status": "ready_to_reseal", "delivery_modified": False, "evidence_written": False}, ensure_ascii=False, indent=2))
            return 0
        evidence = checked["evidence"]
        evidence.mkdir(exist_ok=False)
        write_exclusive(evidence / "original_DELIVERY-MANIFEST.json", checked["original_bytes"])
        for name, raw in checked["doc_bytes"].items():
            write_exclusive(evidence / ("current_" + name), raw)
            require(member(package, name).read_bytes() == raw and member(package, name).stat().st_mtime_ns == next(row["current_mtime_ns"] for row in checked["changes"] if row["path"] == name), "Allowlisted document changed during authentication")
        write_exclusive(evidence / "original_DINO_document_check_receipt.json", checked["prior_bytes"])
        write_exclusive(evidence / "check_receipt.json", (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        require(checked["master"].read_bytes() == checked["original_bytes"] and sha_file(member(project, PRIOR_RECEIPT_RELATIVE)) == PRIOR_RECEIPT_SHA, "Master or prior maintenance receipt changed during authentication")
        checked["master"].write_bytes(checked["updated_bytes"])
        after_sha = sha_file(checked["master"])
        require(after_sha == report["planned_master_sha256"], "Actual master replacement differs from the checked plan")
        final = {**report, "status": "completed", "master_before_sha256": report["original_master_sha256"], "master_after_sha256": after_sha, "evidence_directory": str(evidence), "master_entries_changed": list(KNOWN_DOCS), "other_records_order_and_values_preserved": True, "current_document_bytes_preserved": True, "self_hash_scope": "receipts remain outside the package; no receipt assigns itself a hash"}
        write_exclusive(evidence / "reseal_receipt.json", (json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        print(json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "prior_doc_reconciliation_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
