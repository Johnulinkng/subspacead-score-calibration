"""Small file-only fixtures for the report-edit compatibility verifier."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readable_delivery_metadata", ROOT / "scripts/verify_readable_delivery.py")
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def put(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def seal(root: Path, name: str, header: dict) -> dict:
    files = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": VERIFIER.digest(p)}
             for p in sorted(root.rglob("*")) if p.is_file() and p != root / name]
    value = {**header, "files": files}
    dump(root / name, value)
    return value


class ReadableMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = ROOT / "tmp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="readable-metadata-", dir=parent)
        self.package = Path(self.temp.name).resolve()
        assert self.package.is_relative_to(parent.resolve())
        self.extension = self.package / VERIFIER.EXTENSION
        put(self.package / "项目总览.html", "historic root")
        put(self.package / "report/report.html", "before report")
        put(self.package / "code/src/algorithm.py", "value = 1\n")
        put(self.package / "licenses/LICENSE.txt", "public license")
        old_master = seal(self.package, "DELIVERY-MANIFEST.json", {"status": "completed", "scientific_claim_eligible": False})
        self.historical = self.extension / "history/DELIVERY-MANIFEST.json"
        dump(self.historical, old_master)
        put(self.extension / "history/项目总览.html", "historic root")
        put(self.package / "项目总览.html", "route1 root before editing")
        put(self.extension / "report.html", "before extension report")
        self.extension_header = {"status": "completed", "categories": 27, "metric_rows": 162,
                                 "scientific_claim_eligible": False, "previous_delivery_manifest_sha256": VERIFIER.digest(self.historical),
                                 "sealed_at_utc": "before"}
        self.old_extension = seal(self.extension, "extension_manifest.json", self.extension_header)
        self.base_header = {"status": "completed", "scientific_claim_eligible": False, "updated_at": "before",
                            "route1_extension": {"extension_manifest_sha256": VERIFIER.digest(self.extension / "extension_manifest.json"),
                                                 "categories": 27, "metric_rows": 162}}
        self.baseline = seal(self.package, "DELIVERY-MANIFEST.json", self.base_header)
        self.base_sha = VERIFIER.digest(self.package / "DELIVERY-MANIFEST.json")
        self.baseline_index = VERIFIER.manifest_index(self.baseline)
        records = self.package / VERIFIER.RECORDS
        records.mkdir(parents=True)
        (records / "整理前DELIVERY-MANIFEST.json").write_bytes((self.package / "DELIVERY-MANIFEST.json").read_bytes())
        (records / "整理前extension_manifest.json").write_bytes((self.extension / "extension_manifest.json").read_bytes())
        changed = ["项目总览.html", "report/report.html", VERIFIER.EXTENSION + "/report.html"]
        rows = []
        for relative in changed:
            original = VERIFIER.RECORDS + "/原文/" + relative
            archive = self.package / original
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes((self.package / relative).read_bytes())
            put(self.package / relative, "after author prose: " + relative)
            rows.append({"path": relative, "before_sha256": VERIFIER.digest(archive),
                         "after_sha256": VERIFIER.digest(self.package / relative), "original_copy": original})
        self.ledger = {"schema_version": 1, "status": "completed", "base_master_sha256": self.base_sha, "files": rows}
        self.ledger_path = records / "文稿整理回执.json"
        dump(self.ledger_path, self.ledger)
        put(self.package / "code/代码说明.txt", "A readable guide; the executed source is unchanged.")
        self.reseal()

    def tearDown(self) -> None:
        assert self.package.is_relative_to((ROOT / "tmp").resolve())
        self.temp.cleanup()

    def reseal(self) -> None:
        seal(self.extension, "extension_manifest.json", {**self.extension_header, "sealed_at_utc": "after"})
        current_header = json.loads(json.dumps(self.base_header))
        current_header["updated_at"] = "after"
        current_header["route1_extension"]["extension_manifest_sha256"] = VERIFIER.digest(self.extension / "extension_manifest.json")
        seal(self.package, "DELIVERY-MANIFEST.json", current_header)

    def verify(self) -> dict:
        with patch.object(VERIFIER, "BASE_MASTER_SHA256", self.base_sha):
            return VERIFIER.verify_integrity(self.package, self.extension, True)

    def test_valid_prose_ledger_and_original_root_history(self) -> None:
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["authored_documents_changed"]), 3)
        self.assertEqual(result["historical_authored_documents_archived"], ["report/report.html"])
        self.assertEqual(result["root_documents_archived_before_change"], ["项目总览.html"])
        self.assertFalse(result["executed_sources_models_maps_and_numeric_results_changed"])

    def test_wrong_original_archive_rejected_even_with_new_seal(self) -> None:
        put(self.package / self.ledger["files"][1]["original_copy"], "wrong archive")
        self.reseal()
        with self.assertRaisesRegex(ValueError, "archive differs"):
            self.verify()

    def test_executable_and_provenance_never_allowed_in_ledger(self) -> None:
        for relative in ("code/src/algorithm.py", "licenses/LICENSE.txt", "report/来源说明.md", "优化实验/OAS验证/run/notes.txt"):
            with self.subTest(relative=relative):
                self.ledger["files"][0]["path"] = relative
                dump(self.ledger_path, self.ledger)
                with self.assertRaisesRegex(ValueError, "Only authored"):
                    VERIFIER.verify_editorial_ledger(self.package, self.baseline_index, self.base_sha)

    def test_source_change_cannot_be_covered_by_current_seal(self) -> None:
        put(self.package / "code/src/algorithm.py", "value = 2\n")
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Historic source"):
            self.verify()

    def test_missing_changed_document_ledger_rejected(self) -> None:
        self.ledger["files"].pop()
        dump(self.ledger_path, self.ledger)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "unregistered document changed"):
            self.verify()

    def test_original_root_history_still_required(self) -> None:
        put(self.extension / "history/项目总览.html", "wrong old root")
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Historic source"):
            self.verify()

    def test_noncanonical_ledger_copy_rejected(self) -> None:
        self.ledger["files"][0]["original_copy"] = VERIFIER.RECORDS + "/原文/../项目总览.html"
        dump(self.ledger_path, self.ledger)
        with self.assertRaisesRegex(ValueError, "preserve the relative"):
            VERIFIER.verify_editorial_ledger(self.package, self.baseline_index, self.base_sha)

    def test_manifest_cannot_change_experimental_metadata(self) -> None:
        self.extension_header["scientific_claim_eligible"] = True
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Complete frozen"):
            self.verify()

    def test_base_pin_is_not_accepted_from_ledger_alone(self) -> None:
        with patch.object(VERIFIER, "BASE_MASTER_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Archived base master"):
                VERIFIER.verify_integrity(self.package, self.extension, True)

    def test_master_verification_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the complete master"):
            VERIFIER.verify_integrity(self.package, self.extension, False)

    def test_new_master_extension_pointer_must_match_actual_seal(self) -> None:
        master_path = self.package / "DELIVERY-MANIFEST.json"
        current = VERIFIER.read(master_path)
        current["route1_extension"]["extension_manifest_sha256"] = "0" * 64
        dump(master_path, current)
        with self.assertRaisesRegex(ValueError, "Current master extension pointer"):
            self.verify()

    def test_other_route1_metadata_cannot_advance_with_pointer(self) -> None:
        master_path = self.package / "DELIVERY-MANIFEST.json"
        current = VERIFIER.read(master_path)
        current["route1_extension"]["categories"] = 28
        dump(master_path, current)
        with self.assertRaisesRegex(ValueError, "cannot alter experiment manifest metadata"):
            self.verify()

    def test_old_master_pointer_must_match_authenticated_old_seal(self) -> None:
        current = VERIFIER.read(self.package / "DELIVERY-MANIFEST.json")
        before = json.loads(json.dumps(self.baseline))
        before["route1_extension"]["extension_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Base master extension pointer"):
            VERIFIER.check_master_header(current, before,
                                         VERIFIER.digest(self.package / VERIFIER.RECORDS / "整理前extension_manifest.json"),
                                         VERIFIER.digest(self.extension / "extension_manifest.json"))

    def test_sealed_current_document_is_still_required(self) -> None:
        put(self.extension / "report.html", "edited after sealing")
        with self.assertRaisesRegex(ValueError, "differs from ledger"):
            self.verify()

    def test_untracked_current_file_rejected(self) -> None:
        put(self.package / "untracked.txt", "not in the new master")
        with self.assertRaisesRegex(ValueError, "membership differs"):
            self.verify()

    def test_no_torch_import_in_metadata_checks(self) -> None:
        self.verify()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
