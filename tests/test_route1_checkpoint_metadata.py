"""Tiny metadata fixtures only: no models, arrays, delivery writes, or processes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("route1_finish_metadata_fixture", PROJECT / "scripts/finish_route1_delivery.py")
FINISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINISH)
TIMERS = ("fresh_forward_seconds", "cache_read_validation_seconds", "source_validation_seconds", "cache_write_seconds", "hook_total_seconds")


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class MetadataFixture:
    """Synthetic producer receipts; placeholder hashes never represent NPY data."""

    def __init__(self, root: Path):
        self.project = root.resolve()
        self.extension = self.project / "fixture-extension"
        self.category = ("fixture", "alpha")
        self.execution = "fixture-session"
        self.rows = []
        scripts = self.project / "scripts"
        scripts.mkdir(parents=True)
        for name in ("checkpoint_resume_route1_controls.py", "resume_route1_controls.py"):
            (scripts / name).write_text("# Metadata fixture placeholder, never imported.\n", encoding="utf-8")
        self.helper_sha = FINISH.digest(scripts / "checkpoint_resume_route1_controls.py")
        self.delegate_sha = FINISH.digest(scripts / "resume_route1_controls.py")
        self.config = {
            "source_sha256": {"fixture-core.py": sha("fixture core")},
            "environment_sha256": {"fixture-lock.txt": sha("fixture environment")},
            "vendor_source_sha256": {"fixture-vendor.py": sha("fixture vendor")},
            "input_metadata_sha256": {"fixture-input.json": sha("fixture inputs")},
            "protocol_sha256": sha("fixture protocol"),
            "protocol": {"weight_files_sha256": {"fixture-weights": sha("fixture weights")}},
            "backbone": {"checkpoint_sha256": sha("fixture checkpoint")},
            "expected_matrix": {"categories": [list(self.category)]},
        }
        write_json(self.extension / "run/run_config.json", self.config)
        full_run = self.project / "runs/quick/route1-controls-20261014"
        write_json(self.extension / "extension_manifest.json", {"source_run": str(full_run)})
        self.context = {
            "schema_version": 1, "checkpoint_cache_schema": 1,
            "full_run": str(full_run), "project_root": str(self.project),
            "checkpoint_cache_root": str(self.project / "runs/quick/route1-checkpoint-cache-20261014"),
            "run_config_sha256": FINISH.digest(self.extension / "run/run_config.json"),
            "protocol_sha256": self.config["protocol_sha256"],
            **{field: self.config[field] for field in ("source_sha256", "environment_sha256", "vendor_source_sha256", "input_metadata_sha256")},
            "weight_files_sha256": self.config["protocol"]["weight_files_sha256"],
            "checkpoint_sha256": self.config["backbone"]["checkpoint_sha256"],
            "checkpoint_helper_sha256": self.helper_sha, "delegate_adapter_sha256": self.delegate_sha,
            "categories": self.config["expected_matrix"]["categories"],
            "device": "cpu", "cpu_threads": 4, "feature_dtype": "float32",
            "returned_shape": [32, 32, 384], "stored_shape": [1024, 384], "image_res": 448, "layers": [-4, -5],
            "runtime_package_versions": {name: "fixture-version" for name in ("torch", "torchvision", "numpy", "transformers", "Pillow", "scipy", "scikit-learn", "opencv-python", "threadpoolctl", "safetensors")},
            "python_version": "fixture Python", "python_executable": str(self.project / "fixture-python.exe"), "platform": "fixture platform",
            "thread_environment": {name: "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        }
        self.session = self.extension / "run/checkpoint_execution_history" / self.execution
        self.context_sha = self.seal_context()
        self.cat = self.extension / "run/categories" / Path(*self.category)
        self.train = [{"path": f"fixture-sources/normal-{index}.png", "sha256": sha(f"normal {index}")} for index in range(2)]
        self.test = [{"image_path": str(self.project / f"fixture-sources/test-{index}.png"), "image_sha256": sha(f"test {index}")} for index in range(2)]
        write_json(self.cat / "train_manifest.json", self.train)
        write_json(self.cat / "test_inventory.json", self.test)
        write_json(self.cat / "completion.json", {"status": "completed", "fixture": True})
        self.tokens = [self.token(index) for index in range(2)]
        write_json(self.cat / "token_cache_manifest.json", {"files": [{"sha256": row["token_file_sha256"]} for row in self.tokens]})
        for index, token in enumerate(self.tokens):
            self.rows.append({"type": "cache_commit", "context_sha256": self.context_sha, **token, "dataset": self.category[0], "category": self.category[1], "normal_index": index})
            self.rows.append(self.image("normal", index, index))
        self.rows.extend(self.image("test", index, index + 2) for index in range(2))
        self.save()

    def seal_context(self) -> str:
        canonical = (json.dumps(self.context, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        context_sha = sha(canonical)
        write_json(self.session / "context.json", {"context_sha256": context_sha, "context": self.context})
        return context_sha

    def token(self, index: int) -> dict:
        return {"normal_index": index, "cache_entry": f"{self.category[0]}/{self.category[1]}/{index:06d}", "producer_execution_id": self.execution,
                "metadata_sha256": sha(f"metadata {index}"), "token_file_sha256": sha(f"placeholder file {index}"), "token_content_sha256": sha(f"placeholder content {index}"),
                "source_image_sha256": self.train[index]["sha256"]}

    def image(self, phase: str, index: int, cursor: int) -> dict:
        normal = phase == "normal"
        row = {"type": "image_result", "context_sha256": self.context_sha, "execution_id": self.execution, "event_index": cursor,
               "dataset": self.category[0], "category": self.category[1], "phase": phase, "image_index": index,
               "source_path": str((self.project / self.train[index]["path"]).resolve()) if normal else self.test[index]["image_path"],
               "source_image_sha256": self.train[index]["sha256"] if normal else self.test[index]["image_sha256"],
               "action": "normal_fresh" if normal else "test_fresh", "fresh_forward_seconds": 1.0, "cache_read_validation_seconds": 0.0,
               "source_validation_seconds": 0.1, "cache_write_seconds": 0.2 if normal else 0.0, "hook_total_seconds": 1.3 if normal else 1.1}
        row.update({field: self.tokens[index][field] if normal else None for field in ("cache_entry", "metadata_sha256", "token_file_sha256", "token_content_sha256")})
        return row

    def save(self, tail: bytes = b"") -> bytes:
        raw = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for row in self.rows) + tail
        (self.session / "events.jsonl").write_bytes(raw)
        return raw

    def finalize(self, status: str = "completed", failed_forward_seconds: float = 0.0) -> None:
        images = [row for row in self.rows if row["type"] == "image_result"]
        counts = {name + "_count": sum(row["action"] == name for row in images) for name in ("normal_fresh", "normal_cache_hit", "test_fresh")}
        forward_successes = counts["normal_fresh_count"] + counts["test_fresh_count"]
        summary = {"schema_version": 1, "status": status, "execution_id": self.execution, "context_sha256": self.context_sha,
                   "checkpoint_helper_sha256": self.helper_sha, "delegate_adapter_sha256": self.delegate_sha, **counts,
                   "events_consumed": len(images), "expected_events": 4, "expected_normal_events": 2, "expected_test_events": 2,
                   "queue_complete": len(images) == 4, "forward_attempt_count": forward_successes + bool(failed_forward_seconds),
                   "forward_success_count": forward_successes, "failed_forward_count": int(bool(failed_forward_seconds)),
                   "test_always_fresh": True, "algorithms_and_protocol_unchanged": True, "fit_only_and_fresh_resource_workers_changed": False,
                   "adopted_unsealed_token_count": 0, "normal_extraction_seconds_scope": "includes normal cache read/validation and fresh forwards; not full first-pass DINO time",
                   "elapsed_seconds": 20.0, "startup_context_and_queue_seconds": 0.5,
                   **{field: sum(row[field] for row in images) for field in TIMERS}}
        summary["fresh_forward_seconds"] += failed_forward_seconds
        write_json(self.session / "summary.json", summary)

    def cleanup(self, indexed: bool) -> None:
        relative = f"cache_summary_{self.category[0]}_{self.category[1]}.json"
        path = self.session / relative
        write_json(path, {"dataset": self.category[0], "category": self.category[1], "completion_sha256": FINISH.digest(self.cat / "completion.json"),
                          "context_sha256": self.context_sha, "cache_removed_after_completed_category": True, "authenticated_token_records": self.tokens})
        if indexed:
            write_json(self.session / "completed_cache_summary.json", [{"dataset": self.category[0], "category": self.category[1], "summary_path": relative,
                                                                       "summary_sha256": FINISH.digest(path), "normal_images": 2}])

    def audit(self) -> dict:
        return FINISH.validate_checkpoint_history(self.project, self.extension, self.config)


@pytest.fixture
def metadata():
    scratch = PROJECT / "tmp"
    scratch.mkdir(exist_ok=True)
    assert scratch.resolve() == scratch.absolute() and scratch.resolve().is_relative_to(PROJECT.resolve())
    with tempfile.TemporaryDirectory(prefix="route1-checkpoint-metadata-fixture-", dir=scratch) as directory:
        root = Path(directory).resolve()
        assert root.is_relative_to(scratch.resolve())
        yield MetadataFixture(root)


def test_complete_producer_context_source_and_normal_then_test(metadata):
    metadata.finalize()
    metadata.cleanup(indexed=True)
    audited = metadata.audit()
    session = audited["sessions"][0]
    assert audited["context_sha256"] == metadata.context_sha
    assert audited["producer_commits"] == audited["completed_token_bindings"] == 2
    assert (session["normal_fresh_count"], session["normal_cache_hit_count"], session["test_fresh_count"]) == (2, 0, 2)
    assert session["cleanup_seals"][0]["physical_removal_evidence"] == "completed_by_final_cleanup_index"


def test_resumed_cache_hits_bind_previous_session_producer(metadata):
    metadata.rows = metadata.rows[:4]
    metadata.save()
    history = metadata.session.parent
    rows = []
    for index in range(2):
        row = metadata.image("normal", index, index)
        row.update({"execution_id": "fixture-resumed", "action": "normal_cache_hit", "fresh_forward_seconds": 0.0,
                    "cache_read_validation_seconds": 0.4, "cache_write_seconds": 0.0, "hook_total_seconds": 0.5})
        rows.append(row)
    for index in range(2):
        row = metadata.image("test", index, index + 2)
        row["execution_id"] = "fixture-resumed"
        rows.append(row)
    metadata.execution, metadata.session, metadata.rows = "fixture-resumed", history / "fixture-resumed", rows
    assert metadata.seal_context() == metadata.context_sha
    metadata.save()
    metadata.finalize()
    audited = metadata.audit()
    resumed = next(row for row in audited["sessions"] if row["execution_id"] == "fixture-resumed")
    assert audited["producer_commits"] == 2
    assert (resumed["normal_fresh_count"], resumed["normal_cache_hit_count"], resumed["test_fresh_count"]) == (0, 2, 2)
    assert resumed["returned_image_timing_sums"]["fresh_forward_seconds"] == 2.0


@pytest.mark.parametrize("bad_index", [0, 2])
def test_locally_signed_duplicate_or_jump_is_rejected(metadata, bad_index):
    # Index 2 switches to locally valid Test 0 before all normal images.
    metadata.rows[3] = metadata.image("normal", 0, 1) if bad_index == 0 else metadata.image("test", 0, 1)
    metadata.save()
    with pytest.raises(ValueError, match="order is repeated or discontinuous"):
        metadata.audit()


def test_locally_signed_first_normal_index_skip_is_rejected(metadata):
    metadata.rows[1] = metadata.image("normal", 1, 0)
    metadata.save()
    with pytest.raises(ValueError, match="category order changed"):
        metadata.audit()


@pytest.mark.parametrize("tail", [b'{"type":"image_result"', b'{"error":"' + "中".encode("utf-8")[:2]])
def test_unfinalized_truncated_json_or_utf8_is_retained(metadata, tail):
    original = metadata.save(tail)
    audited = metadata.audit()["sessions"][0]
    assert audited["status"] == "interrupted_without_final_summary"
    assert audited["ignored_truncated_tail_bytes"] == len(tail)
    assert audited["ignored_truncated_tail_sha256"] == sha(tail)
    assert audited["journal_sha256"] == sha(original)
    assert (metadata.session / "events.jsonl").read_bytes() == original
    assert audited["test_fresh_count"] == 2


def test_bad_middle_json_is_rejected(metadata):
    raw = metadata.save()
    first, rest = raw.split(b"\n", 1)
    (metadata.session / "events.jsonl").write_bytes(first + b"\n{broken\n" + rest)
    with pytest.raises(ValueError, match="corrupt completed or intermediate event"):
        metadata.audit()


def test_finalized_bad_tail_is_rejected(metadata):
    metadata.finalize()
    metadata.save(b'{"unfinished":')
    with pytest.raises(ValueError, match="corrupt completed or intermediate event"):
        metadata.audit()


def test_orphan_cleanup_is_authenticated_without_removal_claim(metadata):
    metadata.cleanup(indexed=False)
    audited = metadata.audit()
    session = audited["sessions"][0]
    assert audited["completed_token_bindings"] == 2
    assert session["cleanup_index_present"] is False
    assert session["unindexed_cleanup_seals"] == 1
    assert session["cleanup_seals"][0]["physical_removal_evidence"] == "not_proven_intent_only"


def test_self_signed_wrong_source_dependency_is_rejected(metadata):
    metadata.context["source_sha256"] = {"fixture-core.py": sha("changed fixture core")}
    metadata.seal_context()
    with pytest.raises(ValueError, match="changed execution dependency hashes"):
        metadata.audit()


def test_failed_forward_time_is_preserved_without_inventing_image_results(metadata):
    metadata.rows = metadata.rows[:4]
    metadata.save()
    metadata.finalize(status="failed", failed_forward_seconds=9.0)
    session = metadata.audit()["sessions"][0]
    assert session["returned_image_timing_sums"]["fresh_forward_seconds"] == 2.0
    assert session["reported_summary"]["fresh_forward_seconds"] == 11.0
    assert session["reported_summary"]["failed_forward_count"] == 1
    assert session["test_fresh_count"] == 0


def test_changed_producer_hash_is_rejected(metadata):
    metadata.rows[1]["metadata_sha256"] = sha("other metadata")
    metadata.save()
    with pytest.raises(ValueError, match="lacks an authenticated producer commit"):
        metadata.audit()
