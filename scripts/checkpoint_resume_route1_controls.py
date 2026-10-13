"""Resume frozen route-1 controls with authenticated normal-image checkpoints.

Only this process wraps LocalDinoExtractor.extract, the PIL entry actually
used by the frozen worker. Normal tokens produced by this helper receive an
immediate producer seal; Test always invokes the original forward method.
The existing resume adapter still owns all fitting, evaluation and outputs.
Unsealed tokens from other runners are never adopted. Normal extraction
wall time includes checkpoint validation/reads and must not be presented as
the time for a fresh full DINO extraction. Resource-trial workers are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "runs/quick/route1-checkpoint-cache-20261014"
RETURNED_SHAPE = (32, 32, 384)
STORED_SHAPE = (1024, 384)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    require(not path.exists(), "Refusing to overwrite checkpoint metadata: " + str(path))
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_in(root, relative):
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Path escapes authenticated root: " + str(path))
    return path


def authenticated_context(run, config_path):
    saved = run / "run_config.json"
    config = read(saved)
    require(config["pilot"] is False, "Checkpoint resume requires the full experiment")
    require(digest(config_path) == config["protocol_sha256"] and read(config_path) == config["protocol"], "Frozen protocol changed")
    for name in ("source_sha256", "environment_sha256"):
        for relative, expected in config[name].items():
            require(digest(file_in(ROOT, relative)) == expected, "Frozen context drift: " + relative)
    protocol = config["protocol"]
    source = (ROOT / protocol["source_run"]).resolve(strict=True)
    require(source.is_relative_to(ROOT), "Historical source escapes project")
    for relative, expected in config["input_metadata_sha256"].items():
        require(digest(file_in(source, relative)) == expected, "Historical input metadata changed: " + relative)
    for relative, expected in config["vendor_source_sha256"].items():
        require(digest(file_in(ROOT / protocol["vendor_root"], relative)) == expected, "Vendor source changed")
    for relative, expected in protocol["weight_files_sha256"].items():
        require(digest(file_in(ROOT / protocol["weights_dir"], relative)) == expected, "Weight artifact changed")
    context = {
        "schema_version": 1, "full_run": str(run), "project_root": str(ROOT),
        "checkpoint_cache_root": str(CACHE_ROOT.absolute()),
        "run_config_sha256": digest(saved), "protocol_sha256": config["protocol_sha256"],
        "source_sha256": config["source_sha256"], "environment_sha256": config["environment_sha256"],
        "vendor_source_sha256": config["vendor_source_sha256"],
        "input_metadata_sha256": config["input_metadata_sha256"],
        "weight_files_sha256": protocol["weight_files_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "checkpoint_helper_sha256": digest(Path(__file__).resolve()),
        "delegate_adapter_sha256": digest(ROOT / "scripts/resume_route1_controls.py"),
        "feature_dtype": "float32", "returned_shape": list(RETURNED_SHAPE),
        "stored_shape": list(STORED_SHAPE), "device": "cpu",
        "image_res": protocol["image_res"], "layers": protocol["layers"],
        "cpu_threads": protocol["cpu_threads"], "checkpoint_cache_schema": 1,
        "categories": config["expected_matrix"]["categories"],
        "runtime_package_versions": {name: importlib.metadata.version(name) for name in
                                     ("torch", "torchvision", "numpy", "transformers", "Pillow", "scipy", "scikit-learn", "opencv-python", "threadpoolctl", "safetensors")},
        "python_version": sys.version, "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    return config, source, context


def event_queue(run, source, config):
    events = []
    for dataset, category in config["expected_matrix"]["categories"]:
        require(all(part and part.replace("_", "").isalnum() for part in (dataset, category)), "Invalid category path component")
        completion = run / "categories" / dataset / category / "completion.json"
        if completion.exists():
            require(read(completion)["status"] == "completed", "Existing category completion is invalid")
            continue
        cat = source / "categories" / dataset / category
        train, inventory = read(cat / "train_manifest.json"), read(cat / "test_inventory.json")
        for phase, records in (("normal", train), ("test", inventory)):
            for index, record in enumerate(records):
                path = (ROOT / record["path"]).resolve(strict=True) if phase == "normal" else Path(record["image_path"]).resolve(strict=True)
                require(path.is_relative_to(ROOT), "Image escapes project")
                if phase == "normal":
                    require(path.parent.name == "good" and path.parent.parent.name == "train", "Checkpoint event is not normal training")
                events.append({
                    "event_index": len(events), "dataset": dataset, "category": category,
                    "phase": phase, "image_index": index, "source_path": str(path),
                    "source_image_sha256": record["sha256"] if phase == "normal" else record["image_sha256"],
                })
    return events


class CheckpointHook:
    """Queue-checked extractor wrapper; no constructor or probe forwards."""

    def __init__(self, *, run, cache_root, context, events, execution_id, history):
        self.run, self.cache_root = Path(run).resolve(), Path(cache_root).resolve()
        self.context, self.events, self.execution_id = context, events, execution_id
        self.history = Path(history)
        self.context_sha256 = hashlib.sha256(encoded(context)).hexdigest()
        self.cursor = 0
        self.commits = {}
        self.category_counts = {}
        self.counts = {"normal_fresh_count": 0, "normal_cache_hit_count": 0, "test_fresh_count": 0}
        self.seconds = {name: 0.0 for name in ("fresh_forward_seconds", "cache_read_validation_seconds", "source_validation_seconds", "cache_write_seconds", "hook_total_seconds")}
        self.forward_attempt_count = self.forward_success_count = self.failed_forward_count = 0
        self.cache_root.mkdir(parents=True, exist_ok=True)
        context_path = self.cache_root / "context.json"
        if context_path.exists():
            previous = read(context_path)
            require(previous == {"context_sha256": self.context_sha256, "context": context}, "Independent checkpoint-cache context changed")
        else:
            atomic_json(context_path, {"context_sha256": self.context_sha256, "context": context})
        for journal in sorted((self.run / "checkpoint_execution_history").glob("*/events.jsonl")):
            for line in journal.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "cache_commit" and row.get("context_sha256") == self.context_sha256:
                    key = (row["producer_execution_id"], row["cache_entry"])
                    require(key not in self.commits, "Duplicate checkpoint producer commit")
                    self.commits[key] = row

    def append(self, row):
        with (self.history / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())

    def source_identity(self, image, event):
        from PIL import Image
        import numpy as np

        require(isinstance(image, Image.Image), "Frozen extractor entry requires a PIL image")
        path = Path(event["source_path"]).resolve(strict=True)
        supplied_filename = getattr(image, "filename", None)
        if supplied_filename:
            require(Path(supplied_filename).resolve() == path, "PIL filename does not match the next frozen image path")
        require(digest(path) == event["source_image_sha256"], "Authenticated source image changed")
        with Image.open(path) as expected_image:
            expected = np.asarray(expected_image.convert("RGB"))
        supplied = np.asarray(image.convert("RGB"))
        require(supplied.shape == expected.shape and np.array_equal(supplied, expected), "PIL RGB pixels do not match next frozen image event")
        return {"source_rgb_shape": list(expected.shape), "source_rgb_sha256": hashlib.sha256(memoryview(expected).cast("B")).hexdigest()}

    def cache_paths(self, event):
        entry = self.cache_root / event["dataset"] / event["category"] / f"{event['image_index']:06d}"
        require(entry.resolve().is_relative_to(self.cache_root), "Checkpoint entry escapes dedicated cache root")
        return entry, entry / "tokens.npy", entry / "metadata.json"

    def token_array(self, values, *, returned):
        import numpy as np
        array = np.asarray(values)
        require(array.shape == (RETURNED_SHAPE if returned else STORED_SHAPE) and array.dtype == np.float32 and np.isfinite(array).all(), "Invalid checkpoint float32 token geometry")
        return np.ascontiguousarray(array).reshape(STORED_SHAPE)

    def read_cache(self, event, rgb, entry, token_path, metadata_path):
        import numpy as np
        require(entry.is_dir() and token_path.is_file() and metadata_path.is_file(), "Existing checkpoint entry is incomplete; refusing overwrite")
        require({p.name for p in entry.iterdir()} == {"tokens.npy", "metadata.json"}, "Unexpected checkpoint entry payload")
        metadata = read(metadata_path)
        expected = {
            "schema_version": 1, "status": "committed", "context_sha256": self.context_sha256,
            "dataset": event["dataset"], "category": event["category"], "normal_index": event["image_index"],
            "source_path": event["source_path"], "source_image_sha256": event["source_image_sha256"],
            "stored_shape": list(STORED_SHAPE), "returned_shape": list(RETURNED_SHAPE),
            "token_dtype": "float32", **rgb,
        }
        require(all(metadata.get(key) == value for key, value in expected.items()), "Checkpoint input/context binding mismatch; refusing overwrite")
        relative = entry.relative_to(self.cache_root).as_posix()
        commit = self.commits.get((metadata["producer_execution_id"], relative))
        require(commit is not None and commit["metadata_sha256"] == digest(metadata_path), "Checkpoint producer seal is missing or changed")
        require(digest(token_path) == metadata["token_file_sha256"] == commit["token_file_sha256"], "Checkpoint token file SHA256 mismatch")
        require(token_path.stat().st_size == metadata["token_file_size_bytes"], "Checkpoint token size mismatch")
        values = self.token_array(np.load(token_path, allow_pickle=False), returned=False)
        content = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
        require(content == metadata["token_content_sha256"] == commit["token_content_sha256"], "Checkpoint token content SHA256 mismatch")
        return values.reshape(RETURNED_SHAPE), metadata, digest(metadata_path)

    def write_cache(self, event, rgb, values, forward_seconds, entry):
        import numpy as np
        require(not entry.exists(), "Checkpoint entry already exists; refusing overwrite")
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = entry.parent / (".pending-" + entry.name + "-" + uuid.uuid4().hex)
        temporary.mkdir()
        flat = self.token_array(values, returned=True)
        token_path = temporary / "tokens.npy"
        with token_path.open("xb") as stream:
            np.save(stream, flat, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        metadata = {
            "schema_version": 1, "status": "committed", "context_sha256": self.context_sha256,
            "dataset": event["dataset"], "category": event["category"], "normal_index": event["image_index"],
            "source_path": event["source_path"], "source_image_sha256": event["source_image_sha256"],
            "stored_shape": list(STORED_SHAPE), "returned_shape": list(RETURNED_SHAPE),
            "token_dtype": "float32", "token_file_sha256": digest(token_path),
            "token_content_sha256": hashlib.sha256(memoryview(flat).cast("B")).hexdigest(),
            "token_file_size_bytes": token_path.stat().st_size, "fresh_forward_seconds": forward_seconds,
            "producer_execution_id": self.execution_id, "created_at_utc": now(), **rgb,
        }
        metadata_path = temporary / "metadata.json"
        atomic_json(metadata_path, metadata)
        relative = entry.relative_to(self.cache_root).as_posix()
        commit = {
            "type": "cache_commit", "context_sha256": self.context_sha256,
            "producer_execution_id": self.execution_id, "cache_entry": relative,
            "metadata_sha256": digest(metadata_path), "token_file_sha256": metadata["token_file_sha256"],
            "token_content_sha256": metadata["token_content_sha256"],
            "dataset": event["dataset"], "category": event["category"], "normal_index": event["image_index"],
            "source_image_sha256": event["source_image_sha256"], "committed_at_utc": now(),
        }
        # The producer record precedes the atomic directory commit. A crash
        # leaves only an ignored .pending directory, never a usable half-pair.
        self.append(commit)
        os.rename(temporary, entry)
        self.commits[(self.execution_id, relative)] = commit
        return flat.reshape(RETURNED_SHAPE), metadata, commit["metadata_sha256"]

    def category_account(self, event):
        key = event["dataset"] + "/" + event["category"]
        return self.category_counts.setdefault(key, {**dict.fromkeys(self.counts, 0), **dict.fromkeys(self.seconds, 0.0)})

    def forward(self, original_extract, instance, image, measured, event):
        before = time.perf_counter(); self.forward_attempt_count += 1
        try:
            result = original_extract(instance, image)
            self.forward_success_count += 1
            return result
        except BaseException:
            self.failed_forward_count += 1
            raise
        finally:
            measured["fresh_forward_seconds"] = time.perf_counter() - before
            self.seconds["fresh_forward_seconds"] += measured["fresh_forward_seconds"]
            self.category_account(event)["fresh_forward_seconds"] += measured["fresh_forward_seconds"]

    def extract(self, original_extract, instance, image):
        started = time.perf_counter()
        require(self.cursor < len(self.events), "Unexpected extra extractor call beyond frozen event queue")
        event = self.events[self.cursor]
        measured = {name: 0.0 for name in self.seconds}
        before = time.perf_counter(); rgb = self.source_identity(image, event)
        measured["source_validation_seconds"] = time.perf_counter() - before
        metadata = None; metadata_sha = None; entry = None
        if event["phase"] == "normal":
            entry, token_path, metadata_path = self.cache_paths(event)
            if entry.exists():
                before = time.perf_counter()
                result, metadata, metadata_sha = self.read_cache(event, rgb, entry, token_path, metadata_path)
                measured["cache_read_validation_seconds"] = time.perf_counter() - before
                action, counter = "normal_cache_hit", "normal_cache_hit_count"
            else:
                values = self.forward(original_extract, instance, image, measured, event)
                before = time.perf_counter()
                result, metadata, metadata_sha = self.write_cache(event, rgb, values, measured["fresh_forward_seconds"], entry)
                measured["cache_write_seconds"] = time.perf_counter() - before
                action, counter = "normal_fresh", "normal_fresh_count"
        else:
            result = self.forward(original_extract, instance, image, measured, event)
            self.token_array(result, returned=True)
            action, counter = "test_fresh", "test_fresh_count"
        measured["hook_total_seconds"] = time.perf_counter() - started
        row = {
            "type": "image_result", "execution_id": self.execution_id, "context_sha256": self.context_sha256,
            **event, "action": action, **measured, "completed_at_utc": now(),
            "metadata_sha256": metadata_sha,
            "cache_entry": entry.relative_to(self.cache_root).as_posix() if entry else None,
            "token_file_sha256": metadata["token_file_sha256"] if metadata else None,
            "token_content_sha256": metadata["token_content_sha256"] if metadata else None,
        }
        self.append(row)
        self.cursor += 1; self.counts[counter] += 1
        category = self.category_account(event)
        category[counter] += 1
        for name, value in measured.items():
            if name != "fresh_forward_seconds":
                self.seconds[name] += value; category[name] += value
        return result

    def assert_complete(self):
        require(self.cursor == len(self.events), "Frozen worker did not consume the complete expected image queue")

    def cleanup_completed(self):
        summaries = []
        categories = sorted({tuple(task) for task in self.context.get("categories", [])} or
                            {(event["dataset"], event["category"]) for event in self.events})
        for dataset, category in categories:
            completed = self.run / "categories" / dataset / category
            seal_path = completed / "completion.json"
            target = self.cache_root / dataset / category
            if not target.exists() or not seal_path.exists():
                continue
            seal = read(seal_path)
            require(seal["status"] == "completed", "Cannot clean an incomplete category checkpoint")
            metadata_path = completed / "metadata.json"
            relative_metadata = metadata_path.relative_to(self.run).as_posix()
            require(digest(metadata_path) == seal["sha256"][relative_metadata], "Completed category metadata seal changed")
            producer = read(metadata_path)
            require(producer["dataset"] == dataset and producer["category"] == category, "Completed category identity mismatch")
            train_path = completed / "train_manifest.json"
            normal_manifest_path = completed / "token_cache_manifest.json"
            for path in (train_path, normal_manifest_path):
                require(digest(path) == seal["sha256"][path.relative_to(self.run).as_posix()], "Completed normal-token inventory seal changed")
            train = read(train_path)
            normal_manifest = read(normal_manifest_path)
            # Verify the exact resolved recursive-delete target and reject
            # links/junctions inside this dedicated category before removal.
            resolved = target.resolve(strict=True)
            require(resolved == target and resolved.is_relative_to(self.cache_root) and resolved != self.cache_root, "Unsafe checkpoint cleanup target")
            require(all(path.resolve() == path.absolute() for path in target.rglob("*")), "Checkpoint cleanup tree contains a redirected path")
            records = []
            for index, source_image in enumerate(train):
                event = {"dataset": dataset, "category": category, "phase": "normal", "image_index": index,
                         "source_path": str((Path(self.context.get("project_root", ROOT)) / source_image["path"]).resolve()),
                         "source_image_sha256": source_image["sha256"]}
                entry, token_path, cache_meta = self.cache_paths(event)
                cache_metadata = read(cache_meta)
                rgb = {field: cache_metadata[field] for field in ("source_rgb_shape", "source_rgb_sha256")}
                _, cache_metadata, cache_metadata_sha = self.read_cache(event, rgb, entry, token_path, cache_meta)
                records.append({"normal_index": event["image_index"], "cache_entry": entry.relative_to(self.cache_root).as_posix(),
                                "metadata_sha256": cache_metadata_sha, "token_file_sha256": cache_metadata["token_file_sha256"],
                                "token_content_sha256": cache_metadata["token_content_sha256"],
                                "source_image_sha256": event["source_image_sha256"], "producer_execution_id": cache_metadata["producer_execution_id"]})
            # Checkpoint .npy is flat, exactly the worker's serialization.
            require(len(records) == len(normal_manifest["files"]) and all(record["token_file_sha256"] == normal_manifest["files"][index]["sha256"] for index, record in enumerate(records)), "Completed worker consumed different checkpoint token files")
            summary = {"dataset": dataset, "category": category, "completion_sha256": digest(seal_path),
                       "context_sha256": self.context_sha256, "authenticated_token_records": records,
                       "cache_removed_after_completed_category": True}
            # Persist fingerprints outside the cache before deleting it.
            summary_path = self.history / ("cache_summary_" + dataset + "_" + category + ".json")
            atomic_json(summary_path, summary)
            shutil.rmtree(resolved)
            summaries.append({"dataset": dataset, "category": category, "summary_path": summary_path.name,
                              "summary_sha256": digest(summary_path), "normal_images": len(records)})
        atomic_json(self.history / "completed_cache_summary.json", summaries)
        return summaries

    def summary(self, *, status, error, elapsed_seconds, cleanup):
        return {
            "schema_version": 1, "status": status, "execution_id": self.execution_id,
            "context_sha256": self.context_sha256, "checkpoint_helper_sha256": self.context.get("checkpoint_helper_sha256"),
            "delegate_adapter_sha256": self.context.get("delegate_adapter_sha256"),
            "expected_events": len(self.events), "events_consumed": self.cursor,
            "expected_normal_events": sum(event["phase"] == "normal" for event in self.events),
            "expected_test_events": sum(event["phase"] == "test" for event in self.events),
            "queue_complete": self.cursor == len(self.events), **self.counts, **self.seconds,
            "forward_attempt_count": self.forward_attempt_count, "forward_success_count": self.forward_success_count,
            "failed_forward_count": self.failed_forward_count,
            "per_category": self.category_counts, "elapsed_seconds": elapsed_seconds,
            "test_always_fresh": True, "algorithms_and_protocol_unchanged": True,
            "fit_only_and_fresh_resource_workers_changed": False, "adopted_unsealed_token_count": 0,
            "normal_extraction_seconds_scope": "includes normal cache read/validation and fresh forwards; not full first-pass DINO time",
            "completed_cache_cleanup": cleanup, "error": error, "finished_at_utc": now(),
        }


def self_check():
    """Small synthetic checkpoint checks; never import Torch or real workers."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="route1-checkpoint-stub-") as directory:
        root = Path(directory); run = root / "run"; cache = root / "cache"
        images = []
        events = []
        for index, phase in enumerate(("normal", "normal", "test", "test")):
            path = root / (str(index) + ".png")
            Image.fromarray(np.full((5, 7, 3), index + 11, dtype=np.uint8)).save(path)
            images.append(path)
            events.append({"event_index": index, "dataset": "stub", "category": "sample", "phase": phase,
                           "image_index": index if phase == "normal" else index - 2, "source_path": str(path.resolve()), "source_image_sha256": digest(path)})
        class Stub:
            def __init__(self):
                self.forwards = 0
            def extract(self, image):
                self.forwards += 1
                return np.full(RETURNED_SHAPE, np.asarray(image.convert("RGB"))[0, 0, 0], dtype=np.float32)
        context = {"stub_context": "v1", "checkpoint_helper_sha256": digest(Path(__file__)),
                   "project_root": str(root.resolve()), "categories": [["stub", "sample"]]}
        def hook_for(execution_id):
            history = run / "checkpoint_execution_history" / execution_id; history.mkdir(parents=True)
            return CheckpointHook(run=run, cache_root=cache, context=context, events=events, execution_id=execution_id, history=history)
        first = hook_for("first"); stub = Stub(); require(stub.forwards == 0, "Stub constructor forwarded")
        for path in images:
            with Image.open(path) as image:
                first.extract(Stub.extract, stub, image)
        first.assert_complete(); require(stub.forwards == 4 and first.counts["normal_fresh_count"] == 2 and first.counts["test_fresh_count"] == 2, "Fresh stub call accounting failed")
        second = hook_for("second"); stub = Stub()
        for path in images:
            with Image.open(path) as image:
                second.extract(Stub.extract, stub, image)
        second.assert_complete(); require(stub.forwards == 2 and second.counts["normal_cache_hit_count"] == 2 and second.counts["test_fresh_count"] == 2, "Test forwarded/cache call accounting failed")
        try:
            with Image.open(images[0]) as image:
                second.extract(Stub.extract, stub, image)
        except ValueError:
            pass
        else:
            raise AssertionError("Extra queue call accepted")
        wrong_order = hook_for("wrong_order")
        try:
            with Image.open(images[1]) as image:
                wrong_order.extract(Stub.extract, Stub(), image)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong RGB event order accepted")
        try:
            wrong_order.assert_complete()
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete queue accepted")
        tampered = cache / "stub/sample/000000/tokens.npy"
        with tampered.open("ab") as stream:
            stream.write(b"tamper")
        detector = hook_for("tampered")
        try:
            with Image.open(images[0]) as image:
                detector.extract(Stub.extract, Stub(), image)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered token file accepted")
        altered_context = {**context, "stub_context": "v2"}
        history = run / "checkpoint_execution_history/altered"; history.mkdir()
        try:
            CheckpointHook(run=run, cache_root=cache, context=altered_context, events=events, execution_id="altered", history=history)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed cache context accepted")
        # Simulate a prior session killed after core completion but before
        # finally. The new session's queue is empty; cleanup still discovers
        # the category through the full frozen category collection.
        cleanup_cache = root / "cleanup_cache"
        history = run / "checkpoint_execution_history/cleanup_producer"; history.mkdir()
        producer_hook = CheckpointHook(run=run, cache_root=cleanup_cache, context=context, events=events,
                                       execution_id="cleanup_producer", history=history)
        stub = Stub()
        for path in images:
            with Image.open(path) as image:
                producer_hook.extract(Stub.extract, stub, image)
        history = run / "checkpoint_execution_history/cleanup_new_session"; history.mkdir()
        empty_queue_hook = CheckpointHook(run=run, cache_root=cleanup_cache, context=context, events=[],
                                          execution_id="cleanup_new_session", history=history)
        empty_queue_hook.cleanup_completed()
        require((cleanup_cache / "stub/sample").exists(), "Incomplete category cache was removed")
        completed = run / "categories/stub/sample"; completed.mkdir(parents=True)
        atomic_json(completed / "metadata.json", {"dataset": "stub", "category": "sample"})
        atomic_json(completed / "train_manifest.json", [{"path": str(i) + ".png", "sha256": digest(images[i])} for i in range(2)])
        atomic_json(completed / "token_cache_manifest.json", {"files": [
            {"sha256": read(cleanup_cache / f"stub/sample/{i:06d}/metadata.json")["token_file_sha256"]} for i in range(2)]})
        atomic_json(completed / "completion.json", {"status": "completed", "sha256": {
            (completed / name).relative_to(run).as_posix(): digest(completed / name)
            for name in ("metadata.json", "train_manifest.json", "token_cache_manifest.json")}})
        history = run / "checkpoint_execution_history/cleanup_final"; history.mkdir()
        final_hook = CheckpointHook(run=run, cache_root=cleanup_cache, context=context, events=[],
                                    execution_id="cleanup_final", history=history)
        require(len(final_hook.cleanup_completed()) == 1 and not (cleanup_cache / "stub/sample").exists(), "Previously completed category cache was not cleaned")
    require("torch" not in sys.modules, "Stub check imported Torch")
    print(json.dumps({"status": "passed", "scope": "stub-only checkpoint/order/context/tamper/Test-forward checks", "torch_imported": False}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/route1_controls_20261014.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    require(args.resume and args.output_dir is not None, "Existing --output-dir and --resume required")
    attempt_started = time.perf_counter()
    run = args.output_dir.resolve(strict=True)
    require(run.is_relative_to(ROOT / "runs/quick") and run != CACHE_ROOT.resolve(), "Existing project experiment required")
    require(CACHE_ROOT.resolve() == CACHE_ROOT.absolute() and CACHE_ROOT.resolve().is_relative_to(ROOT / "runs/quick"), "Dedicated checkpoint-cache root is redirected or escapes the workspace")
    require(read(run / "run_state.json")["status"] != "completed", "Completed run needs no resume")
    config, source, context = authenticated_context(run, args.config.resolve(strict=True))
    events = event_queue(run, source, config)
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    history = run / "checkpoint_execution_history" / execution_id; history.mkdir(parents=True)
    atomic_json(history / "context.json", {"context_sha256": hashlib.sha256(encoded(context)).hexdigest(), "context": context})
    hook = CheckpointHook(run=run, cache_root=CACHE_ROOT, context=context, events=events, execution_id=execution_id, history=history)
    sys.path.insert(0, str(ROOT / "src"))
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor
    original_extract = LocalDinoExtractor.extract
    original_argv = sys.argv[:]
    startup_seconds = time.perf_counter() - attempt_started
    status = "failed"; error = None; cleanup = []
    def wrapped_extract(instance, image):
        try:
            return hook.extract(original_extract, instance, image)
        except BaseException as exception:
            hook.append({"type": "image_failure", "execution_id": execution_id,
                         "context_sha256": hook.context_sha256, "event_index": hook.cursor,
                         "expected_event": hook.events[hook.cursor] if hook.cursor < len(hook.events) else None,
                         "error": type(exception).__name__ + ": " + str(exception), "failed_at_utc": now()})
            raise
    LocalDinoExtractor.extract = wrapped_extract
    try:
        delegate_path = ROOT / "scripts/resume_route1_controls.py"
        spec = importlib.util.spec_from_file_location("unchanged_route1_resume_delegate", delegate_path)
        delegate = importlib.util.module_from_spec(spec); spec.loader.exec_module(delegate)
        sys.argv = [str(delegate_path), "--output-dir", str(run), "--config", str(args.config.resolve()), "--resume"]
        result = delegate.main()
        require(result == 0 and read(run / "run_state.json")["status"] == "completed", "Delegate did not complete the full run")
        hook.assert_complete(); status = "completed"
        return result
    except BaseException as exception:
        error = type(exception).__name__ + ": " + str(exception)
        raise
    finally:
        LocalDinoExtractor.extract = original_extract; sys.argv = original_argv
        try:
            cleanup = hook.cleanup_completed()
        except BaseException as exception:
            status = "failed"; error = (error + "; " if error else "") + "checkpoint cleanup: " + type(exception).__name__ + ": " + str(exception)
            summary = hook.summary(status=status, error=error, elapsed_seconds=time.perf_counter()-attempt_started, cleanup=cleanup)
            summary["startup_context_and_queue_seconds"] = startup_seconds
            atomic_json(history / "summary.json", summary)
            raise
        summary = hook.summary(status=status, error=error, elapsed_seconds=time.perf_counter()-attempt_started, cleanup=cleanup)
        summary["startup_context_and_queue_seconds"] = startup_seconds
        atomic_json(history / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
