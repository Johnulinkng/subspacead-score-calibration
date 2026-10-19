"""Verify saved scientific inputs and completed units before resuming computation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/research-q3-20261016"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_child(parent, relative):
    path = (Path(parent) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError("Saved artifact escaped its registered root: " + str(path))
    return path


def audit(output):
    counters = {"sealed_sources": 0, "feature_files": 0, "source_images": 0,
                "test_masks": 0, "map_files": 0, "frames": 0,
                "completed_independent_units": 0, "independent_metric_rows": 0}
    def check(path, expected):
        if sha(path) != expected:
            raise ValueError("Hash mismatch: " + str(path))

    try:
        for folder in ("features", "independent-partitions", "contribution-refits"):
            frozen = read(RUN / folder / "frozen_protocol.json")
            for relative, digest in frozen["sha256"].items():
                check(checked_child(ROOT, relative), digest)
                counters["sealed_sources"] += 1
        queue = read(RUN / "baseline-queue-v2.json")
        source = Path(queue["source_snapshot"])
        for relative, digest in queue["source_sha256"].items():
            check(checked_child(source, relative), digest)
            counters["sealed_sources"] += 1
        if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != queue["source_commit"]:
            raise ValueError("Comparator snapshot commit drift")
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise ValueError("Comparator snapshot has local changes")
        tasks = read(RUN / "features/frozen_protocol.json")["tasks"]
        independent = RUN / "independent-partitions/categories"
        config = read(ROOT / "configs/q3_independent_partitions_20261016.json")
        for dataset, category in tasks:
            cache = RUN / "features/categories" / dataset / category
            complete = read(cache / "completion.json")
            for kind in ("normal", "test"):
                manifest = cache / (kind + "_manifest.json")
                check(manifest, complete[kind + "_manifest_sha256"])
                for row in read(manifest):
                    check(checked_child(cache, row["features"]), row["features_sha256"])
                    counters["feature_files"] += 1
                    check(checked_child(ROOT, row["path"]), row["sha256"])
                    counters["source_images"] += 1
                    if kind == "test":
                        check(Path(row["mask_path"]), row["mask_sha256"])
                        counters["test_masks"] += 1
            for seed in config["seeds"]:
                unit = independent / dataset / category / str(seed)
                marker = unit / "metrics.json"
                if not marker.exists():
                    continue
                rows = read(marker)
                if len(rows) != len(config["variants"]) or {r["variant"] for r in rows} != set(config["variants"]):
                    raise ValueError("Incomplete metric marker: " + str(unit))
                for row in rows:
                    if (row["dataset"], row["category"], row["seed"]) != (dataset, category, seed):
                        raise ValueError("Metric unit identity mismatch")
                    if read(unit / (row["variant"] + ".json")) != row:
                        raise ValueError("Metric marker and individual result disagree")
                    manifest = read(unit / (row["variant"] + "_map_manifest.json"))
                    for relative, digest in manifest["sha256"].items():
                        check(checked_child(unit, relative), digest)
                        counters["map_files"] += 1
                for frame in read(unit / "frames.json"):
                    check(checked_child(unit, frame["path"]), frame["sha256"])
                    counters["frames"] += 1
                groups = list(read(unit / "partitions.json").values())
                hashes = [{r["sha256"] for r in group} for group in groups]
                if any(left & right for i, left in enumerate(hashes) for right in hashes[i+1:]):
                    raise ValueError("Normal partition content overlap")
                counters["completed_independent_units"] += 1
                counters["independent_metric_rows"] += len(rows)
            write(output, {"status": "auditing", "category": f"{dataset}/{category}",
                           "updated_at": now(), **counters})
            print("verified " + dataset + "/" + category, flush=True)
        if counters["feature_files"] != 16175:
            raise ValueError("Unexpected original feature cache denominator")
        result = {"status": "passed", "finished_at": now(), **counters,
                  "queue_sha256": sha(RUN / "baseline-queue-v2.json"),
                  "audit_script_sha256": sha(Path(__file__)),
                  "completed_unit_model_hashes": {
                      str(p.relative_to(RUN)): sha(p)
                      for marker in independent.rglob("metrics.json")
                      for name in ("pca.npz", "calibration.npz", "partitions.json", "metrics.json")
                      for p in [marker.parent / name]},
                  "scope": "Exact saved-file identities and structural consistency; not a rerun of numerical metrics"}
        write(output, result)
        return result
    except BaseException as error:
        write(output, {"status": "failed", "finished_at": now(), "error": repr(error), **counters})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.output)
