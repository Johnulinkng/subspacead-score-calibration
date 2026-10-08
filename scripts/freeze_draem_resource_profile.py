"""Confirm a training admission budget from a completed real train-only S0.

This converts resource evidence, not performance evidence. It does not launch
training and it cannot turn a synthetic diagnostic into a real-data S0 pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitored-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.monitored_run.resolve()
    output = args.output.resolve()
    if not run.is_relative_to(ROOT / "runs/monitored") or not output.is_relative_to(ROOT):
        raise ValueError("Resource evidence must remain inside its project boundary")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing resource profile")
    summary = read(run / "summary.json")
    if summary["mode"] != "experiment" or summary["run_purpose"] != "debug_limited" or summary["status"] != "completed" or summary["child"]["exit_code"] != 0:
        raise ValueError("A completed real train-only S0 is required; synthetic diagnostics are insufficient")
    command = read(run / "command.json")
    evidence = summary["evidence_sha256"]
    for field, filename in (("command", "command.json"), ("resource_samples_jsonl", "resource_samples.jsonl"), ("stdout_log", "stdout.log"), ("stderr_log", "stderr.log")):
        if digest(run / filename) != evidence[field]:
            raise ValueError(f"S0 monitoring evidence drift: {filename}")
    if Path(command["script"]).resolve() != ROOT / "scripts/train_draem.py" or command["script_sha256"] != digest(ROOT / "scripts/train_draem.py"):
        raise ValueError("S0 must use the current project DRAEM training CLI")
    config = read(run / "model/run_config.json")
    state = read(run / "model/run_state.json")
    settings = config["settings"]
    expected = {"dataset": "visa", "category": "candle", "seed": 1887537248,
                "image_size": 256, "batch_size": 8, "workers": 0,
                "pin_memory": False, "device": "cuda:0", "max_steps": 2,
                "run_purpose": "debug_limited", "input_protocol": "visa_train_dev_dtd_v1"}
    if any(settings[key] != value for key, value in expected.items()) or state["status"] != "debug_completed" or state["global_step"] != 2:
        raise ValueError("S0 execution does not match the registered real-data smoke")
    inputs = config["input_evidence"]
    if inputs["formal_test_content_accessed"] or inputs["ground_truth_content_accessed"]:
        raise ValueError("S0 resource confirmation cannot use test/GT content")
    preflight_path = Path(summary["preflight"]["report"]).resolve()
    if not preflight_path.is_relative_to(run) or digest(preflight_path) != evidence["preflight_report"]:
        raise ValueError("S0 preflight evidence drift")
    preflight = read(preflight_path)
    prototype_path = Path(preflight["resource_profile"]["path"]).resolve()
    if not prototype_path.is_relative_to(ROOT) or digest(prototype_path) != preflight["resource_profile"]["sha256"]:
        raise ValueError("S0 bootstrap resource profile drift")
    profile = read(prototype_path)
    if profile["kind"] != "draem_synthetic_capacity_diagnostic" or profile["status"] != "passed" or settings["activation_checkpointing"] != profile["recipe"]["activation_checkpointing"]:
        raise ValueError("S0 does not confirm the measured memory strategy")
    for relative, expected_hash in profile["source_sha256"].items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or digest(source) != expected_hash:
            raise ValueError(f"S0 source drift: {relative}")
    samples = [json.loads(line) for line in (run / "resource_samples.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    peak_rss = max((row["child"].get("peak_working_set_bytes") or 0) / GIB for row in samples)
    peak_commit = max((row["child"].get("peak_private_commit_bytes") or 0) / GIB for row in samples)
    if peak_rss <= 0 or peak_commit <= 0 or not state.get("gpu_peak_reserved_bytes"):
        raise ValueError("Real S0 peak resource measurements are incomplete")
    checkpoints = read(run / "model/checkpoints/manifest.json")
    if checkpoints["global_step"] != 2 or checkpoints["run_purpose"] != "debug_limited":
        raise ValueError("S0 checkpoint evidence is incomplete")
    for name in ("reconstructive", "discriminative"):
        if digest(run / f"model/checkpoints/{name}.pt") != checkpoints["checkpoints"][name]["sha256"]:
            raise ValueError("S0 checkpoint content drift")
    profile.update({
        "kind": "draem_real_s0_capacity_profile",
        "host_peak_working_set_gib": max(profile["host_peak_working_set_gib"], peak_rss),
        "host_peak_private_commit_gib": max(profile["host_peak_private_commit_gib"], peak_commit),
        "gpu_peak_allocated_gib": max(profile["gpu_peak_allocated_gib"], state["gpu_peak_allocated_bytes"] / GIB),
        "gpu_peak_reserved_gib": max(profile["gpu_peak_reserved_gib"], state["gpu_peak_reserved_bytes"] / GIB),
        "real_s0": {"path": run.relative_to(ROOT).as_posix(), "evidence_sha256": {
            name: digest(run / name) for name in ("summary.json", "command.json", "resource_samples.jsonl", "stdout.log", "stderr.log", "model/run_config.json", "model/run_state.json", "model/checkpoints/manifest.json")
        }},
        "dataset_accessed": True,
        "checkpoint_saved": True,
        "parameter_update_learning_rate": settings["learning_rate"],
        "claim_eligible": False,
        "limitations": "Real train-only two-step S0 confirms an admission budget, not long-run stability or performance; registered experiment authority still required.",
    })
    profile["source_sha256"][Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profile": str(output), "sha256": digest(output), "claim_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
