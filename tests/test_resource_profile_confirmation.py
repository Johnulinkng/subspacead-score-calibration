from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_s0_fixture(tmp_path):
    # Metadata-only fixtures; no image data, model execution or GPU access.
    script = tmp_path / "scripts/freeze_draem_resource_profile.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT / "scripts/freeze_draem_resource_profile.py", script)
    spec = importlib.util.spec_from_file_location("fixture_resource_confirmation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = json.loads((PROJECT / "artifacts/resource_profiles/draem_fp32_256_batch8_bounded_cache_20261009.json").read_text(encoding="utf-8"))
    for relative in profile["source_sha256"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, target)
    prototype = tmp_path / "artifacts/prototype.json"
    write(prototype, profile)
    run = tmp_path / "runs/monitored/metadata-fixture-s0"
    run.mkdir(parents=True)
    settings = {"dataset": "visa", "category": "candle", "seed": 1887537248,
                "image_size": 256, "batch_size": 8, "workers": 0, "pin_memory": False,
                "device": "cuda:0", "max_steps": 2, "run_purpose": "debug_limited",
                "input_protocol": "visa_train_dev_dtd_v1", "activation_checkpointing": True,
                "learning_rate": 1e-4}
    write(run / "model/run_config.json", {"settings": settings, "input_evidence": {
        "formal_test_content_accessed": False, "ground_truth_content_accessed": False}})
    write(run / "model/run_state.json", {"status": "debug_completed", "global_step": 2,
        "gpu_peak_allocated_bytes": 6 * GIB, "gpu_peak_reserved_bytes": int(6.2 * GIB)})
    checkpoints = {}
    for name in ("reconstructive", "discriminative"):
        path = run / f"model/checkpoints/{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"metadata-only fixture, not a model")
        checkpoints[name] = {"sha256": module.digest(path)}
    write(run / "model/checkpoints/manifest.json", {"global_step": 2, "run_purpose": "debug_limited", "checkpoints": checkpoints})
    write(run / "command.json", {"script": str(tmp_path / "scripts/train_draem.py"), "script_sha256": module.digest(tmp_path / "scripts/train_draem.py")})
    (run / "resource_samples.jsonl").write_text(json.dumps({"child": {
        "peak_working_set_bytes": 2 * GIB, "peak_private_commit_bytes": 10 * GIB}}) + "\n", encoding="utf-8")
    (run / "stdout.log").write_text("fixture\n", encoding="utf-8")
    (run / "stderr.log").write_text("", encoding="utf-8")
    preflight = run / "preflight/preflight_fixture.json"
    write(preflight, {"resource_profile": {"path": str(prototype), "sha256": module.digest(prototype)}})
    summary = {"mode": "experiment", "run_purpose": "debug_limited", "status": "completed",
               "child": {"exit_code": 0}, "preflight": {"report": str(preflight)},
               "evidence_sha256": {key: module.digest(run / filename) for key, filename in (
                   ("command", "command.json"), ("resource_samples_jsonl", "resource_samples.jsonl"),
                   ("stdout_log", "stdout.log"), ("stderr_log", "stderr.log"))}}
    summary["evidence_sha256"]["preflight_report"] = module.digest(preflight)
    write(run / "summary.json", summary)
    return module, run


def test_real_s0_confirmation_never_reduces_measured_peaks(tmp_path, monkeypatch):
    module, run = make_s0_fixture(tmp_path)
    output = tmp_path / "artifacts/confirmed.json"
    monkeypatch.setattr(sys, "argv", ["freeze", "--monitored-run", str(run), "--output", str(output)])
    assert module.main() == 0
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["kind"] == "draem_real_s0_capacity_profile"
    assert profile["host_peak_working_set_gib"] == 2
    assert profile["host_peak_private_commit_gib"] == 10
    assert profile["gpu_peak_reserved_gib"] == pytest.approx(6.2)
    assert profile["claim_eligible"] is False
    assert profile["parameter_update_learning_rate"] == 1e-4
    assert len(profile["real_s0"]["evidence_sha256"]) == 8


@pytest.mark.parametrize("mutation,reason", [("synthetic", "real train-only S0"), ("tampered_log", "monitoring evidence drift")])
def test_confirmation_rejects_synthetic_or_mutated_evidence(tmp_path, monkeypatch, mutation, reason):
    module, run = make_s0_fixture(tmp_path)
    if mutation == "synthetic":
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        summary["mode"] = "capacity_diagnostic"
        write(run / "summary.json", summary)
    else:
        (run / "stdout.log").write_text("altered", encoding="utf-8")
    output = tmp_path / "artifacts/rejected.json"
    monkeypatch.setattr(sys, "argv", ["freeze", "--monitored-run", str(run), "--output", str(output)])
    with pytest.raises(ValueError, match=reason):
        module.main()
    assert not output.exists()
