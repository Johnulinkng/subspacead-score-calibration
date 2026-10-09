"""A failed allocation probe must not authorize an experiment."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts/Test-TrainingPreflight.ps1"
PROFILE = ROOT / "artifacts/resource_profiles/draem_fp32_256_batch8_bounded_cache_20261009.json"
LAUNCHER = ROOT / "scripts/Invoke-MonitoredExperiment.ps1"


def powershell() -> str:
    executable = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    assert executable
    return executable


def test_measured_budget_uses_peak_instead_of_eight_gib():
    relative = Path("tmp") / f"preflight-test-{uuid.uuid4().hex}"
    output = ROOT / relative
    try:
        completed = subprocess.run(
            [powershell(), "-NoProfile", "-File", str(PREFLIGHT),
             "-ResourceProfile", str(PROFILE), "-OutputDirectory", str(relative)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        assert completed.returncode in {0, 2}, completed.stderr
        reports = list(output.glob("preflight_*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text(encoding="utf-8-sig"))
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        assert report["budget_basis"] == "measured_peak_plus_reserves"
        assert report["thresholds"]["minimum_free_ram_gib"] == pytest.approx(
            round(2.5 + 1.25 * profile["host_peak_working_set_gib"], 3)
        )
        assert report["thresholds"]["minimum_free_ram_gib"] < 8
        assert report["checks"]["free_ram"] == (
            report["host"]["free_ram_gib"] >= report["thresholds"]["minimum_free_ram_gib"]
        )
        assert report["ready_for_training_launch"] == all(report["checks"].values())
        assert report["ready_for_capacity_diagnostic"] is False
    finally:
        if output.exists():
            shutil.rmtree(output)


@pytest.mark.parametrize("mutation,expected", [
    ("failed", "Only a successful"), ("source_drift", "source drift"),
    ("missing_sources", "lacks source evidence"),
])
def test_invalid_profile_cannot_set_budget(mutation, expected):
    relative = Path("tmp") / f"profile-test-{uuid.uuid4().hex}"
    root = ROOT / relative
    root.mkdir(parents=True)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if mutation == "failed":
        profile["status"] = "failed"
    elif mutation == "source_drift":
        profile["source_sha256"]["scripts/train_draem.py"] = "0" * 64
    else:
        profile["source_sha256"] = {}
    source = root / "profile.json"
    source.write_text(json.dumps(profile), encoding="utf-8")
    try:
        completed = subprocess.run(
            [powershell(), "-NoProfile", "-File", str(PREFLIGHT),
             "-ResourceProfile", str(source), "-OutputDirectory", str(relative / "output")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        assert completed.returncode != 0
        assert expected in completed.stderr
        assert not list((root / "output").glob("preflight_*.json"))
    finally:
        shutil.rmtree(root)


def test_synthetic_profile_cannot_authorize_full_training():
    relative = Path("runs/monitored") / f"profile-full-reject-{uuid.uuid4().hex}"
    output = ROOT / relative
    try:
        completed = subprocess.run(
            [powershell(), "-NoProfile", "-File", str(LAUNCHER),
             "-OutputDirectory", str(relative), "-ScriptPath", "scripts/train_draem.py",
             "-RunPurpose", "development_full", "-ResourceProfile", str(PROFILE)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        assert completed.returncode != 0
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8-sig"))
        assert summary["child"]["pid"] is None
        assert summary["lock"]["created"] is False
        assert "cannot authorize full experiments" in summary["failure_message"]
    finally:
        if output.exists():
            shutil.rmtree(output)
