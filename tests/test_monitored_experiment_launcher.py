from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "Invoke-MonitoredExperiment.ps1"
S0_WRAPPER = PROJECT_ROOT / "scripts" / "Invoke-DraemS0Smoke.ps1"
S0_WAITER = PROJECT_ROOT / "scripts" / "Wait-For-DraemS0Window.ps1"


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable is not None
    return executable


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cpu_only_self_test_records_resources_and_releases_lock() -> None:
    relative_output = Path("runs") / "monitored" / f"pytest-selftest-{uuid.uuid4().hex}"
    output = PROJECT_ROOT / relative_output
    lock = PROJECT_ROOT / "runs" / ".training.lock.json"
    assert not lock.exists(), "A pre-existing training lock makes this test unsafe."

    try:
        completed = subprocess.run(
            [
                _powershell(),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                "-SelfTest",
                "-OutputDirectory",
                str(relative_output),
                "-SampleIntervalSeconds",
                "1",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        command = json.loads((output / "command.json").read_text(encoding="utf-8"))
        samples = [
            json.loads(line)
            for line in (output / "resource_samples.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

        assert summary["status"] == "completed"
        assert summary["mode"] == "self_test"
        assert summary["run_purpose"] == "synthetic_self_test"
        assert summary["claim_eligible"] is False
        assert summary["preflight"]["required"] is False
        assert summary["lock"]["created"] is False
        assert summary["child"]["exit_code"] == 0
        assert summary["child"]["pid"] > 0
        assert summary["monitoring"]["sample_count"] >= 2
        assert summary["evidence_sha256"]["command"] == _sha256(output / "command.json")
        assert summary["evidence_sha256"]["resource_samples_jsonl"] == _sha256(
            output / "resource_samples.jsonl"
        )
        assert summary["evidence_sha256"]["stdout_log"] == _sha256(output / "stdout.log")
        assert summary["evidence_sha256"]["stderr_log"] == _sha256(output / "stderr.log")
        assert summary["evidence_sha256"]["mutable_child_logs_omitted"] is False
        assert command["claim_eligible"] is False
        assert command["executable"].casefold() == str(
            PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        ).casefold()
        assert samples
        assert all(sample["child"]["pid"] == summary["child"]["pid"] for sample in samples)
        stdout_rows = [
            json.loads(line)
            for line in (output / "stdout.log").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert stdout_rows[0]["claim_eligible"] is False
        assert stdout_rows[0]["gpu_visible"] is False
        assert stdout_rows[0]["sentinel"] == 'space and "quote" ending\\'
        assert stdout_rows[0]["pid"] == summary["child"]["pid"]
        assert summary["child"]["process_launcher_pid"] > 0
        assert Path(summary["child"]["executable"]).is_relative_to(PROJECT_ROOT)
        assert "synthetic self-test stderr sentinel" in (output / "stderr.log").read_text(
            encoding="utf-8"
        )
        assert not lock.exists()
    finally:
        if output.exists():
            shutil.rmtree(output)


def test_launcher_has_no_general_preflight_bypass_and_rejects_output_escape() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "SkipPreflight" not in source
    assert "BypassPreflight" not in source
    assert 'ValidateSet("debug_limited", "development_full", "author_reproduction", "formal_full")' in source
    start_process_lines = [line for line in source.splitlines() if "Start-Process" in line]
    assert start_process_lines
    assert all("-WindowStyle Hidden" in line for line in start_process_lines)
    assert "[System.IO.FileMode]::CreateNew" in source

    rejection_root = PROJECT_ROOT / "artifacts" / "monitor_rejections"
    before = set(rejection_root.glob("launcher_rejection_*.json"))

    try:
        completed = subprocess.run(
            [
                _powershell(),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                "-SelfTest",
                "-OutputDirectory",
                "artifacts/escaped-monitor-output",
                "-SampleIntervalSeconds",
                "1",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert completed.returncode != 0
        assert not (PROJECT_ROOT / "artifacts" / "escaped-monitor-output").exists()
    finally:
        for path in set(rejection_root.glob("launcher_rejection_*.json")) - before:
            path.unlink()


def test_draem_s0_wrapper_freezes_gpu_resource_smoke_contract() -> None:
    source = S0_WRAPPER.read_text(encoding="utf-8")
    required_tokens = (
        '"--dataset", "visa"',
        '"--category", "candle"',
        '"--seed", "1887537248"',
        '"--run-purpose", "debug_limited"',
        '"--image-size", "256"',
        '"--batch-size", "8"',
        '"--workers", "0"',
        '"--no-pin-memory"',
        '"--device", "cuda:0"',
        '"--max-steps", "2"',
        'Invoke-MonitoredExperiment.ps1',
    )
    assert all(token in source for token in required_tokens)
    assert "SkipPreflight" not in source
    assert "BypassPreflight" not in source


def test_draem_s0_waiter_cannot_bypass_or_preempt_other_gpu_work() -> None:
    source = S0_WAITER.read_text(encoding="utf-8")
    assert "Invoke-DraemS0Smoke.ps1" in source
    assert "Test-TrainingPreflight.ps1" in source
    assert "-ResourceProfile" in source
    assert "-ge 8.0" not in source
    assert "Stop-Process" not in source
    assert "taskkill" not in source.casefold()
    assert "SkipPreflight" not in source
    assert "BypassPreflight" not in source
    assert "[System.IO.FileMode]::CreateNew" in source
    preflight = (PROJECT_ROOT / "scripts/Test-TrainingPreflight.ps1").read_text(encoding="utf-8")
    assert "runs/.training.lock.json" in preflight


def test_launcher_rejects_a_non_project_command_before_preflight() -> None:
    relative_output = Path("runs") / "monitored" / f"pytest-command-reject-{uuid.uuid4().hex}"
    output = PROJECT_ROOT / relative_output
    try:
        completed = subprocess.run(
            [
                _powershell(),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                "-OutputDirectory",
                str(relative_output),
                "-ScriptPath",
                r"C:\Windows\System32\notepad.exe",
                "-RunPurpose",
                "development_full",
                "-SampleIntervalSeconds",
                "1",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert completed.returncode != 0
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "rejected"
        assert "inside the project" in summary["failure_message"]
        assert summary["preflight"]["report"] is None
        assert summary["child"]["pid"] is None
    finally:
        if output.exists():
            shutil.rmtree(output)
