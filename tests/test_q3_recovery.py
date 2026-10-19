from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supervise_q3_recovery as recovery


def sample(ram, commit=20):
    return {"system_free_ram_bytes": ram * recovery.GIB,
            "system_free_commit_bytes": commit * recovery.GIB}


def test_waiting_development_does_not_start_before_models_are_complete():
    assert not recovery.eligible_stage("trainonly-development", set(), sample(12), 8192)
    assert recovery.eligible_stage("trainonly-development", {"independent-partitions"}, sample(4), 2048)


def test_gpu_comparator_and_cpu_fit_have_separate_resource_admission():
    assert recovery.eligible_stage("independent-partitions", set(), sample(4.2), 0)
    assert not recovery.eligible_stage("baselines", set(), sample(4.2), 8192)
    assert not recovery.eligible_stage("baselines", set(), sample(8), 2048)
    assert recovery.eligible_stage("baselines", set(), sample(6), 8192)


def test_final_analysis_requires_all_essential_completed_stages():
    assert not recovery.eligible_stage("final-analysis", {"baselines", "independent-partitions"}, sample(8), 8192)
    assert recovery.eligible_stage("final-analysis", {"baselines", "independent-partitions", "selected-candidate"}, sample(8), 8192)


def test_preservation_rejects_escape_and_existing_destination(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    owned.mkdir()
    monkeypatch.setattr(recovery, "RUN", owned)
    outside = tmp_path / "unrelated.txt"
    outside.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        recovery.preserve(outside, owned / "history/unrelated.txt")
    source, destination = owned / "attempt.txt", owned / "history/attempt.txt"
    source.write_text("failed attempt", encoding="utf-8")
    destination.parent.mkdir()
    destination.write_text("older evidence", encoding="utf-8")
    with pytest.raises(ValueError):
        recovery.preserve(source, destination)
    assert outside.read_text() == "keep"
    assert source.read_text() == "failed attempt"
    assert destination.read_text() == "older evidence"
