from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from industrial_anomaly_lab.rd4ad_runtime import (
    DEFAULT_VENDOR_ROOT,
    _load_vendor_file,
    audit_vendor_source,
    load_student_checkpoint_safely,
    load_teacher_checkpoint_safely,
    rd4ad_anomaly_maps,
    rd4ad_image_scores,
)


def test_batched_anomaly_map_matches_cosine_sum_without_smoothing() -> None:
    teacher = torch.ones((2, 3, 2, 3), dtype=torch.float32)
    student = teacher.clone()
    student[1] = -1.0

    maps = rd4ad_anomaly_maps(
        [teacher, teacher],
        [student, student],
        output_size=(4, 5),
        gaussian_sigma=0,
    )

    assert maps.shape == (2, 4, 5)
    assert maps.dtype == np.float32
    assert np.allclose(maps[0], 0.0, atol=1e-6)
    assert np.allclose(maps[1], 4.0, atol=1e-6)
    assert np.allclose(rd4ad_image_scores(maps), [0.0, 4.0], atol=1e-6)


def test_anomaly_map_rejects_feature_contract_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        rd4ad_anomaly_maps(
            [torch.ones(1, 2, 4, 4)],
            [torch.ones(1, 3, 4, 4)],
            output_size=8,
        )


def test_tensor_only_checkpoint_loaders_and_legacy_key_report(
    tmp_path: Path,
) -> None:
    teacher_source = torch.nn.Linear(3, 2)
    teacher_target = torch.nn.Linear(3, 2)
    teacher_path = tmp_path / "teacher.pt"
    torch.save(teacher_source.state_dict(), teacher_path)

    teacher_report = load_teacher_checkpoint_safely(teacher_path, teacher_target)
    assert teacher_report == {"missing_keys": [], "unexpected_keys": []}
    assert torch.equal(teacher_source.weight, teacher_target.weight)

    bn_source = torch.nn.Linear(2, 2)
    decoder_source = torch.nn.Linear(2, 2)
    bn_target = torch.nn.Linear(2, 2)
    decoder_target = torch.nn.Linear(2, 2)
    bn_state = dict(bn_source.state_dict())
    bn_state["memory.obsolete"] = torch.ones(1)
    student_path = tmp_path / "student.pt"
    torch.save(
        {"bn": bn_state, "decoder": decoder_source.state_dict()}, student_path
    )

    report = load_student_checkpoint_safely(
        student_path, bn_target, decoder_target
    )
    assert report["dropped_legacy_memory_keys"] == ["memory.obsolete"]
    assert torch.equal(bn_source.weight, bn_target.weight)
    assert torch.equal(decoder_source.weight, decoder_target.weight)


def test_vendor_model_modules_import_without_running_downloads() -> None:
    audit = audit_vendor_source()
    encoder_module = _load_vendor_file("test_resnet", "resnet.py", DEFAULT_VENDOR_ROOT)
    decoder_module = _load_vendor_file(
        "test_de_resnet", "de_resnet.py", DEFAULT_VENDOR_ROOT
    )

    assert callable(encoder_module.wide_resnet50_2)
    assert callable(decoder_module.de_wide_resnet50_2)
    assert audit["all_expected_files_match"] is True
