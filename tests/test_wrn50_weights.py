from __future__ import annotations

from pathlib import Path

import pytest
import torch

from industrial_anomaly_lab.patchcore_adapter import (
    PATCHCORE_AUTHOR_COMMIT,
    build_patchcore_backbone_from_local_checkpoint,
    build_patchcore_model_from_local_checkpoint,
)
from industrial_anomaly_lab.run_artifacts import sha256_file
from industrial_anomaly_lab.wrn50_weights import (
    WRN50_IMAGENET1K_V1_FILENAME,
    WRN50_IMAGENET1K_V1_URL,
    build_torchvision_wrn50_from_local_checkpoint,
    inspect_wrn50_checkpoint,
    load_wrn50_checkpoint_into_model,
    official_wrn50_source_metadata,
)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, output_features: int = 2) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, output_features)
        self.register_buffer("counter", torch.tensor(4, dtype=torch.int64))


class _TinyPatchCoreBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(image))


def _write_checkpoint(path: Path, model: torch.nn.Module) -> str:
    torch.save(model.state_dict(), path)
    return sha256_file(path)


def test_official_metadata_matches_installed_torchvision_without_download() -> None:
    metadata = official_wrn50_source_metadata()

    assert metadata["official_url"] == WRN50_IMAGENET1K_V1_URL
    assert metadata["official_filename"] == WRN50_IMAGENET1K_V1_FILENAME
    assert metadata["url_hash_prefix"] == "95faca4d"
    assert metadata["parameter_count"] == 68_883_240
    assert metadata["legacy_pretrained_true_target"].endswith("IMAGENET1K_V1")
    assert metadata["preprocessing"]["center_crop"] == 224
    assert metadata["download_attempted"] is False


def test_hash_checked_tensor_only_strict_load_and_inspection(tmp_path: Path) -> None:
    source = _TinyBackbone()
    checkpoint = tmp_path / WRN50_IMAGENET1K_V1_FILENAME
    expected_sha256 = _write_checkpoint(checkpoint, source)
    target = _TinyBackbone()
    for parameter in target.parameters():
        parameter.data.zero_()

    report = load_wrn50_checkpoint_into_model(
        checkpoint, target, expected_sha256=expected_sha256.upper()
    )
    inspection = inspect_wrn50_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        model_factory=_TinyBackbone,
    )

    assert report["sha256"] == expected_sha256
    assert report["sha256_verified"] is True
    assert report["url_hash_prefix_matched"] is False
    assert report["safe_load"]["weights_only"] is True
    assert report["strict_load"]["passed"] is True
    assert report["dtype_conversion"]["applied_tensor_count"] == 0
    assert inspection["strict_load"]["passed"] is True
    assert inspection["implicit_download_used"] is False
    for expected, actual in zip(
        source.state_dict().values(), target.state_dict().values(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_float16_to_float32_is_explicitly_widened_and_reported(
    tmp_path: Path,
) -> None:
    source = _TinyBackbone()
    state = source.state_dict()
    state["projection.weight"] = state["projection.weight"].to(torch.float16)
    checkpoint = tmp_path / "half-weight.pth"
    torch.save(state, checkpoint)
    expected_sha256 = sha256_file(checkpoint)
    target = _TinyBackbone()

    report = load_wrn50_checkpoint_into_model(
        checkpoint, target, expected_sha256=expected_sha256
    )

    conversion = report["dtype_conversion"]
    assert conversion["policy"] == "allowlisted_widening_only"
    assert conversion["applied_tensor_count"] == 1
    assert conversion["applied"] == [
        {
            "source_dtype": "torch.float16",
            "target_dtype": "torch.float32",
            "reason": "floating_point_widening",
            "key_count": 1,
            "keys": ["projection.weight"],
        }
    ]
    assert target.projection.weight.dtype == torch.float32
    assert torch.equal(
        target.projection.weight,
        state["projection.weight"].to(torch.float32),
    )


def test_local_builder_uses_weights_none_and_patchcore_bypasses_vendor_loader(
    tmp_path: Path, monkeypatch
) -> None:
    source = _TinyBackbone()
    checkpoint = tmp_path / WRN50_IMAGENET1K_V1_FILENAME
    expected_sha256 = _write_checkpoint(checkpoint, source)
    calls: list[dict[str, object]] = []

    def fake_constructor(*, weights, progress):
        calls.append({"weights": weights, "progress": progress})
        return _TinyBackbone()

    monkeypatch.setattr(
        "industrial_anomaly_lab.wrn50_weights.torchvision_models.wide_resnet50_2",
        fake_constructor,
    )
    model, report = build_torchvision_wrn50_from_local_checkpoint(
        checkpoint, expected_sha256=expected_sha256
    )
    patchcore_model, patchcore_report = build_patchcore_backbone_from_local_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        device="cpu",
    )

    assert isinstance(model, _TinyBackbone)
    assert calls == [
        {"weights": None, "progress": False},
        {"weights": None, "progress": False},
    ]
    assert report["implicit_download_used"] is False
    assert patchcore_model.name == "wideresnet50"
    assert patchcore_model.seed is None
    assert patchcore_report["vendor_backbones_load_called"] is False
    assert patchcore_report["vendor_pretrained_true_called"] is False
    assert patchcore_report["patchcore_vendor_commit"] == PATCHCORE_AUTHOR_COMMIT


def test_checkpoint_failures_are_rejected_before_model_mutation(tmp_path: Path) -> None:
    source = _TinyBackbone()
    checkpoint = tmp_path / "checkpoint.pth"
    expected_sha256 = _write_checkpoint(checkpoint, source)
    target = _TinyBackbone(output_features=4)
    before = {key: value.clone() for key, value in target.state_dict().items()}

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_wrn50_checkpoint_into_model(
            checkpoint, target, expected_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="shape_mismatch"):
        load_wrn50_checkpoint_into_model(
            checkpoint, target, expected_sha256=expected_sha256
        )
    for key, value in target.state_dict().items():
        assert torch.equal(value, before[key])


def test_patchcore_model_assembly_uses_local_checkpoint_without_data(
    tmp_path: Path,
) -> None:
    source = _TinyPatchCoreBackbone()
    checkpoint = tmp_path / WRN50_IMAGENET1K_V1_FILENAME
    expected_sha256 = _write_checkpoint(checkpoint, source)

    model, report = build_patchcore_model_from_local_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        device="cpu",
        coreset_device="cpu",
        faiss_workers=1,
        model_factory=_TinyPatchCoreBackbone,
    )

    assert set(model.forward_modules) == {
        "feature_aggregator",
        "preprocessing",
        "preadapt_aggregator",
    }
    assert report["model_constructed"] is True
    assert report["memory_bank_fitted"] is False
    assert report["dataset_content_accessed"] is False
    assert report["implicit_download_used"] is False
    assert report["configuration"]["layers"] == ["layer2", "layer3"]
    assert report["configuration"]["coreset_percentage"] == pytest.approx(0.1)
    assert report["checkpoint"]["vendor_pretrained_true_called"] is False


def test_non_tensor_checkpoint_and_invalid_expected_hash_are_rejected(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "invalid.pth"
    torch.save({"projection.weight": torch.ones(2, 3), "metadata": "unsafe"}, checkpoint)
    actual_sha256 = sha256_file(checkpoint)

    with pytest.raises(ValueError, match="64 hexadecimal"):
        load_wrn50_checkpoint_into_model(
            checkpoint, _TinyBackbone(), expected_sha256="95faca4d"
        )
    with pytest.raises(ValueError, match="non-tensor"):
        load_wrn50_checkpoint_into_model(
            checkpoint, _TinyBackbone(), expected_sha256=actual_sha256
        )


@pytest.mark.parametrize(
    ("key", "unsafe_value"),
    [
        ("projection.weight", lambda value: value.to(torch.float64)),
        ("counter", lambda value: value.to(torch.int32)),
        ("projection.weight", lambda value: torch.full_like(value, float("nan"))),
    ],
)
def test_unsafe_dtype_or_nonfinite_tensor_is_rejected_before_model_mutation(
    tmp_path: Path,
    key: str,
    unsafe_value,
) -> None:
    source = _TinyBackbone()
    state = source.state_dict()
    state[key] = unsafe_value(state[key])
    checkpoint = tmp_path / f"unsafe-{key.replace('.', '-')}.pth"
    torch.save(state, checkpoint)
    expected_sha256 = sha256_file(checkpoint)
    target = _TinyBackbone()
    before = {name: value.clone() for name, value in target.state_dict().items()}

    with pytest.raises(ValueError, match="unsafe_dtype_conversion|nonfinite"):
        load_wrn50_checkpoint_into_model(
            checkpoint, target, expected_sha256=expected_sha256
        )
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])
