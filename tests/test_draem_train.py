from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

import industrial_anomaly_lab.draem_train as draem_train
from industrial_anomaly_lab.draem_train import TrainSettings, run_training


class _MemoryTrainDataset(Dataset[dict[str, object]]):
    """One synthetic train sample with no filesystem-backed image access."""

    def __init__(self) -> None:
        self.image_paths = [Path("memory-normal.png")]
        self.anomaly_source_paths = [Path("memory-texture.jpg")]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        return {
            "image": torch.zeros((3, 4, 4), dtype=torch.float32),
            "augmented_image": torch.ones((3, 4, 4), dtype=torch.float32),
            "anomaly_mask": torch.ones((1, 4, 4), dtype=torch.float32),
            "has_anomaly": torch.ones((1,), dtype=torch.float32),
            "idx": 0,
            "image_path": "memory-normal.png",
        }


class _TinyReconstructive(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


class _TinyDiscriminative(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(6, 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


class _ZeroLoss(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return prediction.sum() * 0.0


class _NonfiniteLoss(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return prediction.sum() * 0.0 + prediction.new_tensor(float("nan"))


class _FiniteValueNonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction: torch.Tensor) -> torch.Tensor:
        ctx.shape = prediction.shape
        return prediction.new_zeros(())

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor) -> torch.Tensor:
        del output_gradient
        return torch.full(ctx.shape, float("nan"))


class _NonfiniteGradientLoss(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return _FiniteValueNonfiniteGradient.apply(prediction)


def _settings(tmp_path: Path, output_name: str) -> TrainSettings:
    return TrainSettings(
        dataset="visa",
        category="candle",
        dataset_root=tmp_path / "unused-dataset",
        anomaly_source_root=tmp_path / "unused-dtd",
        output_dir=tmp_path / output_name,
        seed=17,
        image_size=4,
        batch_size=1,
        epochs=1,
        learning_rate=1e-4,
        workers=0,
        device="cpu",
        max_steps=1,
        run_purpose="debug_limited",
    )


def _patch_memory_training(monkeypatch, *, focal_loss: torch.nn.Module):
    models = (_TinyReconstructive(), _TinyDiscriminative())
    monkeypatch.setattr(
        draem_train,
        "_resolve_training_dataset",
        lambda settings: (
            _MemoryTrainDataset(),
            {
                "input_protocol": settings.input_protocol,
                "formal_test_content_accessed": False,
                "ground_truth_content_accessed": False,
                "training_started": False,
            },
        ),
    )
    monkeypatch.setattr(
        draem_train,
        "build_models",
        lambda device: tuple(model.to(device) for model in models),
    )
    monkeypatch.setattr(draem_train, "DeviceSafeSSIM", lambda: _ZeroLoss())
    monkeypatch.setattr(draem_train, "build_focal_loss", lambda: focal_loss)
    return models


def test_nonfinite_component_loss_fails_before_backward_and_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, "nonfinite-loss")
    models = _patch_memory_training(monkeypatch, focal_loss=_NonfiniteLoss())

    with pytest.raises(
        FloatingPointError,
        match=r"nonfinite DRAEM loss at epoch=1, step=1: .*loss_segmentation",
    ):
        run_training(settings)

    output = Path(settings.output_dir)
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["epoch"] == 1
    assert state["global_step"] == 0
    assert state["failed_step"] == 1
    assert state["checkpoint_cleanup_complete"] is True
    assert all(parameter.grad is None for model in models for parameter in model.parameters())
    assert not (output / "checkpoints/reconstructive.pt").exists()
    assert not (output / "checkpoints/discriminative.pt").exists()
    assert not (output / "checkpoints/manifest.json").exists()

def test_nonfinite_gradient_fails_before_optimizer_step_and_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, "nonfinite-gradient")
    _patch_memory_training(monkeypatch, focal_loss=_NonfiniteGradientLoss())

    with pytest.raises(
        FloatingPointError,
        match=r"nonfinite DRAEM gradients at epoch=1, step=1",
    ):
        run_training(settings)

    output = Path(settings.output_dir)
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["global_step"] == 0
    assert state["failed_step"] == 1
    assert state["checkpoint_cleanup_complete"] is True
    assert not (output / "checkpoints/manifest.json").exists()


def test_failure_after_manifest_removes_all_final_checkpoint_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, "post-save-failure")
    _patch_memory_training(monkeypatch, focal_loss=_ZeroLoss())
    original_write_json = draem_train.write_json
    injected = False

    def fail_completed_state(path: str | Path, payload: object) -> None:
        nonlocal injected
        if (
            not injected
            and Path(path).name == "run_state.json"
            and isinstance(payload, dict)
            and payload.get("status") == "debug_completed"
        ):
            injected = True
            raise OSError("injected failure after checkpoint manifest")
        original_write_json(path, payload)

    monkeypatch.setattr(draem_train, "write_json", fail_completed_state)

    with pytest.raises(OSError, match="injected failure"):
        run_training(settings)

    output = Path(settings.output_dir)
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (output / "train_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert injected is True
    assert state["status"] == "failed"
    assert state["global_step"] == 1
    assert state["failed_step"] is None
    assert state["checkpoint_cleanup_complete"] is True
    assert len(metrics) == 1
    assert math.isfinite(metrics[0]["gradient_norm"])
    assert not (output / "checkpoints/reconstructive.pt").exists()
    assert not (output / "checkpoints/discriminative.pt").exists()
    assert not (output / "checkpoints/manifest.json").exists()


def test_explicit_disabled_pin_memory_is_recorded(
    tmp_path: Path, monkeypatch
) -> None:
    settings = replace(
        _settings(tmp_path, "pin-memory-disabled"), pin_memory=False
    )
    _patch_memory_training(monkeypatch, focal_loss=_ZeroLoss())

    run_training(settings)

    output = Path(settings.output_dir)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "debug_completed"
    assert config["settings"]["pin_memory"] is False
    assert config["runtime"]["pin_memory"] is False
