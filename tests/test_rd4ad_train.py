from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from industrial_anomaly_lab.rd4ad_train import (
    Rd4adTrainSettings,
    inspect_training_inputs,
    rd4ad_cosine_loss,
    run_training,
)
from industrial_anomaly_lab.run_artifacts import sha256_file


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [self.layer(image)]


class _TinyBottleneck(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.layer(features[0])


class _TinyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> list[torch.Tensor]:
        return [self.layer(feature)]


def _write_train_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.arange(24 * 28 * 3, dtype=np.uint8).reshape(24, 28, 3)
    Image.fromarray(pixels, mode="RGB").save(path)


def _inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    dataset_root = tmp_path / "visa"
    _write_train_image(dataset_root / "candle/train/good/000.JPG")
    teacher_path = tmp_path / "teacher.pth"
    torch.save(_TinyEncoder().state_dict(), teacher_path)
    return dataset_root, teacher_path, sha256_file(teacher_path)


def _settings(
    tmp_path: Path,
    *,
    execution_scope: str = "debug_limited",
    max_steps: int | None = 1,
    output_name: str = "run",
) -> Rd4adTrainSettings:
    dataset_root, teacher_path, teacher_sha256 = _inputs(tmp_path)
    return Rd4adTrainSettings(
        dataset="visa",
        category="candle",
        dataset_root=dataset_root,
        teacher_checkpoint=teacher_path,
        teacher_sha256=teacher_sha256,
        output_dir=tmp_path / output_name,
        seed=111,
        image_size=16,
        batch_size=1,
        epochs=1,
        learning_rate=0.005,
        workers=0,
        device="cpu",
        execution_scope=execution_scope,  # type: ignore[arg-type]
        max_steps=max_steps,
    )


def test_max_steps_is_only_accepted_for_debug_limited(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _settings(tmp_path, execution_scope="full", max_steps=1).validate()
    with pytest.raises(ValueError, match="requires a positive max_steps"):
        _settings(tmp_path, max_steps=None).validate()

    full = _settings(tmp_path, execution_scope="full", max_steps=None)
    object.__setattr__(full, "image_size", 256)
    object.__setattr__(full, "batch_size", 3)
    object.__setattr__(full, "epochs", 200)
    with pytest.raises(ValueError, match="batch_size must be 16, 8 or 4"):
        full.validate()


def test_author_cosine_loss_matches_flattened_scale_sum() -> None:
    teacher = torch.ones((2, 3, 2, 2), requires_grad=False)
    same = teacher.clone().requires_grad_(True)
    opposite = (-teacher).clone().requires_grad_(True)

    zero = rd4ad_cosine_loss([teacher], [same])
    four = rd4ad_cosine_loss([teacher, teacher], [opposite, opposite])
    four.backward()

    assert zero.item() == pytest.approx(0.0, abs=1e-6)
    assert four.item() == pytest.approx(4.0, abs=1e-6)
    assert opposite.grad is not None


def test_validate_only_uses_train_good_without_test_tree(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_train._inspect_teacher_checkpoint",
        lambda path, expected: {
            "checkpoint_path": str(path),
            "sha256": expected,
            "sha256_verified": True,
            "implicit_download_used": False,
        },
    )

    report = inspect_training_inputs(settings)

    assert report["input_inventory"]["normal_training_images"] == 1
    assert len(
        report["input_inventory"]["relative_path_size_and_content_sha256"]
    ) == 64
    assert report["input_inventory"]["allowed_split"] == "train/good only"
    assert report["input_inventory"]["test_or_ground_truth_enumerated"] is False
    assert report["training_started"] is False
    assert report["test_split_accessed"] is False
    assert not (Path(settings.dataset_root) / "candle/test").exists()
    assert not Path(settings.output_dir).exists()


def test_debug_training_saves_audited_final_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_train._inspect_teacher_checkpoint",
        lambda path, expected: {
            "checkpoint_path": str(path),
            "sha256": expected,
            "sha256_verified": True,
            "implicit_download_used": False,
        },
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_train.build_models",
        lambda device: (
            _TinyEncoder().to(device),
            _TinyBottleneck().to(device),
            _TinyDecoder().to(device),
        ),
    )

    result = run_training(settings)
    output = Path(settings.output_dir)
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "checkpoints/manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = output / "checkpoints/rd4ad_final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert result["status"] == "debug_completed"
    assert state["global_step"] == 1
    assert config["execution_scope"] == "debug_limited"
    assert config["test_split_accessed"] is False
    assert config["runtime"]["implicit_network_downloads"] is False
    assert config["teacher_vendor_load_report"]["strict_load"]["passed"] is True
    assert manifest["final_checkpoint_only"] is True
    assert manifest["test_selected"] is False
    assert manifest["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert set(payload) == {"bn", "decoder"}
    assert (output / "train_steps.jsonl").is_file()
    assert not (Path(settings.dataset_root) / "candle/test").exists()


def test_nonfinite_loss_fails_without_final_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, output_name="nonfinite")
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_train._inspect_teacher_checkpoint",
        lambda path, expected: {
            "checkpoint_path": str(path),
            "sha256": expected,
            "sha256_verified": True,
            "implicit_download_used": False,
        },
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_train.build_models",
        lambda device: (
            _TinyEncoder().to(device),
            _TinyBottleneck().to(device),
            _TinyDecoder().to(device),
        ),
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_train.rd4ad_cosine_loss",
        lambda teacher, student: torch.tensor(float("nan"), requires_grad=True),
    )

    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_training(settings)

    state = json.loads(
        (Path(settings.output_dir) / "run_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "failed"
    assert state["global_step"] == 0
    assert not (Path(settings.output_dir) / "checkpoints/rd4ad_final.pt").exists()
