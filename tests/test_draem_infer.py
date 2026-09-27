from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from industrial_anomaly_lab.draem_infer import InferenceSettings, run_inference


class _TinyReconstructive(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


class _TinyDiscriminative(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(6, 2, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 12:24] = 255
    Image.fromarray(mask).save(path)


def test_inference_persists_raw_maps_and_evaluates_saved_arrays(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "mvtec"
    _write_rgb(data_root / "bottle/test/good/000.png", 80)
    _write_rgb(data_root / "bottle/test/crack/001.png", 180)
    _write_mask(data_root / "bottle/ground_truth/crack/001_mask.png")

    reconstructive = _TinyReconstructive()
    discriminative = _TinyDiscriminative()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save(reconstructive.state_dict(), checkpoints / "reconstructive.pt")
    torch.save(discriminative.state_dict(), checkpoints / "discriminative.pt")

    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.build_models",
        lambda device: (_TinyReconstructive().to(device), _TinyDiscriminative().to(device)),
    )
    output = tmp_path / "output"
    result = run_inference(
        InferenceSettings(
            dataset="mvtec_ad",
            category="bottle",
            dataset_root=data_root,
            reconstructive_checkpoint=checkpoints / "reconstructive.pt",
            discriminative_checkpoint=checkpoints / "discriminative.pt",
            output_dir=output,
            image_size=32,
            batch_size=2,
        )
    )

    rows = [
        json.loads(line)
        for line in (output / "predictions/index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    raw_map = np.load(output / rows[0]["raw_map"], allow_pickle=False)
    mask = np.load(output / rows[0]["evaluation_mask"], allow_pickle=False)

    assert result["status"] == "completed"
    assert result["num_images"] == 2
    assert raw_map.shape == (32, 32)
    assert raw_map.dtype == np.float32
    assert np.isfinite(raw_map).all()
    assert mask.dtype == np.bool_
    assert all("image_score" in row for row in rows)
    assert (output / "metrics.json").is_file()
    assert (output / "run_config.json").is_file()
