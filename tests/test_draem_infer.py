from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from industrial_anomaly_lab.data_protocols import TestSample as InventoryTestSample
from industrial_anomaly_lab.draem_infer import InferenceSettings, run_inference
from industrial_anomaly_lab.run_artifacts import sha256_file


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
            execution_scope="debug_limited",
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
    assert all("image_score" not in row for row in rows)
    assert all("raw_map_max_score" in row for row in rows)
    assert all("method_native_image_score" in row for row in rows)
    assert rows[0]["raw_map_max_score"] == pytest.approx(float(raw_map.max()))
    assert (output / "metrics.json").is_file()
    assert (output / "run_config.json").is_file()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert config["execution_scope"] == "debug_limited"
    assert config["schema_version"] == 2
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 2
    assert metrics["image_score_contract"]["common_image_score_field"] == (
        "raw_map_max_score"
    )
    assert "image_auroc_method_native" in metrics["metrics"]


def test_formal_inference_refuses_pending_authority_before_test_dataset_access(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    reconstructive = checkpoints / "reconstructive.pt"
    discriminative = checkpoints / "discriminative.pt"
    reconstructive.write_bytes(b"synthetic")
    discriminative.write_bytes(b"synthetic")
    dataset_constructed = False

    def _forbidden_dataset(*args, **kwargs):
        nonlocal dataset_constructed
        dataset_constructed = True
        raise AssertionError("formal gate must run before test dataset construction")

    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.DraemTestDataset", _forbidden_dataset
    )
    with pytest.raises(ValueError, match="formal evaluation protocol is not frozen"):
        run_inference(
            InferenceSettings(
                dataset="visa",
                category="candle",
                dataset_root=tmp_path / "must-not-be-read",
                reconstructive_checkpoint=reconstructive,
                discriminative_checkpoint=discriminative,
                output_dir=tmp_path / "must-not-be-written",
                execution_scope="full",
                seed=1887537248,
                run_purpose="formal_full",
                checkpoint_manifest=checkpoints / "manifest.json",
            )
        )
    assert dataset_constructed is False
    assert not (tmp_path / "must-not-be-written").exists()


def _formal_selection(data_root: Path) -> dict[str, object]:
    image_paths = [
        data_root / "bottle/test/good/000.png",
        data_root / "bottle/test/crack/001.png",
    ]
    mask_path = data_root / "bottle/ground_truth/crack/001_mask.png"
    samples = [
        InventoryTestSample(
            dataset="mvtec_ad",
            category="bottle",
            defect_type="good",
            sample_id="bottle/good/000",
            image_path=image_paths[0],
            mask_path=None,
            label=0,
        ),
        InventoryTestSample(
            dataset="mvtec_ad",
            category="bottle",
            defect_type="crack",
            sample_id="bottle/crack/001",
            image_path=image_paths[1],
            mask_path=mask_path,
            label=1,
        ),
    ]
    entries = [
        {
            "dataset": "mvtec_ad",
            "category": "bottle",
            "sample_id": sample.sample_id,
            "label": sample.label,
            "defect_type": sample.defect_type,
            "source_image": sample.image_path.relative_to(data_root).as_posix(),
            "source_image_sha256": sha256_file(sample.image_path),
            "source_mask": (
                None
                if sample.mask_path is None
                else sample.mask_path.relative_to(data_root).as_posix()
            ),
            "source_mask_sha256": (
                None if sample.mask_path is None else sha256_file(sample.mask_path)
            ),
        }
        for sample in samples
    ]
    evidence = {
        "schema_version": 1,
        "contract": "formal_test_inventory_category_selection_v1",
        "bundle": "manifests/formal/mvtec_ad",
        "inventory_id": "mvtec-ad-synthetic",
        "dataset": "mvtec_ad",
        "split": "test",
        "category": "bottle",
        "inventory_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "entry_count": 2,
        "category_entry_count": 2,
        "dataset_root": "data/mvtec_ad",
    }
    return {"evidence": evidence, "entries": entries, "samples": samples}


def test_formal_inference_uses_inventory_samples_and_binds_all_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "mvtec"
    _write_rgb(data_root / "bottle/test/good/000.png", 80)
    _write_rgb(data_root / "bottle/test/crack/001.png", 180)
    _write_mask(data_root / "bottle/ground_truth/crack/001_mask.png")
    selection = _formal_selection(data_root)

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save(_TinyReconstructive().state_dict(), checkpoints / "reconstructive.pt")
    torch.save(_TinyDiscriminative().state_dict(), checkpoints / "discriminative.pt")
    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.build_models",
        lambda device: (_TinyReconstructive().to(device), _TinyDiscriminative().to(device)),
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.validate_formal_authority",
        lambda **kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.validate_formal_checkpoint_manifest",
        lambda *args, **kwargs: {"validated": True},
    )
    resolver_calls: list[dict[str, object]] = []

    def _resolve(**kwargs):
        resolver_calls.append(kwargs)
        return selection

    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.resolve_formal_test_inventory_category",
        _resolve,
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.data_protocols.build_test_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal dataset must not rediscover test directories")
        ),
    )
    inventory_dir = tmp_path / "project/manifests/formal/mvtec_ad"
    output = tmp_path / "output"
    result = run_inference(
        InferenceSettings(
            dataset="mvtec_ad",
            category="bottle",
            dataset_root=data_root,
            reconstructive_checkpoint=checkpoints / "reconstructive.pt",
            discriminative_checkpoint=checkpoints / "discriminative.pt",
            output_dir=output,
            execution_scope="full",
            run_purpose="formal_full",
            checkpoint_manifest=checkpoints / "manifest.json",
            formal_test_inventory_dir=inventory_dir,
            image_size=32,
            batch_size=2,
        )
    )
    assert result["status"] == "completed"
    assert len(resolver_calls) == 1
    assert resolver_calls[0]["output_dir"] == inventory_dir
    evidence = selection["evidence"]
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert config["formal_provenance"]["formal_test_inventory"] == evidence
    assert config["formal_test_inventory"] == metrics["formal_test_inventory"] == evidence
    assert state["formal_test_inventory"] == evidence
    rows = [
        json.loads(line)
        for line in (output / "predictions/index.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in rows] == [
        "bottle/good/000",
        "bottle/crack/001",
    ]
    assert all(row["formal_test_inventory_id"] == evidence["inventory_id"] for row in rows)


def test_formal_inference_requires_inventory_before_output_creation(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    reconstructive = checkpoints / "reconstructive.pt"
    discriminative = checkpoints / "discriminative.pt"
    reconstructive.write_bytes(b"synthetic")
    discriminative.write_bytes(b"synthetic")
    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.validate_formal_authority",
        lambda **kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_infer.validate_formal_checkpoint_manifest",
        lambda *args, **kwargs: {"validated": True},
    )
    output = tmp_path / "must-not-be-written"
    with pytest.raises(ValueError, match="requires formal_test_inventory_dir"):
        run_inference(
            InferenceSettings(
                dataset="mvtec_ad",
                category="bottle",
                dataset_root=tmp_path / "must-not-be-read",
                reconstructive_checkpoint=reconstructive,
                discriminative_checkpoint=discriminative,
                output_dir=output,
                execution_scope="full",
                run_purpose="formal_full",
                checkpoint_manifest=checkpoints / "manifest.json",
            )
        )
    assert not output.exists()
