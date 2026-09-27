from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from industrial_anomaly_lab.data_protocols import TestSample as InventoryTestSample
from industrial_anomaly_lab.rd4ad_infer import (
    Rd4adInferenceSettings,
    run_rd4ad_inference,
)
from industrial_anomaly_lab.run_artifacts import sha256_file


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [image * self.scale]


class _TinyBottleneck(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return features[0] * self.scale


class _TinyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, feature: torch.Tensor) -> list[torch.Tensor]:
        return [feature * self.scale]


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 30, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[5:17, 8:21] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_rd4ad_inference_persists_raw_maps_and_uses_unified_evaluator(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "visa"
    _write_rgb(data_root / "candle/test/good/000.JPG", 80)
    _write_rgb(data_root / "candle/test/bad/001.JPG", 180)
    _write_mask(data_root / "candle/ground_truth/bad/001.png")

    teacher = _TinyEncoder()
    bottleneck = _TinyBottleneck()
    decoder = _TinyDecoder()
    teacher_path = tmp_path / "teacher.pt"
    student_path = tmp_path / "student.pt"
    torch.save(teacher.state_dict(), teacher_path)
    torch.save(
        {"bn": bottleneck.state_dict(), "decoder": decoder.state_dict()},
        student_path,
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.build_models",
        lambda device: (
            _TinyEncoder().to(device),
            _TinyBottleneck().to(device),
            _TinyDecoder().to(device),
        ),
    )

    output = tmp_path / "output"
    result = run_rd4ad_inference(
        Rd4adInferenceSettings(
            dataset="visa",
            category="candle",
            dataset_root=data_root,
            teacher_checkpoint=teacher_path,
            student_checkpoint=student_path,
            output_dir=output,
            execution_scope="debug_limited",
            image_size=32,
            batch_size=2,
            gaussian_sigma=4.0,
        )
    )

    rows = [
        json.loads(line)
        for line in (output / "predictions/index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    raw_map = np.load(output / rows[0]["raw_map"], allow_pickle=False)
    mask = np.load(output / rows[0]["evaluation_mask"], allow_pickle=False)

    assert result["status"] == "completed"
    assert result["num_images"] == 2
    assert raw_map.shape == (32, 32)
    assert raw_map.dtype == np.float32
    assert np.isfinite(raw_map).all()
    assert mask.dtype == np.bool_
    assert "image_score" not in rows[0]
    assert rows[0]["raw_map_max_score"] == float(raw_map.max())
    assert rows[0]["method_native_image_score"] == float(raw_map.max())
    assert config["prediction_contract"]["raw_map_normalization"] == "none"
    assert config["prediction_contract"]["metrics_reload_saved_arrays"] is True
    assert config["runtime"]["torchvision_pretrained_download"] is False
    assert config["execution_scope"] == "debug_limited"
    assert (output / "metrics.json").is_file()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 2
    assert metrics["image_score_contract"]["common_image_score_field"] == (
        "raw_map_max_score"
    )
    assert metrics["metrics"]["image_auroc"] == pytest.approx(
        metrics["metrics"]["image_auroc_method_native"]
    )


def test_formal_rd4ad_inference_refuses_pending_authority_before_test_access(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = tmp_path / "teacher.pt"
    student = tmp_path / "student.pt"
    teacher.write_bytes(b"synthetic-teacher")
    student.write_bytes(b"synthetic-student")
    dataset_constructed = False

    def _forbidden_dataset(*args, **kwargs):
        nonlocal dataset_constructed
        dataset_constructed = True
        raise AssertionError("formal gate must run before test dataset construction")

    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.Rd4adTestDataset", _forbidden_dataset
    )
    with pytest.raises(ValueError, match="formal evaluation protocol is not frozen"):
        run_rd4ad_inference(
            Rd4adInferenceSettings(
                dataset="visa",
                category="candle",
                dataset_root=tmp_path / "must-not-be-read",
                teacher_checkpoint=teacher,
                student_checkpoint=student,
                output_dir=tmp_path / "must-not-be-written",
                execution_scope="full",
                seed=1887537248,
                run_purpose="formal_full",
                checkpoint_manifest=tmp_path / "manifest.json",
            )
        )
    assert dataset_constructed is False
    assert not (tmp_path / "must-not-be-written").exists()


def _formal_selection(data_root: Path) -> dict[str, object]:
    image_paths = [
        data_root / "candle/test/good/000.JPG",
        data_root / "candle/test/bad/001.JPG",
    ]
    mask_path = data_root / "candle/ground_truth/bad/001.png"
    samples = [
        InventoryTestSample(
            dataset="visa",
            category="candle",
            defect_type="good",
            sample_id="candle/good/000",
            image_path=image_paths[0],
            mask_path=None,
            label=0,
        ),
        InventoryTestSample(
            dataset="visa",
            category="candle",
            defect_type="bad",
            sample_id="candle/bad/001",
            image_path=image_paths[1],
            mask_path=mask_path,
            label=1,
        ),
    ]
    entries = [
        {
            "dataset": "visa",
            "category": "candle",
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
        "bundle": "manifests/formal/visa",
        "inventory_id": "visa-synthetic",
        "dataset": "visa",
        "split": "test",
        "category": "candle",
        "inventory_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
        "entry_count": 2,
        "category_entry_count": 2,
        "dataset_root": "data/visa",
    }
    return {"evidence": evidence, "entries": entries, "samples": samples}


def test_formal_rd4ad_inference_uses_inventory_and_binds_all_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "visa"
    _write_rgb(data_root / "candle/test/good/000.JPG", 80)
    _write_rgb(data_root / "candle/test/bad/001.JPG", 180)
    _write_mask(data_root / "candle/ground_truth/bad/001.png")
    selection = _formal_selection(data_root)

    teacher_path = tmp_path / "teacher.pt"
    student_path = tmp_path / "student.pt"
    torch.save(_TinyEncoder().state_dict(), teacher_path)
    torch.save(
        {
            "bn": _TinyBottleneck().state_dict(),
            "decoder": _TinyDecoder().state_dict(),
        },
        student_path,
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.build_models",
        lambda device: (
            _TinyEncoder().to(device),
            _TinyBottleneck().to(device),
            _TinyDecoder().to(device),
        ),
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.validate_formal_authority",
        lambda **kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.validate_formal_checkpoint_manifest",
        lambda *args, **kwargs: {"validated": True},
    )
    resolver_calls: list[dict[str, object]] = []

    def _resolve(**kwargs):
        resolver_calls.append(kwargs)
        return selection

    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.resolve_formal_test_inventory_category",
        _resolve,
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_data.build_test_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal dataset must not rediscover test directories")
        ),
    )
    inventory_dir = tmp_path / "project/manifests/formal/visa"
    output = tmp_path / "output"
    result = run_rd4ad_inference(
        Rd4adInferenceSettings(
            dataset="visa",
            category="candle",
            dataset_root=data_root,
            teacher_checkpoint=teacher_path,
            student_checkpoint=student_path,
            output_dir=output,
            execution_scope="full",
            run_purpose="formal_full",
            checkpoint_manifest=tmp_path / "manifest.json",
            formal_test_inventory_dir=inventory_dir,
            image_size=32,
            batch_size=2,
            gaussian_sigma=4.0,
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
        "candle/good/000",
        "candle/bad/001",
    ]
    assert all(row["formal_test_inventory_id"] == evidence["inventory_id"] for row in rows)


def test_formal_rd4ad_inference_requires_inventory_before_output_creation(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = tmp_path / "teacher.pt"
    student = tmp_path / "student.pt"
    teacher.write_bytes(b"synthetic-teacher")
    student.write_bytes(b"synthetic-student")
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.validate_formal_authority",
        lambda **kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        "industrial_anomaly_lab.rd4ad_infer.validate_formal_checkpoint_manifest",
        lambda *args, **kwargs: {"validated": True},
    )
    output = tmp_path / "must-not-be-written"
    with pytest.raises(ValueError, match="requires formal_test_inventory_dir"):
        run_rd4ad_inference(
            Rd4adInferenceSettings(
                dataset="visa",
                category="candle",
                dataset_root=tmp_path / "must-not-be-read",
                teacher_checkpoint=teacher,
                student_checkpoint=student,
                output_dir=output,
                execution_scope="full",
                run_purpose="formal_full",
                checkpoint_manifest=tmp_path / "manifest.json",
            )
        )
    assert not output.exists()
