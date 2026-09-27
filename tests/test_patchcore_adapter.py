from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.patchcore_adapter import (
    PATCHCORE_AUTHOR_COMMIT,
    PatchCoreDataset,
    PatchCoreExportSettings,
    PatchCoreGeometry,
    audit_patchcore_runtime,
    build_patchcore_loader,
    export_patchcore_predictions,
    import_patchcore_vendor,
    run_patchcore_synthetic_smoke,
)
from industrial_anomaly_lab.data_protocols import TestSample as DatasetTestSample
from industrial_anomaly_lab.result_aggregation import load_completed_run


def _write_rgb(path: Path, shape: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*shape, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[shape[0] // 4 : shape[0] // 2, shape[1] // 3 : shape[1] // 2] = 255
    Image.fromarray(mask).save(path)


@pytest.mark.parametrize(
    ("dataset_name", "category", "image_suffix", "mask_name"),
    [
        ("mvtec_ad", "bottle", ".png", "001_mask.png"),
        ("visa", "candle", ".JPG", "001.png"),
    ],
)
def test_dataset_supports_both_layouts_and_author_geometry(
    tmp_path: Path,
    dataset_name: str,
    category: str,
    image_suffix: str,
    mask_name: str,
) -> None:
    _write_rgb(tmp_path / category / "train/good" / f"000{image_suffix}", (40, 70), 80)
    _write_rgb(tmp_path / category / "test/good" / f"000{image_suffix}", (40, 70), 90)
    _write_rgb(tmp_path / category / "test/crack" / f"001{image_suffix}", (40, 70), 180)
    _write_mask(tmp_path / category / "ground_truth/crack" / mask_name, (40, 70))

    geometry = PatchCoreGeometry(resize=40, image_size=32, evaluation_size=48)
    train = PatchCoreDataset(
        dataset_name, tmp_path, category, split="train", geometry=geometry
    )
    test = PatchCoreDataset(
        dataset_name, tmp_path, category, split="test", geometry=geometry
    )
    loader = build_patchcore_loader(
        train, batch_size=1, workers=0, seed=0, pin_memory=False
    )

    assert tuple(train[0]["image"].shape) == (3, 32, 32)
    assert tuple(next(iter(loader))["image"].shape) == (1, 3, 32, 32)
    assert [int(test[index]["label"]) for index in range(len(test))] == [1, 0]
    assert tuple(test[0]["mask"].shape) == (1, 48, 48)
    assert set(np.unique(test[0]["mask"].numpy())).issubset({0.0, 1.0})
    assert float(test[1]["mask"].sum()) == 0.0


def test_export_uses_saved_unnormalized_maps_and_dual_image_scores(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "visa"
    _write_rgb(data_root / "candle/test/good/000.JPG", (40, 70), 80)
    _write_rgb(data_root / "candle/test/bad/001.JPG", (40, 70), 180)
    _write_mask(data_root / "candle/ground_truth/bad/001.png", (40, 70))
    geometry = PatchCoreGeometry(resize=40, image_size=32, evaluation_size=48)

    bad_map = np.zeros((32, 32), dtype=np.float64)
    bad_map[10:20, 12:22] = 7.5
    # Deliberately reverse the raw-map-max ordering.  The PatchCore author
    # image score is a separate patch-score aggregation and must stay separate.
    good_map = np.full((32, 32), 8.0, dtype=np.float64)
    output = tmp_path / "run"
    result = export_patchcore_predictions(
        PatchCoreExportSettings(
            dataset="visa",
            category="candle",
            dataset_root=data_root,
            output_dir=output,
            seed=0,
            execution_scope="debug_limited",
            model_provenance={"kind": "synthetic_test_double"},
            geometry=geometry,
        ),
        author_maps=[bad_map, good_map],
        author_image_scores=[9.0, 1.0],
    )

    rows = [
        json.loads(line)
        for line in (output / "predictions/index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    persisted_bad = np.load(output / rows[0]["raw_map"], allow_pickle=False)
    persisted_good = np.load(output / rows[1]["raw_map"], allow_pickle=False)
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert result["num_images"] == 2
    assert persisted_bad.shape == (48, 48)
    assert persisted_bad.dtype == np.float32
    assert persisted_bad.max() == pytest.approx(7.5)
    assert persisted_good.min() == pytest.approx(8.0)
    assert persisted_good.max() == pytest.approx(8.0)
    assert "image_score" not in rows[0]
    assert rows[0]["method_native_image_score"] == pytest.approx(9.0)
    assert rows[0]["raw_map_max_score"] == pytest.approx(7.5)
    assert result["metrics"]["image_auroc_method_native"] == pytest.approx(1.0)
    assert result["metrics"]["image_auroc"] == pytest.approx(0.0)
    assert run_config["prediction_contract"]["raw_map_normalization"] == "none"
    assert (
        run_config["prediction_contract"]["image_scores"][
            "common_image_score_field"
        ]
        == "raw_map_max_score"
    )
    assert run_config["execution_scope"] == "debug_limited"
    assert run_config["run_purpose"] == "debug_limited"
    assert run_config["claim_scope"] == "debug_only"
    assert run_config["formal_matrix_eligible"] is False
    assert run_config["evaluation_protocol"].startswith("synthetic_nonstandard")
    assert run_config["settings"]["vendor_commit"] == PATCHCORE_AUTHOR_COMMIT
    with pytest.raises(ValueError, match="debug-limited"):
        load_completed_run(output)


def test_export_rejects_wrong_map_shape_before_writing_prediction(tmp_path: Path) -> None:
    data_root = tmp_path / "mvtec"
    _write_rgb(data_root / "bottle/test/good/000.png", (32, 32), 80)
    _write_rgb(data_root / "bottle/test/crack/001.png", (32, 32), 180)
    _write_mask(data_root / "bottle/ground_truth/crack/001_mask.png", (32, 32))
    settings = PatchCoreExportSettings(
        dataset="mvtec_ad",
        category="bottle",
        dataset_root=data_root,
        output_dir=tmp_path / "run",
        seed=0,
        execution_scope="debug_limited",
        model_provenance={"kind": "synthetic_test_double"},
        geometry=PatchCoreGeometry(resize=32, image_size=24, evaluation_size=32),
    )

    with pytest.raises(ValueError, match="must have shape"):
        export_patchcore_predictions(
            settings,
            author_maps=[np.zeros((23, 24)), np.zeros((24, 24))],
            author_image_scores=[0.0, 1.0],
        )

    state = json.loads((tmp_path / "run/run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["num_images_written"] == 0


def test_full_patchcore_export_is_not_implicitly_a_formal_run(tmp_path: Path) -> None:
    data_root = tmp_path / "mvtec"
    _write_rgb(data_root / "bottle/test/good/000.png", (32, 40), 80)
    _write_rgb(data_root / "bottle/test/crack/001.png", (32, 40), 180)
    _write_mask(data_root / "bottle/ground_truth/crack/001_mask.png", (32, 40))
    geometry = PatchCoreGeometry()
    anomalous = np.zeros((224, 224), dtype=np.float32)
    anomalous[56:168, 65:159] = 5.0
    normal = np.zeros((224, 224), dtype=np.float32)
    output = tmp_path / "formal-run"

    export_patchcore_predictions(
        PatchCoreExportSettings(
            dataset="mvtec_ad",
            category="bottle",
            dataset_root=data_root,
            output_dir=output,
            seed=1887537248,
            execution_scope="full",
            model_provenance={"kind": "synthetic_contract_fixture"},
            geometry=geometry,
        ),
        author_maps=[anomalous, normal],
        author_image_scores=[5.0, 0.0],
    )

    with pytest.raises(ValueError, match="run_purpose is not formal_full"):
        load_completed_run(output)
    metrics_payload = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics_payload["protocol"] == "industrial_anomaly_raw_map_max_256_v2"
    assert metrics_payload["schema_version"] == 2
    assert metrics_payload["image_score_contract"]["common_image_auroc_metric"] == (
        "image_auroc"
    )
    assert "image_auroc_method_native" in metrics_payload["metrics"]
    run_config = json.loads(
        (output / "run_config.json").read_text(encoding="utf-8")
    )
    run_state = json.loads(
        (output / "run_state.json").read_text(encoding="utf-8")
    )
    assert run_config["run_purpose"] == "author_reproduction"
    assert run_config["formal_matrix_eligible"] is False
    assert run_state["run_config_sha256"]


def test_formal_export_consumes_explicit_inventory_and_binds_all_outputs(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "mvtec"
    normal_path = data_root / "bottle/test/good/000.png"
    anomaly_path = data_root / "bottle/test/crack/001.png"
    mask_path = data_root / "bottle/ground_truth/crack/001_mask.png"
    _write_rgb(normal_path, (32, 40), 80)
    _write_rgb(anomaly_path, (32, 40), 180)
    _write_mask(mask_path, (32, 40))
    samples = [
        DatasetTestSample(
            dataset="mvtec_ad",
            category="bottle",
            defect_type="good",
            sample_id="bottle/good/000",
            image_path=normal_path,
            mask_path=None,
            label=0,
        ),
        DatasetTestSample(
            dataset="mvtec_ad",
            category="bottle",
            defect_type="crack",
            sample_id="bottle/crack/001",
            image_path=anomaly_path,
            mask_path=mask_path,
            label=1,
        ),
    ]
    evidence = {
        "schema_version": 1,
        "contract": "formal_test_inventory_category_selection_v1",
        "bundle": "artifacts/formal_test_inventory/mvtec_ad",
        "inventory_id": "synthetic-inventory",
        "dataset": "mvtec_ad",
        "split": "test",
        "category": "bottle",
        "inventory_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "entry_count": 2,
        "category_entry_count": 2,
        "dataset_root": "data/mvtec_ad",
    }
    entries = [
        {
            "sample_id": samples[0].sample_id,
            "label": 0,
            "defect_type": "good",
            "source_image": "bottle/test/good/000.png",
            "source_mask": None,
            "source_image_sha256": "3" * 64,
            "source_mask_sha256": None,
        },
        {
            "sample_id": samples[1].sample_id,
            "label": 1,
            "defect_type": "crack",
            "source_image": "bottle/test/crack/001.png",
            "source_mask": "bottle/ground_truth/crack/001_mask.png",
            "source_image_sha256": "4" * 64,
            "source_mask_sha256": "5" * 64,
        },
    ]
    normal = np.zeros((224, 224), dtype=np.float32)
    anomalous = np.zeros((224, 224), dtype=np.float32)
    anomalous[60:170, 70:160] = 5.0
    output = tmp_path / "formal"

    result = export_patchcore_predictions(
        PatchCoreExportSettings(
            dataset="mvtec_ad",
            category="bottle",
            dataset_root=data_root,
            output_dir=output,
            seed=1887537248,
            execution_scope="full",
            run_purpose="formal_full",
            formal_provenance={"claim_scope": "formal_matrix_candidate"},
            model_provenance={"kind": "synthetic_contract_fixture"},
        ),
        author_maps=[normal, anomalous],
        author_image_scores=[0.0, 5.0],
        formal_test_selection={
            "evidence": evidence,
            "entries": entries,
            "samples": samples,
            "dataset_root": data_root,
        },
    )

    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output / "predictions/index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["status"] == "completed"
    assert config["formal_test_inventory"] == evidence
    assert state["formal_test_inventory"] == evidence
    assert metrics["formal_test_inventory"] == evidence
    assert state["run_config_sha256"]
    assert state["metrics_sha256"]
    assert state["predictions_index_sha256"]
    assert rows[0]["formal_test_inventory_id"] == "synthetic-inventory"
    assert rows[1]["source_ground_truth_sha256"] == "5" * 64


def test_full_export_rejects_nonstandard_geometry(tmp_path: Path) -> None:
    settings = PatchCoreExportSettings(
        dataset="visa",
        category="candle",
        dataset_root=tmp_path,
        output_dir=tmp_path / "run",
        seed=1887537248,
        execution_scope="full",
        model_provenance={"kind": "synthetic_contract_fixture"},
        geometry=PatchCoreGeometry(resize=32, image_size=24, evaluation_size=32),
    )

    with pytest.raises(ValueError, match="full PatchCore exports require frozen geometry"):
        settings.validate()


def test_runtime_audit_is_read_only_and_missing_dependencies_fail_closed(
    tmp_path: Path,
) -> None:
    audit = audit_patchcore_runtime(tmp_path)

    assert audit["import_attempted"] is False
    assert audit["training_or_test_evaluation_run"] is False
    assert audit["runtime_import_ready"] is False
    with pytest.raises(RuntimeError, match="runtime is not ready"):
        import_patchcore_vendor(tmp_path)


def test_frozen_author_runtime_import_and_synthetic_cpu_smoke() -> None:
    audit = audit_patchcore_runtime()
    modules = import_patchcore_vendor()
    result = run_patchcore_synthetic_smoke(seed=17)

    assert audit["runtime_import_ready"] is True
    assert [module.__name__ for module in modules] == [
        "patchcore.patchcore",
        "patchcore.backbones",
        "patchcore.common",
        "patchcore.sampler",
    ]
    assert result["status"] == "passed"
    assert result["formal_training_run"] is False
    assert result["formal_test_content_accessed"] is False
    assert result["pretrained_weights_requested"] is False
    assert result["author_backbone_loader_called"] is False
    assert result["feature_bank_shape"] == [32, 8]
    assert result["faiss_index_size"] == 32
    assert result["map_shape"] == [1, 8, 8]
    assert result["outputs_finite"] is True
