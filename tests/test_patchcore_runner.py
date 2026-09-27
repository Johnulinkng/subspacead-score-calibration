from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.patchcore_adapter import PatchCoreGeometry
from industrial_anomaly_lab.patchcore_adapter import audit_patchcore_runtime
import industrial_anomaly_lab.patchcore_runner as patchcore_runner_module
from industrial_anomaly_lab.patchcore_runner import (
    FROZEN_CONFIGURATION,
    PatchCoreFitSettings,
    PatchCoreInferenceSettings,
    inspect_fit_inputs,
    inspect_inference_inputs,
    load_fitted_patchcore_state,
    run_patchcore_fit,
    run_patchcore_inference,
    save_fitted_patchcore_state,
)
from industrial_anomaly_lab.result_aggregation import load_completed_run
from industrial_anomaly_lab.run_artifacts import sha256_file, write_json


def _write_rgb(path: Path, value: int, shape: tuple[int, int] = (30, 42)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*shape, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, shape: tuple[int, int] = (30, 42)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:18, 9:25] = 255
    Image.fromarray(mask).save(path)


def _write_registration(path: Path, seeds: list[int] | None = None) -> None:
    path.write_text(
        json.dumps({"model_seeds": seeds or [11, 22, 33]}), encoding="utf-8"
    )


class _FakeNN:
    def __init__(self) -> None:
        self.search_index = None

    def fit(self, features: np.ndarray) -> None:
        self.search_index = faiss.IndexFlatL2(features.shape[1])
        self.search_index.add(features)

    def save(self, filename: str) -> None:
        faiss.write_index(self.search_index, filename)

    def load(self, filename: str) -> None:
        self.search_index = faiss.read_index(filename)


class _FakePatchCore:
    def __init__(self, map_size: int) -> None:
        self.map_size = map_size
        self.anomaly_scorer = SimpleNamespace(
            detection_features=None,
            nn_method=_FakeNN(),
        )
        self.fit_paths: list[str] = []

    def fit(self, loader) -> None:
        chunks: list[np.ndarray] = []
        for batch in loader:
            self.fit_paths.extend(batch["image_path"])
            images = batch["image"].numpy()
            means = images.mean(axis=(1, 2, 3)).astype(np.float32)
            chunks.append(np.stack([means, means + np.float32(1.0)], axis=1))
        features = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        self.anomaly_scorer.detection_features = features
        self.anomaly_scorer.nn_method.fit(features)

    def predict(self, images):
        batch = images.detach().cpu().numpy()
        scores = batch.mean(axis=(1, 2, 3)).astype(np.float32)
        maps = [
            np.full((self.map_size, self.map_size), score, dtype=np.float32)
            for score in scores
        ]
        return list(scores), maps


def _fake_builder(settings: PatchCoreFitSettings):
    return _FakePatchCore(settings.geometry.image_size), {
        "kind": "synthetic_test_double",
        "configuration": FROZEN_CONFIGURATION,
        "implicit_download_used": False,
    }


def _fake_checkpoint_inspector(path: Path, expected_sha256: str):
    actual = sha256_file(path)
    if expected_sha256 != actual:
        raise ValueError("synthetic checkpoint SHA mismatch")
    return {
        "checkpoint_path": str(path),
        "sha256": actual,
        "sha256_verified": True,
        "strict_load": {"passed": True},
        "implicit_download_used": False,
    }


def _fit_settings(
    tmp_path: Path,
    *,
    execution_scope: str = "debug_limited",
    max_steps: int | None = 1,
    seed: int = 11,
) -> PatchCoreFitSettings:
    data = tmp_path / "data"
    for index, value in enumerate((20, 80, 140)):
        _write_rgb(data / f"widget/train/good/{index:03d}.png", value)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    registration = tmp_path / "registration.json"
    _write_registration(registration)
    return PatchCoreFitSettings(
        dataset="mvtec_ad",
        category="widget",
        dataset_root=data,
        output_dir=tmp_path / "fit",
        backbone_checkpoint=checkpoint,
        backbone_sha256=sha256_file(checkpoint),
        seed=seed,
        execution_scope=execution_scope,  # type: ignore[arg-type]
        batch_size=2,
        workers=0,
        max_steps=max_steps,
        registration_path=registration,
        geometry=PatchCoreGeometry(resize=32, image_size=24, evaluation_size=32),
    )


def _fitted_context(
    checkpoint: Path,
    *,
    dataset: str = "mvtec_ad",
    category: str = "widget",
    seed: int = 11,
    execution_scope: str = "debug_limited",
) -> dict[str, object]:
    debug = execution_scope == "debug_limited"
    return {
        "dataset": dataset,
        "category": category,
        "seed": seed,
        "execution_scope": execution_scope,
        "run_purpose": "debug_limited" if debug else "author_reproduction",
        "claim_scope": "debug_only" if debug else "author_reproduction_only",
        "formal_matrix_eligible": False,
        "backbone_sha256": sha256_file(checkpoint),
        "train_inventory_sha256": "b" * 64,
        "train_image_count_total": 2,
        "train_image_count_selected": 1 if debug else 2,
        "max_steps": 1 if debug else None,
        "training_recipe": {"schema_version": 2, "method": "PatchCore"},
        "formal_provenance": None,
        "formal_training_inputs": None,
        "formal_source_provenance": None,
        "vendor_source_audit": audit_patchcore_runtime(),
    }


def test_full_fit_rejects_max_steps_nonstandard_geometry_and_unregistered_seed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="max_steps is forbidden"):
        _fit_settings(
            tmp_path / "max", execution_scope="full", max_steps=1
        ).validate()
    with pytest.raises(ValueError, match="frozen 256->224 geometry"):
        _fit_settings(
            tmp_path / "geometry", execution_scope="full", max_steps=None
        ).validate()

    settings = _fit_settings(
        tmp_path / "seed", execution_scope="full", max_steps=None, seed=44
    )
    object.__setattr__(settings, "geometry", PatchCoreGeometry())
    with pytest.raises(ValueError, match="not pre-registered"):
        settings.validate()


def test_validate_only_hashes_train_good_without_enumerating_test(tmp_path: Path) -> None:
    settings = _fit_settings(tmp_path)
    # If a test-manifest path were consulted, this deliberately malformed item
    # could not be interpreted as a normal test tree.
    (settings.dataset_root / "widget/test").write_text("must stay unread", encoding="utf-8")

    report = inspect_fit_inputs(
        settings,
        checkpoint_inspector=_fake_checkpoint_inspector,
        require_project_environment=False,
    )

    assert report["fit_started"] is False
    assert report["test_split_accessed"] is False
    assert report["input_inventory"]["normal_training_images"] == 3
    assert report["input_inventory"]["allowed_split"] == "train/good only"
    assert len(report["train_manifest_rows"]) == 3
    assert all("/train/good/" in f"/{row['relative_path']}" for row in report["train_manifest_rows"])


def test_safe_fitted_state_roundtrip_and_hash_tamper_rejection(tmp_path: Path) -> None:
    features = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    source = _FakePatchCore(map_size=24)
    source.anomaly_scorer.detection_features = features
    source.anomaly_scorer.nn_method.fit(features)
    model_dir = tmp_path / "model"
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    manifest = save_fitted_patchcore_state(
        source,
        model_dir,
        context=_fitted_context(
            checkpoint, dataset="visa", category="candle", seed=11
        ),
        backbone_checkpoint=checkpoint,
    )

    target = _FakePatchCore(map_size=24)
    report = load_fitted_patchcore_state(
        target,
        model_dir,
        backbone_checkpoint=checkpoint,
        expected_context={"dataset": "visa", "category": "candle", "seed": 11},
    )
    np.testing.assert_array_equal(target.anomaly_scorer.detection_features, features)
    assert manifest["schema_version"] == 2
    assert set(manifest["checkpoints"]) == {
        "memory_bank",
        "faiss_index",
        "backbone",
    }
    assert manifest["external_hashes"]["backbone"] == sha256_file(checkpoint)
    assert report["faiss"]["reconstruction_verified"] is True
    assert manifest["serialization"]["pickle_used"] is False
    assert not list(model_dir.glob("*.pkl"))

    with (model_dir / "memory_bank.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_fitted_patchcore_state(
            _FakePatchCore(24), model_dir, backbone_checkpoint=checkpoint
        )


def test_inference_scope_allows_full_fit_for_debug_but_rejects_debug_fit_for_full(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    registration = tmp_path / "registration.json"
    _write_registration(registration)
    features = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    def save_state(directory: Path, scope: str) -> None:
        source = _FakePatchCore(map_size=224)
        source.anomaly_scorer.detection_features = features
        source.anomaly_scorer.nn_method.fit(features)
        save_fitted_patchcore_state(
            source,
            directory,
            context=_fitted_context(
                checkpoint,
                execution_scope=scope,
            ),
            backbone_checkpoint=checkpoint,
        )

    full_model = tmp_path / "full-model"
    save_state(full_model, "full")
    debug_settings = PatchCoreInferenceSettings(
        dataset="mvtec_ad",
        category="widget",
        dataset_root=tmp_path / "data",
        fitted_model_dir=full_model,
        output_dir=tmp_path / "debug-output",
        backbone_checkpoint=checkpoint,
        backbone_sha256=sha256_file(checkpoint),
        seed=11,
        execution_scope="debug_limited",
        max_steps=1,
        registration_path=registration,
    )
    _model, report = inspect_inference_inputs(
        debug_settings,
        model_builder=_fake_builder,
        require_project_environment=False,
    )
    assert report["fitted_state"]["context"]["execution_scope"] == "full"

    debug_model = tmp_path / "debug-model"
    save_state(debug_model, "debug_limited")
    full_settings = PatchCoreInferenceSettings(
        dataset="mvtec_ad",
        category="widget",
        dataset_root=tmp_path / "data",
        fitted_model_dir=debug_model,
        output_dir=tmp_path / "full-output",
        backbone_checkpoint=checkpoint,
        backbone_sha256=sha256_file(checkpoint),
        seed=11,
        execution_scope="full",
        max_steps=None,
        registration_path=registration,
    )
    with pytest.raises(ValueError, match="debug_limited fitted state"):
        inspect_inference_inputs(
            full_settings,
            model_builder=_fake_builder,
            require_project_environment=False,
        )


def test_fit_then_independent_debug_inference_uses_safe_state_and_export_contract(
    tmp_path: Path,
) -> None:
    fit_settings = _fit_settings(tmp_path)
    fit_state = run_patchcore_fit(
        fit_settings,
        model_builder=_fake_builder,
        checkpoint_inspector=_fake_checkpoint_inspector,
        require_project_environment=False,
    )
    assert fit_state["status"] == "debug_completed"
    assert fit_state["train_images_selected"] == 2
    assert fit_state["test_split_accessed"] is False
    assert (fit_settings.output_dir / "model/memory_bank.npy").is_file()
    assert (fit_settings.output_dir / "model/nnscorer_search_index.faiss").is_file()
    assert not list(fit_settings.output_dir.rglob("*.pkl"))
    events = [
        json.loads(line)
        for line in (fit_settings.output_dir / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["event"] for row in events] == [
        "fit_started",
        "memory_bank_fit_started",
        "fit_completed",
    ]

    _write_rgb(fit_settings.dataset_root / "widget/test/bad/100.png", 220)
    _write_mask(fit_settings.dataset_root / "widget/ground_truth/bad/100_mask.png")
    _write_rgb(fit_settings.dataset_root / "widget/test/good/101.png", 30)
    inference = PatchCoreInferenceSettings(
        dataset="mvtec_ad",
        category="widget",
        dataset_root=fit_settings.dataset_root,
        fitted_model_dir=fit_settings.output_dir / "model",
        output_dir=tmp_path / "inference",
        backbone_checkpoint=fit_settings.backbone_checkpoint,
        backbone_sha256=fit_settings.backbone_sha256,
        seed=fit_settings.seed,
        execution_scope="debug_limited",
        batch_size=1,
        workers=0,
        max_steps=1,
        registration_path=fit_settings.registration_path,
        geometry=fit_settings.geometry,
    )
    _model, validation = inspect_inference_inputs(
        inference,
        model_builder=_fake_builder,
        require_project_environment=False,
    )
    assert validation["test_manifest_enumerated"] is False
    assert validation["fitted_state"]["pickle_loaded"] is False

    result = run_patchcore_inference(
        inference,
        model_builder=_fake_builder,
        require_project_environment=False,
    )
    assert result["status"] == "completed"
    assert result["num_images"] == 1
    assert result["metrics"] == {}
    config = json.loads(
        (inference.output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (inference.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (inference.output_dir / "predictions/index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    raw_map = np.load(inference.output_dir / rows[0]["raw_map"], allow_pickle=False)
    assert config["execution_scope"] == "debug_limited"
    assert config["prediction_contract"]["raw_map_normalization"] == "none"
    assert config["prediction_contract"]["complete_category"] is False
    assert config["settings"]["model_provenance"]["pickle_loaded"] is False
    assert metrics["metrics_status"] == "unavailable_for_partial_debug_selection"
    assert raw_map.dtype == np.float32
    assert raw_map.shape == (32, 32)
    with pytest.raises(ValueError, match="not completed|debug-limited"):
        load_completed_run(inference.output_dir)


def test_fitted_state_rejects_old_schema_and_path_escape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    features = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    def fitted(directory: Path) -> None:
        source = _FakePatchCore(map_size=24)
        source.anomaly_scorer.detection_features = features
        source.anomaly_scorer.nn_method.fit(features)
        save_fitted_patchcore_state(
            source,
            directory,
            context=_fitted_context(checkpoint),
            backbone_checkpoint=checkpoint,
        )

    old = tmp_path / "old"
    fitted(old)
    old_manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    old_manifest["schema_version"] = 1
    write_json(old / "manifest.json", old_manifest)
    with pytest.raises(ValueError, match="unsupported PatchCore fitted-state manifest"):
        load_fitted_patchcore_state(
            _FakePatchCore(24), old, backbone_checkpoint=checkpoint
        )

    escaped = tmp_path / "escaped"
    fitted(escaped)
    escaped_manifest = json.loads(
        (escaped / "manifest.json").read_text(encoding="utf-8")
    )
    escaped_manifest["memory_bank"]["path"] = "../outside.npy"
    write_json(escaped / "manifest.json", escaped_manifest)
    with pytest.raises(ValueError, match="path is unsafe"):
        load_fitted_patchcore_state(
            _FakePatchCore(24), escaped, backbone_checkpoint=checkpoint
        )


def test_formal_fit_uses_only_manifest_resolved_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    selected = data / "widget/train/good/selected.png"
    unregistered = data / "widget/train/good/unregistered.png"
    _write_rgb(selected, 20)
    _write_rgb(unregistered, 80)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    registration = tmp_path / "registration.json"
    _write_registration(registration)
    project = tmp_path / "configs/project.json"
    project.parent.mkdir(parents=True)
    project.write_text("{}", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "contract": "formal_training_input_bundle_v1",
        "method": "PatchCore",
        "dataset": "mvtec_ad",
        "category": "widget",
    }
    monkeypatch.setattr(
        patchcore_runner_module,
        "resolve_frozen_formal_training_inputs",
        lambda **_kwargs: SimpleNamespace(
            normal_image_paths=(selected,), anomaly_source_paths=(), evidence=evidence
        ),
    )
    monkeypatch.setattr(
        patchcore_runner_module,
        "validate_formal_authority",
        lambda **_kwargs: {"run_purpose": "formal_full"},
    )
    monkeypatch.setattr(
        patchcore_runner_module,
        "capture_formal_source_provenance",
        lambda **_kwargs: {"contract": "synthetic_clean_source"},
    )
    settings = PatchCoreFitSettings(
        dataset="mvtec_ad",
        category="widget",
        dataset_root=data,
        output_dir=tmp_path / "fit",
        backbone_checkpoint=checkpoint,
        backbone_sha256=sha256_file(checkpoint),
        seed=11,
        execution_scope="full",
        run_purpose="formal_full",
        max_steps=None,
        registration_path=registration,
        project_config_path=project,
    )

    report = inspect_fit_inputs(
        settings,
        checkpoint_inspector=_fake_checkpoint_inspector,
        require_project_environment=False,
    )

    assert report["input_inventory"]["normal_training_images"] == 1
    assert report["train_manifest_rows"][0]["relative_path"].endswith(
        "selected.png"
    )
    assert all(
        "unregistered.png" not in row["relative_path"]
        for row in report["train_manifest_rows"]
    )
    assert report["formal_matrix_eligible"] is True


def test_formal_fit_fails_closed_while_registry_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    selected = data / "widget/train/good/selected.png"
    _write_rgb(selected, 20)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    registration = tmp_path / "configs/registration.json"
    registration.parent.mkdir(parents=True)
    _write_registration(registration)
    project = tmp_path / "configs/project.json"
    project.write_text(
        json.dumps(
            {
                "evaluation": {
                    "formal_protocol_frozen": False,
                    "minimum_seeds": 3,
                    "seeds": [11, 22, 33],
                },
                "datasets": {
                    "mvtec_ad": {"categories": ["widget"]},
                    "visa": {"categories": ["candle"]},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        patchcore_runner_module,
        "resolve_frozen_formal_training_inputs",
        lambda **_kwargs: SimpleNamespace(
            normal_image_paths=(selected,),
            anomaly_source_paths=(),
            evidence={
                "schema_version": 1,
                "contract": "formal_training_input_bundle_v1",
                "method": "PatchCore",
                "dataset": "mvtec_ad",
                "category": "widget",
            },
        ),
    )
    settings = PatchCoreFitSettings(
        dataset="mvtec_ad",
        category="widget",
        dataset_root=data,
        output_dir=tmp_path / "fit",
        backbone_checkpoint=checkpoint,
        backbone_sha256=sha256_file(checkpoint),
        seed=11,
        execution_scope="full",
        run_purpose="formal_full",
        max_steps=None,
        registration_path=registration,
        project_config_path=project,
    )
    with pytest.raises(ValueError, match="formal evaluation protocol is not frozen"):
        inspect_fit_inputs(
            settings,
            checkpoint_inspector=_fake_checkpoint_inspector,
            require_project_environment=False,
        )
    assert not settings.output_dir.exists()


def test_failed_fit_removes_final_fitted_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _fit_settings(tmp_path)

    def fail_after_writing(_model, model_dir, **_kwargs):
        directory = Path(model_dir)
        directory.mkdir(parents=True)
        (directory / "memory_bank.npy").write_bytes(b"partial")
        (directory / "nnscorer_search_index.faiss").write_bytes(b"partial")
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("synthetic post-fit persistence failure")

    monkeypatch.setattr(
        patchcore_runner_module,
        "save_fitted_patchcore_state",
        fail_after_writing,
    )
    with pytest.raises(RuntimeError, match="synthetic post-fit"):
        run_patchcore_fit(
            settings,
            model_builder=_fake_builder,
            checkpoint_inspector=_fake_checkpoint_inspector,
            require_project_environment=False,
        )
    assert not (settings.output_dir / "model").exists()
    state = json.loads(
        (settings.output_dir / "run_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "failed"
    assert sorted(state["removed_final_fitted_artifacts"]) == [
        "manifest.json",
        "memory_bank.npy",
        "nnscorer_search_index.faiss",
    ]
