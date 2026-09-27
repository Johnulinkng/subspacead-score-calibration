from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import industrial_anomaly_lab.development_protocol as development_protocol
from industrial_anomaly_lab.development_protocol import build_protocol_artifacts
from industrial_anomaly_lab.draem_train import TrainSettings, inspect_training_inputs
from industrial_anomaly_lab.synthetic_development import (
    resolve_synthetic_development_inputs,
)
from industrial_anomaly_lab.training_inputs import (
    VISA_DEVELOPMENT_PROTOCOL,
    _selection_hash,
    resolve_manifest_training_inputs,
)


REGISTERED_SEEDS = [1887537248, 1337854654, 1609762728]
SEED_SHA256 = "f08188604fbe0abedff307a881d63b5c32a99bc27c1fdf5a294b9098cbab4055"


def _write_sources(directory: Path, count: int, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:03d}.JPG").write_bytes(
            f"{prefix}-original-{index}".encode("ascii")
        )


def _prepare_frozen_protocol(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    visa_root = tmp_path / "data/visa"
    dtd_root = tmp_path / "data/dtd"
    for category in ("a", "b"):
        _write_sources(visa_root / category / "train/good", 10, f"visa-{category}")
        sentinel = visa_root / category / "test/bad/SENTINEL.JPG"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"formal-test-content-must-not-be-read")
    for texture in ("banded", "woven"):
        _write_sources(dtd_root / texture, 10, f"dtd-{texture}")

    config_path = tmp_path / "configs/development_protocol.json"
    config_path.parent.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "protocol_id": VISA_DEVELOPMENT_PROTOCOL,
        "status": "frozen_before_model_development",
        "model_seeds": REGISTERED_SEEDS,
        "model_seed_registration": {
            "derivation_string": "industrial-anomaly-lab|formal-v1|2026-09-27",
            "sha256": SEED_SHA256,
            "hex_chunk_offsets": [0, 8, 16],
            "derived_seeds": REGISTERED_SEEDS,
        },
        "visa": {
            "root": "data/visa",
            "categories": ["a", "b"],
            "dev_fraction": 0.2,
            "split_salt": "visa-test-salt",
        },
        "dtd": {
            "root": "data/dtd",
            "dev_fraction": 0.2,
            "split_salt": "dtd-test-salt",
        },
        "synthetic_development": {
            "renderer_id": "draem_synthetic_dev_renderer_v2",
            "normal_examples_per_dev_image": 1,
            "recipe_salt": "recipe-test-salt",
            "positive_variants_per_dev_image": 2,
            "image_size": 256,
            "augmenters_per_positive": 3,
            "perlin_scales": [1, 2, 4, 8, 16, 32],
            "perlin_threshold": 0.5,
            "blend_beta_range": [0.0, 0.8],
            "normal_false_positive_threshold": 0.5,
            "target_rotation_probability": 0.3,
            "recipe_schema_version": 3,
            "render_policy": "test frozen renderer policy",
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_root = tmp_path / "manifests/development/visa_v1"
    build_protocol_artifacts(tmp_path, config_path, manifest_root)
    manifest_path = manifest_root / "manifest.json"
    config["frozen_manifest"] = {
        "path": manifest_path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, manifest_root, visa_root, dtd_root


def _resign_category_bundle(
    tmp_path: Path,
    config_path: Path,
    manifest_root: Path,
    mutate,
) -> None:
    category_path = manifest_root / "visa_a_split.jsonl"
    rows = [json.loads(line) for line in category_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[0])
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    category_path.write_bytes(payload)

    summary_path = manifest_root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in summary["artifact_files"]
        if Path(item["path"]).name == category_path.name
    )
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["frozen_manifest"]["sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")


def _resolve(
    config_path: Path, visa_root: Path, dtd_root: Path, seed: int = REGISTERED_SEEDS[0]
):
    return resolve_manifest_training_inputs(
        dataset="visa",
        category="a",
        dataset_root=visa_root,
        anomaly_source_root=dtd_root,
        seed=seed,
        protocol_config_path=config_path,
    )


def test_validate_only_uses_only_frozen_train_splits(tmp_path: Path) -> None:
    config_path, _, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    settings = TrainSettings(
        dataset="visa",
        category="a",
        dataset_root=visa_root,
        anomaly_source_root=dtd_root,
        output_dir=tmp_path / "must_not_be_created",
        seed=REGISTERED_SEEDS[0],
        input_protocol=VISA_DEVELOPMENT_PROTOCOL,
        development_protocol_config=config_path,
    )

    result = inspect_training_inputs(settings)

    assert result["training_started"] is False
    assert result["normal_training_images"] == 8
    assert result["anomaly_source_images"] == 16
    assert result["input_evidence"]["selected_file_size_and_sha256_verified"] is True
    assert result["input_evidence"]["development_protocol_eligible"] is True
    assert result["input_evidence"]["formal_test_content_accessed"] is False
    assert result["input_evidence"]["renderer_provenance_sha256"] == _selection_hash(
        [result["input_evidence"]["renderer_provenance"]]
    )
    assert not settings.output_dir.exists()
    resolved = _resolve(config_path, visa_root, dtd_root)
    assert all("/train/good/" in path.as_posix() for path in resolved.normal_image_paths)
    assert all("/test/" not in path.as_posix() for path in resolved.normal_image_paths)
    assert len(resolved.anomaly_source_paths) == 16


@pytest.mark.parametrize(
    "drift_relative",
    [
        "src/industrial_anomaly_lab/training_inputs.py",
        "src/industrial_anomaly_lab/draem_runtime.py",
        "src/industrial_anomaly_lab/draem_infer.py",
        "src/industrial_anomaly_lab/draem_train.py",
        "src/industrial_anomaly_lab/run_artifacts.py",
        "scripts/train_draem.py",
        "scripts/evaluate_draem_synthetic_dev.py",
    ],
)
def test_registered_runtime_file_drift_rejects_both_manifest_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift_relative: str
) -> None:
    assert drift_relative in development_protocol.RENDERER_PROVENANCE_FILES
    config_path, _, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    project_root = Path(development_protocol.__file__).resolve().parents[2]
    drift_target = (project_root / drift_relative).resolve()
    original_sha256_file = development_protocol.sha256_file

    def drifted_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = original_sha256_file(path, chunk_size)
        if Path(path).resolve() == drift_target:
            return "0" * 64 if digest != "0" * 64 else "1" * 64
        return digest

    monkeypatch.setattr(development_protocol, "sha256_file", drifted_sha256)

    with pytest.raises(ValueError, match="renderer provenance"):
        _resolve(config_path, visa_root, dtd_root)
    with pytest.raises(ValueError, match="renderer provenance"):
        resolve_synthetic_development_inputs(
            category="a",
            dataset_root=visa_root,
            anomaly_source_root=dtd_root,
            seed=REGISTERED_SEEDS[0],
            protocol_config_path=config_path,
        )


def test_default_author_input_mode_remains_cli_compatible(tmp_path: Path) -> None:
    _, _, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    settings = TrainSettings(
        dataset="visa",
        category="a",
        dataset_root=visa_root,
        anomaly_source_root=dtd_root,
        output_dir=tmp_path / "must_not_be_created",
    )

    result = inspect_training_inputs(settings)

    assert result["input_protocol"] == "author_full_train"
    assert result["normal_training_images"] == 10
    assert result["anomaly_source_images"] == 20
    assert result["input_evidence"]["manifest_backed"] is False
    assert result["input_evidence"]["development_protocol_eligible"] is False
    assert not settings.output_dir.exists()


def test_root_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    config_path, manifest_root, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    with (manifest_root / "manifest.json").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="root manifest SHA-256 mismatch"):
        _resolve(config_path, visa_root, dtd_root)


def test_child_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    config_path, manifest_root, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    with (manifest_root / "visa_a_split.jsonl").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        _resolve(config_path, visa_root, dtd_root)


def test_resigned_root_with_basename_only_artifact_redirect_fails_closed(
    tmp_path: Path,
) -> None:
    config_path, manifest_root, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    manifest_path = manifest_root / "manifest.json"
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary["artifact_files"][0]["path"] = (
        "unregistered/location/" + Path(summary["artifact_files"][0]["path"]).name
    )
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["frozen_manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact path mismatch"):
        _resolve(config_path, visa_root, dtd_root)


def test_resigned_train_dev_swap_fails_deterministic_split_derivation(
    tmp_path: Path,
) -> None:
    config_path, manifest_root, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    category_path = manifest_root / "visa_a_split.jsonl"
    rows = [json.loads(line) for line in category_path.read_text().splitlines()]
    train_row = next(row for row in rows if row["split"] == "train")
    dev_row = next(row for row in rows if row["split"] == "dev")
    train_row["split"], dev_row["split"] = dev_row["split"], train_row["split"]
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    category_path.write_bytes(payload)
    manifest_path = manifest_root / "manifest.json"
    summary = json.loads(manifest_path.read_text())
    next(
        item for item in summary["artifact_files"] if Path(item["path"]).name == category_path.name
    )["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config = json.loads(config_path.read_text())
    config["frozen_manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="source split does not exactly match"):
        _resolve(config_path, visa_root, dtd_root)


def test_selected_source_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    config_path, _, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    resolved = _resolve(config_path, visa_root, dtd_root)
    target = resolved.normal_image_paths[0]
    original = target.read_bytes()
    target.write_bytes(b"X" * len(original))

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        _resolve(config_path, visa_root, dtd_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset", "mvtec_ad", "dataset must be visa"),
        ("category", "b", "category mismatch"),
        ("split", "training", "split must be train or dev"),
        ("relative_path", "a/test/good/000.JPG", "outside a/train/good"),
    ],
)
def test_resigned_invalid_manifest_fields_still_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    config_path, manifest_root, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)
    _resign_category_bundle(
        tmp_path,
        config_path,
        manifest_root,
        lambda row: row.__setitem__(field, value),
    )

    with pytest.raises(ValueError, match=message):
        _resolve(config_path, visa_root, dtd_root)


def test_unregistered_seed_fails_before_dataset_construction(tmp_path: Path) -> None:
    config_path, _, visa_root, dtd_root = _prepare_frozen_protocol(tmp_path)

    with pytest.raises(ValueError, match="not one of the three registered"):
        _resolve(config_path, visa_root, dtd_root, seed=42)
