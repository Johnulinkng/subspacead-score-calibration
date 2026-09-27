from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_anomaly_lab.development_protocol import build_protocol_artifacts


def _write_fake_images(directory: Path, count: int, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:03d}.JPG").write_bytes(
            f"{prefix}-original-{index}".encode("ascii")
        )


def _config(path: Path) -> None:
    value = {
        "schema_version": 1,
        "protocol_id": "test_protocol_v1",
        "status": "frozen_before_model_development",
        "model_seeds": [1887537248, 1337854654, 1609762728],
        "model_seed_registration": {
            "derivation_string": "industrial-anomaly-lab|formal-v1|2026-09-27",
            "sha256": "f08188604fbe0abedff307a881d63b5c32a99bc27c1fdf5a294b9098cbab4055",
            "hex_chunk_offsets": [0, 8, 16],
            "derived_seeds": [1887537248, 1337854654, 1609762728],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_manifests_are_deterministic_content_disjoint_and_test_blind(tmp_path: Path) -> None:
    for category in ("a", "b"):
        _write_fake_images(tmp_path / f"data/visa/{category}/train/good", 10, category)
        # A sentinel under formal test must never enter inventory or recipes.
        (tmp_path / f"data/visa/{category}/test/bad").mkdir(parents=True)
        (tmp_path / f"data/visa/{category}/test/bad/SENTINEL.JPG").write_bytes(
            b"formal-test-content"
        )
    for texture in ("banded", "woven"):
        _write_fake_images(tmp_path / f"data/dtd/{texture}", 10, texture)
    config = tmp_path / "configs/development_protocol.json"
    _config(config)

    first = build_protocol_artifacts(tmp_path, config, tmp_path / "out/first")
    second = build_protocol_artifacts(tmp_path, config, tmp_path / "out/second")

    assert first["formal_test_content_accessed"] is False
    assert first["training_started"] is False
    assert first["model_seeds"] == [1887537248, 1337854654, 1609762728]
    assert first["visa"]["train"] == 16
    assert first["visa"]["dev"] == 4
    assert first["dtd"]["train"] == 16
    assert first["dtd"]["dev"] == 4
    assert first["synthetic_development"]["synthetic_positive_recipes"] == 8
    for name in (
        "visa_a_split.jsonl",
        "visa_b_split.jsonl",
        "dtd_source_split.jsonl",
        "synthetic_dev_recipes.jsonl",
    ):
        assert (tmp_path / "out/first" / name).read_bytes() == (
            tmp_path / "out/second" / name
        ).read_bytes()
        assert b"SENTINEL" not in (tmp_path / "out/first" / name).read_bytes()


def test_exact_duplicates_stay_on_the_same_side(tmp_path: Path) -> None:
    _write_fake_images(tmp_path / "data/visa/a/train/good", 10, "a")
    _write_fake_images(tmp_path / "data/visa/b/train/good", 10, "b")
    duplicate = tmp_path / "data/visa/a/train/good/009.JPG"
    duplicate.write_bytes((tmp_path / "data/visa/a/train/good/000.JPG").read_bytes())
    for texture in ("banded", "woven"):
        _write_fake_images(tmp_path / f"data/dtd/{texture}", 10, texture)
    config = tmp_path / "configs/development_protocol.json"
    _config(config)

    build_protocol_artifacts(tmp_path, config, tmp_path / "out")
    rows = _read_jsonl(tmp_path / "out/visa_a_split.jsonl")
    by_name = {Path(str(row["relative_path"])).name: row for row in rows}

    assert by_name["000.JPG"]["sha256"] == by_name["009.JPG"]["sha256"]
    assert by_name["000.JPG"]["split"] == by_name["009.JPG"]["split"]
    train_hashes = {row["sha256"] for row in rows if row["split"] == "train"}
    dev_hashes = {row["sha256"] for row in rows if row["split"] == "dev"}
    assert train_hashes.isdisjoint(dev_hashes)


def test_hand_selected_seed_edit_is_rejected_before_inventory(tmp_path: Path) -> None:
    config = tmp_path / "configs/development_protocol.json"
    _config(config)
    value = json.loads(config.read_text(encoding="utf-8"))
    value["model_seeds"] = [1, 2, 3]
    config.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="registered derivation"):
        build_protocol_artifacts(tmp_path, config, tmp_path / "out")

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad_root", ["../outside", "C:/outside", "data\\visa"])
def test_source_root_escape_is_rejected_before_inventory(
    tmp_path: Path, bad_root: str
) -> None:
    config = tmp_path / "configs/development_protocol.json"
    _config(config)
    value = json.loads(config.read_text(encoding="utf-8"))
    value["visa"]["root"] = bad_root
    config.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="POSIX relative|remain inside (the )?project"):
        build_protocol_artifacts(tmp_path, config, tmp_path / "out")

    assert not (tmp_path / "out").exists()
