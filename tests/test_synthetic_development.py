from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
import cv2
import imgaug.augmenters as iaa
from PIL import Image

from industrial_anomaly_lab.development_protocol import (
    _perlin_noise,
    build_protocol_artifacts,
    derive_expected_recipe,
    render_perlin_mask,
)
from industrial_anomaly_lab.draem_runtime import DEFAULT_VENDOR_ROOT, _load_vendor_file
from industrial_anomaly_lab.draem_synthetic_dev import (
    SyntheticDevEvaluationSettings,
    SyntheticDevSettings,
    inspect_synthetic_development,
    run_render_integrity_audit,
    run_synthetic_development_evaluation,
)
from industrial_anomaly_lab.synthetic_development import (
    SyntheticDevelopmentDataset,
    _decode_verified,
    blend_synthetic_anomaly,
    render_synthetic_recipe,
    resolve_synthetic_development_inputs,
)
from industrial_anomaly_lab.training_inputs import _selection_hash


REGISTERED_SEEDS = [1887537248, 1337854654, 1609762728]
SEED_SHA256 = "f08188604fbe0abedff307a881d63b5c32a99bc27c1fdf5a294b9098cbab4055"


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


def _write_image(path: Path, value: int, marker: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:32, :32]
    image = np.stack(
        (
            (value + xx * (marker + 1)) % 256,
            (value + yy * (marker + 2)) % 256,
            (value + xx + yy + marker * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


def _prepare_frozen_protocol(
    tmp_path: Path, *, target_rotation_probability: float = 0.3
) -> tuple[Path, Path, Path, Path]:
    visa_root = tmp_path / "data/visa"
    dtd_root = tmp_path / "data/dtd"
    for category_index, category in enumerate(("a", "b")):
        for index in range(10):
            _write_image(
                visa_root / category / "train/good" / f"{index:03d}.png",
                20 + category_index * 50,
                index,
            )
        sentinel = visa_root / category / "test/bad/SENTINEL.png"
        _write_image(sentinel, 250, 99)
    for texture_index, texture in enumerate(("banded", "woven")):
        for index in range(10):
            _write_image(
                dtd_root / texture / f"{index:03d}.png",
                80 + texture_index * 70,
                index + 20,
            )
    config_path = tmp_path / "configs/development_protocol.json"
    config_path.parent.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "protocol_id": "visa_train_dev_dtd_v1",
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
            "image_size": 32,
            "augmenters_per_positive": 3,
            "perlin_scales": [1, 2, 4, 8, 16],
            "perlin_threshold": 0.5,
            "blend_beta_range": [0.0, 0.8],
            "normal_false_positive_threshold": 0.5,
            "target_rotation_probability": target_rotation_probability,
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


def _basic_settings(config: Path, visa: Path, dtd: Path) -> SyntheticDevSettings:
    return SyntheticDevSettings(
        category="a",
        dataset_root=visa,
        anomaly_source_root=dtd,
        protocol_config_path=config,
        seed=REGISTERED_SEEDS[0],
    )


def _resign_recipes(config_path: Path, manifest_root: Path, mutate) -> None:
    recipe_path = manifest_root / "synthetic_dev_recipes.jsonl"
    rows = [json.loads(line) for line in recipe_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[0])
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    recipe_path.write_bytes(payload)
    manifest_path = manifest_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["artifact_files"] if Path(item["path"]).name == recipe_path.name
    )
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["frozen_manifest"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_resolver_and_renderer_are_manifest_backed_and_deterministic(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    inputs = resolve_synthetic_development_inputs(
        category="a",
        dataset_root=visa,
        anomaly_source_root=dtd,
        seed=REGISTERED_SEEDS[0],
        protocol_config_path=config,
    )
    first_image, first_mask = render_synthetic_recipe(inputs.recipes[0])
    second_image, second_mask = render_synthetic_recipe(inputs.recipes[0])
    dataset = SyntheticDevelopmentDataset(inputs)

    assert len(inputs.normal_sources) == 2
    assert len(inputs.recipes) == 4
    assert len(dataset) == 6
    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_mask, second_mask)
    assert first_image.dtype == np.float32
    assert first_mask.dtype == np.bool_
    assert first_mask.any()
    normal = dataset[0]
    assert int(normal["label"]) == 0
    assert not bool(normal["mask"].any())
    assert inputs.evidence["selected_file_size_and_sha256_verified"] is True
    assert inputs.evidence["formal_test_content_accessed"] is False
    assert "test" not in " ".join(source.relative_path for source in inputs.normal_sources)


def test_selected_dev_source_tamper_fails_closed(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    inputs = resolve_synthetic_development_inputs(
        category="a",
        dataset_root=visa,
        anomaly_source_root=dtd,
        seed=REGISTERED_SEEDS[0],
        protocol_config_path=config,
    )
    target = inputs.normal_sources[0].path
    payload = target.read_bytes()
    target.write_bytes(b"X" * len(payload))

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_selected_dev_leaf_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    inputs = resolve_synthetic_development_inputs(
        category="a",
        dataset_root=visa,
        anomaly_source_root=dtd,
        seed=REGISTERED_SEEDS[0],
        protocol_config_path=config,
    )
    target = inputs.normal_sources[0].path
    backing = target.with_suffix(target.suffix + ".real")
    target.replace(backing)
    try:
        target.symlink_to(backing)
    except OSError as error:
        backing.replace(target)
        pytest.skip(f"file symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="linked/reparse component"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_resigned_recipe_with_wrong_seed_fails_closed(tmp_path: Path) -> None:
    config, manifest_root, visa, dtd = _prepare_frozen_protocol(tmp_path)
    _resign_recipes(
        config,
        manifest_root,
        lambda row: row.__setitem__("perlin_seed", (row["perlin_seed"] + 1) % 2**32),
    )

    with pytest.raises(ValueError, match="does not exactly match deterministic derivation"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_resigned_numeric_type_drift_fails_strict_canonical_derivation(
    tmp_path: Path,
) -> None:
    config, manifest_root, visa, dtd = _prepare_frozen_protocol(tmp_path)
    _resign_recipes(
        config,
        manifest_root,
        lambda row: row.__setitem__("image_size", float(row["image_size"])),
    )
    with pytest.raises(ValueError, match="does not exactly match deterministic derivation"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("normal_examples_per_dev_image", 2, "exactly one normal"),
        ("positive_variants_per_dev_image", "2", "positive integer"),
        ("perlin_threshold", "0.5", "float in"),
    ],
)
def test_synthetic_config_semantic_or_type_drift_fails_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    payload = json.loads(config.read_text())
    payload["synthetic_development"][field] = value
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_draft_protocol_status_is_rejected(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    payload = json.loads(config.read_text())
    payload["status"] = "draft"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not frozen_before_model_development"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_resigned_recipe_cannot_swap_to_another_valid_dtd_dev_member(
    tmp_path: Path,
) -> None:
    config, manifest_root, visa, dtd = _prepare_frozen_protocol(tmp_path)
    dtd_rows = [
        json.loads(line)
        for line in (manifest_root / "dtd_source_split.jsonl").read_text().splitlines()
    ]
    dev_rows = [row for row in dtd_rows if row["split"] == "dev"]

    def swap_source(row: dict[str, object]) -> None:
        replacement = next(
            item for item in dev_rows if item["relative_path"] != row["dtd_relative_path"]
        )
        row["dtd_relative_path"] = replacement["relative_path"]
        row["dtd_sha256"] = replacement["sha256"]

    _resign_recipes(config, manifest_root, swap_source)
    with pytest.raises(ValueError, match="does not exactly match deterministic derivation"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_global_recipe_tamper_is_rejected_when_resolving_another_category(
    tmp_path: Path,
) -> None:
    config, manifest_root, visa, dtd = _prepare_frozen_protocol(tmp_path)

    def mutate_last(rows: list[dict[str, object]]) -> None:
        assert rows[-1]["category"] == "b"
        rows[-1]["blend_beta"] = 0.12345678

    recipe_path = manifest_root / "synthetic_dev_recipes.jsonl"
    rows = [json.loads(line) for line in recipe_path.read_text().splitlines()]
    mutate_last(rows)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    recipe_path.write_bytes(payload)
    manifest_path = manifest_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    next(
        item for item in manifest["artifact_files"] if Path(item["path"]).name == recipe_path.name
    )["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config_payload = json.loads(config.read_text())
    config_payload["frozen_manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    config.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match deterministic derivation"):
        inspect_synthetic_development(_basic_settings(config, visa, dtd))


def test_local_perlin_rng_matches_pinned_draem_vendor_formula() -> None:
    vendor = _load_vendor_file("test_perlin_parity", "perlin.py", DEFAULT_VENDOR_ROOT)
    original_state = np.random.get_state()
    try:
        np.random.seed(123456789)
        expected = vendor.rand_perlin_2d_np((32, 32), (4, 8))
    finally:
        np.random.set_state(original_state)
    actual = _perlin_noise((32, 32), (4, 8), seed=123456789)
    assert np.array_equal(actual, expected)


def test_rotated_perlin_mask_matches_vendor_formula_and_fixed_affine() -> None:
    vendor = _load_vendor_file("test_rotated_perlin_parity", "perlin.py", DEFAULT_VENDOR_ROOT)
    original_state = np.random.get_state()
    try:
        np.random.seed(99887766)
        perlin = vendor.rand_perlin_2d_np((32, 32), (4, 8))
    finally:
        np.random.set_state(original_state)
    expected = iaa.Affine(rotate=37.125, order=1, cval=0, mode="constant")(
        image=perlin
    ) > 0.5
    actual = render_perlin_mask(
        image_size=32,
        scale_x=4,
        scale_y=8,
        rotation_degrees=37.125,
        threshold=0.5,
        seed=99887766,
    )
    assert np.array_equal(actual, expected)


def test_target_rotation_is_applied_before_normalization(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(
        tmp_path, target_rotation_probability=1.0
    )
    inputs = resolve_synthetic_development_inputs(
        category="a",
        dataset_root=visa,
        anomaly_source_root=dtd,
        seed=REGISTERED_SEEDS[0],
        protocol_config_path=config,
    )
    recipe = inputs.recipes[0]
    assert recipe.row["target_rotation_applied"] is True
    rendered, mask = render_synthetic_recipe(recipe)
    target = cv2.resize(
        _decode_verified(recipe.target), (32, 32), interpolation=cv2.INTER_LINEAR
    )
    expected_target = iaa.Affine(
        rotate=float(recipe.row["target_rotation_degrees"]),
        order=1,
        cval=0,
        mode="constant",
    )(image=target).astype(np.float32) / 255.0
    assert np.array_equal(rendered[~mask], expected_target[~mask])


def test_blend_helper_matches_frozen_renderer_v2_formula_at_float32_boundary() -> None:
    target = np.linspace(0.0, 1.0, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    mask_2d = np.asarray(
        [
            [False, True, False, True, False],
            [True, True, False, False, True],
            [False, False, True, True, False],
            [True, False, True, False, True],
        ]
    )
    beta = 0.31415927
    mask = mask_2d.astype(np.float32)[:, :, None]
    source_region = source.astype(np.float32) * mask / 255.0
    expected = (
        mask
        * (
            target * (1.0 - mask)
            + (1.0 - beta) * source_region
            + beta * target * mask
        )
        + (1.0 - mask) * target
    ).astype(np.float32)
    actual = blend_synthetic_anomaly(target, source, mask_2d, beta=beta)
    assert np.array_equal(actual, expected)


def test_recipe_derivation_has_a_golden_canonical_hash() -> None:
    target = {
        "dataset": "visa",
        "category": "a",
        "relative_path": "a/train/good/007.png",
        "sha256": "1" * 64,
        "split": "dev",
    }
    dtd = [
        {
            "dataset": "dtd",
            "category": "banded",
            "relative_path": "banded/001.png",
            "sha256": "2" * 64,
            "split": "dev",
        },
        {
            "dataset": "dtd",
            "category": "woven",
            "relative_path": "woven/002.png",
            "sha256": "3" * 64,
            "split": "dev",
        },
    ]
    recipe = derive_expected_recipe(
        target=target,
        variant=1,
        ordered_dtd_dev_records=dtd,
        protocol_id="visa_train_dev_dtd_v1",
        recipe_salt="recipe-test-salt",
        image_size=32,
        augmenters_per_positive=3,
        perlin_scales=[1, 2, 4, 8, 16],
        perlin_threshold=0.5,
        blend_beta_range=(0.0, 0.8),
        target_rotation_probability=0.3,
        recipe_schema_version=3,
    )
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(canonical).hexdigest()
        == "95c514ed5c83c73c40d9ee89d8ece20c2b85984b6c350a7b46f5e8d760958241"
    )


def test_render_audit_saves_float32_inputs_bool_masks_and_no_metrics(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    output = tmp_path / "audit"
    result = run_render_integrity_audit(
        _basic_settings(config, visa, dtd), output_dir=output, count=2
    )
    rows = [json.loads(line) for line in (output / "index.jsonl").read_text().splitlines()]
    rendered = np.load(output / rows[0]["rendered_input"], allow_pickle=False)
    mask = np.load(output / rows[0]["mask"], allow_pickle=False)

    assert result["recipes_rendered"] == 2
    assert result["claim_eligible"] is False
    assert result["performance_evaluated"] is False
    assert rendered.dtype == np.float32
    assert mask.dtype == np.bool_
    assert not (output / "metrics.json").exists()


def test_threaded_renderer_retains_one_byte_contract(tmp_path: Path) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    inputs = resolve_synthetic_development_inputs(
        category="a",
        dataset_root=visa,
        anomaly_source_root=dtd,
        seed=REGISTERED_SEEDS[0],
        protocol_config_path=config,
    )
    recipe = inputs.recipes[1]

    def render_hash(_: int) -> str:
        image, mask = render_synthetic_recipe(recipe)
        return hashlib.sha256(image.tobytes() + mask.tobytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(render_hash, range(32)))
    assert len(set(hashes)) == 1


def test_model_substitute_evaluation_includes_normal_false_positives_and_raw_maps(
    tmp_path: Path, monkeypatch
) -> None:
    config, _, visa, dtd = _prepare_frozen_protocol(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save(_TinyReconstructive().state_dict(), checkpoints / "reconstructive.pt")
    torch.save(_TinyDiscriminative().state_dict(), checkpoints / "discriminative.pt")
    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_synthetic_dev.build_models",
        lambda device: (_TinyReconstructive().to(device), _TinyDiscriminative().to(device)),
    )
    output = tmp_path / "evaluation"
    result = run_synthetic_development_evaluation(
        SyntheticDevEvaluationSettings(
            category="a",
            dataset_root=visa,
            anomaly_source_root=dtd,
            protocol_config_path=config,
            seed=REGISTERED_SEEDS[0],
            reconstructive_checkpoint=checkpoints / "reconstructive.pt",
            discriminative_checkpoint=checkpoints / "discriminative.pt",
            output_dir=output,
            batch_size=2,
        )
    )
    rows = [
        json.loads(line)
        for line in (output / "predictions/index.jsonl").read_text().splitlines()
    ]
    raw_map = np.load(output / rows[0]["raw_map"], allow_pickle=False)
    mask = np.load(output / rows[0]["evaluation_mask"], allow_pickle=False)
    metrics = json.loads((output / "metrics.json").read_text())

    assert result["num_images"] == 6
    assert result["num_normal_dev"] == 2
    assert result["num_synthetic_positive"] == 4
    assert result["claim_eligible"] is False
    assert result["evaluation_scope"] == "synthetic-development-only"
    assert result["formal_test_content_accessed"] is False
    assert result["performance_evaluated"] is True
    assert raw_map.dtype == np.float32
    assert mask.dtype == np.bool_
    assert sum(row["sample_kind"] == "normal_dev" for row in rows) == 2
    assert all("image_score" not in row for row in rows)
    assert all("raw_map_max_score" in row for row in rows)
    assert all("method_native_image_score" in row for row in rows)
    assert rows[0]["raw_map_max_score"] == pytest.approx(float(raw_map.max()))
    assert metrics["schema_version"] == 2
    assert metrics["image_score_contract"]["common_image_score_field"] == (
        "raw_map_max_score"
    )
    assert "image_auroc_method_native" in metrics["metrics"]
    assert "normal_false_positive_metrics" in metrics
    assert "method_native_normal_false_positive_metrics" in metrics
    assert metrics["performance_evaluated"] is True
    assert metrics["computed_from"].startswith("predictions/index.jsonl")
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["performance_evaluated"] is True
    input_evidence = run_config["input_evidence"]
    assert input_evidence["renderer_provenance_sha256"] == _selection_hash(
        [input_evidence["renderer_provenance"]]
    )
