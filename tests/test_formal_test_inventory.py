from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.formal_test_inventory import (
    build_formal_test_inventory_bundle,
    resolve_formal_test_inventory_category,
    validate_formal_prediction_rows,
    validate_formal_test_inventory_bundle,
    validate_formal_test_inventory_evidence,
)
from industrial_anomaly_lab.run_artifacts import sha256_file, write_json


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, side: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((16, 16), dtype=np.uint8)
    array[:side, :side] = 255
    Image.fromarray(array, mode="L").save(path)


def _fixture(tmp_path: Path, dataset: str = "mvtec_ad") -> dict[str, Path]:
    project = tmp_path / "project"
    config_dir = project / "configs"
    manifest_dir = project / "manifests"
    category = "bottle" if dataset == "mvtec_ad" else "candle"
    dataset_root = project / "data" / dataset
    image_suffix = ".png" if dataset == "mvtec_ad" else ".JPG"
    for index, side in enumerate((1, 4, 8), start=1):
        _write_rgb(
            dataset_root / category / "test" / "crack" / f"{index:03d}{image_suffix}",
            30 + index,
        )
        mask_name = f"{index:03d}_mask.png" if dataset == "mvtec_ad" else f"{index:03d}.png"
        _write_mask(
            dataset_root / category / "ground_truth" / "crack" / mask_name,
            side,
        )
    _write_rgb(dataset_root / category / "test" / "good" / f"000{image_suffix}", 90)
    prepared_path = manifest_dir / f"{dataset}_prepared.json"
    if dataset == "mvtec_ad":
        prepared = {
            "schema_version": 1,
            "dataset": "MVTec AD",
            "layout": {
                "root": f"data/{dataset}",
                "categories": {
                    category: {
                        "test_good_images": 1,
                        "test_anomaly_images": 3,
                        "ground_truth_masks": 3,
                    }
                },
            },
        }
    else:
        prepared = {
            "schema_version": 1,
            "dataset": "VisA",
            "prepared": {"root": f"data/{dataset}"},
            "split": {
                "partition_counts": {
                    category: {
                        "train_normal": 2,
                        "test_normal": 1,
                        "test_anomaly": 3,
                    }
                }
            },
        }
    write_json(prepared_path, prepared)
    project_path = config_dir / "project.json"
    write_json(
        project_path,
        {
            "datasets": {
                dataset: {
                    "categories": [category],
                    "prepared_manifest": prepared_path.relative_to(project).as_posix(),
                }
            },
            "evaluation": {"formal_protocol_frozen": True},
        },
    )
    registration_path = config_dir / "registration.json"
    methods = ["DRAEM", "Improved"]
    write_json(
        registration_path,
        {
            "registration_id": "synthetic-f0",
            "formal_method_registry": {
                "status": "frozen_before_formal_evaluation",
                "methods": methods,
                "baseline": "DRAEM",
                "candidate": "Improved",
                "recipes": {
                    method: {
                        "recipe_id": f"{method.lower()}-v1",
                        "training": {"epochs": 1},
                        "inference": {"image_size": 256},
                    }
                    for method in methods
                },
            },
        },
    )
    protocol_path = config_dir / "inventory_protocol.json"
    write_json(
        protocol_path,
        {
            "schema_version": 1,
            "kind": "formal_test_inventory_protocol",
            "status": "frozen_before_formal_test_enumeration",
            "datasets": ["mvtec_ad", "visa"],
            "mask_interpretation": {
                "positive_rule": "pixel_value_gt_0",
                "area_denominator": "all_native_mask_pixels",
            },
            "anomaly_scale_thresholds": {
                "small_max_fraction": 0.01,
                "medium_max_fraction": 0.1,
            },
        },
    )
    return {
        "project": project,
        "dataset_root": dataset_root,
        "project_config": project_path,
        "registration": registration_path,
        "protocol": protocol_path,
        "prepared": prepared_path,
    }


def _build(fixture: dict[str, Path], output: Path, dataset: str) -> dict[str, object]:
    return build_formal_test_inventory_bundle(
        dataset=dataset,
        dataset_root=fixture["dataset_root"],
        output_dir=output,
        project_root=fixture["project"],
        project_config_path=fixture["project_config"],
        registration_path=fixture["registration"],
        protocol_path=fixture["protocol"],
        prepared_manifest_path=fixture["prepared"],
    )


@pytest.mark.parametrize("dataset", ["mvtec_ad", "visa"])
def test_build_and_revalidate_complete_inventory(tmp_path: Path, dataset: str) -> None:
    fixture = _fixture(tmp_path, dataset)
    first_dir = fixture["project"] / "manifests" / "formal" / "first"
    second_dir = fixture["project"] / "manifests" / "formal" / "second"
    receipt = _build(fixture, first_dir, dataset)
    repeated = _build(fixture, second_dir, dataset)
    inventory = json.loads((first_dir / "inventory.json").read_text(encoding="utf-8"))

    assert (first_dir / "inventory.json").read_bytes() == (
        second_dir / "inventory.json"
    ).read_bytes()
    assert receipt == repeated
    assert receipt["entry_count"] == 4
    assert receipt["formal_test_content_accessed"] is True
    assert receipt["model_predictions_accessed"] is False
    assert receipt["result_driven_selection_performed"] is False
    assert {entry["anomaly_scale"] for entry in inventory["entries"]} == {
        "normal",
        "small",
        "medium",
        "large",
    }
    assert all(not Path(entry["source_image"]).is_absolute() for entry in inventory["entries"])
    assert validate_formal_test_inventory_bundle(
        output_dir=first_dir, project_root=fixture["project"]
    ) == receipt


def test_pending_formal_freeze_rejects_before_test_access(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    project = json.loads(fixture["project_config"].read_text(encoding="utf-8"))
    project["evaluation"]["formal_protocol_frozen"] = False
    write_json(fixture["project_config"], project)
    output = fixture["project"] / "manifests" / "formal" / "blocked"

    with pytest.raises(ValueError, match="formal evaluation protocol is not frozen"):
        _build(fixture, output, "mvtec_ad")
    assert not output.exists()


def test_incomplete_registry_rejects_before_test_access(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registration = json.loads(fixture["registration"].read_text(encoding="utf-8"))
    registration["formal_method_registry"]["candidate"] = None
    write_json(fixture["registration"], registration)
    output = fixture["project"] / "manifests" / "formal" / "blocked"

    with pytest.raises(ValueError, match="formal candidate"):
        _build(fixture, output, "mvtec_ad")
    assert not output.exists()


def test_source_and_inventory_tamper_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "manifests" / "formal" / "built"
    _build(fixture, output, "mvtec_ad")
    image = fixture["dataset_root"] / "bottle" / "test" / "good" / "000.png"
    _write_rgb(image, 240)
    with pytest.raises(ValueError, match="differs from current complete test split"):
        validate_formal_test_inventory_bundle(
            output_dir=output, project_root=fixture["project"]
        )

    clean_fixture = _fixture(tmp_path / "other")
    clean_output = clean_fixture["project"] / "manifests" / "formal" / "built"
    _build(clean_fixture, clean_output, "mvtec_ad")
    inventory = clean_output / "inventory.json"
    inventory.write_text(inventory.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory SHA-256 mismatch"):
        validate_formal_test_inventory_bundle(
            output_dir=clean_output, project_root=clean_fixture["project"]
        )


def test_count_drift_and_nonbinary_mask_fail_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prepared = json.loads(fixture["prepared"].read_text(encoding="utf-8"))
    prepared["layout"]["categories"]["bottle"]["test_anomaly_images"] = 4
    write_json(fixture["prepared"], prepared)
    output = fixture["project"] / "manifests" / "formal" / "bad-count"
    with pytest.raises(ValueError, match="formal-test counts differ"):
        _build(fixture, output, "mvtec_ad")
    assert not output.exists()

    fixture = _fixture(tmp_path / "mask")
    bad_mask = (
        fixture["dataset_root"]
        / "bottle"
        / "ground_truth"
        / "crack"
        / "001_mask.png"
    )
    array = np.zeros((16, 16), dtype=np.uint8)
    array[0, 0] = 127
    Image.fromarray(array, mode="L").save(bad_mask)
    output = fixture["project"] / "manifests" / "formal" / "bad-mask"
    with pytest.raises(ValueError, match="binary 0/255"):
        _build(fixture, output, "mvtec_ad")
    assert not output.exists()


def test_output_escape_and_existing_output_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must stay below project root"):
        _build(fixture, tmp_path / "outside", "mvtec_ad")

    output = fixture["project"] / "manifests" / "formal" / "occupied"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        _build(fixture, output, "mvtec_ad")
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("dataset", "category"),
    [("mvtec_ad", "bottle"), ("visa", "candle")],
)
def test_category_resolver_returns_hash_bound_explicit_samples(
    tmp_path: Path, dataset: str, category: str
) -> None:
    fixture = _fixture(tmp_path, dataset)
    output = fixture["project"] / "manifests" / "formal" / "built"
    _build(fixture, output, dataset)

    selection = resolve_formal_test_inventory_category(
        output_dir=output,
        dataset=dataset,
        category=category,
        dataset_root=fixture["dataset_root"],
        project_root=fixture["project"],
    )
    evidence = selection["evidence"]
    assert evidence["contract"] == "formal_test_inventory_category_selection_v1"
    assert evidence["bundle"] == "manifests/formal/built"
    assert evidence["category_entry_count"] == 4
    assert len(selection["samples"]) == len(selection["entries"]) == 4
    assert all(sample.category == category for sample in selection["samples"])

    revalidated = validate_formal_test_inventory_evidence(
        evidence,
        dataset=dataset,
        category=category,
        project_config_path=fixture["project_config"],
    )
    assert revalidated["evidence"] == evidence

    rows = []
    for index, entry in enumerate(selection["entries"]):
        rows.append(
            {
                "index": index,
                "sample_id": entry["sample_id"],
                "label": entry["label"],
                "defect_type": entry["defect_type"],
                "input_image": entry["source_image"],
                "source_ground_truth": entry["source_mask"],
                "source_image_sha256": entry["source_image_sha256"],
                "source_ground_truth_sha256": entry["source_mask_sha256"],
                "formal_test_inventory_id": evidence["inventory_id"],
                "formal_test_inventory_sha256": evidence["inventory_sha256"],
                "formal_test_inventory_receipt_sha256": evidence["receipt_sha256"],
            }
        )
    validate_formal_prediction_rows(rows, selection)
    rows[0]["sample_id"] = f"{category}/good/tampered"
    with pytest.raises(ValueError, match="differs from inventory field sample_id"):
        validate_formal_prediction_rows(rows, selection)


def test_category_resolver_rejects_wrong_identity_and_path_escape(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "manifests" / "formal" / "built"
    _build(fixture, output, "mvtec_ad")

    with pytest.raises(ValueError, match="inventory dataset mismatch"):
        resolve_formal_test_inventory_category(
            output_dir=output,
            dataset="visa",
            category="bottle",
            project_root=fixture["project"],
        )
    with pytest.raises(ValueError, match="category is absent"):
        resolve_formal_test_inventory_category(
            output_dir=output,
            dataset="mvtec_ad",
            category="cable",
            project_root=fixture["project"],
        )
    with pytest.raises(ValueError, match="supplied dataset root differs"):
        resolve_formal_test_inventory_category(
            output_dir=output,
            dataset="mvtec_ad",
            category="bottle",
            dataset_root=fixture["project"] / "data" / "other",
            project_root=fixture["project"],
        )

    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["dataset_root"] = "../escape"
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="safe relative path"):
        resolve_formal_test_inventory_category(
            output_dir=output,
            dataset="mvtec_ad",
            category="bottle",
            project_root=fixture["project"],
        )


def test_category_resolver_rejects_incomplete_inventory_and_receipt_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "manifests" / "formal" / "built"
    _build(fixture, output, "mvtec_ad")
    selection = resolve_formal_test_inventory_category(
        output_dir=output,
        dataset="mvtec_ad",
        category="bottle",
        project_root=fixture["project"],
    )

    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["result_driven_selection_performed"] = True
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="receipt flag is invalid"):
        validate_formal_test_inventory_evidence(
            selection["evidence"],
            dataset="mvtec_ad",
            category="bottle",
            project_config_path=fixture["project_config"],
        )

    clean = _fixture(tmp_path / "clean")
    clean_output = clean["project"] / "manifests" / "formal" / "built"
    _build(clean, clean_output, "mvtec_ad")
    inventory_path = clean_output / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["entries"].pop()
    write_json(inventory_path, inventory)
    receipt_path = clean_output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inventory_sha256"] = sha256_file(inventory_path)
    receipt["entry_count"] -= 1
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="differs from current complete test split"):
        resolve_formal_test_inventory_category(
            output_dir=clean_output,
            dataset="mvtec_ad",
            category="bottle",
            project_root=clean["project"],
        )
