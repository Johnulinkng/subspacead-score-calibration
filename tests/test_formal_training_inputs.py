from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import industrial_anomaly_lab.formal_source_provenance as source_provenance_module
import industrial_anomaly_lab.formal_training_inputs as formal_inputs
from industrial_anomaly_lab.formal_training_inputs import (
    build_formal_training_input_bundle,
    resolve_frozen_formal_training_inputs,
    resolve_formal_training_inputs,
)
from industrial_anomaly_lab.formal_provenance import (
    EXPECTED_FORMAL_VENDOR_SOURCES,
    validate_formal_authority,
    validate_formal_checkpoint_manifest,
)
from industrial_anomaly_lab.run_artifacts import sha256_file


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "project"
    mvtec = root / "data/mvtec_ad"
    visa = root / "data/visa/1cls"
    dtd = root / "data/dtd/images"
    _write(mvtec / "bottle/train/good/000.png", b"mvtec-a")
    _write(mvtec / "bottle/train/good/001.png", b"mvtec-b")
    _write(mvtec / "bottle/test/crack/forbidden.png", b"must-not-be-read")
    _write(mvtec / "bottle/ground_truth/crack/forbidden_mask.png", b"must-not-be-read")
    _write(visa / "candle/train/good/000.JPG", b"visa-a")
    _write(visa / "candle/test/bad/forbidden.JPG", b"must-not-be-read")
    _write(dtd / "banded/banded_0001.jpg", b"dtd-a")
    _write(dtd / "striped/striped_0001.jpg", b"dtd-b")
    for name in ("mvtec_ad", "visa", "dtd"):
        _write_json(root / f"manifests/{name}.json", {"source": name})
    _write(root / "src/builder.py", b"builder-v1")
    _write(root / "src/validator.py", b"validator-v1")
    return {
        "root": root,
        "mvtec": mvtec,
        "visa": visa,
        "dtd": dtd,
        "output": root / "manifests/formal/v1",
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    root = paths["root"]
    return build_formal_training_input_bundle(
        project_root=root,
        output_dir=paths["output"],
        dataset_roots={"mvtec_ad": paths["mvtec"], "visa": paths["visa"]},
        dtd_root=paths["dtd"],
        categories={"mvtec_ad": ["bottle"], "visa": ["candle"]},
        source_manifests={
            name: root / f"manifests/{name}.json"
            for name in ("mvtec_ad", "visa", "dtd")
        },
        provenance_files={
            "builder": root / "src/builder.py",
            "validator": root / "src/validator.py",
        },
    )


def _resolve(paths: dict[str, Path], *, method: str = "DRAEM"):
    return resolve_formal_training_inputs(
        method=method,
        dataset="mvtec_ad",
        category="bottle",
        dataset_root=paths["mvtec"],
        dtd_root=paths["dtd"] if method == "DRAEM" else None,
        bundle_manifest_path=paths["output"] / "manifest.json",
        project_root=paths["root"],
    )


def test_builder_and_resolver_never_enumerate_test_or_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    original_scandir = formal_inputs.os.scandir
    visited: list[Path] = []

    def guarded_scandir(path):
        current = Path(path)
        visited.append(current)
        forbidden = {part.casefold() for part in current.parts} & {"test", "ground_truth"}
        if forbidden:
            raise AssertionError(f"forbidden enumeration: {current}")
        return original_scandir(path)

    monkeypatch.setattr(formal_inputs.os, "scandir", guarded_scandir)
    build = _build(paths)
    draem = _resolve(paths)
    patchcore = _resolve(paths, method="PatchCore")
    rd4ad = _resolve(paths, method="RD4AD")

    assert build["normal_training_images"] == 3
    assert build["dtd_images"] == 2
    assert len(draem.normal_image_paths) == 2
    assert len(draem.anomaly_source_paths) == 2
    assert len(rd4ad.normal_image_paths) == 2
    assert rd4ad.anomaly_source_paths == ()
    assert len(patchcore.normal_image_paths) == 2
    assert patchcore.anomaly_source_paths == ()
    assert patchcore.evidence["method"] == "PatchCore"
    assert draem.evidence["formal_test_content_accessed"] is False
    assert draem.evidence["test_or_ground_truth_enumerated"] is False
    assert all("test" not in {p.casefold() for p in path.parts} for path in visited)


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra"])
def test_resolver_rejects_changed_training_file_set_or_bytes(
    tmp_path: Path, mutation: str
) -> None:
    paths = _fixture(tmp_path)
    _build(paths)
    target = paths["mvtec"] / "bottle/train/good/000.png"
    if mutation == "tamper":
        target.write_bytes(b"changed")
    elif mutation == "missing":
        target.unlink()
    else:
        _write(paths["mvtec"] / "bottle/train/good/extra.png", b"extra")

    with pytest.raises((ValueError, FileNotFoundError), match="mismatch|changed|missing"):
        _resolve(paths)


def test_resolver_rejects_wrong_category_without_fallback_discovery(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _build(paths)
    with pytest.raises(ValueError, match="not registered"):
        resolve_formal_training_inputs(
            method="RD4AD",
            dataset="mvtec_ad",
            category="cable",
            dataset_root=paths["mvtec"],
            bundle_manifest_path=paths["output"] / "manifest.json",
            project_root=paths["root"],
        )


def test_resolver_rejects_child_path_escape_even_if_child_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _build(paths)
    child = paths["output"] / "mvtec_ad__bottle__train_good.jsonl"
    rows = [json.loads(line) for line in child.read_text(encoding="utf-8").splitlines()]
    rows[0]["project_relative_path"] = "../outside.png"
    raw = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    child.write_bytes(raw)
    manifest_path = paths["output"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["children"] if item["path"].endswith(child.name))
    entry["size_bytes"] = len(raw)
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsafe"):
        _resolve(paths)


def test_builder_rejects_linked_training_entry(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    outside = paths["root"] / "outside.png"
    outside.write_bytes(b"outside")
    link = paths["mvtec"] / "bottle/train/good/link.png"
    try:
        os.symlink(outside, link)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(ValueError, match="linked/reparse"):
        _build(paths)


def test_resolver_rejects_provenance_and_source_manifest_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _build(paths)
    (paths["root"] / "src/validator.py").write_bytes(b"validator-v2")
    with pytest.raises(ValueError, match="provenance.*(size|SHA-256) mismatch"):
        _resolve(paths)

    paths = _fixture(tmp_path / "source-drift")
    _build(paths)
    _write_json(paths["root"] / "manifests/mvtec_ad.json", {"changed": True})
    with pytest.raises(ValueError, match="source manifest.*(size|SHA-256) mismatch"):
        _resolve(paths)


def test_formal_checkpoint_revalidates_portable_training_input_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_validation = {
        "schema_version": 1,
        "contract": "clean_git_source_provenance_v1",
        "repositories": [
            {
                "name": name,
                "path": path,
                "head_commit": commit,
                "expected_commit": commit,
                "worktree_clean": True,
            }
            for name, (path, commit) in EXPECTED_FORMAL_VENDOR_SOURCES.items()
        ],
    }
    monkeypatch.setattr(
        source_provenance_module,
        "validate_formal_source_provenance",
        lambda evidence, *, project_root: source_validation,
    )
    paths = _fixture(tmp_path)
    build = _build(paths)
    evidence = _resolve(paths).evidence
    serialized = json.dumps(evidence, sort_keys=True)
    assert evidence["path_base"] == "project_root"
    assert evidence["project_root"] == "."
    assert paths["root"].as_posix() not in serialized
    assert not Path(str(evidence["root_manifest_path"])).is_absolute()

    root = paths["root"]
    training_recipe = {"schema_version": 1, "method": "DRAEM", "epochs": 700}
    project = {
        "baseline": {"formal_training_recipe_frozen": True},
        "formal_training_inputs": {
            "status": "frozen_train_only_before_formal_evaluation",
            "contract": "formal_training_input_bundle_v1",
            "manifest": "manifests/formal/v1/manifest.json",
            "sha256": build["manifest_sha256"],
            "formal_test_content_accessed": False,
            "ground_truth_content_accessed": False,
            "test_or_ground_truth_enumerated": False,
            "training_started": False,
        },
        "evaluation": {
            "minimum_seeds": 3,
            "seeds": [11, 22, 33],
            "formal_protocol_frozen": True,
        },
        "datasets": {"mvtec_ad": {"categories": ["bottle"]}},
    }
    registration = {
        "registration_id": "portable-formal-input-v1",
        "model_seeds": [11, 22, 33],
        "formal_method_registry": {
            "status": "frozen_before_formal_evaluation",
            "methods": ["DRAEM"],
            "recipes": {
                "DRAEM": {
                    "recipe_id": "draem-test-v1",
                    "training": training_recipe,
                    "inference": {"schema_version": 1, "method": "DRAEM"},
                }
            },
        },
    }
    project_path = root / "configs/project.json"
    registration_path = root / "configs/registration.json"
    _write_json(project_path, project)
    _write_json(registration_path, registration)
    authority = validate_formal_authority(
        method="DRAEM",
        dataset="mvtec_ad",
        category="bottle",
        seed=11,
        phase="training",
        recipe=training_recipe,
        training_input_evidence=evidence,
        formal_input_project_root=root,
        project_config_path=project_path,
        registration_path=registration_path,
    )
    assert authority["training_input_validation"][  # type: ignore[index]
        "current_file_set_and_content_sha256_verified"
    ] is True

    checkpoint_dir = root / "runs/checkpoints"
    _write(checkpoint_dir / "reconstructive.pt", b"r")
    _write(checkpoint_dir / "discriminative.pt", b"d")
    checkpoints = {
        role: {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in {
            "reconstructive": checkpoint_dir / "reconstructive.pt",
            "discriminative": checkpoint_dir / "discriminative.pt",
        }.items()
    }
    checkpoint_manifest = {
        "schema_version": 2,
        "method": "DRAEM",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "seed": 11,
        "run_purpose": "formal_full",
        "execution_scope": "full",
        "claim_scope": "formal_matrix_candidate",
        "checkpoint_origin": "project_training_run",
        "final_checkpoint_only": True,
        "test_selected": False,
        "training_recipe": training_recipe,
        "formal_provenance": authority,
        "formal_source_provenance": {"synthetic_fixture": True},
        "formal_training_inputs": evidence,
        "checkpoints": checkpoints,
        "external_hashes": {},
    }
    checkpoint_manifest_path = checkpoint_dir / "manifest.json"
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    report = validate_formal_checkpoint_manifest(
        checkpoint_manifest_path,
        method="DRAEM",
        dataset="mvtec_ad",
        category="bottle",
        seed=11,
        checkpoint_files={
            "reconstructive": checkpoint_dir / "reconstructive.pt",
            "discriminative": checkpoint_dir / "discriminative.pt",
        },
        project_config_path=project_path,
        registration_path=registration_path,
    )
    assert report["training_input_validation"][  # type: ignore[index]
        "root_manifest_sha256"
    ] == evidence["root_manifest_sha256"]

    (paths["mvtec"] / "bottle/train/good/001.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch|changed"):
        validate_formal_checkpoint_manifest(
            checkpoint_manifest_path,
            method="DRAEM",
            dataset="mvtec_ad",
            category="bottle",
            seed=11,
            checkpoint_files={
                "reconstructive": checkpoint_dir / "reconstructive.pt",
                "discriminative": checkpoint_dir / "discriminative.pt",
            },
            project_config_path=project_path,
            registration_path=registration_path,
        )


def test_frozen_reference_rejects_a_valid_rebuilt_bundle_with_a_different_root(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    first = _build(paths)
    project_path = paths["root"] / "configs/project.json"
    _write_json(
        project_path,
        {
            "formal_training_inputs": {
                "status": "frozen_train_only_before_formal_evaluation",
                "contract": "formal_training_input_bundle_v1",
                "manifest": "manifests/formal/v1/manifest.json",
                "sha256": first["manifest_sha256"],
                "formal_test_content_accessed": False,
                "ground_truth_content_accessed": False,
                "test_or_ground_truth_enumerated": False,
                "training_started": False,
            }
        },
    )

    _write(paths["mvtec"] / "bottle/train/good/002.png", b"new-valid-input")
    second_output = paths["root"] / "manifests/formal/v2"
    second = build_formal_training_input_bundle(
        project_root=paths["root"],
        output_dir=second_output,
        dataset_roots={"mvtec_ad": paths["mvtec"], "visa": paths["visa"]},
        dtd_root=paths["dtd"],
        categories={"mvtec_ad": ["bottle"], "visa": ["candle"]},
        source_manifests={
            name: paths["root"] / f"manifests/{name}.json"
            for name in ("mvtec_ad", "visa", "dtd")
        },
        provenance_files={
            "builder": paths["root"] / "src/builder.py",
            "validator": paths["root"] / "src/validator.py",
        },
    )
    assert second["manifest_sha256"] != first["manifest_sha256"]

    with pytest.raises(ValueError, match="differs from the frozen project pin"):
        resolve_frozen_formal_training_inputs(
            method="RD4AD",
            dataset="mvtec_ad",
            category="bottle",
            dataset_root=paths["mvtec"],
            bundle_manifest_path=second_output / "manifest.json",
            project_config_path=project_path,
            project_root=paths["root"],
        )
