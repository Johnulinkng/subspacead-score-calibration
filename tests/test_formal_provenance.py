from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_anomaly_lab.formal_provenance import (
    canonical_sha256,
    classify_run_purpose,
    validate_development_authority,
    validate_formal_authority,
    validate_formal_checkpoint_manifest,
)
from industrial_anomaly_lab.draem_train import TrainSettings, run_training
from industrial_anomaly_lab.run_artifacts import sha256_file
import industrial_anomaly_lab.formal_source_provenance as source_provenance_module


SEEDS = [11, 22, 33]
FORMAL_METHOD = "SYNTHETIC_FORMAL"


def _synthetic_source_validation() -> dict[str, object]:
    commits = {
        "DRAEM": ("third_party/DRAEM", "2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef"),
        "PatchCore": ("third_party/PatchCore", "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"),
        "RD4AD": ("third_party/RD4AD", "6554076872c65f8784f6ece8cfb39ce77e1aee12"),
    }
    return {
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
            for name, (path, commit) in commits.items()
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _authority_files(
    tmp_path: Path,
    *,
    method: str = FORMAL_METHOD,
    training_recipe: dict[str, object] | None = None,
    inference_recipe: dict[str, object] | None = None,
    formal_frozen: bool = True,
    registry_frozen: bool = True,
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    training = training_recipe or {"schema_version": 1, "method": method, "epochs": 700}
    inference = inference_recipe or {
        "schema_version": 1,
        "method": method,
        "evaluation_protocol": "draem_256_unified_v1",
    }
    project = {
        "baseline": {"formal_training_recipe_frozen": True},
        "evaluation": {
            "minimum_seeds": 3,
            "seeds": SEEDS,
            "formal_protocol_frozen": formal_frozen,
        },
        "datasets": {
            "visa": {"categories": ["candle"]},
            "mvtec_ad": {"categories": ["bottle"]},
        },
    }
    registration = {
        "registration_id": "synthetic-formal-v1",
        "model_seeds": SEEDS,
        "formal_method_registry": {
            "status": (
                "frozen_before_formal_evaluation"
                if registry_frozen
                else "pending_candidate_and_ablation_freeze"
            ),
            "methods": [method] if registry_frozen else None,
            "recipes": (
                {
                    method: {
                        "recipe_id": f"{method.lower()}-synthetic-v1",
                        "training": training,
                        "inference": inference,
                    }
                }
                if registry_frozen
                else None
            ),
        },
    }
    development = {"status": "frozen_before_model_development"}
    project_path = tmp_path / "project.json"
    registration_path = tmp_path / "registration.json"
    development_path = tmp_path / "development.json"
    _write_json(project_path, project)
    _write_json(registration_path, registration)
    _write_json(development_path, development)
    return project_path, registration_path, development_path, training, inference


def test_run_purpose_is_explicit_and_never_inferred_from_full_scope() -> None:
    assert classify_run_purpose("formal_full", max_steps=None) == {
        "run_purpose": "formal_full",
        "execution_scope": "full",
        "claim_scope": "formal_matrix_candidate",
        "formal_matrix_eligible": True,
    }
    assert classify_run_purpose("development_full", max_steps=None)["claim_scope"] == (
        "development_only"
    )
    assert classify_run_purpose("author_reproduction", max_steps=None)[
        "formal_matrix_eligible"
    ] is False
    with pytest.raises(ValueError, match="positive max_steps"):
        classify_run_purpose("debug_limited", max_steps=None)
    with pytest.raises(ValueError, match="forbidden"):
        classify_run_purpose("formal_full", max_steps=1)
    with pytest.raises(ValueError, match="forbidden"):
        classify_run_purpose("development_full", max_steps=1)


def test_development_full_allows_pending_formal_registry_but_only_registered_seed(
    tmp_path: Path,
) -> None:
    project, registration, development, _, _ = _authority_files(
        tmp_path, formal_frozen=False, registry_frozen=False
    )
    authority = validate_development_authority(
        method="DRAEM",
        dataset="visa",
        category="candle",
        seed=SEEDS[0],
        project_config_path=project,
        registration_path=registration,
        development_protocol_path=development,
    )
    assert authority["registered_seed_count"] == 3
    assert authority["claim_scope"] == "development_only"
    assert authority["formal_matrix_eligible"] is False

    with pytest.raises(ValueError, match="not preregistered"):
        validate_development_authority(
            method="DRAEM",
            dataset="visa",
            category="candle",
            seed=999,
            project_config_path=project,
            registration_path=registration,
            development_protocol_path=development,
        )


def test_draem_development_full_keeps_700_epoch_recipe_and_ignores_formal_freeze(
    tmp_path: Path, monkeypatch
) -> None:
    project, registration, development, _, _ = _authority_files(
        tmp_path, formal_frozen=False, registry_frozen=False
    )
    settings = TrainSettings(
        dataset="visa",
        category="candle",
        dataset_root=tmp_path / "must-not-be-read",
        anomaly_source_root=tmp_path / "must-not-be-read-dtd",
        output_dir=tmp_path / "must-not-be-written",
        seed=SEEDS[0],
        image_size=256,
        batch_size=8,
        epochs=700,
        learning_rate=1e-4,
        workers=0,
        device="cpu",
        max_steps=None,
        input_protocol="visa_train_dev_dtd_v1",
        development_protocol_config=development,
        run_purpose="development_full",
        project_config_path=project,
        registration_path=registration,
    )
    reached_dataset_resolution = False

    def _stop_before_data(*args, **kwargs):
        nonlocal reached_dataset_resolution
        reached_dataset_resolution = True
        raise RuntimeError("synthetic dataset sentinel")

    monkeypatch.setattr(
        "industrial_anomaly_lab.draem_train._resolve_training_dataset",
        _stop_before_data,
    )
    with pytest.raises(RuntimeError, match="synthetic dataset sentinel"):
        run_training(settings)
    assert reached_dataset_resolution is True
    assert not settings.output_dir.exists()

    object.__setattr__(settings, "epochs", 699)
    with pytest.raises(ValueError, match="700-epoch"):
        run_training(settings)


def test_formal_authority_requires_frozen_protocol_registry_three_seeds_and_recipe(
    tmp_path: Path,
) -> None:
    project, registration, _, training, _ = _authority_files(tmp_path)
    authority = validate_formal_authority(
        method=FORMAL_METHOD,
        dataset="visa",
        category="candle",
        seed=SEEDS[1],
        phase="training",
        recipe=training,
        project_config_path=project,
        registration_path=registration,
    )
    assert authority["registered_seed_count"] == 3
    assert authority["recipe_sha256"] == canonical_sha256(training)

    altered = dict(training)
    altered["epochs"] = 699
    with pytest.raises(ValueError, match="differs from registration"):
        validate_formal_authority(
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[1],
            phase="training",
            recipe=altered,
            project_config_path=project,
            registration_path=registration,
        )

    payload = json.loads(registration.read_text(encoding="utf-8"))
    payload["model_seeds"] = [SEEDS[0]]
    _write_json(registration, payload)
    with pytest.raises(ValueError, match="at least 3 unique"):
        validate_formal_authority(
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            phase="training",
            recipe=training,
            project_config_path=project,
            registration_path=registration,
        )


@pytest.mark.parametrize("method", ["DRAEM", "RD4AD"])
def test_production_methods_remain_formal_blocked_without_hashed_training_inputs(
    tmp_path: Path, method: str
) -> None:
    authority_root = tmp_path / method.lower()
    authority_root.mkdir()
    project, registration, _, training, _ = _authority_files(
        authority_root, method=method
    )
    with pytest.raises(ValueError, match="training-input provenance"):
        validate_formal_authority(
            method=method,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            phase="training",
            recipe=training,
            project_config_path=project,
            registration_path=registration,
        )


def test_checkpoint_manifest_binds_identity_recipe_and_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_validation = _synthetic_source_validation()
    monkeypatch.setattr(
        source_provenance_module,
        "validate_formal_source_provenance",
        lambda evidence, *, project_root: source_validation,
    )
    project, registration, _, training, _ = _authority_files(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    reconstructive = checkpoint_dir / "reconstructive.pt"
    discriminative = checkpoint_dir / "discriminative.pt"
    reconstructive.write_bytes(b"synthetic-reconstructive")
    discriminative.write_bytes(b"synthetic-discriminative")
    authority = validate_formal_authority(
        method=FORMAL_METHOD,
        dataset="visa",
        category="candle",
        seed=SEEDS[0],
        phase="training",
        recipe=training,
        project_config_path=project,
        registration_path=registration,
    )
    manifest = {
        "schema_version": 2,
        "method": FORMAL_METHOD,
        "dataset": "visa",
        "category": "candle",
        "seed": SEEDS[0],
        "run_purpose": "formal_full",
        "execution_scope": "full",
        "claim_scope": "formal_matrix_candidate",
        "checkpoint_origin": "project_training_run",
        "final_checkpoint_only": True,
        "test_selected": False,
        "training_recipe": training,
        "formal_provenance": authority,
        "formal_source_provenance": {"synthetic_fixture": True},
        "checkpoints": {
            "reconstructive": {
                "path": reconstructive.name,
                "size_bytes": reconstructive.stat().st_size,
                "sha256": sha256_file(reconstructive),
            },
            "discriminative": {
                "path": discriminative.name,
                "size_bytes": discriminative.stat().st_size,
                "sha256": sha256_file(discriminative),
            },
        },
        "external_hashes": {},
    }
    manifest_path = checkpoint_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    report = validate_formal_checkpoint_manifest(
        manifest_path,
        method=FORMAL_METHOD,
        dataset="visa",
        category="candle",
        seed=SEEDS[0],
        checkpoint_files={
            "reconstructive": reconstructive,
            "discriminative": discriminative,
        },
        project_config_path=project,
        registration_path=registration,
    )
    assert report["validated"] is True
    assert report["manifest_sha256"] == sha256_file(manifest_path)

    unexpected_external = json.loads(json.dumps(manifest))
    unexpected_external["external_hashes"] = {"teacher": "0" * 64}
    _write_json(manifest_path, unexpected_external)
    with pytest.raises(ValueError, match="external_hashes roles do not match"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={
                "reconstructive": reconstructive,
                "discriminative": discriminative,
            },
            project_config_path=project,
            registration_path=registration,
        )

    missing_source = json.loads(json.dumps(manifest))
    del missing_source["formal_source_provenance"]
    _write_json(manifest_path, missing_source)
    with pytest.raises(ValueError, match="missing formal source provenance"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={
                "reconstructive": reconstructive,
                "discriminative": discriminative,
            },
            project_config_path=project,
            registration_path=registration,
        )

    missing_vendor = dict(source_validation)
    missing_vendor["repositories"] = [
        row
        for row in source_validation["repositories"]
        if row["name"] != "RD4AD"
    ]
    monkeypatch.setattr(
        source_provenance_module,
        "validate_formal_source_provenance",
        lambda evidence, *, project_root: missing_vendor,
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="frozen clean RD4AD vendor"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={
                "reconstructive": reconstructive,
                "discriminative": discriminative,
            },
            project_config_path=project,
            registration_path=registration,
        )
    monkeypatch.setattr(
        source_provenance_module,
        "validate_formal_source_provenance",
        lambda evidence, *, project_root: source_validation,
    )

    for field, value in (
        ("dataset", "mvtec_ad"),
        ("category", "capsules"),
        ("seed", SEEDS[1]),
        ("execution_scope", "debug_limited"),
    ):
        tampered_manifest = json.loads(json.dumps(manifest))
        tampered_manifest[field] = value
        _write_json(manifest_path, tampered_manifest)
        with pytest.raises(ValueError, match=f"{field} mismatch"):
            validate_formal_checkpoint_manifest(
                manifest_path,
                method=FORMAL_METHOD,
                dataset="visa",
                category="candle",
                seed=SEEDS[0],
                checkpoint_files={
                    "reconstructive": reconstructive,
                    "discriminative": discriminative,
                },
                project_config_path=project,
                registration_path=registration,
            )

    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["formal_provenance"]["project_config_sha256"] = "0" * 64
    _write_json(manifest_path, tampered_manifest)
    with pytest.raises(ValueError, match="formal provenance mismatch"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={
                "reconstructive": reconstructive,
                "discriminative": discriminative,
            },
            project_config_path=project,
            registration_path=registration,
        )

    _write_json(manifest_path, manifest)
    discriminative.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={
                "reconstructive": reconstructive,
                "discriminative": discriminative,
            },
            project_config_path=project,
            registration_path=registration,
        )


def test_checkpoint_manifest_rejects_author_reproduction_disguised_as_full(
    tmp_path: Path,
) -> None:
    project, registration, _, training, _ = _authority_files(tmp_path)
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"student")
    authority = validate_formal_authority(
        method=FORMAL_METHOD,
        dataset="visa",
        category="candle",
        seed=SEEDS[0],
        phase="training",
        recipe=training,
        project_config_path=project,
        registration_path=registration,
    )
    manifest = {
        "schema_version": 2,
        "method": FORMAL_METHOD,
        "dataset": "visa",
        "category": "candle",
        "seed": SEEDS[0],
        "run_purpose": "author_reproduction",
        "execution_scope": "full",
        "checkpoint_origin": "project_training_run",
        "final_checkpoint_only": True,
        "test_selected": False,
        "training_recipe": training,
        "formal_provenance": authority,
        "checkpoints": {
            "student": {
                "path": checkpoint.name,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            }
        },
        "external_hashes": {},
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="run_purpose mismatch"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={"student": checkpoint},
            project_config_path=project,
            registration_path=registration,
        )

    manifest["run_purpose"] = "formal_full"
    manifest["claim_scope"] = "formal_matrix_candidate"
    manifest["checkpoint_origin"] = "official_pretrained"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not produced by a project training run"):
        validate_formal_checkpoint_manifest(
            manifest_path,
            method=FORMAL_METHOD,
            dataset="visa",
            category="candle",
            seed=SEEDS[0],
            checkpoint_files={"student": checkpoint},
            project_config_path=project,
            registration_path=registration,
        )
