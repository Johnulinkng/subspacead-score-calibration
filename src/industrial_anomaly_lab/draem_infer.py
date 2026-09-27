"""Deterministic DRAEM inference with raw prediction artifacts."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_protocols import DraemTestDataset
from .draem_runtime import (
    build_models,
    configure_reproducibility,
    load_state_dict_safely,
    make_generator,
    seed_dataloader_worker,
)
from .formal_provenance import (
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_REGISTRATION,
    RunPurpose,
    classify_run_purpose,
    validate_formal_authority,
    validate_formal_checkpoint_manifest,
)
from .formal_test_inventory import (
    resolve_formal_test_inventory_category,
    validate_formal_prediction_rows,
)
from .metrics import (
    EVALUATION_PROTOCOL,
    METRICS_SCHEMA_VERSION,
    draem_image_scores,
    evaluate_category,
    image_auroc_from_scores,
    image_score_contract,
    raw_map_max_image_scores,
)
from .run_artifacts import (
    prepare_output_directory,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class InferenceSettings:
    dataset: str
    category: str
    dataset_root: Path
    reconstructive_checkpoint: Path
    discriminative_checkpoint: Path
    output_dir: Path
    execution_scope: Literal["full", "debug_limited"]
    seed: int = 42
    image_size: int = 256
    batch_size: int = 8
    workers: int = 0
    device: str = "cpu"
    run_purpose: RunPurpose | None = None
    checkpoint_manifest: Path | None = None
    formal_test_inventory_dir: Path | None = None
    project_config_path: Path = DEFAULT_PROJECT_CONFIG
    registration_path: Path = DEFAULT_REGISTRATION

    def resolved_run_purpose(self) -> RunPurpose:
        if self.run_purpose is not None:
            return self.run_purpose
        return "debug_limited" if self.execution_scope == "debug_limited" else "author_reproduction"

    def validate(self) -> None:
        if self.dataset not in {"mvtec_ad", "visa"}:
            raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
        if not self.category:
            raise ValueError("category must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.execution_scope not in {"full", "debug_limited"}:
            raise ValueError("execution_scope must be 'full' or 'debug_limited'")
        purpose = classify_run_purpose(
            self.resolved_run_purpose(),
            max_steps=1 if self.execution_scope == "debug_limited" else None,
        )
        if purpose["execution_scope"] != self.execution_scope:
            raise ValueError("run_purpose and execution_scope disagree")
        if self.resolved_run_purpose() == "development_full":
            raise ValueError(
                "development_full must use the synthetic development evaluator, "
                "not the formal test inference entry point"
            )
        if (
            self.resolved_run_purpose() != "formal_full"
            and self.formal_test_inventory_dir is not None
        ):
            raise ValueError(
                "formal_test_inventory_dir is only valid for formal_full inference"
            )
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError("image_size and batch_size must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        for checkpoint in (
            self.reconstructive_checkpoint,
            self.discriminative_checkpoint,
        ):
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "dataset_root",
            "reconstructive_checkpoint",
            "discriminative_checkpoint",
            "output_dir",
            "project_config_path",
            "registration_path",
        ):
            payload[key] = str(payload[key])
        payload["checkpoint_manifest"] = (
            None if self.checkpoint_manifest is None else str(self.checkpoint_manifest)
        )
        payload["formal_test_inventory_dir"] = (
            None
            if self.formal_test_inventory_dir is None
            else str(self.formal_test_inventory_dir)
        )
        return payload


def draem_inference_recipe(settings: InferenceSettings) -> dict[str, object]:
    return {
        "schema_version": 2,
        "method": "DRAEM",
        "image_size": settings.image_size,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "raw_map": "segmentation_softmax_anomaly_channel",
        "common_image_score": "max_of_persisted_raw_anomaly_map",
        "method_native_image_score": "max_after_21x21_average_pool",
        "mask_resize": "nearest",
        "raw_map_normalization": "none",
    }


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative_to_root(value: str, root: Path) -> str | None:
    if not value:
        return None
    return Path(value).resolve().relative_to(root).as_posix()


def run_inference(settings: InferenceSettings) -> dict[str, Any]:
    """Infer one full category, then evaluate by reloading the saved arrays."""

    settings.validate()
    run_purpose = settings.resolved_run_purpose()
    purpose = classify_run_purpose(
        run_purpose,
        max_steps=1 if settings.execution_scope == "debug_limited" else None,
    )
    formal_provenance: dict[str, object] | None = None
    formal_test_selection: dict[str, Any] | None = None
    formal_test_evidence: dict[str, object] | None = None
    if run_purpose == "formal_full":
        inference_recipe = draem_inference_recipe(settings)
        inference_authority = validate_formal_authority(
            method="DRAEM",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            phase="inference",
            recipe=inference_recipe,
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
        )
        manifest_path = settings.checkpoint_manifest or (
            Path(settings.reconstructive_checkpoint).resolve().parent / "manifest.json"
        )
        checkpoint_validation = validate_formal_checkpoint_manifest(
            manifest_path,
            method="DRAEM",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            checkpoint_files={
                "reconstructive": settings.reconstructive_checkpoint,
                "discriminative": settings.discriminative_checkpoint,
            },
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
        )
        if settings.formal_test_inventory_dir is None:
            raise ValueError(
                "formal_full DRAEM inference requires formal_test_inventory_dir"
            )
        config_path = Path(settings.project_config_path).resolve()
        project_root = (
            config_path.parent.parent
            if config_path.parent.name == "configs"
            else config_path.parent
        )
        formal_test_selection = resolve_formal_test_inventory_category(
            output_dir=settings.formal_test_inventory_dir,
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=settings.dataset_root,
            project_root=project_root,
        )
        formal_test_evidence = formal_test_selection["evidence"]
        formal_provenance = {
            "claim_scope": "formal_matrix_candidate",
            "inference_recipe": inference_recipe,
            "inference_authority": inference_authority,
            "checkpoint_validation": checkpoint_validation,
            "formal_test_inventory": formal_test_evidence,
        }
    configure_reproducibility(settings.seed, deterministic=True)
    device = _resolve_device(settings.device)
    evaluation_protocol = (
        EVALUATION_PROTOCOL
        if settings.image_size == 256
        else f"synthetic_nonstandard_{settings.image_size}px"
    )
    output_dir = prepare_output_directory(settings.output_dir)
    raw_dir = output_dir / "predictions" / "raw_maps"
    mask_dir = output_dir / "predictions" / "evaluation_masks"
    dataset_root = Path(settings.dataset_root).resolve()

    dataset = DraemTestDataset(
        settings.dataset,
        dataset_root,
        settings.category,
        image_size=settings.image_size,
        samples=(
            None
            if formal_test_selection is None
            else formal_test_selection["samples"]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.workers,
        worker_init_fn=seed_dataloader_worker,
        generator=make_generator(settings.seed),
        pin_memory=device.type == "cuda",
        persistent_workers=settings.workers > 0,
    )

    checkpoint_provenance = {
        "reconstructive": {
            "path": str(Path(settings.reconstructive_checkpoint).resolve()),
            "sha256": sha256_file(settings.reconstructive_checkpoint),
        },
        "discriminative": {
            "path": str(Path(settings.discriminative_checkpoint).resolve()),
            "sha256": sha256_file(settings.discriminative_checkpoint),
        },
    }
    run_config = {
        "schema_version": 2,
        "method": "DRAEM",
        "execution_scope": settings.execution_scope,
        "run_purpose": run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        "formal_provenance": formal_provenance,
        "evaluation_protocol": evaluation_protocol,
        "settings": settings.to_dict(),
        "checkpoint_provenance": checkpoint_provenance,
        "prediction_contract": {
            "raw_map_dtype": "float32",
            "raw_map_normalization": "none",
            "mask_resize": "nearest",
            "image_scores": image_score_contract(
                "draem_max_after_21x21_average_pool"
            ),
            "metrics_reload_saved_arrays": True,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "created_at_utc": utc_now(),
    }
    if formal_test_evidence is not None:
        run_config["formal_test_inventory"] = formal_test_evidence
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "run_state.json", {"status": "running"})

    reconstructive, discriminative = build_models(device)
    load_state_dict_safely(settings.reconstructive_checkpoint, reconstructive)
    load_state_dict_safely(settings.discriminative_checkpoint, discriminative)
    reconstructive.eval()
    discriminative.eval()

    rows: list[dict[str, Any]] = []
    sample_index = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                reconstruction = reconstructive(images)
                logits = discriminative(torch.cat((reconstruction, images), dim=1))
                anomaly_maps = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                masks = batch["mask"][:, 0].cpu().numpy().astype(bool)
                labels = batch["label"].cpu().numpy().astype(np.int64)

                for within_batch in range(anomaly_maps.shape[0]):
                    raw_path = raw_dir / f"{sample_index:06d}.npy"
                    mask_path = mask_dir / f"{sample_index:06d}.npy"
                    raw_map = anomaly_maps[within_batch].astype(np.float32, copy=False)
                    evaluation_mask = masks[within_batch]
                    _atomic_save_npy(raw_path, raw_map)
                    _atomic_save_npy(mask_path, evaluation_mask)
                    row = {
                        "index": sample_index,
                        "sample_id": batch["sample_id"][within_batch],
                        "label": int(labels[within_batch]),
                        "defect_type": batch["defect_type"][within_batch],
                        "input_image": _relative_to_root(
                            batch["image_path"][within_batch], dataset_root
                        ),
                        "source_ground_truth": _relative_to_root(
                            batch["mask_path"][within_batch], dataset_root
                        ),
                        "raw_map": raw_path.relative_to(output_dir).as_posix(),
                        "raw_map_sha256": sha256_file(raw_path),
                        "evaluation_mask": mask_path.relative_to(output_dir).as_posix(),
                        "evaluation_mask_sha256": sha256_file(mask_path),
                        "original_height": int(batch["original_height"][within_batch]),
                        "original_width": int(batch["original_width"][within_batch]),
                    }
                    if formal_test_selection is not None:
                        entry = formal_test_selection["entries"][sample_index]
                        row.update(
                            {
                                "source_image_sha256": entry["source_image_sha256"],
                                "source_ground_truth_sha256": entry[
                                    "source_mask_sha256"
                                ],
                                "formal_test_inventory_id": formal_test_evidence[
                                    "inventory_id"
                                ],
                                "formal_test_inventory_sha256": formal_test_evidence[
                                    "inventory_sha256"
                                ],
                                "formal_test_inventory_receipt_sha256": formal_test_evidence[
                                    "receipt_sha256"
                                ],
                            }
                        )
                    rows.append(row)
                    sample_index += 1

        write_jsonl(output_dir / "predictions" / "index.jsonl", rows)

        # Reload exactly the persisted arrays.  Metrics and later figures thereby
        # share one prediction source rather than parallel in-memory values.
        saved_maps = np.stack(
            [np.load(output_dir / row["raw_map"], allow_pickle=False) for row in rows]
        )
        saved_masks = np.stack(
            [
                np.load(output_dir / row["evaluation_mask"], allow_pickle=False)
                for row in rows
            ]
        )
        saved_labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        raw_map_max_scores = raw_map_max_image_scores(saved_maps)
        method_native_scores = draem_image_scores(saved_maps)
        for row, common_score, native_score in zip(
            rows, raw_map_max_scores, method_native_scores, strict=True
        ):
            row["raw_map_max_score"] = float(common_score)
            row["method_native_image_score"] = float(native_score)
        if formal_test_selection is not None:
            validate_formal_prediction_rows(rows, formal_test_selection)
        write_jsonl(output_dir / "predictions" / "index.jsonl", rows)

        metrics = evaluate_category(saved_maps, saved_masks, saved_labels).to_dict()
        metrics["image_auroc_method_native"] = image_auroc_from_scores(
            saved_labels, method_native_scores
        )
        metrics_payload = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "dataset": settings.dataset,
            "category": settings.category,
            "num_images": len(rows),
            "protocol": evaluation_protocol,
            "image_score_contract": image_score_contract(
                "draem_max_after_21x21_average_pool"
            ),
            "metrics": metrics,
            "computed_from": "predictions/index.jsonl raw_map and evaluation_mask files",
            "computed_at_utc": utc_now(),
        }
        if formal_test_evidence is not None:
            metrics_payload["formal_test_inventory"] = formal_test_evidence
        write_json(output_dir / "metrics.json", metrics_payload)
        final_state = {
            "status": "completed",
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "formal_matrix_eligible": purpose["formal_matrix_eligible"],
            "num_images": len(rows),
            "run_config_sha256": sha256_file(output_dir / "run_config.json"),
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
            "predictions_index_sha256": sha256_file(
                output_dir / "predictions" / "index.jsonl"
            ),
            "updated_at_utc": utc_now(),
        }
        if formal_test_evidence is not None:
            final_state["formal_test_inventory"] = formal_test_evidence
        write_json(output_dir / "run_state.json", final_state)
        return {**final_state, "metrics": metrics}
    except BaseException as error:
        write_json(
            output_dir / "run_state.json",
            {
                "status": "failed",
                "num_images_written": len(rows),
                "updated_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
