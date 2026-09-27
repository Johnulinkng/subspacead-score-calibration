"""DRAEM validation, render audit and evaluation on the frozen synthetic dev set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .draem_infer import _atomic_save_npy, _resolve_device
from .draem_runtime import (
    build_models,
    configure_reproducibility,
    load_state_dict_safely,
    make_generator,
    seed_dataloader_worker,
)
from .metrics import (
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
from .synthetic_development import (
    SyntheticDevelopmentDataset,
    render_synthetic_recipe,
    resolve_synthetic_development_inputs,
)


@dataclass(frozen=True)
class SyntheticDevSettings:
    category: str
    dataset_root: Path
    anomaly_source_root: Path
    protocol_config_path: Path
    seed: int

    def validate(self) -> None:
        if not self.category:
            raise ValueError("category must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("dataset_root", "anomaly_source_root", "protocol_config_path"):
            payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class SyntheticDevEvaluationSettings(SyntheticDevSettings):
    reconstructive_checkpoint: Path
    discriminative_checkpoint: Path
    output_dir: Path
    batch_size: int = 8
    workers: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        super().validate()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        for checkpoint in (
            self.reconstructive_checkpoint,
            self.discriminative_checkpoint,
        ):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "dataset_root",
            "anomaly_source_root",
            "protocol_config_path",
            "reconstructive_checkpoint",
            "discriminative_checkpoint",
            "output_dir",
        ):
            payload[key] = str(payload[key])
        return payload


def inspect_synthetic_development(settings: SyntheticDevSettings) -> dict[str, Any]:
    """Validate the frozen bundle and every selected dev file without rendering."""

    settings.validate()
    inputs = resolve_synthetic_development_inputs(
        category=settings.category,
        dataset_root=settings.dataset_root,
        anomaly_source_root=settings.anomaly_source_root,
        seed=settings.seed,
        protocol_config_path=settings.protocol_config_path,
    )
    return {
        "status": "validated",
        **inputs.evidence,
        "images_rendered": 0,
        "model_loaded": False,
    }


def run_render_integrity_audit(
    settings: SyntheticDevSettings,
    *,
    output_dir: Path,
    count: int,
) -> dict[str, Any]:
    """Render a bounded recipe prefix twice and persist equality/hash evidence."""

    settings.validate()
    if count <= 0:
        raise ValueError("render audit count must be positive")
    inputs = resolve_synthetic_development_inputs(
        category=settings.category,
        dataset_root=settings.dataset_root,
        anomaly_source_root=settings.anomaly_source_root,
        seed=settings.seed,
        protocol_config_path=settings.protocol_config_path,
    )
    if count > len(inputs.recipes):
        raise ValueError("render audit count exceeds the frozen recipe count")
    destination = prepare_output_directory(output_dir)
    input_dir = destination / "rendered_inputs"
    mask_dir = destination / "rendered_masks"
    rows: list[dict[str, Any]] = []
    write_json(
        destination / "run_config.json",
        {
            "schema_version": 1,
            "mode": "limited-render-integrity-audit",
            "evaluation_scope": "synthetic-development-only",
            "claim_eligible": False,
            "settings": settings.to_dict(),
            "requested_recipe_count": count,
            "input_evidence": inputs.evidence,
            "formal_test_content_accessed": False,
            "ground_truth_content_accessed": False,
            "training_started": False,
            "performance_evaluated": False,
            "created_at_utc": utc_now(),
        },
    )
    for index, recipe in enumerate(inputs.recipes[:count]):
        first_image, first_mask = render_synthetic_recipe(recipe)
        second_image, second_mask = render_synthetic_recipe(recipe)
        if not np.array_equal(first_image, second_image) or not np.array_equal(
            first_mask, second_mask
        ):
            raise RuntimeError(f"recipe is not byte-deterministic: {recipe.row['sample_id']}")
        image_path = input_dir / f"{index:06d}.npy"
        mask_path = mask_dir / f"{index:06d}.npy"
        _atomic_save_npy(image_path, first_image.astype(np.float32, copy=False))
        _atomic_save_npy(mask_path, first_mask.astype(bool, copy=False))
        rows.append(
            {
                "index": index,
                "sample_id": recipe.row["sample_id"],
                "recipe_seed": recipe.row["recipe_seed"],
                "target_relative_path": recipe.target.relative_path,
                "dtd_relative_path": recipe.dtd_source.relative_path,
                "rendered_input": image_path.relative_to(destination).as_posix(),
                "rendered_input_sha256": sha256_file(image_path),
                "mask": mask_path.relative_to(destination).as_posix(),
                "mask_sha256": sha256_file(mask_path),
                "mask_foreground_pixels": int(first_mask.sum()),
                "repeat_render_equal": True,
            }
        )
    write_jsonl(destination / "index.jsonl", rows)
    result = {
        "status": "completed",
        "mode": "limited-render-integrity-audit",
        "evaluation_scope": "synthetic-development-only",
        "claim_eligible": False,
        "recipes_rendered": len(rows),
        "all_repeat_renders_equal": True,
        "performance_evaluated": False,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "training_started": False,
        "updated_at_utc": utc_now(),
    }
    write_json(destination / "result.json", result)
    return result


def _normal_false_positive_metrics(
    maps: np.ndarray,
    labels: np.ndarray,
    image_scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    normal = labels == 0
    if not bool(normal.any()):
        raise ValueError("normal false-positive metrics require normal dev samples")
    normal_scores = image_scores[normal]
    normal_maps = maps[normal]
    return {
        "threshold": float(threshold),
        "normal_image_false_positive_rate": float(np.mean(normal_scores >= threshold)),
        "normal_pixel_false_positive_rate": float(np.mean(normal_maps >= threshold)),
        "normal_image_score_mean": float(np.mean(normal_scores)),
        "normal_image_score_p95": float(np.quantile(normal_scores, 0.95)),
        "normal_image_score_max": float(np.max(normal_scores)),
    }


def run_synthetic_development_evaluation(
    settings: SyntheticDevEvaluationSettings,
) -> dict[str, Any]:
    """Evaluate every normal/positive dev sample and reload persisted raw arrays."""

    settings.validate()
    inputs = resolve_synthetic_development_inputs(
        category=settings.category,
        dataset_root=settings.dataset_root,
        anomaly_source_root=settings.anomaly_source_root,
        seed=settings.seed,
        protocol_config_path=settings.protocol_config_path,
    )
    false_positive_threshold = float(
        inputs.evidence["normal_false_positive_threshold"]
    )
    if not 0.0 <= false_positive_threshold <= 1.0:
        raise ValueError("frozen normal false-positive threshold must be in [0, 1]")
    configure_reproducibility(settings.seed, deterministic=True)
    device = _resolve_device(settings.device)
    destination = prepare_output_directory(settings.output_dir)
    raw_dir = destination / "predictions" / "raw_maps"
    mask_dir = destination / "predictions" / "evaluation_masks"
    dataset = SyntheticDevelopmentDataset(inputs)
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
    provenance = {
        "reconstructive": {
            "path": str(settings.reconstructive_checkpoint.resolve()),
            "sha256": sha256_file(settings.reconstructive_checkpoint),
            "size_bytes": settings.reconstructive_checkpoint.stat().st_size,
        },
        "discriminative": {
            "path": str(settings.discriminative_checkpoint.resolve()),
            "sha256": sha256_file(settings.discriminative_checkpoint),
            "size_bytes": settings.discriminative_checkpoint.stat().st_size,
        },
    }
    run_config = {
        "schema_version": 2,
        "method": "DRAEM",
        "mode": "frozen-synthetic-development-evaluation",
        "evaluation_scope": "synthetic-development-only",
        "claim_eligible": False,
        "settings": settings.to_dict(),
        "image_size": inputs.image_size,
        "input_evidence": inputs.evidence,
        "checkpoint_provenance": provenance,
        "prediction_contract": {
            "raw_map_dtype": "float32",
            "raw_map_normalization": "none",
            "mask_dtype": "bool",
            "image_scores": image_score_contract(
                "draem_max_after_21x21_average_pool"
            ),
            "metrics_reload_saved_arrays": True,
            "normal_samples_included": True,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "training_started": False,
        "performance_evaluated": True,
        "created_at_utc": utc_now(),
    }
    write_json(destination / "run_config.json", run_config)
    write_json(destination / "run_state.json", {"status": "running"})

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
                    evaluation_mask = masks[within_batch].astype(bool, copy=False)
                    _atomic_save_npy(raw_path, raw_map)
                    _atomic_save_npy(mask_path, evaluation_mask)
                    rows.append(
                        {
                            "index": sample_index,
                            "sample_id": batch["sample_id"][within_batch],
                            "label": int(labels[within_batch]),
                            "sample_kind": "normal_dev" if labels[within_batch] == 0 else "synthetic_positive",
                            "recipe_id": batch["recipe_id"][within_batch] or None,
                            "input_relative_path": batch["input_relative_path"][within_batch],
                            "dtd_relative_path": batch["dtd_relative_path"][within_batch] or None,
                            "raw_map": raw_path.relative_to(destination).as_posix(),
                            "raw_map_sha256": sha256_file(raw_path),
                            "evaluation_mask": mask_path.relative_to(destination).as_posix(),
                            "evaluation_mask_sha256": sha256_file(mask_path),
                        }
                    )
                    sample_index += 1
        write_jsonl(destination / "predictions" / "index.jsonl", rows)
        saved_maps = np.stack(
            [np.load(destination / row["raw_map"], allow_pickle=False) for row in rows]
        )
        saved_masks = np.stack(
            [np.load(destination / row["evaluation_mask"], allow_pickle=False) for row in rows]
        )
        saved_labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        if saved_maps.dtype != np.float32 or saved_masks.dtype != np.bool_:
            raise RuntimeError("persisted prediction dtype contract was violated")
        raw_map_max_scores = raw_map_max_image_scores(saved_maps)
        method_native_scores = draem_image_scores(saved_maps)
        for row, common_score, native_score in zip(
            rows, raw_map_max_scores, method_native_scores, strict=True
        ):
            row["raw_map_max_score"] = float(common_score)
            row["method_native_image_score"] = float(native_score)
        write_jsonl(destination / "predictions" / "index.jsonl", rows)
        common_metrics = evaluate_category(saved_maps, saved_masks, saved_labels).to_dict()
        common_metrics["image_auroc_method_native"] = image_auroc_from_scores(
            saved_labels, method_native_scores
        )
        false_positive_metrics = _normal_false_positive_metrics(
            saved_maps,
            saved_labels,
            raw_map_max_scores,
            threshold=false_positive_threshold,
        )
        method_native_false_positive_metrics = _normal_false_positive_metrics(
            saved_maps,
            saved_labels,
            method_native_scores,
            threshold=false_positive_threshold,
        )
        metrics_payload = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "dataset": "visa_frozen_train_dev_synthetic",
            "category": settings.category,
            "evaluation_scope": "synthetic-development-only",
            "claim_eligible": False,
            "num_images": len(rows),
            "num_normal_dev": int(np.sum(saved_labels == 0)),
            "num_synthetic_positive": int(np.sum(saved_labels == 1)),
            "image_score_contract": image_score_contract(
                "draem_max_after_21x21_average_pool"
            ),
            "metrics": common_metrics,
            "normal_false_positive_metrics": false_positive_metrics,
            "method_native_normal_false_positive_metrics": (
                method_native_false_positive_metrics
            ),
            "computed_from": "predictions/index.jsonl raw_map and evaluation_mask files",
            "formal_test_content_accessed": False,
            "ground_truth_content_accessed": False,
            "performance_evaluated": True,
            "computed_at_utc": utc_now(),
        }
        write_json(destination / "metrics.json", metrics_payload)
        result = {
            "status": "completed",
            "evaluation_scope": "synthetic-development-only",
            "claim_eligible": False,
            "num_images": len(rows),
            "num_normal_dev": int(np.sum(saved_labels == 0)),
            "num_synthetic_positive": int(np.sum(saved_labels == 1)),
            "metrics": common_metrics,
            "normal_false_positive_metrics": false_positive_metrics,
            "method_native_normal_false_positive_metrics": (
                method_native_false_positive_metrics
            ),
            "formal_test_content_accessed": False,
            "ground_truth_content_accessed": False,
            "training_started": False,
            "performance_evaluated": True,
            "updated_at_utc": utc_now(),
        }
        write_json(destination / "run_state.json", result)
        return result
    except BaseException as error:
        write_json(
            destination / "run_state.json",
            {
                "status": "failed",
                "evaluation_scope": "synthetic-development-only",
                "claim_eligible": False,
                "num_images_written": len(rows),
                "formal_test_content_accessed": False,
                "ground_truth_content_accessed": False,
                "training_started": False,
                "performance_evaluated": False,
                "updated_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
