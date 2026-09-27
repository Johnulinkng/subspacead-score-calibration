"""Deterministic DRAEM inference with raw prediction artifacts."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from .metrics import draem_image_scores, evaluate_category
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
    seed: int = 42
    image_size: int = 256
    batch_size: int = 8
    workers: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.dataset not in {"mvtec_ad", "visa"}:
            raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
        if not self.category:
            raise ValueError("category must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
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
        ):
            payload[key] = str(payload[key])
        return payload


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
    configure_reproducibility(settings.seed, deterministic=True)
    device = _resolve_device(settings.device)
    output_dir = prepare_output_directory(settings.output_dir)
    raw_dir = output_dir / "predictions" / "raw_maps"
    mask_dir = output_dir / "predictions" / "evaluation_masks"
    dataset_root = Path(settings.dataset_root).resolve()

    dataset = DraemTestDataset(
        settings.dataset,
        dataset_root,
        settings.category,
        image_size=settings.image_size,
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
        "schema_version": 1,
        "method": "DRAEM",
        "evaluation_protocol": "draem_256_unified_v1",
        "settings": settings.to_dict(),
        "checkpoint_provenance": checkpoint_provenance,
        "prediction_contract": {
            "raw_map_dtype": "float32",
            "raw_map_normalization": "none",
            "mask_resize": "nearest",
            "image_score": "max_after_21x21_average_pool",
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
                    rows.append(
                        {
                            "index": sample_index,
                            "sample_id": batch["sample_id"][within_batch],
                            "label": int(labels[within_batch]),
                            "defect_type": batch["sample_id"][within_batch].split("/")[1],
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
                            "original_height": int(
                                batch["original_height"][within_batch]
                            ),
                            "original_width": int(batch["original_width"][within_batch]),
                        }
                    )
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
        image_scores = draem_image_scores(saved_maps)
        for row, image_score in zip(rows, image_scores, strict=True):
            row["image_score"] = float(image_score)
        write_jsonl(output_dir / "predictions" / "index.jsonl", rows)

        metrics = evaluate_category(
            saved_maps,
            saved_masks,
            saved_labels,
            image_scores=image_scores,
        ).to_dict()
        metrics_payload = {
            "schema_version": 1,
            "dataset": settings.dataset,
            "category": settings.category,
            "num_images": len(rows),
            "protocol": "draem_256_unified_v1",
            "metrics": metrics,
            "computed_from": "predictions/index.jsonl raw_map and evaluation_mask files",
            "computed_at_utc": utc_now(),
        }
        write_json(output_dir / "metrics.json", metrics_payload)
        final_state = {
            "status": "completed",
            "num_images": len(rows),
            "updated_at_utc": utc_now(),
        }
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
