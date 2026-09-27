"""Project-layer DRAEM training entry point.

The implementation leaves the author repository untouched and records every
run below a new, initially empty output directory.  It intentionally keeps the
author loss, initialization, BGR input convention, and random-resampling data
behavior.  ``max_steps`` exists only for smoke tests and marks the run as
ineligible for performance claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.utils.data import DataLoader

from .data_protocols import DraemSyntheticTrainDataset
from .draem_runtime import (
    DeviceSafeSSIM,
    build_focal_loss,
    build_models,
    configure_reproducibility,
    initialize_author_weights,
    make_generator,
    seed_dataloader_worker,
)
from .run_artifacts import (
    append_jsonl,
    prepare_output_directory,
    save_state_dict,
    utc_now,
    write_json,
)


@dataclass(frozen=True)
class TrainSettings:
    dataset: str
    category: str
    dataset_root: Path
    anomaly_source_root: Path
    output_dir: Path
    seed: int = 42
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 700
    learning_rate: float = 1e-4
    workers: int = 0
    device: str = "cpu"
    max_steps: int | None = None

    def validate(self) -> None:
        if self.dataset not in {"mvtec_ad", "visa"}:
            raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
        if not self.category:
            raise ValueError("category must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.image_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size, batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("dataset_root", "anomaly_source_root", "output_dir"):
            payload[key] = str(payload[key])
        return payload


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def inspect_training_inputs(settings: TrainSettings) -> dict[str, Any]:
    """Validate inputs without allocating a model or starting optimization."""

    settings.validate()
    dataset = DraemSyntheticTrainDataset(
        settings.dataset_root,
        settings.category,
        settings.anomaly_source_root,
        image_size=settings.image_size,
    )
    return {
        "dataset": settings.dataset,
        "category": settings.category,
        "normal_training_images": len(dataset.image_paths),
        "anomaly_source_images": len(dataset.anomaly_source_paths),
        "image_size": settings.image_size,
        "validated_at_utc": utc_now(),
        "training_started": False,
    }


def run_training(settings: TrainSettings) -> dict[str, Any]:
    """Run one category and save tensor-only final checkpoints."""

    settings.validate()
    configure_reproducibility(settings.seed, deterministic=True)
    device = _resolve_device(settings.device)
    output_dir = prepare_output_directory(settings.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    logs_path = output_dir / "train_metrics.jsonl"

    dataset = DraemSyntheticTrainDataset(
        settings.dataset_root,
        settings.category,
        settings.anomaly_source_root,
        image_size=settings.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.workers,
        worker_init_fn=seed_dataloader_worker,
        generator=make_generator(settings.seed),
        pin_memory=device.type == "cuda",
        persistent_workers=settings.workers > 0,
    )

    execution_scope = "debug_limited" if settings.max_steps is not None else "full"
    run_config = {
        "schema_version": 1,
        "method": "DRAEM",
        "implementation": "project_adapter_over_author_official",
        "evaluation_protocol": "draem_256_unified_v1",
        "execution_scope": execution_scope,
        "standalone_performance_claim_eligible": False,
        "claim_boundary": (
            "A complete multi-category, multi-seed frozen-protocol matrix is "
            "required before performance claims."
        ),
        "settings": settings.to_dict(),
        "input_inventory": {
            "normal_training_images": len(dataset.image_paths),
            "anomaly_source_images": len(dataset.anomaly_source_paths),
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
    write_json(
        output_dir / "run_state.json",
        {"status": "running", "updated_at_utc": utc_now(), "global_step": 0},
    )

    reconstructive, discriminative = build_models(device)
    reconstructive.apply(initialize_author_weights)
    discriminative.apply(initialize_author_weights)
    reconstructive.train()
    discriminative.train()

    optimizer = torch.optim.Adam(
        [
            {"params": reconstructive.parameters(), "lr": settings.learning_rate},
            {"params": discriminative.parameters(), "lr": settings.learning_rate},
        ]
    )
    milestones = sorted(
        {max(1, int(settings.epochs * 0.8)), max(1, int(settings.epochs * 0.9))}
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = DeviceSafeSSIM().to(device)
    focal_loss = build_focal_loss().to(device)

    global_step = 0
    last_epoch = 0
    try:
        stop = False
        for epoch in range(1, settings.epochs + 1):
            last_epoch = epoch
            for batch in loader:
                target = batch["image"].to(device, non_blocking=True)
                augmented = batch["augmented_image"].to(device, non_blocking=True)
                target_mask = batch["anomaly_mask"].to(device, non_blocking=True)

                reconstruction = reconstructive(augmented)
                logits = discriminative(torch.cat((reconstruction, augmented), dim=1))
                probabilities = torch.softmax(logits, dim=1)
                loss_l2 = mse_loss(reconstruction, target)
                loss_ssim = ssim_loss(reconstruction, target)
                loss_segmentation = focal_loss(probabilities, target_mask)
                total_loss = loss_l2 + loss_ssim + loss_segmentation

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                global_step += 1

                append_jsonl(
                    logs_path,
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "loss_total": float(total_loss.detach().cpu()),
                        "loss_l2": float(loss_l2.detach().cpu()),
                        "loss_ssim": float(loss_ssim.detach().cpu()),
                        "loss_segmentation": float(loss_segmentation.detach().cpu()),
                    },
                )
                if settings.max_steps is not None and global_step >= settings.max_steps:
                    stop = True
                    break
            scheduler.step()
            if stop:
                break

        reconstructive_info = save_state_dict(
            checkpoints_dir / "reconstructive.pt", reconstructive
        )
        discriminative_info = save_state_dict(
            checkpoints_dir / "discriminative.pt", discriminative
        )
        checkpoint_manifest = {
            "schema_version": 1,
            "epoch": last_epoch,
            "global_step": global_step,
            "reconstructive": reconstructive_info,
            "discriminative": discriminative_info,
            "saved_at_utc": utc_now(),
        }
        write_json(checkpoints_dir / "manifest.json", checkpoint_manifest)
        final_state = {
            "status": "debug_completed" if settings.max_steps is not None else "completed",
            "standalone_performance_claim_eligible": False,
            "updated_at_utc": utc_now(),
            "epoch": last_epoch,
            "global_step": global_step,
        }
        write_json(output_dir / "run_state.json", final_state)
        return final_state
    except BaseException as error:
        write_json(
            output_dir / "run_state.json",
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "epoch": last_epoch,
                "global_step": global_step,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
