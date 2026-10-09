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
from .draem_checkpointing import enable_activation_checkpointing
from .draem_runtime import (
    DeviceSafeSSIM,
    build_focal_loss,
    build_models,
    configure_reproducibility,
    initialize_author_weights,
    make_generator,
    seed_dataloader_worker,
)
from .formal_provenance import (
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_REGISTRATION,
    RunPurpose,
    classify_run_purpose,
    validate_development_authority,
    validate_formal_authority,
)
from .formal_source_provenance import capture_formal_source_provenance
from .formal_training_inputs import (
    DEFAULT_FORMAL_INPUT_MANIFEST,
    PROJECT_ROOT as FORMAL_PROJECT_ROOT,
    resolve_frozen_formal_training_inputs,
)
from .metrics import EVALUATION_PROTOCOL
from .run_artifacts import (
    append_jsonl,
    prepare_output_directory,
    save_state_dict,
    utc_now,
    write_json,
)
from .training_inputs import (
    AUTHOR_FULL_TRAIN,
    SUPPORTED_INPUT_PROTOCOLS,
    VISA_DEVELOPMENT_PROTOCOL,
    resolve_manifest_training_inputs,
)


DRAEM_AUTHOR_IMAGE_SIZE = 256
DRAEM_AUTHOR_BATCH_SIZE = 8
DRAEM_AUTHOR_EPOCHS = 700
DRAEM_AUTHOR_LEARNING_RATE = 1e-4


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
    pin_memory: bool | None = None
    activation_checkpointing: bool = False
    device: str = "cpu"
    max_steps: int | None = None
    input_protocol: str = AUTHOR_FULL_TRAIN
    development_protocol_config: Path | None = None
    formal_input_manifest: Path = DEFAULT_FORMAL_INPUT_MANIFEST
    run_purpose: RunPurpose = "author_reproduction"
    project_config_path: Path = DEFAULT_PROJECT_CONFIG
    registration_path: Path = DEFAULT_REGISTRATION

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
        if self.pin_memory is not None and type(self.pin_memory) is not bool:
            raise ValueError("pin_memory must be true, false or null")
        if type(self.activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be true or false")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        classify_run_purpose(self.run_purpose, max_steps=self.max_steps)
        if self.input_protocol not in SUPPORTED_INPUT_PROTOCOLS:
            raise ValueError(f"unsupported input_protocol: {self.input_protocol}")
        if self.input_protocol == VISA_DEVELOPMENT_PROTOCOL:
            if self.dataset != "visa":
                raise ValueError("the frozen development protocol supports only VisA")
            if self.development_protocol_config is None:
                raise ValueError(
                    "development_protocol_config is required for a development run"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "dataset_root",
            "anomaly_source_root",
            "output_dir",
            "project_config_path",
            "registration_path",
            "formal_input_manifest",
        ):
            payload[key] = str(payload[key])
        payload["development_protocol_config"] = (
            None
            if self.development_protocol_config is None
            else str(self.development_protocol_config)
        )
        return payload


def draem_training_recipe(settings: TrainSettings) -> dict[str, object]:
    """Canonical settings that must exactly match the frozen F0 recipe."""

    milestones = sorted(
        {max(1, int(settings.epochs * 0.8)), max(1, int(settings.epochs * 0.9))}
    )
    return {
        "schema_version": 1,
        "method": "DRAEM",
        "implementation": "project_adapter_over_author_official",
        "input_protocol": settings.input_protocol,
        "image_size": settings.image_size,
        "batch_size": settings.batch_size,
        "epochs": settings.epochs,
        "learning_rate": settings.learning_rate,
        "activation_checkpointing": settings.activation_checkpointing,
        "optimizer": "Adam",
        "scheduler": {
            "name": "MultiStepLR",
            "milestones": milestones,
            "gamma": 0.2,
        },
        "loss": "MSE_plus_SSIM_plus_Focal",
        "checkpoint_rule": "final_epoch_only_no_test_selection",
    }


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _resolve_pin_memory(requested: bool | None, device: torch.device) -> bool:
    """Resolve the transport-only pinned-memory policy for the run record."""

    if requested is None:
        return device.type == "cuda"
    return requested


def _resolve_training_dataset(
    settings: TrainSettings,
) -> tuple[DraemSyntheticTrainDataset, dict[str, Any]]:
    if settings.input_protocol == VISA_DEVELOPMENT_PROTOCOL:
        assert settings.development_protocol_config is not None
        inputs = resolve_manifest_training_inputs(
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=settings.dataset_root,
            anomaly_source_root=settings.anomaly_source_root,
            seed=settings.seed,
            protocol_config_path=settings.development_protocol_config,
        )
        dataset = DraemSyntheticTrainDataset(
            settings.dataset_root,
            settings.category,
            settings.anomaly_source_root,
            image_size=settings.image_size,
            normal_image_paths=inputs.normal_image_paths,
            anomaly_source_paths=inputs.anomaly_source_paths,
        )
        return dataset, inputs.evidence

    if settings.run_purpose == "formal_full":
        inputs = resolve_frozen_formal_training_inputs(
            method="DRAEM",
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=settings.dataset_root,
            dtd_root=settings.anomaly_source_root,
            bundle_manifest_path=settings.formal_input_manifest,
            project_config_path=settings.project_config_path,
            project_root=FORMAL_PROJECT_ROOT,
        )
        dataset = DraemSyntheticTrainDataset(
            settings.dataset_root,
            settings.category,
            settings.anomaly_source_root,
            image_size=settings.image_size,
            normal_image_paths=inputs.normal_image_paths,
            anomaly_source_paths=inputs.anomaly_source_paths,
        )
        return dataset, inputs.evidence

    dataset = DraemSyntheticTrainDataset(
        settings.dataset_root,
        settings.category,
        settings.anomaly_source_root,
        image_size=settings.image_size,
    )
    evidence = {
        "input_protocol": AUTHOR_FULL_TRAIN,
        "manifest_backed": False,
        "development_protocol_eligible": False,
        "normal_training_images": len(dataset.image_paths),
        "anomaly_source_images": len(dataset.anomaly_source_paths),
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "training_started": False,
    }
    return dataset, evidence


def inspect_training_inputs(settings: TrainSettings) -> dict[str, Any]:
    """Validate inputs without allocating a model or starting optimization."""

    settings.validate()
    dataset, input_evidence = _resolve_training_dataset(settings)
    return {
        "dataset": settings.dataset,
        "category": settings.category,
        "normal_training_images": len(dataset.image_paths),
        "anomaly_source_images": len(dataset.anomaly_source_paths),
        "image_size": settings.image_size,
        "input_protocol": settings.input_protocol,
        "input_evidence": input_evidence,
        "validated_at_utc": utc_now(),
        "training_started": False,
    }


def _require_finite_losses(
    losses: dict[str, torch.Tensor], *, epoch: int, step: int
) -> None:
    """Stop before backward when any named DRAEM loss is non-finite."""

    nonfinite = [
        name
        for name, value in losses.items()
        if not bool(torch.isfinite(value.detach()).all().item())
    ]
    if nonfinite:
        names = ", ".join(nonfinite)
        raise FloatingPointError(
            f"nonfinite DRAEM loss at epoch={epoch}, step={step}: {names}"
        )


def _finite_gradient_norm(
    parameters: list[torch.nn.Parameter], *, epoch: int, step: int
) -> torch.Tensor:
    """Return the global gradient norm while failing on NaN or infinity."""

    try:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        raise FloatingPointError(
            f"nonfinite DRAEM gradients at epoch={epoch}, step={step}: {error}"
        ) from error
    if not bool(torch.isfinite(gradient_norm.detach()).all().item()):
        raise FloatingPointError(
            f"nonfinite DRAEM gradient norm at epoch={epoch}, step={step}"
        )
    return gradient_norm


def _remove_incomplete_checkpoint_artifacts(checkpoints_dir: Path) -> list[str]:
    """Remove final or temporary checkpoint artifacts after any failed run."""

    paths = (
        checkpoints_dir / "manifest.json",
        checkpoints_dir / "reconstructive.pt",
        checkpoints_dir / "discriminative.pt",
        checkpoints_dir / ".manifest.json.tmp",
        checkpoints_dir / ".reconstructive.pt.tmp",
        checkpoints_dir / ".discriminative.pt.tmp",
    )
    errors: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    return errors


def run_training(settings: TrainSettings) -> dict[str, Any]:
    """Run one category and save tensor-only final checkpoints."""

    settings.validate()
    purpose = classify_run_purpose(settings.run_purpose, max_steps=settings.max_steps)
    dataset: DraemSyntheticTrainDataset | None = None
    input_evidence: dict[str, Any] | None = None
    source_provenance: dict[str, object] | None = None
    if settings.run_purpose == "development_full":
        if settings.input_protocol != VISA_DEVELOPMENT_PROTOCOL:
            raise ValueError(
                "development_full DRAEM requires visa_train_dev_dtd_v1"
            )
        if settings.development_protocol_config is None:
            raise ValueError("development_full requires development_protocol_config")
        expected_development_recipe = {
            "image_size": DRAEM_AUTHOR_IMAGE_SIZE,
            "batch_size": DRAEM_AUTHOR_BATCH_SIZE,
            "epochs": DRAEM_AUTHOR_EPOCHS,
            "learning_rate": DRAEM_AUTHOR_LEARNING_RATE,
        }
        observed_development_recipe = {
            "image_size": settings.image_size,
            "batch_size": settings.batch_size,
            "epochs": settings.epochs,
            "learning_rate": settings.learning_rate,
        }
        if observed_development_recipe != expected_development_recipe:
            raise ValueError(
                "development_full DRAEM must use the frozen 700-epoch baseline "
                f"recipe: expected={expected_development_recipe}, "
                f"observed={observed_development_recipe}"
            )
        authority: dict[str, object] | None = validate_development_authority(
            method="DRAEM",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
            development_protocol_path=settings.development_protocol_config,
        )
    elif settings.run_purpose == "formal_full":
        if settings.input_protocol != AUTHOR_FULL_TRAIN:
            raise ValueError("formal_full DRAEM requires author_full_train")
        dataset, input_evidence = _resolve_training_dataset(settings)
        authority = validate_formal_authority(
            method="DRAEM",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            phase="training",
            recipe=draem_training_recipe(settings),
            training_input_evidence=input_evidence,
            formal_input_project_root=FORMAL_PROJECT_ROOT,
            project_config_path=settings.project_config_path,
            registration_path=settings.registration_path,
        )
        config_path = Path(settings.project_config_path).resolve()
        if config_path.parent.name != "configs":
            raise ValueError("formal project config must be stored below <project>/configs")
        source_provenance = capture_formal_source_provenance(
            project_root=config_path.parent.parent,
            scope_path=config_path.parent / "formal_source_provenance.json",
        )
    else:
        authority = None
    if dataset is None or input_evidence is None:
        dataset, input_evidence = _resolve_training_dataset(settings)
    configure_reproducibility(settings.seed, deterministic=True)
    device = _resolve_device(settings.device)
    resolved_pin_memory = _resolve_pin_memory(settings.pin_memory, device)
    cuda_allocator_cap_bytes: int | None = None
    if device.type == "cuda" and settings.activation_checkpointing:
        free_gpu, total_gpu = torch.cuda.mem_get_info(device)
        # Bound the cache as well as live tensors, leaving dedicated memory
        # for the desktop and avoiding Windows shared-memory spill.
        cuda_allocator_cap_bytes = min(
            int(6.5 * 1024**3), free_gpu - int(0.75 * 1024**3),
            int(0.90 * total_gpu),
        )
        if cuda_allocator_cap_bytes < 5 * 1024**3:
            raise RuntimeError("Insufficient dedicated VRAM for checkpointed DRAEM")
        torch.cuda.set_per_process_memory_fraction(
            cuda_allocator_cap_bytes / total_gpu, device
        )
    output_dir = prepare_output_directory(settings.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    logs_path = output_dir / "train_metrics.jsonl"

    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.workers,
        worker_init_fn=seed_dataloader_worker,
        generator=make_generator(settings.seed),
        pin_memory=resolved_pin_memory,
        persistent_workers=settings.workers > 0,
    )

    execution_scope = str(purpose["execution_scope"])
    run_config = {
        "schema_version": 1,
        "method": "DRAEM",
        "implementation": "project_adapter_over_author_official",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "execution_scope": execution_scope,
        "run_purpose": settings.run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_matrix_eligible": purpose["formal_matrix_eligible"],
        "formal_provenance": authority,
        "formal_source_provenance": source_provenance,
        "standalone_performance_claim_eligible": False,
        "claim_boundary": (
            "A complete multi-category, multi-seed frozen-protocol matrix is "
            "required before performance claims."
        ),
        "settings": settings.to_dict(),
        "input_evidence": input_evidence,
        "input_inventory": {
            "normal_training_images": len(dataset.image_paths),
            "anomaly_source_images": len(dataset.anomaly_source_paths),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "pin_memory": resolved_pin_memory,
            "activation_checkpointing": settings.activation_checkpointing,
            "cuda_allocator_cap_bytes": cuda_allocator_cap_bytes,
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

    global_step = 0
    last_epoch = 0
    active_step: int | None = None
    try:
        # Resource failures during initialization must also produce a failed
        # state, rather than leaving an apparently running experiment behind.
        reconstructive, discriminative = build_models(device)
        reconstructive.apply(initialize_author_weights)
        discriminative.apply(initialize_author_weights)
        if settings.activation_checkpointing:
            enable_activation_checkpointing(reconstructive)
            enable_activation_checkpointing(discriminative)
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
        trainable_parameters = [
            parameter
            for model in (reconstructive, discriminative)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        stop = False
        for epoch in range(1, settings.epochs + 1):
            last_epoch = epoch
            for batch in loader:
                active_step = global_step + 1
                optimizer.zero_grad(set_to_none=True)
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

                _require_finite_losses(
                    {
                        "loss_l2": loss_l2,
                        "loss_ssim": loss_ssim,
                        "loss_segmentation": loss_segmentation,
                        "loss_total": total_loss,
                    },
                    epoch=epoch,
                    step=active_step,
                )
                total_loss.backward()
                gradient_norm = _finite_gradient_norm(
                    trainable_parameters,
                    epoch=epoch,
                    step=active_step,
                )
                optimizer.step()
                global_step += 1
                active_step = None

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
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                    },
                )
                # These tensors are no longer used after logging. Keeping the
                # previous batch's outputs alive increases the next forward's
                # peak without changing the training calculation.
                del (
                    batch, target, augmented, target_mask, reconstruction,
                    logits, probabilities, loss_l2, loss_ssim,
                    loss_segmentation, total_loss, gradient_norm,
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
            "schema_version": 2,
            "method": "DRAEM",
            "dataset": settings.dataset,
            "category": settings.category,
            "seed": settings.seed,
            "run_purpose": settings.run_purpose,
            "execution_scope": execution_scope,
            "claim_scope": purpose["claim_scope"],
            "checkpoint_origin": "project_training_run",
            "final_checkpoint_only": True,
            "test_selected": False,
            "epoch": last_epoch,
            "global_step": global_step,
            "training_recipe": draem_training_recipe(settings),
            "execution_runtime": run_config["runtime"],
            "formal_provenance": authority,
            "formal_source_provenance": source_provenance,
            "formal_training_inputs": (
                input_evidence if settings.run_purpose == "formal_full" else None
            ),
            "checkpoints": {
                "reconstructive": reconstructive_info,
                "discriminative": discriminative_info,
            },
            "external_hashes": {},
            "saved_at_utc": utc_now(),
        }
        write_json(checkpoints_dir / "manifest.json", checkpoint_manifest)
        final_state = {
            "status": "debug_completed" if settings.max_steps is not None else "completed",
            "run_purpose": settings.run_purpose,
            "claim_scope": purpose["claim_scope"],
            "standalone_performance_claim_eligible": False,
            "updated_at_utc": utc_now(),
            "epoch": last_epoch,
            "global_step": global_step,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        }
        write_json(output_dir / "run_state.json", final_state)
        return final_state
    except BaseException as error:
        cleanup_errors = _remove_incomplete_checkpoint_artifacts(checkpoints_dir)
        write_json(
            output_dir / "run_state.json",
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "epoch": last_epoch,
                "global_step": global_step,
                "failed_step": active_step,
                "error_type": type(error).__name__,
                "error": str(error),
                "checkpoint_cleanup_complete": not cleanup_errors,
                "checkpoint_cleanup_errors": cleanup_errors,
            },
        )
        raise
