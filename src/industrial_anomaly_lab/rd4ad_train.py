"""Fail-closed, project-owned training entry point for frozen RD4AD."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .draem_runtime import (
    configure_reproducibility,
    make_generator,
    seed_dataloader_worker,
)
from .formal_provenance import (
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_REGISTRATION,
    RunPurpose,
    classify_run_purpose,
    validate_formal_authority,
)
from .formal_source_provenance import capture_formal_source_provenance
from .formal_training_inputs import (
    DEFAULT_FORMAL_INPUT_MANIFEST,
    PROJECT_ROOT as FORMAL_PROJECT_ROOT,
    resolve_frozen_formal_training_inputs,
)
from .rd4ad_data import Rd4adNormalTrainDataset
from .rd4ad_runtime import (
    RD4AD_VENDOR_COMMIT,
    audit_vendor_source,
    build_models,
)
from .run_artifacts import (
    append_jsonl,
    prepare_output_directory,
    sha256_file,
    utc_now,
    write_json,
)
from .wrn50_weights import (
    inspect_wrn50_checkpoint,
    load_wrn50_checkpoint_into_model,
)


ExecutionScope = Literal["full", "debug_limited"]
AUTHOR_EPOCHS = 200
AUTHOR_LEARNING_RATE = 0.005
AUTHOR_BETAS = (0.5, 0.999)
AUTHOR_IMAGE_SIZE = 256
AUTHOR_BATCH_SIZE = 16
FORMAL_BATCH_SIZES = frozenset({16, 8, 4})


@dataclass(frozen=True)
class Rd4adTrainSettings:
    dataset: str
    category: str
    dataset_root: Path
    teacher_checkpoint: Path
    teacher_sha256: str
    output_dir: Path
    seed: int = 111
    image_size: int = AUTHOR_IMAGE_SIZE
    batch_size: int = AUTHOR_BATCH_SIZE
    epochs: int = AUTHOR_EPOCHS
    learning_rate: float = AUTHOR_LEARNING_RATE
    workers: int = 0
    device: str = "cpu"
    execution_scope: ExecutionScope = "full"
    max_steps: int | None = None
    run_purpose: RunPurpose | None = None
    project_config_path: Path = DEFAULT_PROJECT_CONFIG
    registration_path: Path = DEFAULT_REGISTRATION
    formal_input_manifest: Path = DEFAULT_FORMAL_INPUT_MANIFEST

    def resolved_run_purpose(self) -> RunPurpose:
        if self.run_purpose is not None:
            return self.run_purpose
        return "debug_limited" if self.execution_scope == "debug_limited" else "author_reproduction"

    def validate(self) -> None:
        if self.dataset not in {"mvtec_ad", "visa"}:
            raise ValueError("dataset must be 'mvtec_ad' or 'visa'")
        if (
            not self.category
            or Path(self.category).name != self.category
            or "/" in self.category
            or "\\" in self.category
        ):
            raise ValueError("category must be one safe directory name")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.image_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size, batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.execution_scope not in {"full", "debug_limited"}:
            raise ValueError("execution_scope must be 'full' or 'debug_limited'")
        purpose = classify_run_purpose(
            self.resolved_run_purpose(), max_steps=self.max_steps
        )
        if purpose["execution_scope"] != self.execution_scope:
            raise ValueError("run_purpose and execution_scope disagree")
        if self.resolved_run_purpose() == "development_full":
            raise ValueError(
                "RD4AD development_full is unavailable until its train/dev "
                "manifest adapter is implemented"
            )
        if len(self.teacher_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in self.teacher_sha256
        ):
            raise ValueError("teacher_sha256 must be a full 64-character SHA-256")
        if not Path(self.teacher_checkpoint).is_file():
            raise FileNotFoundError(self.teacher_checkpoint)

        if self.execution_scope == "full":
            if self.max_steps is not None:
                raise ValueError("max_steps is forbidden for a full RD4AD run")
            author_fields = {
                "image_size": (self.image_size, AUTHOR_IMAGE_SIZE),
                "epochs": (self.epochs, AUTHOR_EPOCHS),
                "learning_rate": (
                    self.learning_rate,
                    AUTHOR_LEARNING_RATE,
                ),
            }
            mismatches = {
                name: {"actual": actual, "required": required}
                for name, (actual, required) in author_fields.items()
                if actual != required
            }
            if mismatches:
                raise ValueError(
                    f"full RD4AD baseline must use author protocol: {mismatches}"
                )
            if self.batch_size not in FORMAL_BATCH_SIZES:
                raise ValueError(
                    "full RD4AD batch_size must be 16, 8 or 4; select and freeze "
                    "one only after the registered resource smoke"
                )
        else:
            if self.max_steps is None or self.max_steps <= 0:
                raise ValueError(
                    "debug_limited requires a positive max_steps; uncapped debug is forbidden"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "dataset_root",
            "teacher_checkpoint",
            "output_dir",
            "project_config_path",
            "registration_path",
            "formal_input_manifest",
        ):
            payload[key] = str(payload[key])
        return payload


def rd4ad_training_recipe(settings: Rd4adTrainSettings) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": "RD4AD",
        "implementation": "project_adapter_over_author_official",
        "architecture": "wide_resnet50_2_encoder_bn_reverse_decoder",
        "image_size": settings.image_size,
        "batch_size": settings.batch_size,
        "epochs": settings.epochs,
        "learning_rate": settings.learning_rate,
        "optimizer": "Adam",
        "betas": list(AUTHOR_BETAS),
        "loss": "sum_per_scale_flattened_cosine_distance",
        "checkpoint_rule": "final_epoch_only_no_test_selection",
    }


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _training_inventory(dataset: Rd4adNormalTrainDataset, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in dataset.image_paths:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"normal training image is empty: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\x00")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "normal_training_images": len(dataset.image_paths),
        "normal_training_file_bytes": total_bytes,
        "relative_path_size_and_content_sha256": digest.hexdigest(),
        "allowed_split": "train/good only",
        "test_or_ground_truth_enumerated": False,
    }


def _inspect_teacher_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    # Shared with PatchCore: this constructs torchvision WRN50-2 with
    # weights=None, performs a hash-gated tensor-only strict load, and never
    # calls a download API.
    return inspect_wrn50_checkpoint(path, expected_sha256=expected_sha256)


def _inspect_training_inputs_with_dataset(
    settings: Rd4adTrainSettings,
) -> tuple[dict[str, Any], Rd4adNormalTrainDataset]:
    """Resolve authenticated train-only inputs and return their inspection."""

    settings.validate()
    root = Path(settings.dataset_root).resolve()
    formal_evidence: dict[str, Any] | None = None
    explicit_paths: Sequence[Path] | None = None
    if settings.resolved_run_purpose() == "formal_full":
        formal_inputs = resolve_frozen_formal_training_inputs(
            method="RD4AD",
            dataset=settings.dataset,
            category=settings.category,
            dataset_root=root,
            bundle_manifest_path=settings.formal_input_manifest,
            project_config_path=settings.project_config_path,
            project_root=FORMAL_PROJECT_ROOT,
        )
        formal_evidence = formal_inputs.evidence
        explicit_paths = formal_inputs.normal_image_paths
    dataset = Rd4adNormalTrainDataset(
        root,
        settings.category,
        image_size=settings.image_size,
        image_paths=explicit_paths,
    )
    report = {
        "schema_version": 1,
        "method": "RD4AD",
        "dataset": settings.dataset,
        "category": settings.category,
        "execution_scope": settings.execution_scope,
        "settings": settings.to_dict(),
        "input_inventory": _training_inventory(dataset, root),
        "formal_training_inputs": formal_evidence,
        "teacher_checkpoint": _inspect_teacher_checkpoint(
            Path(settings.teacher_checkpoint).resolve(), settings.teacher_sha256
        ),
        "vendor_source": audit_vendor_source(),
        "training_started": False,
        "test_split_accessed": False,
        "validated_at_utc": utc_now(),
    }
    return report, dataset


def inspect_training_inputs(settings: Rd4adTrainSettings) -> dict[str, Any]:
    """Validate train-only data, frozen source and teacher without optimization."""

    report, _ = _inspect_training_inputs_with_dataset(settings)
    return report


def rd4ad_cosine_loss(
    teacher_features: Sequence[torch.Tensor],
    student_features: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Author loss: sum of per-scale, per-image flattened cosine distances."""

    if not teacher_features or len(teacher_features) != len(student_features):
        raise ValueError("teacher and student must provide the same non-zero scales")
    total: torch.Tensor | None = None
    for index, (teacher, student) in enumerate(
        zip(teacher_features, student_features, strict=True)
    ):
        if teacher.shape != student.shape or teacher.ndim < 2:
            raise ValueError(
                f"feature scale {index} mismatch: {teacher.shape} != {student.shape}"
            )
        batch = teacher.shape[0]
        scale_loss = torch.mean(
            1.0
            - nn.functional.cosine_similarity(
                teacher.reshape(batch, -1),
                student.reshape(batch, -1),
                dim=1,
            )
        )
        total = scale_loss if total is None else total + scale_loss
    assert total is not None
    return total


def _save_final_checkpoint(
    path: Path,
    bottleneck: nn.Module,
    decoder: nn.Module,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    state = {
        "bn": {
            name: tensor.detach().cpu()
            for name, tensor in bottleneck.state_dict().items()
        },
        "decoder": {
            name: tensor.detach().cpu() for name, tensor in decoder.state_dict().items()
        },
    }
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_only_nested_state_dict": True,
        "keys": ["bn", "decoder"],
    }


def run_training(settings: Rd4adTrainSettings) -> dict[str, Any]:
    """Train one category without reading its test or ground-truth directories."""

    settings.validate()
    run_purpose = settings.resolved_run_purpose()
    purpose = classify_run_purpose(run_purpose, max_steps=settings.max_steps)
    inspection, dataset = _inspect_training_inputs_with_dataset(settings)
    source_provenance: dict[str, object] | None = None
    if run_purpose == "formal_full":
        authority: dict[str, object] | None = validate_formal_authority(
            method="RD4AD",
            dataset=settings.dataset,
            category=settings.category,
            seed=settings.seed,
            phase="training",
            recipe=rd4ad_training_recipe(settings),
            training_input_evidence=inspection["formal_training_inputs"],
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
    configure_reproducibility(settings.seed, deterministic=True)
    device = _resolve_device(settings.device)
    output_dir = prepare_output_directory(settings.output_dir)
    checkpoint_path = output_dir / "checkpoints" / "rd4ad_final.pt"
    log_path = output_dir / "train_steps.jsonl"
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
    complete_protocol = settings.execution_scope == "full"
    run_config = {
        "schema_version": 1,
        "method": "RD4AD",
        "implementation": "project_adapter_over_frozen_author_source",
        "vendor_commit": RD4AD_VENDOR_COMMIT,
        "execution_scope": settings.execution_scope,
        "run_purpose": run_purpose,
        "claim_scope": purpose["claim_scope"],
        "formal_provenance": authority,
        "formal_source_provenance": source_provenance,
        "standalone_performance_claim_eligible": False,
        "formal_run_protocol_eligible": run_purpose == "formal_full",
        "formal_matrix_complete": False,
        "settings": settings.to_dict(),
        "author_protocol": {
            "architecture": "WideResNet50-2 teacher + BN_layer + reverse decoder",
            "image_size": AUTHOR_IMAGE_SIZE,
            "batch_size": AUTHOR_BATCH_SIZE,
            "allowed_resource_smoke_batch_sizes": sorted(
                FORMAL_BATCH_SIZES, reverse=True
            ),
            "epochs": AUTHOR_EPOCHS,
            "learning_rate": AUTHOR_LEARNING_RATE,
            "optimizer": "Adam",
            "betas": list(AUTHOR_BETAS),
            "loss": "sum_per_scale_flattened_cosine_distance",
            "checkpoint_rule": "final_epoch_only_no_test_selection",
        },
        "deliberate_compatibility_changes": [
            "teacher weights are explicit local tensor-only state_dict",
            "teacher parameters are frozen and forward runs under no_grad",
            "author every-10-epoch test evaluation is removed",
            "only the final checkpoint is persisted",
        ],
        "input_inventory": inspection["input_inventory"],
        "formal_training_inputs": inspection["formal_training_inputs"],
        "teacher_checkpoint": inspection["teacher_checkpoint"],
        "vendor_source": inspection["vendor_source"],
        "test_split_accessed": False,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "deterministic_algorithms": True,
            "tf32": False,
            "implicit_network_downloads": False,
        },
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "running",
            "epoch": 0,
            "global_step": 0,
            "updated_at_utc": utc_now(),
        },
    )

    global_step = 0
    completed_epoch = 0
    try:
        encoder, bottleneck, decoder = build_models(device)
        teacher_load_report = load_wrn50_checkpoint_into_model(
            settings.teacher_checkpoint,
            encoder,
            expected_sha256=settings.teacher_sha256,
        )
        run_config["teacher_vendor_load_report"] = teacher_load_report
        write_json(output_dir / "run_config.json", run_config)

        encoder.eval()
        encoder.requires_grad_(False)
        bottleneck.train()
        decoder.train()
        trainable_parameters = list(decoder.parameters()) + list(
            bottleneck.parameters()
        )
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=settings.learning_rate,
            betas=AUTHOR_BETAS,
        )

        stopped = False
        for epoch in range(1, settings.epochs + 1):
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                with torch.no_grad():
                    teacher_features = encoder(images)
                student_features = decoder(bottleneck(teacher_features))
                loss = rd4ad_cosine_loss(teacher_features, student_features)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"nonfinite RD4AD loss at epoch={epoch}, step={global_step + 1}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=float("inf"),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                global_step += 1
                append_jsonl(
                    log_path,
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                )
                if settings.max_steps is not None and global_step >= settings.max_steps:
                    stopped = True
                    break

            completed_epoch = epoch
            write_json(
                output_dir / "run_state.json",
                {
                    "status": "running",
                    "epoch": completed_epoch,
                    "global_step": global_step,
                    "updated_at_utc": utc_now(),
                },
            )
            if stopped:
                break

        if global_step == 0:
            raise RuntimeError("RD4AD training completed no optimizer steps")
        if complete_protocol and completed_epoch != AUTHOR_EPOCHS:
            raise RuntimeError(
                f"full RD4AD run stopped at epoch {completed_epoch}, expected {AUTHOR_EPOCHS}"
            )

        checkpoint = _save_final_checkpoint(
            checkpoint_path,
            bottleneck,
            decoder,
        )
        checkpoint_manifest = {
            "schema_version": 2,
            "method": "RD4AD",
            "dataset": settings.dataset,
            "category": settings.category,
            "seed": settings.seed,
            "run_purpose": run_purpose,
            "execution_scope": settings.execution_scope,
            "claim_scope": purpose["claim_scope"],
            "checkpoint_origin": "project_training_run",
            "epoch": completed_epoch,
            "global_step": global_step,
            "final_checkpoint_only": True,
            "test_selected": False,
            "teacher_sha256": inspection["teacher_checkpoint"]["sha256"],
            "training_recipe": rd4ad_training_recipe(settings),
            "formal_provenance": authority,
            "formal_source_provenance": source_provenance,
            "formal_training_inputs": (
                inspection["formal_training_inputs"]
                if run_purpose == "formal_full"
                else None
            ),
            "checkpoints": {"student": checkpoint},
            "external_hashes": {
                "teacher": inspection["teacher_checkpoint"]["sha256"]
            },
            "checkpoint": checkpoint,
            "saved_at_utc": utc_now(),
        }
        write_json(output_dir / "checkpoints" / "manifest.json", checkpoint_manifest)
        final_state = {
            "status": (
                "completed" if settings.execution_scope == "full" else "debug_completed"
            ),
            "execution_scope": settings.execution_scope,
            "run_purpose": run_purpose,
            "claim_scope": purpose["claim_scope"],
            "standalone_performance_claim_eligible": False,
            "epoch": completed_epoch,
            "global_step": global_step,
            "checkpoint_sha256": checkpoint["sha256"],
            "updated_at_utc": utc_now(),
        }
        write_json(output_dir / "run_state.json", final_state)
        return final_state
    except BaseException as error:
        # A failed run must not leave an apparently usable final checkpoint.
        for incomplete in (
            checkpoint_path,
            output_dir / "checkpoints" / "manifest.json",
        ):
            if incomplete.exists():
                incomplete.unlink()
        write_json(
            output_dir / "run_state.json",
            {
                "status": "failed",
                "execution_scope": settings.execution_scope,
                "epoch": completed_epoch,
                "global_step": global_step,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at_utc": utc_now(),
            },
        )
        raise
