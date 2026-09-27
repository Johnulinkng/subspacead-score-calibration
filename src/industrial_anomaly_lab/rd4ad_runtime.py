"""Modern PyTorch compatibility layer for the frozen RD4AD author source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "third_party" / "RD4AD"
RD4AD_VENDOR_COMMIT = "6554076872c65f8784f6ece8cfb39ce77e1aee12"
RD4AD_SOURCE_SHA256 = {
    "dataset.py": "2814901e5087cc924eca1939f529debdda5fab43c371c47603427be27683415a",
    "de_resnet.py": "c2740b70cd38453c8521151c539f3976d6e51adf64336224d02d4187f8cce4ee",
    "main.py": "9bbe543c3a45e1368df5c33761fb161051773266d86e2d88be4049d403ef0de4",
    "resnet.py": "d70cc2de79f6d253cf4397ebe41550c2bf7a15b43427f1228006368fe6a01c1b",
    "test.py": "0e29666d6586e57c0bf2558cc09568a8fe7d8d82b153c31356387ce4624d05ef",
}


def audit_vendor_source(
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
) -> dict[str, object]:
    """Fail closed if any frozen RD4AD source file changed or disappeared."""

    root = Path(vendor_root)
    actual: dict[str, str] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for filename, expected in RD4AD_SOURCE_SHA256.items():
        path = root / filename
        if not path.is_file():
            missing.append(filename)
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        actual[filename] = digest
        if digest != expected:
            mismatched.append(filename)
    if missing or mismatched:
        raise RuntimeError(
            "frozen RD4AD source audit failed: "
            f"missing={missing}, sha256_mismatch={mismatched}"
        )
    return {
        "commit": RD4AD_VENDOR_COMMIT,
        "source_sha256": actual,
        "all_expected_files_match": True,
    }


def _load_vendor_file(module_name: str, filename: str, vendor_root: Path) -> ModuleType:
    path = Path(vendor_root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"RD4AD vendor file is missing: {path}")
    qualified_name = f"industrial_anomaly_lab._rd4ad_vendor_{module_name}"
    module = ModuleType(qualified_name)
    module.__file__ = str(path)
    module.__package__ = "industrial_anomaly_lab"
    # Executing compiled source avoids depositing __pycache__ inside the frozen
    # third-party checkout, so source-integrity checks stay clean.
    code = compile(path.read_bytes(), str(path), "exec")
    exec(code, module.__dict__)
    return module


def build_models(
    device: str | torch.device,
    *,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Construct WideResNet50-2 RD4AD models without implicit downloads."""

    audit_vendor_source(vendor_root)
    encoder_module = _load_vendor_file("resnet", "resnet.py", vendor_root)
    decoder_module = _load_vendor_file("de_resnet", "de_resnet.py", vendor_root)
    encoder, bottleneck = encoder_module.wide_resnet50_2(pretrained=False)
    decoder = decoder_module.de_wide_resnet50_2(pretrained=False)
    resolved = torch.device(device)
    return encoder.to(resolved), bottleneck.to(resolved), decoder.to(resolved)


def _load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {resolved}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"checkpoint contains a non-string key: {resolved}")
    return payload


def _validate_state_dict(state: object, *, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a non-empty state dict")
    if not all(isinstance(key, str) for key in state):
        raise ValueError(f"{name} contains a non-string key")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{name} contains a non-tensor value")
    return dict(state)


def load_teacher_checkpoint_safely(
    checkpoint_path: str | Path,
    encoder: nn.Module,
) -> dict[str, object]:
    """Load an explicit local ImageNet state dict into the frozen teacher."""

    state = _validate_state_dict(
        _load_checkpoint(checkpoint_path), name="teacher checkpoint"
    )
    result = encoder.load_state_dict(state, strict=True)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def load_student_checkpoint_safely(
    checkpoint_path: str | Path,
    bottleneck: nn.Module,
    decoder: nn.Module,
) -> dict[str, object]:
    """Load the author's tensor-only ``bn``/``decoder`` checkpoint.

    Older author checkpoints may contain obsolete bottleneck keys whose name
    includes ``memory``.  The original evaluation code removes those keys.  We
    preserve that behavior but return the exact dropped-key list as provenance.
    """

    payload = _load_checkpoint(checkpoint_path)
    if "bn" not in payload or "decoder" not in payload:
        raise ValueError("student checkpoint must contain 'bn' and 'decoder'")
    bottleneck_state = _validate_state_dict(payload["bn"], name="bn state dict")
    decoder_state = _validate_state_dict(
        payload["decoder"], name="decoder state dict"
    )
    dropped = sorted(key for key in bottleneck_state if "memory" in key)
    for key in dropped:
        del bottleneck_state[key]
    bn_result = bottleneck.load_state_dict(bottleneck_state, strict=True)
    decoder_result = decoder.load_state_dict(decoder_state, strict=True)
    return {
        "dropped_legacy_memory_keys": dropped,
        "bn_missing_keys": list(bn_result.missing_keys),
        "bn_unexpected_keys": list(bn_result.unexpected_keys),
        "decoder_missing_keys": list(decoder_result.missing_keys),
        "decoder_unexpected_keys": list(decoder_result.unexpected_keys),
    }


def _resolve_output_size(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        resolved = (value, value)
    elif len(value) == 2:
        resolved = (int(value[0]), int(value[1]))
    else:
        raise ValueError("output_size must be an integer or a (height, width) pair")
    if resolved[0] <= 0 or resolved[1] <= 0:
        raise ValueError("output_size dimensions must be positive")
    return resolved


def rd4ad_anomaly_maps(
    teacher_features: Sequence[torch.Tensor],
    student_features: Sequence[torch.Tensor],
    *,
    output_size: int | tuple[int, int],
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """Return unnormalized, post-Gaussian author anomaly maps for a batch.

    Each scale contributes ``1 - cosine_similarity``.  Maps are resized with
    the author's ``align_corners=True`` setting, summed, and smoothed per image.
    Unlike the author helper, this function handles every sample in a batch.
    """

    if not teacher_features or len(teacher_features) != len(student_features):
        raise ValueError("teacher and student must provide the same non-zero scales")
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative")
    size = _resolve_output_size(output_size)
    total: torch.Tensor | None = None
    batch_size: int | None = None
    for index, (teacher, student) in enumerate(
        zip(teacher_features, student_features, strict=True)
    ):
        if teacher.ndim != 4 or student.ndim != 4:
            raise ValueError(f"feature scale {index} must have NCHW shape")
        if teacher.shape != student.shape:
            raise ValueError(
                f"feature scale {index} shape mismatch: {teacher.shape} != {student.shape}"
            )
        if batch_size is None:
            batch_size = teacher.shape[0]
        elif teacher.shape[0] != batch_size:
            raise ValueError("all feature scales must have the same batch size")
        distance = 1.0 - F.cosine_similarity(teacher, student, dim=1)
        resized = F.interpolate(
            distance.unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=True,
        )[:, 0]
        total = resized if total is None else total + resized

    assert total is not None
    maps = total.detach().to(device="cpu", dtype=torch.float32).numpy()
    if gaussian_sigma > 0:
        maps = gaussian_filter(
            maps,
            sigma=(0.0, float(gaussian_sigma), float(gaussian_sigma)),
        )
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or not np.isfinite(maps).all():
        raise RuntimeError("RD4AD produced an invalid anomaly-map batch")
    return maps


def rd4ad_image_scores(anomaly_maps: np.ndarray) -> np.ndarray:
    """Author image score: raw maximum of each post-Gaussian anomaly map."""

    maps = np.asarray(anomaly_maps)
    if maps.ndim != 3 or maps.size == 0:
        raise ValueError("anomaly_maps must have shape (num_images, height, width)")
    if not np.isfinite(maps).all():
        raise ValueError("anomaly_maps contains NaN or infinite values")
    return maps.max(axis=(1, 2)).astype(np.float64, copy=False)
