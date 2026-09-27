"""Explicit local WideResNet50-2 ImageNet V1 checkpoint handling.

The frozen PatchCore and RD4AD sources use legacy ``pretrained=True`` calls,
which can download into an implicit Torch Hub cache.  This project layer never
calls that path: it constructs torchvision WideResNet50-2 with ``weights=None``
and injects a tensor-only local state dict after a full SHA-256 check.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models as torchvision_models

from .run_artifacts import sha256_file


WRN50_IMAGENET1K_V1_URL = (
    "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"
)
WRN50_IMAGENET1K_V1_FILENAME = "wide_resnet50_2-95faca4d.pth"
WRN50_IMAGENET1K_V1_TORCHVISION_ENUM = (
    "Wide_ResNet50_2_Weights.IMAGENET1K_V1"
)
WRN50_IMAGENET1K_V1_URL_HASH_PREFIX = "95faca4d"
WRN50_PARAMETER_COUNT = 68_883_240
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_DTYPE_CONVERSIONS = {
    (torch.float16, torch.float32): "floating_point_widening",
}


def official_wrn50_source_metadata() -> dict[str, object]:
    """Return immutable source metadata without fetching the checkpoint."""

    enum = torchvision_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
    if enum.url != WRN50_IMAGENET1K_V1_URL:
        raise RuntimeError(
            "installed torchvision WideResNet50-2 V1 URL differs from the frozen URL"
        )
    num_params = int(enum.meta["num_params"])
    if num_params != WRN50_PARAMETER_COUNT:
        raise RuntimeError(
            "installed torchvision WideResNet50-2 parameter count differs from the frozen value"
        )
    return {
        "architecture": "wide_resnet50_2",
        "torchvision_weight_enum": WRN50_IMAGENET1K_V1_TORCHVISION_ENUM,
        "official_url": WRN50_IMAGENET1K_V1_URL,
        "official_filename": WRN50_IMAGENET1K_V1_FILENAME,
        "url_hash_prefix": WRN50_IMAGENET1K_V1_URL_HASH_PREFIX,
        "parameter_count": WRN50_PARAMETER_COUNT,
        "legacy_pretrained_true_target": WRN50_IMAGENET1K_V1_TORCHVISION_ENUM,
        "preprocessing": {
            "resize_short_edge": 256,
            "center_crop": 224,
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "license_context": "torchvision model weights; retain upstream attribution and terms",
        "download_attempted": False,
    }


def _normalize_expected_sha256(expected_sha256: str) -> str:
    normalized = expected_sha256.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return normalized


def _read_tensor_state_dict(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256_file(path)
    normalized_expected: str | None = None
    if expected_sha256 is not None:
        normalized_expected = _normalize_expected_sha256(expected_sha256)
        if actual_sha256 != normalized_expected:
            raise ValueError(
                "WideResNet50-2 checkpoint SHA-256 mismatch: "
                f"expected {normalized_expected}, got {actual_sha256}"
            )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("WideResNet50-2 checkpoint must be a non-empty state dict")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("WideResNet50-2 checkpoint contains a non-string key")
    non_tensor_keys = [
        key for key, value in payload.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensor_keys:
        raise ValueError(
            "WideResNet50-2 checkpoint contains non-tensor values: "
            + ", ".join(sorted(non_tensor_keys)[:5])
        )
    state = dict(payload)
    report = {
        **official_wrn50_source_metadata(),
        "checkpoint_path": str(path),
        "checkpoint_filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "url_hash_prefix_matched": actual_sha256.startswith(
            WRN50_IMAGENET1K_V1_URL_HASH_PREFIX
        ),
        "expected_sha256": normalized_expected,
        "sha256_verified": normalized_expected is not None,
        "safe_load": {
            "map_location": "cpu",
            "weights_only": True,
            "top_level_mapping": True,
            "string_keys_only": True,
            "tensor_values_only": True,
        },
        "state_tensor_count": len(state),
        "state_tensor_numel": int(sum(tensor.numel() for tensor in state.values())),
        "source_dtype_counts": dict(
            sorted(Counter(str(tensor.dtype) for tensor in state.values()).items())
        ),
        "implicit_download_used": False,
    }
    return state, report


def _prepare_state_for_strict_load(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Validate all tensors, then apply only allowlisted widening conversions."""

    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    shape_mismatches = sorted(
        key
        for key in set(target).intersection(state)
        if target[key].shape != state[key].shape
    )
    shared_keys = sorted(set(target).intersection(state))
    nonfinite = sorted(
        key
        for key in shared_keys
        if (state[key].is_floating_point() or state[key].is_complex())
        and not bool(torch.isfinite(state[key]).all())
    )
    unsafe_dtype_mismatches: list[str] = []
    conversion_groups: dict[tuple[torch.dtype, torch.dtype], list[str]] = {}
    for key in shared_keys:
        source_dtype = state[key].dtype
        target_dtype = target[key].dtype
        if source_dtype == target_dtype:
            continue
        conversion = (source_dtype, target_dtype)
        if conversion not in _SAFE_DTYPE_CONVERSIONS:
            unsafe_dtype_mismatches.append(
                f"{key}:{source_dtype}->{target_dtype}"
            )
            continue
        conversion_groups.setdefault(conversion, []).append(key)

    if missing or unexpected or shape_mismatches or unsafe_dtype_mismatches or nonfinite:
        raise ValueError(
            "WideResNet50-2 checkpoint is not strict-compatible: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={shape_mismatches[:5]}, "
            f"unsafe_dtype_conversion={unsafe_dtype_mismatches[:5]}, "
            f"nonfinite={nonfinite[:5]}"
        )

    prepared = dict(state)
    applied: list[dict[str, object]] = []
    for (source_dtype, target_dtype), keys in sorted(
        conversion_groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        for key in keys:
            prepared[key] = state[key].to(dtype=target_dtype)
        applied.append(
            {
                "source_dtype": str(source_dtype),
                "target_dtype": str(target_dtype),
                "reason": _SAFE_DTYPE_CONVERSIONS[(source_dtype, target_dtype)],
                "key_count": len(keys),
                "keys": keys,
            }
        )
    conversion_report = {
        "policy": "allowlisted_widening_only",
        "allowed": [
            {
                "source_dtype": str(source_dtype),
                "target_dtype": str(target_dtype),
                "reason": reason,
            }
            for (source_dtype, target_dtype), reason in sorted(
                _SAFE_DTYPE_CONVERSIONS.items(),
                key=lambda item: (str(item[0][0]), str(item[0][1])),
            )
        ],
        "applied_tensor_count": sum(len(keys) for keys in conversion_groups.values()),
        "applied": applied,
        "unsafe_or_narrowing_conversion_allowed": False,
    }
    return prepared, conversion_report


def load_wrn50_checkpoint_into_model(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Hash-check and strict-load a local tensor state dict into ``model``."""

    state, report = _read_tensor_state_dict(
        checkpoint_path, expected_sha256=expected_sha256
    )
    prepared_state, conversion_report = _prepare_state_for_strict_load(model, state)
    result = model.load_state_dict(prepared_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict WideResNet50-2 load returned incompatible keys")
    return {
        **report,
        "dtype_conversion": conversion_report,
        "strict_load": {
            "passed": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
            "unsafe_dtype_mismatches": [],
            "nonfinite_tensors": [],
        },
    }


def _new_uninitialized_torchvision_wrn50() -> nn.Module:
    # ``weights=None`` is explicit and never consults Torch Hub or the network.
    return torchvision_models.wide_resnet50_2(weights=None, progress=False)


def build_torchvision_wrn50_from_local_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
    model_factory: Callable[[], nn.Module] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    """Construct without weights, then inject the verified local checkpoint."""

    factory = model_factory or _new_uninitialized_torchvision_wrn50
    model = factory()
    report = load_wrn50_checkpoint_into_model(
        checkpoint_path,
        model,
        expected_sha256=expected_sha256,
    )
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    return model, {
        **report,
        "constructor": "torchvision.models.wide_resnet50_2(weights=None, progress=False)",
        "target_device": str(resolved_device),
    }


def inspect_wrn50_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict[str, object]:
    """Compute provenance and prove strict compatibility without retaining a model."""

    state, report = _read_tensor_state_dict(
        checkpoint_path, expected_sha256=expected_sha256
    )
    factory = model_factory or _new_uninitialized_torchvision_wrn50
    model = factory()
    prepared_state, conversion_report = _prepare_state_for_strict_load(model, state)
    result = model.load_state_dict(prepared_state, strict=True)
    return {
        **report,
        "dtype_conversion": conversion_report,
        "strict_load": {
            "passed": not result.missing_keys and not result.unexpected_keys,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "shape_mismatches": [],
            "unsafe_dtype_mismatches": [],
            "nonfinite_tensors": [],
        },
        "constructor": "torchvision.models.wide_resnet50_2(weights=None, progress=False)",
        "audit_retained_model": False,
    }
