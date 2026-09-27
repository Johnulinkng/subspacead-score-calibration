"""Compatibility layer around the unmodified DRAEM author repository."""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "third_party" / "DRAEM"


def _load_vendor_file(module_name: str, filename: str, vendor_root: Path) -> ModuleType:
    path = vendor_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"DRAEM vendor file is missing: {path}")
    qualified_name = f"industrial_anomaly_lab._vendor_{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DRAEM vendor module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed every random source used by the DRAEM author implementation."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import imgaug

        imgaug.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy, Python, Torch and imgaug inside a spawned DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import imgaug

        imgaug.seed(worker_seed)
    except ImportError:
        pass

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    for offset, augmenter in enumerate(getattr(dataset, "augmenters", [])):
        if hasattr(augmenter, "seed_"):
            augmenter.seed_(worker_seed + offset + 1)
    rotation = getattr(dataset, "rot", None)
    if rotation is not None and hasattr(rotation, "seed_"):
        rotation.seed_(worker_seed + 1000)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_models(
    device: str | torch.device,
    *,
    vendor_root: Path = DEFAULT_VENDOR_ROOT,
) -> tuple[nn.Module, nn.Module]:
    """Build the author's reconstruction and segmentation networks."""

    model_module = _load_vendor_file("model_unet", "model_unet.py", vendor_root)
    resolved_device = torch.device(device)
    reconstructive = model_module.ReconstructiveSubNetwork(
        in_channels=3, out_channels=3
    ).to(resolved_device)
    discriminative = model_module.DiscriminativeSubNetwork(
        in_channels=6, out_channels=2
    ).to(resolved_device)
    return reconstructive, discriminative


def initialize_author_weights(module: nn.Module) -> None:
    """Apply the convolution and batch-normalization initializer from DRAEM."""

    class_name = module.__class__.__name__
    if "Conv" in class_name and hasattr(module, "weight"):
        module.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm" in class_name:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()


def build_focal_loss(*, vendor_root: Path = DEFAULT_VENDOR_ROOT) -> nn.Module:
    """Build the author's focal loss without importing its CUDA-only SSIM."""

    loss_module = _load_vendor_file("loss", "loss.py", vendor_root)
    return loss_module.FocalLoss()


class DeviceSafeSSIM(nn.Module):
    """The author's SSIM formula without constructor-time CUDA allocation."""

    def __init__(
        self,
        window_size: int = 11,
        *,
        vendor_root: Path = DEFAULT_VENDOR_ROOT,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self._vendor_loss = _load_vendor_file("loss", "loss.py", vendor_root)
        self.register_buffer(
            "window",
            self._vendor_loss.create_window(window_size),
            persistent=False,
        )

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        if image_a.shape != image_b.shape:
            raise ValueError("SSIM inputs must have identical shapes")
        channel = image_a.shape[1]
        window = self._vendor_loss.create_window(
            self.window_size, channel=channel
        ).to(device=image_a.device, dtype=image_a.dtype)
        score, _ = self._vendor_loss.ssim(
            image_a,
            image_b,
            window=window,
            window_size=self.window_size,
            size_average=True,
        )
        return 1.0 - score


def load_state_dict_safely(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    strict: bool = True,
) -> Any:
    """Load a tensor-only state dict on CPU and validate its basic structure."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")
    if not all(isinstance(key, str) for key in state):
        raise ValueError(f"checkpoint contains a non-string key: {path}")
    return model.load_state_dict(state, strict=strict)
