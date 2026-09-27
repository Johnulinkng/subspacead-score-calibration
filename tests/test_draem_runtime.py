from __future__ import annotations

import numpy as np
import pytest

from industrial_anomaly_lab.draem_runtime import (
    DeviceSafeSSIM,
    build_models,
    configure_reproducibility,
)


def test_reproducibility_seed_controls_numpy_and_torch() -> None:
    torch = pytest.importorskip("torch")
    configure_reproducibility(42)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)
    configure_reproducibility(42)

    assert np.array_equal(np.random.random(4), first_numpy)
    assert torch.equal(torch.rand(4), first_torch)


def test_cpu_models_forward_at_smoke_resolution() -> None:
    torch = pytest.importorskip("torch")
    configure_reproducibility(42)
    reconstructive, discriminative = build_models("cpu")
    reconstructive.eval()
    discriminative.eval()
    image = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        reconstruction = reconstructive(image)
        logits = discriminative(torch.cat((reconstruction, image), dim=1))

    assert reconstruction.shape == image.shape
    assert logits.shape == (1, 2, 64, 64)


def test_device_safe_ssim_runs_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    loss = DeviceSafeSSIM()
    image = torch.rand(2, 3, 32, 32)

    value = loss(image, image)

    assert value.item() == pytest.approx(0.0, abs=1e-6)
