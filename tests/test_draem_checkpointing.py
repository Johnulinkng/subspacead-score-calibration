from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from industrial_anomaly_lab.draem_checkpointing import (
    _preserve_batchnorm_buffers,
    enable_activation_checkpointing,
)
from industrial_anomaly_lab.draem_runtime import DEFAULT_VENDOR_ROOT, _load_vendor_file


@pytest.mark.parametrize("momentum", [0.1, None])
def test_recomputation_preserves_outputs_gradients_optimizer_and_bn(momentum):
    torch.manual_seed(917)
    original = nn.Sequential(
        nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8, momentum=momentum),
                      nn.ReLU(inplace=True), nn.Conv2d(8, 8, 3, padding=1),
                      nn.BatchNorm2d(8, momentum=momentum), nn.ReLU(inplace=True)),
        nn.Conv2d(8, 2, 1),
    )
    recomputed = copy.deepcopy(original)
    # Exactly the state_dict shape used by official checkpoints is retained.
    keys = list(recomputed.state_dict())
    assert enable_activation_checkpointing(recomputed) >= 1
    assert enable_activation_checkpointing(recomputed) == 0
    assert keys == list(recomputed.state_dict())
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4) for model in (original, recomputed)]
    for step in range(3):
        image = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 2, 16, 16)
        outputs = []
        for model, optimizer in zip((original, recomputed), optimizers):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            outputs.append(prediction.detach())
            (prediction - target).square().mean().backward()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for left, right in zip(original.parameters(), recomputed.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        for key, left in original.state_dict().items():
            torch.testing.assert_close(left, recomputed.state_dict()[key], rtol=0, atol=0)
        for model in (original, recomputed):
            assert all(module.num_batches_tracked.item() == step + 1 for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    original.eval()
    recomputed.eval()
    with torch.no_grad():
        torch.testing.assert_close(original(image), recomputed(image), rtol=0, atol=0)


def test_recomputation_restores_buffers_when_interrupted():
    block = nn.Sequential(nn.BatchNorm2d(3))
    before = copy.deepcopy(block.state_dict())
    with pytest.raises(RuntimeError, match="interrupted"):
        with _preserve_batchnorm_buffers(block):
            block(torch.rand(2, 3, 4, 4))
            raise RuntimeError("interrupted")
    for key, expected in before.items():
        torch.testing.assert_close(block.state_dict()[key], expected, rtol=0, atol=0)


def test_author_unet_topology_keeps_forward_backward_and_bn_identical():
    # Exercise the complete author topology and reconstruction-to-segmentation
    # gradient path on a small fixture width. Capacity of the full author width
    # is measured separately by the GPU diagnostic, not by this unit test.
    vendor = _load_vendor_file("model_unet", "model_unet.py", DEFAULT_VENDOR_ROOT)
    torch.manual_seed(31)
    networks = nn.ModuleList([
        vendor.ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=4),
        vendor.DiscriminativeSubNetwork(in_channels=6, out_channels=2, base_channels=4),
    ])
    recomputed = copy.deepcopy(networks)
    assert [enable_activation_checkpointing(model) for model in recomputed] == [13, 16]
    original_keys = list(networks.state_dict())
    assert list(recomputed.state_dict()) == original_keys
    image = torch.rand(2, 3, 32, 32)
    outputs = []
    for models in (networks, recomputed):
        reconstruction = models[0](image)
        logits = models[1](torch.cat((reconstruction, image), dim=1))
        outputs.append((reconstruction.detach(), logits.detach()))
        (reconstruction.square().mean() + logits.square().mean()).backward()
    for left, right in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(networks.parameters(), recomputed.parameters()):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    for key, left in networks.state_dict().items():
        torch.testing.assert_close(left, recomputed.state_dict()[key], rtol=0, atol=0)
    # Applying the memory policy preserves strict official checkpoint loading.
    recomputed.load_state_dict(networks.state_dict(), strict=True)
