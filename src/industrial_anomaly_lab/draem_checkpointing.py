"""Activation recomputation without changing author weights or BN updates.

Recomputing a training BatchNorm block would normally update its running
buffers twice. Restore just those buffers after recomputation, including on
PyTorch's early-stop exception. Do not switch BN to eval or change its formula.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import MethodType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def _preserve_batchnorm_buffers(block: nn.Module):
    buffers = [
        (buffer, buffer.detach().clone())
        for module in block.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for buffer in (module.running_mean, module.running_var, module.num_batches_tracked)
        if buffer is not None
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)


def enable_activation_checkpointing(model: nn.Module) -> int:
    """Patch BN-containing Sequential blocks in place; state_dict keys persist.

    Use once before training and before compiling. Grad-disabled inference
    follows the original forward. Blocks and checkpoint settings are uniform
    for every category and seed.
    """

    count = 0
    for block in model.modules():
        if not isinstance(block, nn.Sequential):
            continue
        if not any(isinstance(child, nn.modules.batchnorm._BatchNorm) for child in block.modules()):
            continue
        if getattr(block, "_iad_activation_checkpointed", False):
            continue
        original_forward = block.forward

        def make_forward(original, module):
            def forward(_self, *args):
                if not module.training or not torch.is_grad_enabled():
                    return original(*args)
                return checkpoint(
                    original, *args, use_reentrant=False, preserve_rng_state=True,
                    context_fn=lambda: (nullcontext(), _preserve_batchnorm_buffers(module)),
                )
            return forward

        block.forward = MethodType(make_forward(original_forward, block), block)
        block._iad_activation_checkpointed = True
        count += 1
    return count
