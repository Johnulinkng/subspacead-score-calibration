"""Bounded, synthetic DRAEM capacity diagnostic; no dataset or checkpoints.

The learning rate is zero. Two forward/backward/Adam iterations exercise the
author-shaped allocation path without fitting a model. Successful diagnostics
are admission estimates, not scientific results or a long-run stability test.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GIB = 1024**3


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD), ("load", wintypes.DWORD),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_commit", ctypes.c_ulonglong),
        ("available_commit", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


class ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD), ("page_faults", wintypes.DWORD),
        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
        ("peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
        ("peak_nonpaged_pool", ctypes.c_size_t), ("nonpaged_pool", ctypes.c_size_t),
        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
        ("private_bytes", ctypes.c_size_t),
    ]


def windows_memory() -> dict[str, float]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    counters = ProcessMemory(size=ctypes.sizeof(ProcessMemory))
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "free_ram_gib": status.available_physical / GIB,
        "free_commit_gib": status.available_commit / GIB,
        "working_set_gib": counters.working_set / GIB,
        "peak_working_set_gib": counters.peak_working_set / GIB,
        "private_commit_gib": counters.private_bytes / GIB,
        "peak_private_commit_gib": counters.peak_pagefile / GIB,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-checkpointing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "runs" / "monitored"):
        raise ValueError("Diagnostic output must remain below runs/monitored")
    output.mkdir(parents=True, exist_ok=False)
    baseline_memory = windows_memory()
    if baseline_memory["free_ram_gib"] < 4 or baseline_memory["free_commit_gib"] < 4:
        raise RuntimeError("Diagnostic needs 2 GiB estimated process memory plus 2 GiB system reserve")

    # Import Torch only after checking the initial host reserve.
    import torch
    from industrial_anomaly_lab.draem_runtime import (
        DeviceSafeSSIM, build_focal_loss, build_models,
        configure_reproducibility, initialize_author_weights,
    )

    device = torch.device("cuda:0")
    configure_reproducibility(1887537248, deterministic=True)
    free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    # Fail inside the allocator instead of silently spilling to Windows shared
    # GPU memory. This is a diagnostic cap, not a new training hyperparameter.
    gpu_cap = min(free_gpu - int(.75 * GIB), int(.90 * total_gpu))
    if gpu_cap < 5 * GIB:
        raise RuntimeError("Insufficient free dedicated VRAM for the bounded diagnostic")
    torch.cuda.set_per_process_memory_fraction(gpu_cap / total_gpu, device)
    torch.cuda.reset_peak_memory_stats(device)

    samples: list[dict[str, float]] = []
    stop = threading.Event()
    pressure = threading.Event()
    sample_errors: list[str] = []

    def sample_memory() -> None:
        while not stop.is_set():
            try:
                current = windows_memory()
                current["elapsed_seconds"] = time.monotonic() - started
                samples.append(current)
                if current["free_ram_gib"] < 1.5 or current["free_commit_gib"] < 2:
                    pressure.set()
            except Exception as error:
                sample_errors.append(f"{type(error).__name__}: {error}")
                pressure.set()
            stop.wait(.2)

    started = time.monotonic()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    completed = 0
    stage = "model_allocation"
    error_detail = None
    result = {
        "schema_version": 1,
        "kind": "draem_synthetic_capacity_diagnostic",
        "claim_eligible": False,
        "dataset_accessed": False,
        "formal_test_content_accessed": False,
        "ground_truth_content_accessed": False,
        "checkpoint_saved": False,
        "parameter_update_learning_rate": 0.0,
        "recipe": {"image_size": 256, "batch_size": 8, "workers": 0,
                   "pin_memory": False, "precision": "float32", "device": "cuda:0",
                   "activation_checkpointing": args.activation_checkpointing},
        "baseline_host": baseline_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_gib": total_gpu / GIB,
        "allocator_cap_gib": gpu_cap / GIB,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in (
            ROOT / "pyproject.toml", ROOT / "uv.lock",
            Path(__file__), ROOT / "src/industrial_anomaly_lab/draem_runtime.py",
            ROOT / "src/industrial_anomaly_lab/draem_train.py", ROOT / "scripts/train_draem.py",
            ROOT / "src/industrial_anomaly_lab/draem_checkpointing.py",
            ROOT / "third_party/DRAEM/model_unet.py", ROOT / "third_party/DRAEM/loss.py",
        )},
    }
    try:
        reconstructive, discriminative = build_models(device)
        reconstructive.apply(initialize_author_weights)
        discriminative.apply(initialize_author_weights)
        if args.activation_checkpointing:
            from industrial_anomaly_lab.draem_checkpointing import enable_activation_checkpointing
            result["checkpointed_blocks"] = {
                "reconstructive": enable_activation_checkpointing(reconstructive),
                "discriminative": enable_activation_checkpointing(discriminative),
            }
        parameters = list(reconstructive.parameters()) + list(discriminative.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.0)
        mse = torch.nn.MSELoss()
        ssim = DeviceSafeSSIM().to(device)
        focal = build_focal_loss().to(device)
        for step in range(2):
            if pressure.is_set():
                raise MemoryError("Host safety reserve breached; diagnostic stopped")
            optimizer.zero_grad(set_to_none=True)
            target = torch.rand(8, 3, 256, 256, device=device)
            augmented = torch.rand_like(target)
            mask = (torch.rand(8, 1, 256, 256, device=device) > .8).float()
            stage = "forward"
            reconstruction = reconstructive(augmented)
            probabilities = torch.softmax(discriminative(torch.cat((reconstruction, augmented), dim=1)), dim=1)
            loss = mse(reconstruction, target) + ssim(reconstruction, target) + focal(probabilities, mask)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Synthetic diagnostic loss is nonfinite")
            stage = "backward"
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
            if pressure.is_set():
                raise MemoryError("Host safety reserve breached; diagnostic stopped")
            stage = "adam_allocation_at_zero_learning_rate"
            optimizer.step()
            torch.cuda.synchronize(device)
            completed += 1
            print(json.dumps({"diagnostic_step": completed, "loss": loss.item(),
                              "gradient_norm": norm.item(), "claim_eligible": False}), flush=True)
            del target, augmented, mask, reconstruction, probabilities, loss, norm
    except Exception as error:
        error_detail = {"type": type(error).__name__, "message": str(error), "stage": stage}
    finally:
        stop.set()
        sampler.join(timeout=2)
        samples.append(windows_memory())
        result.update({
            "status": "passed" if error_detail is None and completed == 2 and not sample_errors else "failed",
            "completed_steps": completed,
            "error": error_detail,
            "sample_errors": sample_errors,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
            "host_peak_working_set_gib": max(row["peak_working_set_gib"] for row in samples),
            "host_peak_private_commit_gib": max(row["peak_private_commit_gib"] for row in samples),
            "host_min_free_ram_gib": min(row["free_ram_gib"] for row in samples),
            "host_min_free_commit_gib": min(row["free_commit_gib"] for row in samples),
            "elapsed_seconds": time.monotonic() - started,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": "Synthetic allocation probe only; real loader, checkpoint saving and long-run stability still require S0. No performance evidence.",
        })
        (output / "profile.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (output / "host_samples.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
