"""One authenticated CPU-only normal-fit resource trial; writes one receipt."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
GIB = 1024**3
MODES = ("one_lazy", "two_lazy", "one_eager", "two_eager", "attention_on", "attention_off")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ProcessCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
        )
    ]


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
        )
    ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


class ResourceViolation(RuntimeError):
    pass


class Monitor:
    """PSAPI sampling at 2 Hz, with cooperative cancellation of this process."""

    def __init__(self, policy: dict) -> None:
        if os.name != "nt":
            raise RuntimeError("This resource protocol requires Windows PSAPI measurements")
        self.policy = policy
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessCounters), wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
        self.psapi.GetPerformanceInfo.restype = wintypes.BOOL
        self.handle = self.kernel.GetCurrentProcess()
        self.samples: list[dict] = []
        self.violation: str | None = None
        self.stage = "initializing"
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="route1-own-resource-monitor", daemon=True)

    def sample(self) -> dict:
        counters = ProcessCounters()
        counters.cb = ctypes.sizeof(counters)
        performance = PerformanceInfo()
        performance.cb = ctypes.sizeof(performance)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        if not self.psapi.GetPerformanceInfo(ctypes.byref(performance), performance.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        page = int(performance.PageSize)
        row = {
            "elapsed_seconds": time.monotonic() - self.started, "stage": self.stage,
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage),
            "process_commit_bytes": int(counters.PagefileUsage),
            "peak_process_commit_bytes": int(counters.PeakPagefileUsage),
            "system_free_ram_bytes": int(performance.PhysicalAvailable) * page,
            "system_free_commit_bytes": int(performance.CommitLimit - performance.CommitTotal) * page,
        }
        self.samples.append(row)
        if row["working_set_bytes"] > self.policy["max_working_set_gib"] * GIB:
            self.violation = self.violation or "Owned process working set exceeded maximum"
        if row["system_free_ram_bytes"] < self.policy["reserve_free_ram_gib"] * GIB:
            self.violation = self.violation or "System free RAM fell below reserve"
        if row["system_free_commit_bytes"] < self.policy["reserve_free_commit_gib"] * GIB:
            self.violation = self.violation or "System free commit fell below reserve"
        return row

    def _run(self) -> None:
        while not self.stop_event.wait(0.5):
            try:
                self.sample()
            except BaseException as error:
                self.violation = self.violation or f"Resource monitoring failed: {type(error).__name__}: {error}"
                return

    def check(self) -> None:
        if self.violation:
            raise ResourceViolation(self.violation)

    def set_stage(self, name: str) -> None:
        self.stage = name
        self.sample()
        self.check()

    def close(self) -> dict:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.sample()
        rows = self.samples
        fitting = [r for r in rows if r["stage"] == "fit"]
        return {
            "sampling_interval_seconds": 0.5, "sampling_api": "Windows PSAPI current process + performance info",
            "process_peak_working_set_bytes": max(r["peak_working_set_bytes"] for r in rows),
            "process_peak_commit_bytes": max(r["peak_process_commit_bytes"] for r in rows),
            "sampled_peak_private_bytes": max(r["private_bytes"] for r in rows),
            "fit_sampled_peak_working_set_bytes": max((r["working_set_bytes"] for r in fitting), default=None),
            "fit_sampled_peak_private_bytes": max((r["private_bytes"] for r in fitting), default=None),
            "system_min_free_ram_bytes": min(r["system_free_ram_bytes"] for r in rows),
            "system_min_free_commit_bytes": min(r["system_free_commit_bytes"] for r in rows),
            "resource_violation": self.violation, "memory_samples": rows,
            "phase_peak_notice": "Phase peaks are 2Hz samples; whole-process peaks are OS counters and include loading/warmup",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a fresh receipt path inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    wall_start, cpu_start = time.monotonic(), time.process_time()
    receipt = {
        "schema_version": 1, "status": "initializing", "phase": "normal_fit_end_to_end",
        "pid": os.getpid(), "parent_pid": os.getppid(), "dataset": args.dataset,
        "category": args.category, "mode": args.mode, "repeat": args.repeat,
        "method": args.mode, "trial_index": args.repeat,
        "experiment": "attention" if args.mode in ("attention_on", "attention_off") else "end_to_end",
        "device": "cpu", "gpu_used": False, "gpu_memory": "not_used", "test_content_accessed": False,
        "scientific_claim_eligible": False,
        "scope": "fixed first normal-training subset only; not full-category or full-dataset end-to-end runtime",
    }
    write(output, receipt)
    monitor = None
    exit_code = 1
    try:
        config_path = args.config.resolve(strict=True)
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        config = raw_config.get("protocol", raw_config)
        config_sha = sha(config_path)
        threads = config["cpu_threads"]
        if config["device"] != "cpu" or threads != 4:
            raise ValueError("Resource trials require the frozen CPU backend and four threads")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(threads)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        pairs = config.get("resource_categories", [["mvtec_ad", "bottle"], ["visa", "cashew"]])
        if [args.dataset, args.category] not in pairs:
            raise ValueError("Category is not a frozen representative resource category")
        count = config.get("resource_sample_count", 64)
        repeats = config.get("resource_repeats", 3)
        if count != 64 or not 0 <= args.repeat < repeats:
            raise ValueError("Expected 64 fixed normal samples and zero-based registered repeat")
        policy = {
            "initial_free_ram_gib": 3.25, "initial_free_commit_gib": 3.25,
            "reserve_free_ram_gib": 1.5, "reserve_free_commit_gib": 2.0,
            "max_working_set_gib": 2.5, "minimum_disk_gib": 10.0,
            **config.get("resource_policy", {}),
        }
        # CPU trials never consult or relax the old CUDA admission threshold.
        monitor = Monitor(policy)
        admission = monitor.sample()
        if admission["system_free_ram_bytes"] < policy["initial_free_ram_gib"] * GIB:
            raise ResourceViolation("Initial free RAM below admission threshold")
        if admission["system_free_commit_bytes"] < policy["initial_free_commit_gib"] * GIB:
            raise ResourceViolation("Initial free commit below admission threshold")
        import shutil
        if shutil.disk_usage(output.parent).free < policy["minimum_disk_gib"] * GIB:
            raise ResourceViolation("Initial disk free space below admission threshold")
        monitor.thread.start()
        source_run = (ROOT / config["source_run"]).resolve(strict=True)
        if not source_run.is_relative_to(ROOT / "runs"):
            raise ValueError("Source run must stay inside project runs")
        manifest_path = source_run / "categories" / args.dataset / args.category / "train_manifest.json"
        source_completion = json.loads((manifest_path.parent / "completion.json").read_text(encoding="utf-8"))
        manifest_key = manifest_path.relative_to(source_run).as_posix()
        if source_completion["status"] != "completed" or sha(manifest_path) != source_completion["sha256"][manifest_key]:
            raise ValueError("Historical normal manifest differs from its completed source seal")
        records = json.loads(manifest_path.read_text(encoding="utf-8"))[:count]
        if len(records) != count:
            raise ValueError("Frozen normal manifest has fewer than 64 samples")
        normal_root = ROOT / "data" / ("mvtec_ad" if args.dataset == "mvtec_ad" else "visa/1cls") / args.category / "train/good"
        normal_root = normal_root.resolve(strict=True)
        paths = []
        for record in records:
            path = (ROOT / record["path"]).resolve(strict=True)
            if not path.is_relative_to(normal_root) or sha(path) != record["sha256"]:
                raise ValueError("Unauthenticated or non-normal image")
            paths.append(path)
        source_paths = [Path(__file__), ROOT / "src/industrial_anomaly_lab/subspace_features.py"]
        source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
        expected_sources = raw_config.get("source_sha256", config.get("source_sha256", {}))
        for name, value in source_hashes.items():
            if name in expected_sources and expected_sources[name] != value:
                raise ValueError(f"Frozen source mismatch: {name}")
        receipt.update(
            protocol_id=config["protocol_id"], protocol_sha256=config_sha,
            source_sha256=source_hashes, train_manifest_sha256=sha(manifest_path),
            sample_count=count, subset_count=count, samples=[{"path": r["path"], "sha256": r["sha256"]} for r in records],
            resource_policy=policy, admission=admission, cpu_threads=threads,
            attention_outputs=args.mode == "attention_on", input_loading="eager" if "eager" in args.mode else "lazy",
            pca_algorithm="numpy_float64_two_pass" if args.mode.startswith("two") else "existing_float64_Chan_one_pass",
            resource_reference_notice="two-pass resource implementation uses project NumPy on the same CPU; full accuracy controls call vendor PCAModel separately",
        )
        write(output, receipt)
        monitor.set_stage("loading")
        load_start = time.monotonic()
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        import PIL
        import transformers
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.subspace_features import LocalDinoExtractor, StreamingPCA
        threadpool_limits(limits=threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        cv2.setNumThreads(threads)
        extractor = LocalDinoExtractor(
            ROOT / config["weights_dir"], expected_checkpoint_sha256=config["checkpoint_sha256"],
            expected_artifact_sha256=config["weight_files_sha256"], image_res=448,
            layers=(-4, -5), device="cpu", cpu_threads=threads,
        )
        receipt["extractor"] = extractor.metadata
        receipt["load_seconds"] = time.monotonic() - load_start
        monitor.check()

        def forward(image, attention: bool):
            if not attention:
                return extractor.extract(image)
            inputs = extractor.processor(
                images=[image.convert("RGB")], return_tensors="pt", do_resize=True,
                size={"height": 448, "width": 448}, do_center_crop=False,
                crop_size={"height": 448, "width": 448},
            ).to(extractor.device)
            with torch.inference_mode():
                outputs = extractor.model(**inputs, output_hidden_states=True, output_attentions=True)
                if outputs.attentions is None or len(outputs.attentions) != 12:
                    raise ValueError("Attention-on reference did not retain 12 layer outputs")
                selected = [outputs.hidden_states[i][:, 1:, :].reshape(1, 32, 32, 384) for i in (-4, -5)]
                fused = torch.stack(selected, dim=0).mean(dim=0)
                result = fused.cpu().numpy().copy()[0]
            if result.shape != (32, 32, 384) or result.dtype != np.float32 or not np.isfinite(result).all():
                raise ValueError("Attention reference emitted invalid tokens")
            return result

        attention = args.mode == "attention_on"
        monitor.set_stage("warmup")
        warm_start = time.monotonic()
        with Image.open(paths[0]) as handle:
            warm_image = handle.convert("RGB")
        try:
            warm_tokens = forward(warm_image, attention)
            baseline_tokens = forward(warm_image, False) if args.mode in ("attention_on", "attention_off") else warm_tokens
        finally:
            warm_image.close()
        receipt["warmup_seconds"] = time.monotonic() - warm_start
        receipt["warmup_feature_sha256"] = hashlib.sha256(warm_tokens.tobytes()).hexdigest()
        receipt["warmup_attention_reference_max_abs_error"] = float(np.max(np.abs(warm_tokens - baseline_tokens)))
        receipt["warmup_attention_reference_exact_equal"] = bool(np.array_equal(warm_tokens, baseline_tokens))
        if not receipt["warmup_attention_reference_exact_equal"]:
            raise ValueError("Attention return setting changed normal-image features")
        del warm_tokens, baseline_tokens
        monitor.check()
        monitor.set_stage("fit")
        fit_start, fit_cpu_start = time.monotonic(), time.process_time()
        timers = {"image_decode_seconds": 0.0, "feature_forward_seconds": 0.0, "pca_accumulation_seconds": 0.0}
        counts = {"generator_calls": 0, "image_decodes": 0, "feature_forwards": 0, "normal_patch_tokens_consumed": 0}
        pass_feature_hashes = []

        def generator():
            counts["generator_calls"] += 1
            content = hashlib.sha256()
            eager_images = []
            try:
                if "eager" in args.mode:
                    for path in paths:
                        monitor.check()
                        started = time.monotonic()
                        with Image.open(path) as handle:
                            image = handle.convert("RGB")
                        eager_images.append(image)
                        timers["image_decode_seconds"] += time.monotonic() - started
                        counts["image_decodes"] += 1
                for index, path in enumerate(paths):
                    monitor.check()
                    if eager_images:
                        image = eager_images[index]
                    else:
                        started = time.monotonic()
                        with Image.open(path) as handle:
                            image = handle.convert("RGB")
                        timers["image_decode_seconds"] += time.monotonic() - started
                        counts["image_decodes"] += 1
                    try:
                        started = time.monotonic()
                        tokens = forward(image, attention).reshape(1024, 384)
                        timers["feature_forward_seconds"] += time.monotonic() - started
                    finally:
                        if not eager_images:
                            image.close()
                    monitor.check()
                    counts["feature_forwards"] += 1
                    counts["normal_patch_tokens_consumed"] += len(tokens)
                    content.update(tokens.tobytes())
                    yield tokens
                pass_feature_hashes.append(content.hexdigest())
            finally:
                for image in eager_images:
                    image.close()

        estimator = StreamingPCA(384)
        if args.mode.startswith("two"):
            total = count * 1024
            mean = np.zeros(384, dtype=np.float64)
            scatter = np.zeros((384, 384), dtype=np.float64)
            for tokens in generator():
                started = time.monotonic()
                mean += np.asarray(tokens, dtype=np.float64).sum(axis=0)
                timers["pca_accumulation_seconds"] += time.monotonic() - started
            mean /= total
            for tokens in generator():
                started = time.monotonic()
                centered = np.asarray(tokens, dtype=np.float64) - mean
                scatter += centered.T @ centered
                timers["pca_accumulation_seconds"] += time.monotonic() - started
            estimator.mean[:] = mean
            estimator.scatter[:] = scatter
            estimator.count = total
            if len(pass_feature_hashes) != 2 or pass_feature_hashes[0] != pass_feature_hashes[1]:
                raise ValueError("The two PCA passes consumed different feature content")
        else:
            for tokens in generator():
                started = time.monotonic()
                estimator.update(tokens)
                timers["pca_accumulation_seconds"] += time.monotonic() - started
        eig_start = time.monotonic()
        model = estimator.finalize(explained_variance=.99, eps=1e-6)
        eig_seconds = time.monotonic() - eig_start
        fit_seconds, fit_cpu_seconds = time.monotonic() - fit_start, time.process_time() - fit_cpu_start
        monitor.set_stage("finishing")
        expected_passes = 2 if args.mode.startswith("two") else 1
        if counts != {"generator_calls": expected_passes, "image_decodes": count * expected_passes,
                      "feature_forwards": count * expected_passes, "normal_patch_tokens_consumed": count * expected_passes * 1024}:
            raise ValueError("Actual feature/decode/pass counts differ from registered method")
        if sha(config_path) != config_sha or any(sha(ROOT / p) != h for p, h in source_hashes.items()):
            raise ValueError("Protocol or source changed during trial")
        receipt.update(
            status="completed", fit_seconds=fit_seconds, fit_cpu_seconds=fit_cpu_seconds,
            **timers, pca_eigendecomposition_seconds=eig_seconds,
            pca_total_kernel_seconds=timers["pca_accumulation_seconds"] + eig_seconds,
            formal_feature_and_loader_seconds=timers["image_decode_seconds"] + timers["feature_forward_seconds"],
            fit_other_seconds=fit_seconds - sum(timers.values()) - eig_seconds,
            operation_counts=counts, pass_feature_sha256=pass_feature_hashes,
            pca_rank=model.k, model_patch_count=model.count, retained_variance=model.retained_variance,
            model_mean_sha256=hashlib.sha256(model.mean.tobytes()).hexdigest(),
            model_projector_sha256=hashlib.sha256((model.components @ model.components.T).tobytes()).hexdigest(),
            dependency_versions={"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                                 "opencv": cv2.__version__, "pillow": PIL.__version__, "transformers": transformers.__version__},
            checkpoint_sha256=config["checkpoint_sha256"],
        )
        exit_code = 0
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if monitor is not None:
            try:
                receipt.update(monitor.close())
                if receipt["resource_violation"]:
                    receipt.update(status="failed", error=receipt["resource_violation"])
                    exit_code = 1
            except BaseException as error:
                receipt.update(status="failed", monitoring_error=f"{type(error).__name__}: {error}")
                exit_code = 1
        receipt.update(total_process_seconds=time.monotonic() - wall_start,
                       total_process_cpu_seconds=time.process_time() - cpu_start, exit_code=exit_code)
        write(output, receipt)
        summary = {k: receipt.get(k) for k in ("status", "dataset", "category", "mode", "repeat", "fit_seconds", "pca_rank", "error")}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
