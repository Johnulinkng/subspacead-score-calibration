"""Bounded-memory, training-only DINOv2-S / SubspaceAD reproduction.

The feature and scoring choices follow CLendering/SubspaceAD commit
ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7. In particular, selected hidden
states are averaged *before* one global PCA is fitted, and anomaly scores are
the squared reconstruction residual. Tokens are not L2-normalized. This is a
reproduction adapter of public methods, not a new detection method.

The upstream two-pass PCA is replaced by a mathematically equivalent Chan
mean/scatter accumulator in float64. No rank is discarded during streaming,
and no complete training feature bank is held in memory. Rank selection uses
only the final normal-training covariance. Dataset validation, input seals,
and resource admission belong to the caller.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


UPSTREAM_COMMIT = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
DEFAULT_LAYERS = (-4, -5)
MODEL_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_matrix(features: np.ndarray, feature_dim: int) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != feature_dim:
        raise ValueError(f"features must have shape [N,{feature_dim}]")
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("features must be float32 or float64")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    return values


class LocalDinoExtractor:
    """Frozen, local-only DINOv2-small feature extraction, one image at a time.

    The checkpoint is authenticated before Transformers can open it. Local
    loading, no remote code, eager float32 attention, and the slow PIL image
    processor reproduce the upstream feature path. Attention outputs are
    omitted because this experiment does not use a saliency/background mask;
    this avoids retaining attention matrices while leaving hidden states
    unchanged.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        expected_checkpoint_sha256: str,
        checkpoint_filename: str = "model.safetensors",
        expected_artifact_sha256: Mapping[str, str] | None = None,
        image_res: int = 448,
        layers: Sequence[int] = DEFAULT_LAYERS,
        device: str = "cpu",
        cpu_threads: int = 4,
        max_cuda_memory_gib: float = 2.0,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        directory = Path(model_dir).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("model_dir must be a local directory")
        if checkpoint_filename not in ("model.safetensors", "pytorch_model.bin"):
            raise ValueError("only an unsharded named local checkpoint is supported")
        checkpoint = (directory / checkpoint_filename).resolve(strict=True)
        if not checkpoint.is_relative_to(directory):
            raise ValueError("checkpoint escapes model_dir")
        if len(expected_checkpoint_sha256) != 64:
            raise ValueError("expected checkpoint SHA256 must be supplied")
        actual_sha = _sha256(checkpoint)
        if actual_sha != expected_checkpoint_sha256.lower():
            raise ValueError("DINO checkpoint SHA256 mismatch")
        for filename, expected in (expected_artifact_sha256 or {}).items():
            artifact = (directory / filename).resolve(strict=True)
            if not artifact.is_relative_to(directory) or not artifact.is_file():
                raise ValueError("model artifact escapes local model directory")
            if _sha256(artifact) != expected.lower():
                raise ValueError(f"DINO artifact SHA256 mismatch: {filename}")
        if isinstance(image_res, bool) or image_res <= 0 or image_res % 14:
            raise ValueError("image_res must be a positive multiple of 14")
        if not layers or any(isinstance(x, bool) or not isinstance(x, int) for x in layers):
            raise ValueError("layers must be a nonempty sequence of integer indices")
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")
        if device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("device must be cpu or the first CUDA device")
        torch.set_num_threads(cpu_threads)
        self.device = torch.device(device)
        cuda_fraction = None
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("requested CUDA device is unavailable")
            if not np.isfinite(max_cuda_memory_gib) or max_cuda_memory_gib <= 0:
                raise ValueError("CUDA memory budget must be positive and finite")
            total_bytes = torch.cuda.get_device_properties(self.device).total_memory
            cuda_fraction = min(1.0, max_cuda_memory_gib * 1024**3 / total_bytes)
            torch.cuda.set_per_process_memory_fraction(cuda_fraction, self.device)
        self.processor = AutoImageProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_fast=False,
        )
        self.model, loading_info = AutoModel.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=checkpoint_filename.endswith(".safetensors"),
            torch_dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        loading_fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        rejected = {name: loading_info.get(name) for name in loading_fields if loading_info.get(name)}
        if rejected:
            raise ValueError(f"Incomplete DINO checkpoint loading: {rejected}")
        self.model.eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        if (
            cfg.model_type != "dinov2"
            or cfg.hidden_size != 384
            or cfg.num_hidden_layers != 12
            or cfg.patch_size != 14
            or getattr(cfg, "num_register_tokens", 0) != 0
        ):
            raise ValueError("checkpoint must be the plain 12-layer DINOv2-small")
        self.model.to(self.device)
        self.image_res = image_res
        self.layers = tuple(layers)
        state_count = cfg.num_hidden_layers + 1
        if any(not -state_count <= index < state_count for index in self.layers):
            raise ValueError("hidden-state index is out of bounds")
        self.grid_size = (image_res // cfg.patch_size,) * 2
        self.feature_dim = cfg.hidden_size
        self.metadata: dict[str, Any] = {
            "backbone": "facebook/dinov2-small",
            "checkpoint_sha256": actual_sha,
            "checkpoint_filename": checkpoint_filename,
            "subspacead_commit": UPSTREAM_COMMIT,
            "model_resolution": image_res,
            "hidden_state_layers": list(self.layers),
            "hidden_state_count": state_count,
            "resolved_hidden_state_indices": [index % state_count for index in self.layers],
            "aggregation": "mean_before_pca",
            "feature_dim": self.feature_dim,
            "feature_grid": list(self.grid_size),
            "drop_front_tokens": 1,
            "token_normalization": "none",
            "center_crop": False,
            "clahe": False,
            "dtype": "float32",
            "device": str(self.device),
            "cpu_threads": cpu_threads,
            "cuda_memory_fraction": cuda_fraction,
            "max_cuda_memory_gib": max_cuda_memory_gib if cuda_fraction else None,
            "attention": "eager_without_retained_attention_outputs",
            "checkpoint_loading": {
                "strict": True,
                "state_dict_key_count": len(self.model.state_dict()),
                **{name: len(loading_info.get(name) or ()) for name in loading_fields},
            },
        }

    def extract_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        if len(images) != 1:
            raise ValueError("resource-bounded extractor requires exactly one image")
        if not isinstance(images[0], Image.Image):
            raise TypeError("extract_batch expects PIL images")
        inputs = self.processor(
            images=[images[0].convert("RGB")],
            return_tensors="pt",
            do_resize=True,
            size={"height": self.image_res, "width": self.image_res},
            do_center_crop=False,
            crop_size={"height": self.image_res, "width": self.image_res},
        ).to(self.device)
        if tuple(inputs.pixel_values.shape) != (1, 3, self.image_res, self.image_res):
            raise ValueError("processor did not produce the requested full-frame geometry")
        with torch.inference_mode():
            outputs = self.model(
                **inputs, output_hidden_states=True, output_attentions=False,
            )
            hidden_states = outputs.hidden_states
            if len(hidden_states) != self.metadata["hidden_state_count"]:
                raise ValueError("unexpected DINO hidden-state indexing")
            patch_count = self.grid_size[0] * self.grid_size[1]
            selected = []
            for index in self.layers:
                state = hidden_states[index]
                if tuple(state.shape) != (1, patch_count + 1, self.feature_dim):
                    raise ValueError("unexpected patch-token geometry")
                selected.append(state[:, 1:, :].reshape(1, *self.grid_size, self.feature_dim))
            fused = torch.stack(selected, dim=0).mean(dim=0)
            result = fused.cpu().numpy().copy()
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("DINO produced invalid float32 tokens")
        return result

    def extract(self, image: Image.Image) -> np.ndarray:
        return self.extract_batch([image])[0]

    def __call__(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            return self.extract(image)


class StreamingPCA:
    """Exact-rank float64 sufficient statistics for global PCA, one pass."""

    def __init__(self, feature_dim: int = 384) -> None:
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.count = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.scatter = np.zeros((feature_dim, feature_dim), dtype=np.float64)

    def update(self, tokens: np.ndarray) -> None:
        values = _finite_matrix(tokens, self.feature_dim)
        batch_count = values.shape[0]
        if batch_count == 0:
            raise ValueError("empty feature batches are not allowed")
        batch = np.asarray(values, dtype=np.float64)
        batch_mean = batch.mean(axis=0, dtype=np.float64)
        centered = batch - batch_mean
        batch_scatter = centered.T @ centered
        if self.count == 0:
            self.mean[:] = batch_mean
            self.scatter[:] = batch_scatter
            self.count = batch_count
            return
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.scatter += batch_scatter
        self.scatter += np.outer(delta, delta) * (self.count * batch_count / combined_count)
        self.mean += delta * (batch_count / combined_count)
        self.count = combined_count

    def finalize(self, explained_variance: float = 0.99, eps: float = 1e-6) -> SubspaceModel:
        if self.count < 2:
            raise ValueError("PCA requires at least two normal-training tokens")
        if not np.isfinite(explained_variance) or not 0 < explained_variance <= 1:
            raise ValueError("explained_variance must be within (0,1]")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        covariance = self.scatter / (self.count - 1)
        # Remove only roundoff asymmetry; do not truncate an intermediate rank.
        covariance = (covariance + covariance.T) * 0.5
        evals, evecs = np.linalg.eigh(covariance)
        eigenvalues = evals[::-1]
        tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * np.finfo(np.float64).eps * self.feature_dim * 20
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError("training covariance is not positive semidefinite")
        eigenvalues = np.maximum(eigenvalues, 0)
        total_variance = float(eigenvalues.sum())
        if total_variance == 0:
            # Explicit definition for a degenerate constant training set.
            k = 1
            retained = 1.0
        else:
            cumulative = np.cumsum(eigenvalues) / total_variance
            k = min(self.feature_dim, int(np.searchsorted(cumulative, explained_variance, side="left")) + 1)
            retained = float(cumulative[k - 1])
        return SubspaceModel(
            mean=self.mean.copy(),
            components=evecs[:, ::-1][:, :k].copy(),
            eigenvalues=eigenvalues[:k].copy(),
            count=self.count,
            variance_target=float(explained_variance),
            retained_variance=retained,
            total_variance=total_variance,
            eps=float(eps),
            metadata={
                "fit_algorithm": "float64_chan_mean_scatter_no_intermediate_rank_truncation",
                "covariance_denominator": "normal_patch_token_count-1",
                "subspacead_commit": UPSTREAM_COMMIT,
                "token_normalization": "none",
                "aggregation": "mean_before_pca",
                "scoring": "squared_reconstruction_residual",
                "whiten": False,
                "drop_k": 0,
            },
        )


@dataclass
class SubspaceModel:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    count: int
    variance_target: float
    retained_variance: float
    total_variance: float
    eps: float = 1e-6
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.components = np.asarray(self.components, dtype=np.float64)
        self.eigenvalues = np.asarray(self.eigenvalues, dtype=np.float64)
        dimension = self.mean.size
        if self.mean.ndim != 1 or dimension == 0:
            raise ValueError("PCA mean must be a nonempty vector")
        if self.components.ndim != 2 or self.components.shape[0] != dimension:
            raise ValueError("invalid PCA component shape")
        if not 1 <= self.components.shape[1] <= dimension:
            raise ValueError("invalid PCA rank")
        if self.eigenvalues.shape != (self.components.shape[1],):
            raise ValueError("invalid eigenvalue shape")
        if any(not np.isfinite(values).all() for values in (self.mean, self.components, self.eigenvalues)):
            raise ValueError("PCA arrays must be finite")
        if np.any(self.eigenvalues < 0) or np.any(np.diff(self.eigenvalues) > 0):
            raise ValueError("PCA eigenvalues must be nonnegative and descending")
        if not np.allclose(self.components.T @ self.components, np.eye(self.k), rtol=1e-9, atol=1e-9):
            raise ValueError("PCA components must be orthonormal")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 2:
            raise ValueError("PCA count must be at least two")
        if not 0 < self.variance_target <= 1 or not np.isfinite(self.variance_target):
            raise ValueError("invalid PCA variance target")
        if not 0 <= self.retained_variance <= 1 + 1e-12 or not np.isfinite(self.retained_variance):
            raise ValueError("invalid PCA retained variance")
        if self.total_variance < 0 or not np.isfinite(self.total_variance):
            raise ValueError("invalid PCA total variance")
        if self.eps <= 0 or not np.isfinite(self.eps):
            raise ValueError("invalid PCA epsilon")
        self.metadata = dict(self.metadata or {})
        for values in (self.mean, self.components, self.eigenvalues):
            values.flags.writeable = False

    @property
    def k(self) -> int:
        return self.components.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.mean.size

    def reconstruction_scores(self, tokens: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray:
        """Use upstream reconstruction arithmetic at the supplied feature dtype."""
        values = _finite_matrix(tokens, self.feature_dim)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        result = np.empty(values.shape[0], dtype=values.dtype)
        mean = np.asarray(self.mean, dtype=values.dtype)
        components = np.asarray(self.components, dtype=values.dtype)
        for start in range(0, values.shape[0], chunk_size):
            batch = values[start : start + chunk_size]
            projected = (batch - mean) @ components
            reconstructed = projected @ components.T + mean
            result[start : start + chunk_size] = np.sum((batch - reconstructed) ** 2, axis=1)
        if not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("invalid reconstruction scores")
        return result


def render_subspace_map(
    score_grid: np.ndarray, *, model_size: int = 448, output_size: int = 256,
) -> np.ndarray:
    """Upstream OpenCV resize/3x3 blur, then common evaluation-plane resize."""
    import cv2

    scores = np.asarray(score_grid)
    if scores.ndim != 2 or 0 in scores.shape or not np.isfinite(scores).all():
        raise ValueError("score_grid must be a finite nonempty 2D map")
    if model_size < 1 or output_size < 1:
        raise ValueError("map dimensions must be positive")
    resized = cv2.resize(scores.astype(np.float32), (model_size, model_size), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized, (3, 3), 4.0)
    if output_size != model_size:
        blurred = cv2.resize(blurred, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return np.asarray(blurred, dtype=np.float32)


def image_aggregates(raw_map: np.ndarray, *, fraction: float = 0.01) -> dict[str, float | int]:
    """Max and top-percent mean on one raw map; no per-image normalization."""
    values = np.asarray(raw_map)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("raw_map must be a finite nonempty 2D map")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be within (0,1]")
    flat = values.ravel()
    count = max(1, int(flat.size * fraction))
    highest = np.partition(flat, flat.size - count)[-count:]
    # Float64 mean is the existing repository's common evaluation convention.
    return {"max": float(flat.max()), "top1p": float(highest.mean(dtype=np.float64)), "top_count": count}


def save_subspace_model(
    model: SubspaceModel, path: str | Path, metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save sufficient fitted state without pickles or a training feature bank."""
    destination = Path(path)
    combined_metadata = {**model.metadata, **dict(metadata or {})}
    text = json.dumps(combined_metadata, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with destination.open("xb") as stream:
        np.savez(
            stream,
            schema_version=np.asarray(MODEL_SCHEMA_VERSION, dtype=np.int64),
            mean=model.mean,
            components=model.components,
            eigenvalues=model.eigenvalues,
            count=np.asarray(model.count, dtype=np.int64),
            variance_target=np.asarray(model.variance_target, dtype=np.float64),
            retained_variance=np.asarray(model.retained_variance, dtype=np.float64),
            total_variance=np.asarray(model.total_variance, dtype=np.float64),
            eps=np.asarray(model.eps, dtype=np.float64),
            metadata_json=np.asarray(text),
        )


def load_subspace_model(path: str | Path) -> SubspaceModel:
    expected_fields = {
        "schema_version", "mean", "components", "eigenvalues", "count", "variance_target",
        "retained_variance", "total_variance", "eps", "metadata_json",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise ValueError("invalid Subspace model fields")
        if int(archive["schema_version"]) != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported Subspace model schema")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("invalid model metadata")
        return SubspaceModel(
            mean=archive["mean"].copy(),
            components=archive["components"].copy(),
            eigenvalues=archive["eigenvalues"].copy(),
            count=int(archive["count"]),
            variance_target=float(archive["variance_target"]),
            retained_variance=float(archive["retained_variance"]),
            total_variance=float(archive["total_variance"]),
            eps=float(archive["eps"]),
            metadata=metadata,
        )
