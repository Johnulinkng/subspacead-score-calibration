"""Bounded-memory PaDiM-style ResNet18 feature statistics and raw maps.

This project implementation reuses an official ImageNet-pretrained torchvision
ResNet18. The statistical model follows the PaDiM paper (Defard et al., 2020,
https://arxiv.org/abs/2011.08785): a multivariate Gaussian at each feature
position, with 100 fixed sampled feature dimensions. It is a reproduction
variant, not a claim to reproduce a published score.

Only a caller-supplied stream of normal training features is consumed by
``fit_statistics``. The caller owns dataset split validation; this module does
not discover datasets, inspect labels, select settings, or access test files.
Both the baseline and fixed shrinkage variant share the fitted statistics,
feature dimensions, and geometry. No per-image score normalization is used.

Typical usage::

    extractor = FeatureExtractor(weights, expected_sha256=sha, seed=123)
    stats = fit_statistics(
        (extractor(path) for path in normal_train_paths),
        channel_indices=extractor.channel_indices,
        metadata=extractor.metadata,
    )
    save_statistics(stats, category_stats_path)
    baseline = prepare_scorer(stats, shrinkage_alpha=0.0)
    candidate = prepare_scorer(stats, shrinkage_alpha=0.1)
    feature = extractor(test_path)
    baseline_map = baseline(feature)[0]
    candidate_map = candidate(feature)[0]

The default score geometry follows the model's 224-pixel plane: bilinear
upsampling from 56 to 224, Gaussian sigma=4 on that plane, then a bilinear
resize to the common 256-pixel evaluation plane. Scores remain raw distances.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


RESNET18_WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
FEATURE_LAYERS = ("layer1", "layer2", "layer3")
FEATURE_CHANNELS = 448
DEFAULT_FEATURE_SIZE = 56
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_channels(seed: int, dimensions: int = 100) -> np.ndarray:
    """Choose one immutable feature subset, independent of image contents."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(dimensions, int) or not 1 <= dimensions <= FEATURE_CHANNELS:
        raise ValueError(f"dimensions must be within 1..{FEATURE_CHANNELS}")
    return np.random.default_rng(seed).choice(
        FEATURE_CHANNELS, dimensions, replace=False
    ).astype(np.int64)


class FeatureExtractor:
    """Local-checkpoint-only frozen ResNet18; one image at a time.

    ``__call__`` accepts a path or a PIL image and returns float32[D,56,56]
    on the CPU. CUDA is optional and its PyTorch allocator is capped; CPU
    execution uses two threads by default and never starts loader workers.
    The complete checkpoint is strict-loaded before truncating the forward
    computation at layer3. There is no network or weight download path.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str,
        seed: int = 1887537248,
        dimensions: int = 100,
        device: str = "cpu",
        cpu_threads: int = 2,
        cuda_memory_limit_gib: float = 2.0,
        backbone: str = "resnet18",
        input_size: int = 224,
    ) -> None:
        if backbone != "resnet18" or input_size != 224:
            raise ValueError("this frozen reproduction supports ResNet18 at 224 only")
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        if cuda_memory_limit_gib <= 0:
            raise ValueError("cuda_memory_limit_gib must be positive")
        self.channel_indices = select_channels(seed, dimensions)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        torch.set_num_threads(cpu_threads)
        checkpoint = Path(checkpoint_path).resolve(strict=True)
        expected_sha256 = expected_sha256.lower().strip()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected_sha256 must be a complete SHA-256")
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != expected_sha256:
            raise ValueError("ResNet18 checkpoint SHA-256 mismatch")
        self.model = models.resnet18(weights=None, progress=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError("checkpoint must be a tensor-only state dict")
        if any(
            not bool(torch.isfinite(value).all())
            for value in state.values()
            if value.is_floating_point()
        ):
            raise ValueError("checkpoint contains nonfinite weights")
        self.model.load_state_dict(state, strict=True)
        del state
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA device is unavailable")
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, cuda_memory_limit_gib * 1024**3 / total), self.device
            )
        self.model.to(self.device).eval().requires_grad_(False)
        self._indices = torch.from_numpy(self.channel_indices).to(self.device)
        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.metadata: dict[str, Any] = {
            "method": "PaDiM-style ResNet18 reproduction variant",
            "method_paper": "https://arxiv.org/abs/2011.08785",
            "backbone": "torchvision.models.resnet18",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": actual_sha256,
            "checkpoint_source": RESNET18_WEIGHT_URL,
            "checkpoint_loading": "strict tensor-only weights_only=True",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "seed": seed,
            "dimensions": dimensions,
            "feature_channels": FEATURE_CHANNELS,
            "feature_layers": list(FEATURE_LAYERS),
            "feature_alignment": "nearest neighbor to layer1 geometry",
            "feature_size": [56, 56],
            "input_size": [224, 224],
            "input_color": "RGB",
            "resize": "PIL bilinear full-plane resize to 224x224",
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "batch_size": 1,
            "loader_workers": 0,
            "cpu_threads": cpu_threads,
            "device": str(self.device),
            "cuda_allocator_cap_gib": (
                cuda_memory_limit_gib if self.device.type == "cuda" else None
            ),
            "gradient_training": False,
            "implicit_download": False,
        }

    @torch.inference_mode()
    def __call__(self, image: str | Path | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            tensor = self._transform(image.convert("RGB"))
        else:
            with Image.open(image) as opened:
                tensor = self._transform(opened.convert("RGB"))
        hidden = tensor.unsqueeze(0).to(self.device)
        hidden = self.model.conv1(hidden)
        hidden = self.model.bn1(hidden)
        hidden = self.model.relu(hidden)
        hidden = self.model.maxpool(hidden)
        layer1 = self.model.layer1(hidden)
        layer2 = self.model.layer2(layer1)
        layer3 = self.model.layer3(layer2)
        features = torch.cat(
            [
                layer1,
                functional.interpolate(layer2, size=layer1.shape[-2:], mode="nearest"),
                functional.interpolate(layer3, size=layer1.shape[-2:], mode="nearest"),
            ],
            dim=1,
        ).index_select(1, self._indices)
        result = features[0].cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise ValueError("feature extraction produced nonfinite values")
        return result

    def extract(self, image: str | Path | Image.Image) -> np.ndarray:
        """Named alias for ``__call__``."""
        return self(image)


@dataclass(frozen=True)
class GaussianStatistics:
    """Float64 sample covariance per location, before regularization.

    Mean layout is [H*W,D]; covariance is [H*W,D,D]. The stored sample count
    must be at least two. Regularization/shrinkage are applied by the scorer,
    allowing paired methods to use exactly the same fitted training evidence.
    """

    mean: np.ndarray
    covariance: np.ndarray
    count: int
    feature_shape: tuple[int, int, int]
    channel_indices: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        dimensions, height, width = self.feature_shape
        if min(dimensions, height, width) < 1 or self.count < 2:
            raise ValueError("statistics require positive dimensions and >=2 samples")
        if self.mean.shape != (height * width, dimensions):
            raise ValueError("invalid mean shape")
        if self.covariance.shape != (height * width, dimensions, dimensions):
            raise ValueError("invalid covariance shape")
        if self.mean.dtype != np.float64 or self.covariance.dtype != np.float64:
            raise ValueError("statistics must use float64")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise ValueError("statistics contain nonfinite values")
        if self.channel_indices.shape != (dimensions,):
            raise ValueError("invalid channel-index shape")
        if not np.issubdtype(self.channel_indices.dtype, np.integer):
            raise ValueError("channel indices must be integers")
        if len(np.unique(self.channel_indices)) != dimensions:
            raise ValueError("channel indices must be unique")
        if (self.channel_indices < 0).any() or (
            self.channel_indices >= FEATURE_CHANNELS
        ).any():
            raise ValueError("channel indices outside the ResNet18 feature plane")
        # Symmetry is checked per chunk so the check does not allocate a second
        # full covariance tensor (~251 MB for the standard 100 dimensions).
        for begin in range(0, height * width, 64):
            covariance = self.covariance[begin : begin + 64]
            if not np.allclose(
                covariance, covariance.transpose(0, 2, 1), atol=1e-10, rtol=1e-10
            ):
                raise ValueError("covariance must be symmetric")


def _feature_batch(features: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    array = np.asarray(features)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or min(array.shape) < 1:
        raise ValueError("features must have shape [D,H,W] or [B,D,H,W]")
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("features must contain finite floating-point values")
    return array


def fit_statistics(
    features: Iterable[np.ndarray | torch.Tensor],
    *,
    channel_indices: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    batch_size: int = 16,
    position_chunk_size: int = 64,
    cpu_threads: int = 2,
    progress_callback: Callable[[int], None] | None = None,
) -> GaussianStatistics:
    """Fit all supplied normal training features with float64 Welford merges.

    Only ``batch_size`` feature images plus one covariance tensor and one
    small position chunk are retained. This is independent of train-set size;
    the iterator is consumed exactly once. Test/GT paths are not accepted or
    discovered: the caller supplies a validated normal-training stream.
    """
    if min(batch_size, position_chunk_size, cpu_threads) < 1:
        raise ValueError("fit sizes and cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    count = 0
    shape: tuple[int, int, int] | None = None
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    pending: list[np.ndarray] = []

    def merge_pending() -> None:
        nonlocal count, mean, m2
        if not pending or shape is None:
            return
        dimensions, height, width = shape
        values = torch.from_numpy(np.stack(pending)).to(torch.float64)
        # [P,D,N] supports small-position batched matrix multiplication.
        values = values.reshape(len(pending), dimensions, height * width).permute(2, 1, 0)
        incoming = len(pending)
        new_count = count + incoming
        if mean is None:
            mean = torch.zeros((height * width, dimensions), dtype=torch.float64)
            m2 = torch.zeros(
                (height * width, dimensions, dimensions), dtype=torch.float64
            )
        assert m2 is not None
        for begin in range(0, height * width, position_chunk_size):
            block = values[begin : begin + position_chunk_size]
            block_mean = block.mean(dim=-1)
            centered = block - block_mean.unsqueeze(-1)
            block_m2 = torch.bmm(centered, centered.transpose(1, 2))
            current_mean = mean[begin : begin + position_chunk_size]
            delta = block_mean - current_mean
            if count:
                block_m2.add_(
                    delta.unsqueeze(-1) * delta.unsqueeze(-2),
                    alpha=count * incoming / new_count,
                )
            m2[begin : begin + position_chunk_size].add_(block_m2)
            current_mean.add_(delta, alpha=incoming / new_count)
        count = new_count
        pending.clear()
        if progress_callback is not None:
            progress_callback(count)

    for supplied in features:
        batch = _feature_batch(supplied)
        incoming_shape = tuple(int(size) for size in batch.shape[1:])
        if shape is None:
            shape = incoming_shape
        elif incoming_shape != shape:
            raise ValueError("feature geometry changed within the training stream")
        for sample in batch:
            pending.append(np.ascontiguousarray(sample).copy())
            if len(pending) == batch_size:
                merge_pending()
    merge_pending()
    if count < 2 or shape is None or mean is None or m2 is None:
        raise ValueError("at least two normal training feature images are required")
    dimensions = shape[0]
    indices = (
        np.arange(dimensions, dtype=np.int64)
        if channel_indices is None
        else np.asarray(channel_indices, dtype=np.int64).copy()
    )
    m2.div_(count - 1)
    stats = GaussianStatistics(
        mean=mean.numpy(),
        covariance=m2.numpy(),
        count=count,
        feature_shape=shape,
        channel_indices=indices,
        metadata={
            **dict(metadata or {}),
            "fitting": "float64 chunked Welford sample covariance",
            "covariance_denominator": "n-1",
            "sample_count": count,
            "fit_dataset_access": "caller-supplied normal training stream only",
            "fit_batch_size": batch_size,
            "fit_position_chunk_size": position_chunk_size,
        },
    )
    stats.validate()
    return stats


def save_statistics(stats: GaussianStatistics, path: str | Path) -> None:
    """Save an exclusive tensor/number/text NPZ; never use pickle."""
    stats.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez(
            stream,
            schema_version=np.asarray(1, dtype=np.int64),
            mean=stats.mean,
            covariance=stats.covariance,
            count=np.asarray(stats.count, dtype=np.int64),
            feature_shape=np.asarray(stats.feature_shape, dtype=np.int64),
            channel_indices=stats.channel_indices,
            metadata_json=np.asarray(
                json.dumps(stats.metadata, ensure_ascii=False, sort_keys=True)
            ),
        )


def load_statistics(path: str | Path) -> GaussianStatistics:
    """Load a safe NPZ without executing serialized Python objects."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema_version", "mean", "covariance", "count", "feature_shape",
            "channel_indices", "metadata_json",
        }
        if set(archive.files) != required or int(archive["schema_version"]) != 1:
            raise ValueError("unsupported statistics schema")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("statistics metadata must be a JSON object")
        stats = GaussianStatistics(
            mean=archive["mean"].copy(),
            covariance=archive["covariance"].copy(),
            count=int(archive["count"]),
            feature_shape=tuple(int(value) for value in archive["feature_shape"]),
            channel_indices=archive["channel_indices"].copy(),
            metadata=metadata,
        )
    stats.validate()
    return stats


class GaussianScorer:
    """Precompute one regularized precision matrix per feature location."""

    def __init__(
        self,
        stats: GaussianStatistics,
        *,
        shrinkage_alpha: float = 0.0,
        regularization: float = 0.01,
        sigma: float = 4.0,
        model_size: int = 224,
        output_size: int = 256,
        position_chunk_size: int = 64,
        cpu_threads: int = 2,
    ) -> None:
        stats.validate()
        if not np.isfinite(shrinkage_alpha) or not 0 <= shrinkage_alpha <= 1:
            raise ValueError("shrinkage_alpha must be within [0,1]")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if min(model_size, output_size, position_chunk_size, cpu_threads) < 1:
            raise ValueError("score sizes and cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
        self.feature_shape = stats.feature_shape
        self.mean = torch.from_numpy(stats.mean)
        self.precision = torch.empty_like(torch.from_numpy(stats.covariance))
        self.sigma = sigma
        self.model_size = model_size
        self.output_size = output_size
        self.position_chunk_size = position_chunk_size
        dimensions = self.feature_shape[0]
        for begin in range(0, len(stats.mean), position_chunk_size):
            covariance = torch.from_numpy(
                stats.covariance[begin : begin + position_chunk_size]
            ).clone()
            diagonal = covariance.diagonal(dim1=-2, dim2=-1)
            target_scale = diagonal.mean(dim=-1).clone()
            covariance.mul_(1.0 - shrinkage_alpha)
            diagonal.add_(target_scale[:, None] * shrinkage_alpha + regularization)
            cholesky = torch.linalg.cholesky(covariance)
            self.precision[begin : begin + position_chunk_size] = (
                torch.cholesky_inverse(cholesky)
            )
        self.metadata: dict[str, Any] = {
            "method": "PaDiM Gaussian" if shrinkage_alpha == 0 else "fixed-shrinkage PaDiM Gaussian",
            "shrinkage_alpha": shrinkage_alpha,
            "shrinkage_target": "trace(covariance)/dimensions times identity",
            "regularization": regularization,
            "distance": "sqrt((feature-mean)^T precision (feature-mean))",
            "dimensions": dimensions,
            "spatial_smoothing": {"sigma": sigma, "plane_size": model_size, "mode": "reflect"},
            "output_size": output_size,
            "map_interpolation": "bilinear align_corners=False",
            "per_image_normalization": False,
        }

    @torch.inference_mode()
    def distance_maps(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return unsmoothed float32[B,model_size,model_size] distances.

        Reuse these maps for sigma=0 and sigma=4 to avoid storing or computing
        duplicate precision tensors for the smoothing ablation.
        """
        batch = _feature_batch(features)
        if tuple(batch.shape[1:]) != self.feature_shape:
            raise ValueError("prediction feature shape differs from fitted statistics")
        dimensions, height, width = self.feature_shape
        output: list[np.ndarray] = []
        for sample in batch:
            vector = torch.from_numpy(np.ascontiguousarray(sample)).to(torch.float64)
            vector = vector.reshape(dimensions, height * width).transpose(0, 1)
            squared = torch.empty(height * width, dtype=torch.float64)
            for begin in range(0, height * width, self.position_chunk_size):
                difference = vector[begin : begin + self.position_chunk_size] - self.mean[
                    begin : begin + self.position_chunk_size
                ]
                product = torch.bmm(
                    self.precision[begin : begin + self.position_chunk_size],
                    difference.unsqueeze(-1),
                ).squeeze(-1)
                squared[begin : begin + self.position_chunk_size] = (
                    difference * product
                ).sum(dim=-1)
            distance = squared.clamp_min_(0).sqrt_().reshape(1, 1, height, width)
            model_map = functional.interpolate(
                distance.float(), size=(self.model_size, self.model_size),
                mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            if not np.isfinite(model_map).all():
                raise ValueError("prediction produced nonfinite anomaly scores")
            output.append(model_map.astype(np.float32, copy=False))
        return np.stack(output)

    def distance_map(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Alias returning the same batched native-plane distances."""
        return self.distance_maps(features)

    def __call__(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        return render_maps(
            self.distance_maps(features), sigma=self.sigma,
            output_size=self.output_size,
        )


def render_maps(
    distance_maps: np.ndarray,
    *,
    sigma: float = 4.0,
    output_size: int = 256,
) -> np.ndarray:
    """Smooth native-plane distances then resize, preserving their scale.

    Returns float32[B,output_size,output_size]. An unbatched native map [H,W]
    is accepted as a batch of one. Gaussian smoothing is never applied across
    the batch dimension.
    """
    array = np.asarray(distance_maps)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError("distance maps must have shape [H,W] or [B,H,W]")
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("distance maps must contain finite floating-point values")
    if not np.isfinite(sigma) or sigma < 0 or output_size < 1:
        raise ValueError("sigma must be nonnegative and output_size positive")
    output = []
    for model_map in array:
        model_map = model_map.astype(np.float32, copy=False)
        if sigma:
            model_map = gaussian_filter(model_map, sigma=sigma, mode="reflect")
        if model_map.shape != (output_size, output_size):
            model_map = functional.interpolate(
                torch.from_numpy(np.ascontiguousarray(model_map))[None, None],
                size=(output_size, output_size), mode="bilinear", align_corners=False,
            )[0, 0].numpy()
        output.append(model_map)
    return np.stack(output)


def prepare_scorer(stats: GaussianStatistics, **settings: Any) -> GaussianScorer:
    """Construct once per category/variant, then reuse for every test image."""
    return GaussianScorer(stats, **settings)


def predict_maps(
    features: np.ndarray | torch.Tensor,
    stats: GaussianStatistics,
    **settings: Any,
) -> np.ndarray:
    """Convenience API returning raw float32[B,output_size,output_size].

    For a complete category, use ``prepare_scorer`` once to avoid repeatedly
    factoring the covariance matrices.
    """
    return prepare_scorer(stats, **settings)(features)
