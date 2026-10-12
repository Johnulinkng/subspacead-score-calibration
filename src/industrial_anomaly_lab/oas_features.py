"""Training-statistic-only OAS covariance scoring for the quick reproduction.

The Oracle Approximating Shrinkage (OAS) coefficient is computed separately
at every feature position. This implementation follows the scikit-learn
1.6.1 formula, including its single-feature special case and omitted 2/D
factor. It is not an independent invention or an exact equation-23 estimator.
The formula reference is pinned to:
https://github.com/scikit-learn/scikit-learn/blob/102e5daf66759cf896beeee229d1bf7c80ab2ba4/sklearn/covariance/_shrunk_covariance.py

``GaussianStatistics`` stores unbiased covariance, C = M2/(n-1). OAS expects
the maximum-likelihood covariance S = M2/n. We convert C to S before computing
the coefficient, and add the existing positive diagonal regularization once,
after shrinkage. Test images, labels and metric feedback are never inputs to
coefficient estimation. Distance calculation and rendering are inherited from
the frozen quick reproduction so only covariance estimation changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .quick_features import GaussianScorer, GaussianStatistics


OAS_REFERENCE_URL = (
    "https://github.com/scikit-learn/scikit-learn/blob/"
    "102e5daf66759cf896beeee229d1bf7c80ab2ba4/"
    "sklearn/covariance/_shrunk_covariance.py"
)


class OASGaussianScorer(GaussianScorer):
    """Reuse Gaussian distances with one OAS coefficient per position.

    Only one precision tensor and the supplied fitted statistics are retained.
    The ML covariance, squared entries and Cholesky factors are computed in
    small position chunks rather than as additional full covariance tensors.
    ``alphas`` is a read-only float64 vector in flattened H*W position order.
    """

    def __init__(
        self,
        stats: GaussianStatistics,
        *,
        regularization: float = 0.01,
        sigma: float = 4.0,
        model_size: int = 224,
        output_size: int = 256,
        position_chunk_size: int = 64,
        cpu_threads: int = 2,
        mode: str = "oas",
    ) -> None:
        stats.validate()
        if (
            isinstance(stats.count, bool)
            or not isinstance(stats.count, (int, np.integer))
            or stats.count < 2
        ):
            raise ValueError("OAS sample count must be an integer of at least two")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        for size in (model_size, output_size, position_chunk_size, cpu_threads):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("score sizes and cpu_threads must be positive integers")
        if mode not in ("oas", "ml_no_shrinkage"):
            raise ValueError("mode must be oas or ml_no_shrinkage")

        torch.set_num_threads(cpu_threads)
        self.feature_shape = stats.feature_shape
        self.mean = torch.from_numpy(stats.mean)
        self.precision = torch.empty_like(torch.from_numpy(stats.covariance))
        self.sigma = sigma
        self.model_size = model_size
        self.output_size = output_size
        self.position_chunk_size = position_chunk_size
        dimensions = self.feature_shape[0]
        positions = len(stats.mean)
        alphas = np.empty(positions, dtype=np.float64)
        diagonal_indices = np.arange(dimensions)
        ml_factor = (stats.count - 1) / stats.count

        for begin in range(0, positions, position_chunk_size):
            end = min(begin + position_chunk_size, positions)
            covariance = stats.covariance[begin:end].copy()
            covariance *= ml_factor
            target_scale = np.trace(covariance, axis1=1, axis2=2) / dimensions
            if dimensions == 1 or mode == "ml_no_shrinkage":
                # One-feature OAS and the denominator-only control use zero.
                coefficients = np.zeros(end - begin, dtype=np.float64)
            else:
                squared_mean = np.mean(covariance * covariance, axis=(1, 2))
                mean_squared = target_scale * target_scale
                numerator = squared_mean + mean_squared
                denominator = (stats.count + 1) * (
                    squared_mean - mean_squared / dimensions
                )
                coefficients = np.ones(end - begin, dtype=np.float64)
                np.divide(
                    numerator, denominator, out=coefficients,
                    where=denominator != 0,
                )
                np.minimum(coefficients, 1.0, out=coefficients)
            if not np.isfinite(coefficients).all():
                raise ValueError("OAS coefficients must be finite")
            alphas[begin:end] = coefficients
            covariance *= (1.0 - coefficients)[:, None, None]
            covariance[:, diagonal_indices, diagonal_indices] += (
                coefficients * target_scale + regularization
            )[:, None]
            cholesky = torch.linalg.cholesky(torch.from_numpy(covariance))
            self.precision[begin:end] = torch.cholesky_inverse(cholesky)

        alphas.setflags(write=False)
        self.alphas = alphas
        self.metadata: dict[str, Any] = {
            "method": (
                "PaDiM Gaussian with per-position OAS covariance"
                if mode == "oas" else "PaDiM Gaussian with ML covariance and no shrinkage"
            ),
            "mode": mode,
            "coefficient_estimation": (
                "per-position OAS from normal-training statistics"
                if mode == "oas" else "fixed zero, ML-denominator ablation"
            ),
            "oas_reference": OAS_REFERENCE_URL,
            "oas_formula": "scikit-learn 1.6.1, omitted 2/D factor",
            "oas_coefficient_input": "normal-training covariance and count only",
            "oas_coefficient_formula": (
                "a=mean(S*S), mu=trace(S)/D, "
                "alpha=min((a+mu^2)/((n+1)*(a-mu^2/D)),1); "
                "zero denominator -> 1; D=1 -> 0"
            ),
            "input_covariance_denominator": "n-1",
            "oas_covariance_denominator": "n",
            "ml_covariance_factor": ml_factor,
            "sample_count": int(stats.count),
            "shrinkage_target": "trace(ML covariance)/dimensions times identity",
            "alpha_summary": {
                "count": int(positions),
                "minimum": float(alphas.min()),
                "mean": float(alphas.mean()),
                "median": float(np.median(alphas)),
                "maximum": float(alphas.max()),
                "standard_deviation": float(alphas.std()),
                "zero_count": int(np.count_nonzero(alphas == 0)),
                "one_count": int(np.count_nonzero(alphas == 1)),
            },
            "regularization": regularization,
            "regularization_applied": "once, after OAS covariance shrinkage",
            "distance": "sqrt((feature-mean)^T precision (feature-mean))",
            "dimensions": dimensions,
            "spatial_smoothing": {
                "sigma": sigma, "plane_size": model_size, "mode": "reflect",
            },
            "output_size": output_size,
            "map_interpolation": "bilinear align_corners=False",
            "per_image_normalization": False,
        }


def prepare_oas_scorer(
    stats: GaussianStatistics, **settings: Any,
) -> OASGaussianScorer:
    """Prepare once per category from existing normal-training statistics."""
    return OASGaussianScorer(stats, mode="oas", **settings)


def prepare_ml_scorer(
    stats: GaussianStatistics, **settings: Any,
) -> OASGaussianScorer:
    """Isolate the n-denominator change without applying OAS shrinkage.

    This control shares the OAS covariance conversion, regularization and map
    geometry. Its coefficients are fixed at zero, so its covariance is
    ``(n-1)/n * C + epsilon * I``.
    """
    return OASGaussianScorer(stats, mode="ml_no_shrinkage", **settings)
