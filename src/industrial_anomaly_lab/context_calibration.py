"""Content-conditioned normal-tail adapter prepared for a paired experiment.

Centroids are fitted on normal descriptors only. Descriptor precision and
distance arithmetic are explicit; no test label, image or mask enters fit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .subspace_features import render_subspace_map

VARIANTS = ("baseline", "global", "conditional_tail", "conditioning_removed")


def quantized_descriptor(tokens, pca, dimension=16):
    tokens = np.asarray(tokens, dtype=np.float32)
    dimension = min(dimension, pca.k)
    if tokens.ndim != 2 or tokens.shape[1] != pca.feature_dim or not np.isfinite(tokens).all():
        raise ValueError("Invalid raw feature tokens")
    values = ((tokens - pca.mean.astype(np.float32)) @ pca.components[:, :dimension].astype(np.float32))
    values /= np.sqrt(pca.eigenvalues[:dimension].astype(np.float32) + np.float32(1e-6))
    if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
        raise ValueError("Descriptor cannot be represented as float16")
    return values.astype(np.float16)


def nearest_centers(descriptor, centers, chunk_size=2048):
    descriptor = np.asarray(descriptor)
    centers = np.asarray(centers, dtype=np.float64)
    if descriptor.ndim != 2 or centers.ndim != 2 or descriptor.shape[1] != centers.shape[1]:
        raise ValueError("Descriptor/centroid geometry mismatch")
    if not np.isfinite(descriptor).all() or not np.isfinite(centers).all():
        raise ValueError("Nonfinite content descriptors")
    labels = np.empty(len(descriptor), dtype=np.uint8)
    if len(centers) > 256:
        raise ValueError("Too many centers for label storage")
    for start in range(0, len(descriptor), chunk_size):
        values = descriptor[start:start + chunk_size].astype(np.float64)
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:start + len(values)] = distances.argmin(axis=1)
    return labels


@dataclass
class ContextCalibration:
    centers: np.ndarray
    tails: np.ndarray
    global_tail: float
    global_center: float
    global_scale: float

    def apply(self, residual_grid, descriptor, variant):
        raw = np.asarray(residual_grid, dtype=np.float32)
        if raw.shape != (32, 32) or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Expected a finite 32x32 raw residual grid")
        if variant == "baseline":
            return render_subspace_map(raw)
        if variant == "global":
            return (render_subspace_map(raw).astype(np.float64) - self.global_center) / self.global_scale
        if variant == "conditional_tail":
            labels = nearest_centers(descriptor, self.centers)
            if len(labels) != raw.size:
                raise ValueError("Content descriptor token count mismatch")
            envelope = self.tails[labels].reshape(raw.shape)
        elif variant == "conditioning_removed":
            envelope = self.global_tail
        else:
            raise ValueError("Unknown calibration variant")
        scores = raw.astype(np.float64) * np.minimum(1., raw / envelope)
        return render_subspace_map(scores)

    @property
    def parameter_bytes(self):
        return self.centers.nbytes + self.tails.nbytes + 3 * 8


def fit_context_from_cache(cache, minimum_group_tokens=256, tail_quantile=.95):
    z = np.asarray(cache["latent"])
    r = np.asarray(cache["residual_grid"])
    partition = np.asarray(cache["statistics_partition"], dtype=bool)
    centers = np.asarray(cache["centers"], dtype=np.float32)
    if z.dtype != np.float16 or r.dtype != np.float32 or len(z) != len(r) or len(z) != len(partition):
        raise ValueError("Normal cache precision/count differs from the frozen pilot")
    labels = nearest_centers(z[partition].reshape(-1, z.shape[-1]), centers)
    fit_scores = r[partition].ravel()
    global_tail = max(float(np.quantile(fit_scores, tail_quantile, method="higher")), 1e-12)
    tails = []
    counts = []
    for k in range(len(centers)):
        values = fit_scores[labels == k]
        tail = float(np.quantile(values, tail_quantile, method="higher")) if len(values) >= minimum_group_tokens else global_tail
        tails.append(max(global_tail, tail))
        counts.append(len(values))
    maps = np.stack([render_subspace_map(grid) for grid in r[partition]]).astype(np.float64)
    center = float(np.median(maps))
    scale = max(float(1.4826 * np.median(np.abs(maps - center))), abs(center) * 1e-6, 1e-12)
    return ContextCalibration(centers, np.asarray(tails, dtype=np.float64), global_tail, center, scale), {"fit_cluster_counts": counts, "statistics_images": int(partition.sum()), "threshold_images": int((~partition).sum())}
