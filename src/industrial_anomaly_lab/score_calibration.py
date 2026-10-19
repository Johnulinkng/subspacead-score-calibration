"""Normal-only residual calibration; an exploratory adapter, not a novelty claim.

The local candidate smooths positional robust statistics and shrinks them
towards global statistics. Its ablation removes only this shrinkage. Inputs
are raw, consistently rendered residual maps, never display heatmaps or GT.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

VARIANTS = ("baseline", "global", "local_shrunk", "local_no_shrink")


def valid_map(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite nonempty 2D raw score map")
    return values


@dataclass
class NormalCalibration:
    global_center: float
    global_scale: float
    local_center: np.ndarray
    local_scale: np.ndarray
    reliability: float
    count: int
    floor_ratio: float

    def apply(self, values, variant):
        values = valid_map(values)
        if values.shape != self.local_center.shape:
            raise ValueError("Map geometry differs from fitted calibration")
        if variant == "baseline":
            return values.copy()
        if variant == "global":
            return (values - self.global_center) / self.global_scale
        if variant not in ("local_shrunk", "local_no_shrink"):
            raise ValueError("Unknown calibration variant")
        alpha = self.reliability if variant == "local_shrunk" else 1.0
        center = alpha * self.local_center + (1 - alpha) * self.global_center
        scale = alpha * self.local_scale + (1 - alpha) * self.global_scale
        scale = np.maximum(scale, self.floor_ratio * self.global_scale)
        return (values - center) / scale

    @property
    def parameter_bytes(self):
        return self.local_center.nbytes + self.local_scale.nbytes + 5 * 8


def fit_normal_calibration(normal_maps, *, prior_count=32, smoothing_sigma=8.0,
                           floor_ratio=0.05):
    maps = np.asarray(normal_maps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[0] < 8 or min(maps.shape[1:]) < 2:
        raise ValueError("At least 8 finite normal maps with fixed geometry required")
    if not np.isfinite(maps).all() or np.any(maps < 0):
        raise ValueError("Training residual maps must be finite and nonnegative")
    if prior_count <= 0 or smoothing_sigma < 0 or not 0 < floor_ratio <= 1:
        raise ValueError("Invalid frozen calibration settings")
    global_center = float(np.median(maps))
    global_scale = float(1.4826 * np.median(np.abs(maps - global_center)))
    global_scale = max(global_scale, abs(global_center) * 1e-6, 1e-12)
    center = np.median(maps, axis=0)
    scale = 1.4826 * np.median(np.abs(maps - center[None]), axis=0)
    center = gaussian_filter(center, smoothing_sigma, mode="reflect")
    scale = gaussian_filter(scale, smoothing_sigma, mode="reflect")
    return NormalCalibration(global_center, global_scale, center, scale,
                             len(maps) / (len(maps) + prior_count), len(maps), floor_ratio)


def save_calibration(path, model):
    np.savez_compressed(path, global_center=model.global_center, global_scale=model.global_scale,
                        local_center=model.local_center, local_scale=model.local_scale,
                        reliability=model.reliability, count=model.count,
                        floor_ratio=model.floor_ratio)


def load_calibration(path):
    with np.load(path, allow_pickle=False) as a:
        return NormalCalibration(float(a["global_center"]), float(a["global_scale"]),
                                 a["local_center"].copy(), a["local_scale"].copy(),
                                 float(a["reliability"]), int(a["count"]), float(a["floor_ratio"]))
