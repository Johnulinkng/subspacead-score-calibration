"""Normal-only bounded suppression with identity above a local normal envelope.

An exploratory score adapter. Quantile estimation and spatial calibration
are established tools; this implementation does not establish originality.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

VARIANTS = ("baseline", "global", "local_guarded", "protection_removed")


@dataclass
class TailCalibration:
    global_center: float
    global_scale: float
    global_tail: float
    envelope: np.ndarray
    count: int

    def __post_init__(self):
        self.envelope = np.asarray(self.envelope, dtype=np.float64)
        if self.envelope.ndim != 2 or not self.envelope.size:
            raise ValueError("Envelope must be a nonempty 2D array")
        if not all(np.isfinite(v) for v in (self.global_center, self.global_scale, self.global_tail)):
            raise ValueError("Nonfinite calibration parameters")
        if self.global_scale <= 0 or self.global_tail <= 0 or not np.isfinite(self.envelope).all():
            raise ValueError("Positive finite scales are required")
        if np.any(self.envelope < self.global_tail) or self.count < 8:
            raise ValueError("Local envelope must be at least the global normal tail")

    def apply(self, values, variant):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.envelope.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Expected finite nonnegative raw residuals with fitted geometry")
        if variant == "baseline":
            return values.copy()
        if variant == "global":
            return (values - self.global_center) / self.global_scale
        if variant == "local_guarded":
            # Bounded factor: no amplification; exact identity for r >= T(p).
            return values * np.minimum(1.0, values / self.envelope)
        if variant == "protection_removed":
            # Remove only score-dependent tail protection. All statistics,
            # spatial smoothing and global lower bound remain identical.
            return values * (self.global_tail / self.envelope)
        raise ValueError("Unknown calibration variant")

    @property
    def parameter_bytes(self):
        return self.envelope.nbytes + 4 * 8


def fit_tail_calibration(normal_maps, *, tail_quantile=0.95, smoothing_sigma=8.0):
    values = np.asarray(normal_maps, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 8 or min(values.shape[1:]) < 2:
        raise ValueError("At least eight normal residual maps required")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Normal maps must be finite nonnegative residuals")
    if not 0 < tail_quantile < 1 or smoothing_sigma < 0:
        raise ValueError("Invalid frozen settings")
    center = float(np.median(values))
    scale = max(float(1.4826 * np.median(np.abs(values - center))), abs(center) * 1e-6, 1e-12)
    global_tail = max(float(np.quantile(values, tail_quantile, method="higher")), 1e-12)
    spatial_tail = np.quantile(values, tail_quantile, axis=0, method="higher")
    envelope = np.maximum(global_tail, gaussian_filter(spatial_tail, smoothing_sigma, mode="reflect"))
    return TailCalibration(center, scale, global_tail, envelope, len(values))


def save_tail_calibration(path, model):
    np.savez_compressed(path, global_center=model.global_center, global_scale=model.global_scale,
                        global_tail=model.global_tail, envelope=model.envelope, count=model.count)


def load_tail_calibration(path):
    with np.load(path, allow_pickle=False) as a:
        return TailCalibration(float(a["global_center"]), float(a["global_scale"]),
                               float(a["global_tail"]), a["envelope"].copy(), int(a["count"]))
