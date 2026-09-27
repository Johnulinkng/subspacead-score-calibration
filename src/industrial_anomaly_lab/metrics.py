"""Frozen metric definitions for the DRAEM 256 px unified protocol.

The functions in this module operate on unquantized anomaly scores. They do
not normalize individual maps, because that would change ranking across
images. AUPRO follows the MVTec definition: 8-connected regions are weighted
equally, false-positive rate is computed over all background pixels, and the
curve is integrated up to a fixed FPR limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CategoryMetrics:
    """Metrics for one complete MVTec category and one model run."""

    image_auroc: float
    pixel_auroc: float
    pixel_ap: float
    aupro_0_30: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _as_map_stack(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (num_images, height, width)")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_binary_masks(values: ArrayLike, expected_shape: tuple[int, ...]) -> BoolArray:
    raw = np.asarray(values)
    if raw.shape != expected_shape:
        raise ValueError(
            f"masks shape {raw.shape} does not match score shape {expected_shape}"
        )
    if raw.size == 0:
        raise ValueError("masks must not be empty")
    if not np.isin(raw, (0, 1, False, True)).all():
        raise ValueError("masks must be binary")
    return raw.astype(bool, copy=False)


def _as_binary_labels(values: ArrayLike, count: int) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.shape != (count,):
        raise ValueError(f"image labels must have shape ({count},)")
    if not np.isin(labels, (0, 1, False, True)).all():
        raise ValueError("image labels must be binary")
    labels = labels.astype(np.int64, copy=False)
    if np.unique(labels).size != 2:
        raise ValueError("image AUROC requires both normal and anomalous images")
    return labels


def draem_image_scores(anomaly_maps: ArrayLike, kernel_size: int = 21) -> FloatArray:
    """Return DRAEM image scores from raw pixel anomaly maps.

    The author implementation applies a stride-one average pool with zero
    padding and then takes the maximum. ``uniform_filter`` with constant zero
    padding has the same fixed-denominator behavior for odd kernels.
    """

    maps = _as_map_stack(anomaly_maps, "anomaly_maps")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    pooled = ndimage.uniform_filter(
        maps,
        size=(1, kernel_size, kernel_size),
        mode="constant",
        cval=0.0,
    )
    return pooled.max(axis=(1, 2))


def _region_weights(masks: BoolArray) -> tuple[FloatArray, int]:
    """Assign each foreground pixel its equal-region PRO contribution."""

    weights = np.zeros(masks.shape, dtype=np.float64)
    region_count = 0
    connectivity_8 = np.ones((3, 3), dtype=np.uint8)

    for image_index, mask in enumerate(masks):
        labels, image_region_count = ndimage.label(mask, structure=connectivity_8)
        for region_id in range(1, image_region_count + 1):
            region = labels == region_id
            region_size = int(region.sum())
            if region_size == 0:
                raise RuntimeError("connected-component labeling returned an empty region")
            weights[image_index, region] = 1.0 / region_size
            region_count += 1

    if region_count == 0:
        raise ValueError("AUPRO requires at least one anomalous region")

    weights /= region_count
    return weights, region_count


def aupro(
    anomaly_maps: ArrayLike,
    masks: ArrayLike,
    *,
    fpr_limit: float = 0.30,
) -> float:
    """Compute exact equal-region AUPRO normalized by ``fpr_limit``.

    Equal anomaly scores are added as one threshold group. If a group crosses
    the requested false-positive boundary, the curve is linearly interpolated
    at that boundary, matching the MVTec integration convention.
    """

    scores = _as_map_stack(anomaly_maps, "anomaly_maps")
    binary_masks = _as_binary_masks(masks, scores.shape)
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError("fpr_limit must be in (0, 1]")

    background = ~binary_masks
    background_count = int(background.sum())
    if background_count == 0:
        raise ValueError("AUPRO requires at least one background pixel")

    pro_weights, _ = _region_weights(binary_masks)
    flat_scores = scores.ravel()
    flat_background = background.ravel()
    flat_weights = pro_weights.ravel()

    order = np.argsort(-flat_scores, kind="stable")
    sorted_scores = flat_scores[order]
    cumulative_fpr = np.cumsum(flat_background[order], dtype=np.float64)
    cumulative_fpr /= background_count
    cumulative_pro = np.cumsum(flat_weights[order], dtype=np.float64)

    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    fpr = np.r_[0.0, cumulative_fpr[group_ends]]
    pro = np.r_[0.0, cumulative_pro[group_ends]]
    fpr = np.clip(fpr, 0.0, 1.0)
    pro = np.clip(pro, 0.0, 1.0)

    crossing = int(np.searchsorted(fpr, fpr_limit, side="left"))
    if crossing >= fpr.size:
        raise RuntimeError("AUPRO curve did not reach the requested FPR limit")

    if fpr[crossing] == fpr_limit:
        last_at_limit = int(np.searchsorted(fpr, fpr_limit, side="right"))
        truncated_fpr = fpr[:last_at_limit]
        truncated_pro = pro[:last_at_limit]
    else:
        lower = crossing - 1
        if lower < 0 or fpr[crossing] <= fpr[lower]:
            raise RuntimeError("cannot interpolate AUPRO at the FPR boundary")
        fraction = (fpr_limit - fpr[lower]) / (fpr[crossing] - fpr[lower])
        boundary_pro = pro[lower] + fraction * (pro[crossing] - pro[lower])
        truncated_fpr = np.r_[fpr[:crossing], fpr_limit]
        truncated_pro = np.r_[pro[:crossing], boundary_pro]

    area = float(np.trapz(truncated_pro, truncated_fpr))
    return area / fpr_limit


def evaluate_category(
    anomaly_maps: ArrayLike,
    masks: ArrayLike,
    image_labels: ArrayLike,
    *,
    image_scores: ArrayLike | None = None,
    fpr_limit: float = 0.30,
) -> CategoryMetrics:
    """Evaluate one complete category under ``draem_256_unified_v1``."""

    maps = _as_map_stack(anomaly_maps, "anomaly_maps")
    binary_masks = _as_binary_masks(masks, maps.shape)
    labels = _as_binary_labels(image_labels, maps.shape[0])

    if image_scores is None:
        resolved_image_scores = draem_image_scores(maps)
    else:
        resolved_image_scores = np.asarray(image_scores, dtype=np.float64)
        if resolved_image_scores.shape != (maps.shape[0],):
            raise ValueError(f"image_scores must have shape ({maps.shape[0]},)")
        if not np.isfinite(resolved_image_scores).all():
            raise ValueError("image_scores contains NaN or infinite values")

    flat_masks = binary_masks.ravel().astype(np.uint8)
    if np.unique(flat_masks).size != 2:
        raise ValueError("pixel metrics require foreground and background pixels")
    flat_scores = maps.ravel()

    return CategoryMetrics(
        image_auroc=float(roc_auc_score(labels, resolved_image_scores)),
        pixel_auroc=float(roc_auc_score(flat_masks, flat_scores)),
        pixel_ap=float(average_precision_score(flat_masks, flat_scores)),
        aupro_0_30=aupro(maps, binary_masks, fpr_limit=fpr_limit),
    )
