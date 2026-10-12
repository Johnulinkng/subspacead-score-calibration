"""Exact category metrics from files with bounded sorting storage.

Only background scores and foreground (score, equal-region weight) records
are sorted, in temporary memory-mapped files.  Pair counts give pixel AUROC;
foreground threshold groups give non-interpolated AP.  AUPRO integrates each
foreground group's contribution analytically, including ties crossing the
FPR boundary.  Scores are never normalized, quantized, or downsampled here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage

from .metrics import image_auroc_from_scores


_FOREGROUND_DTYPE = np.dtype([("score", "<f8"), ("weight", "<f8")])
_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def _close(array: np.ndarray) -> None:
    if isinstance(array, np.memmap):
        array._mmap.close()


def _load_pair(map_file: str | Path, mask_file: str | Path):
    scores = np.load(map_file, mmap_mode="r", allow_pickle=False)
    try:
        mask = np.load(mask_file, mmap_mode="r", allow_pickle=False)
    except BaseException:
        _close(scores)
        raise
    try:
        if scores.ndim != 2 or not scores.size:
            raise ValueError("anomaly maps must be nonempty two-dimensional arrays")
        if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
            raise ValueError("anomaly maps must contain real numeric scores")
        if not np.isfinite(scores).all():
            raise ValueError("anomaly maps contain NaN or infinite values")
        if mask.shape != scores.shape:
            raise ValueError("mask shape does not match anomaly map shape")
        if not np.isin(mask, (0, 1, False, True)).all():
            raise ValueError("masks must be binary")
    except BaseException:
        _close(scores)
        _close(mask)
        raise
    return scores, mask


def evaluate_from_files(
    map_files: Sequence[str | Path],
    mask_files: Sequence[str | Path],
    labels: Sequence[int],
    *,
    fpr_limit: float = 0.30,
    chunk_size: int = 262_144,
    temp_root: str | Path | None = None,
) -> dict[str, float]:
    """Match the existing complete-category metric definitions exactly.

    Temporary disk storage is ``8 * background_pixels + 16 * foreground_pixels``
    bytes, at most 350 MiB for 350 maps of 256x256 pixels.  In-place quicksort
    avoids full-size order/cumulative arrays; threshold work uses bounded chunks.
    The default temporary directory is beside the first prediction file.
    """

    map_files, mask_files = list(map_files), list(mask_files)
    if not map_files or len(map_files) != len(mask_files):
        raise ValueError("map and mask file lists must be nonempty and equally sized")
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError("fpr_limit must be in (0, 1]")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    image_scores: list[float] = []
    foreground_count = background_count = region_count = 0
    expected_shape = None
    for map_file, mask_file in zip(map_files, mask_files, strict=True):
        scores, raw_mask = _load_pair(map_file, mask_file)
        try:
            if expected_shape is None:
                expected_shape = scores.shape
            if scores.shape != expected_shape:
                raise ValueError("all category anomaly maps must have the same shape")
            mask = np.asarray(raw_mask, dtype=bool)
            positive_count = int(mask.sum())
            foreground_count += positive_count
            background_count += mask.size - positive_count
            region_count += int(ndimage.label(mask, structure=_CONNECTIVITY)[1])
            image_scores.append(float(scores.max()))
        finally:
            _close(scores)
            _close(raw_mask)
    image_auroc = image_auroc_from_scores(labels, image_scores)
    if foreground_count == 0 or background_count == 0:
        raise ValueError("pixel metrics require foreground and background pixels")
    if region_count == 0:
        raise ValueError("AUPRO requires at least one anomalous region")

    temporary_parent = Path(temp_root) if temp_root else Path(map_files[0]).parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".quick-metrics-", dir=temporary_parent) as temporary:
        background = np.memmap(
            Path(temporary) / "background.bin", dtype="<f8", mode="w+",
            shape=(background_count,),
        )
        foreground = np.memmap(
            Path(temporary) / "foreground.bin", dtype=_FOREGROUND_DTYPE, mode="w+",
            shape=(foreground_count,),
        )
        try:
            b_offset = f_offset = 0
            for map_file, mask_file in zip(map_files, mask_files, strict=True):
                scores, raw_mask = _load_pair(map_file, mask_file)
                try:
                    mask = np.asarray(raw_mask, dtype=bool)
                    region_labels, _ = ndimage.label(mask, structure=_CONNECTIVITY)
                    sizes = np.bincount(region_labels.ravel())
                    positive_scores = scores[mask]
                    negative_scores = scores[~mask]
                    f_stop, b_stop = f_offset + positive_scores.size, b_offset + negative_scores.size
                    foreground["score"][f_offset:f_stop] = positive_scores
                    weights = 1.0 / sizes[region_labels[mask]].astype(np.float64)
                    foreground["weight"][f_offset:f_stop] = weights / region_count
                    background[b_offset:b_stop] = negative_scores
                    f_offset, b_offset = f_stop, b_stop
                finally:
                    _close(scores)
                    _close(raw_mask)
            if (f_offset, b_offset) != (foreground_count, background_count):
                raise ValueError("prediction or mask files changed during evaluation")

            background.sort(kind="quicksort")
            foreground.sort(order="score", kind="quicksort")
            true_positives = 0
            auroc_numerator = ap = pro_area = 0.0

            def accumulate(group_scores, group_counts, group_weights):
                nonlocal true_positives, auroc_numerator, ap, pro_area
                if not group_scores.size:
                    return
                left = np.searchsorted(background, group_scores, side="left")
                right = np.searchsorted(background, group_scores, side="right")
                auroc_numerator += float(np.sum(group_counts * (left + 0.5 * (right - left))))
                cumulative_tp = np.cumsum(group_counts, dtype=np.int64) + true_positives
                false_positives = background_count - left
                ap += float(np.sum(
                    (group_counts / foreground_count)
                    * cumulative_tp / (cumulative_tp + false_positives)
                ))
                true_positives = int(cumulative_tp[-1])

                x_left = (background_count - right) / background_count
                x_right = (background_count - left) / background_count
                width = x_right - x_left
                partial_width = np.clip(fpr_limit - x_left, 0.0, width)
                triangle = np.divide(
                    0.5 * partial_width * partial_width, width,
                    out=np.zeros_like(width), where=width > 0.0,
                )
                contributions = np.maximum(fpr_limit - x_right, 0.0) + triangle
                pro_area += float(np.sum(group_weights * contributions))

            # Keep the last threshold group across chunk boundaries so a tied
            # score always enters AP/AUPRO as one complete threshold group.
            pending_score = None
            pending_count = 0
            pending_weight = 0.0
            upper = foreground_count
            while upper > 0:
                lower = max(0, upper - chunk_size)
                records = foreground[lower:upper][::-1]
                score_chunk, weight_chunk = records["score"], records["weight"]
                starts = np.r_[0, np.flatnonzero(score_chunk[1:] != score_chunk[:-1]) + 1]
                group_scores = np.array(score_chunk[starts], dtype=np.float64)
                group_counts = np.diff(np.r_[starts, score_chunk.size])
                group_weights = np.add.reduceat(weight_chunk, starts)
                if pending_score is not None:
                    if group_scores[0] == pending_score:
                        group_counts[0] += pending_count
                        group_weights[0] += pending_weight
                    else:
                        accumulate(
                            np.array([pending_score]), np.array([pending_count]),
                            np.array([pending_weight]),
                        )
                pending_score = float(group_scores[-1])
                pending_count = int(group_counts[-1])
                pending_weight = float(group_weights[-1])
                accumulate(group_scores[:-1], group_counts[:-1], group_weights[:-1])
                upper = lower
            accumulate(
                np.array([pending_score]), np.array([pending_count]), np.array([pending_weight]),
            )
            if true_positives != foreground_count:
                raise RuntimeError("foreground threshold grouping lost pixels")
            # Suppress only floating-point summation roundoff at the unit
            # interval endpoints, as the reference AUPRO curve also clips.
            return {
                "image_auroc": image_auroc,
                "pixel_auroc": float(np.clip(auroc_numerator / (foreground_count * background_count), 0.0, 1.0)),
                "pixel_ap": float(np.clip(ap, 0.0, 1.0)),
                "aupro_0_30": float(np.clip(pro_area / fpr_limit, 0.0, 1.0)),
            }
        finally:
            _close(background)
            _close(foreground)
