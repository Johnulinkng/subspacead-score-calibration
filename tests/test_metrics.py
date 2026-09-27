from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from industrial_anomaly_lab.metrics import aupro, draem_image_scores, evaluate_category


def test_perfect_prediction_has_unit_metrics() -> None:
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[1, 1:3, 1:3] = 1
    maps = masks.astype(np.float64)

    result = evaluate_category(maps, masks, np.array([0, 1]))

    assert result.image_auroc == pytest.approx(1.0)
    assert result.pixel_auroc == pytest.approx(1.0)
    assert result.pixel_ap == pytest.approx(1.0)
    assert result.aupro_0_30 == pytest.approx(1.0)


def test_reverse_pixel_ranking_has_zero_auroc_and_aupro() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    maps = 1.0 - masks

    assert aupro(maps, masks) == pytest.approx(0.0)


def test_pixel_ap_is_not_trapezoidal_pr_auc() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.8, 0.4, 0.35], dtype=np.float64)
    masks = labels.reshape(2, 1, 2)
    maps = scores.reshape(2, 1, 2)
    precision, recall, _ = precision_recall_curve(labels, scores)

    result = evaluate_category(
        maps,
        masks,
        np.array([0, 1]),
        image_scores=np.array([0.0, 1.0]),
    )

    assert result.pixel_auroc == pytest.approx(0.5)
    assert result.pixel_ap == pytest.approx(7.0 / 12.0)
    assert float(auc(recall, precision)) == pytest.approx(5.0 / 12.0)


def test_all_equal_scores_have_expected_aupro() -> None:
    masks = np.zeros((1, 2, 2), dtype=np.uint8)
    masks[0, 0, 0] = 1
    maps = np.ones_like(masks, dtype=np.float64)

    assert aupro(maps, masks) == pytest.approx(0.15)


def test_regions_are_equal_weighted_and_use_eight_connectivity() -> None:
    masks = np.zeros((1, 4, 5), dtype=np.uint8)
    masks[0, 0, 0] = 1
    masks[0, 2, 2:5] = 1
    maps = np.zeros_like(masks, dtype=np.float64)
    maps[0, 0, 0] = 1.0
    maps[0, 2, 2:5] = -1.0

    assert aupro(maps, masks) == pytest.approx(0.5)

    diagonal = np.zeros((1, 5, 5), dtype=np.uint8)
    diagonal[0, 0, 0] = 1
    diagonal[0, 1, 1] = 1
    diagonal[0, 4, 4] = 1
    diagonal_scores = np.zeros_like(diagonal, dtype=np.float64)
    diagonal_scores[0, 0, 0] = 1.0
    diagonal_scores[0, 1, 1] = -1.0
    diagonal_scores[0, 4, 4] = 1.0
    assert aupro(diagonal_scores, diagonal) == pytest.approx(0.75)


def test_normal_image_background_changes_aupro() -> None:
    anomalous_mask = np.zeros((1, 2, 2), dtype=np.uint8)
    anomalous_mask[0, 0, 0] = 1
    anomalous_map = anomalous_mask.astype(np.float64)

    normal_mask = np.zeros((1, 2, 2), dtype=np.uint8)
    high_normal_map = np.ones((1, 2, 2), dtype=np.float64) * 2.0

    without_normal = aupro(anomalous_map, anomalous_mask)
    with_normal = aupro(
        np.concatenate([anomalous_map, high_normal_map]),
        np.concatenate([anomalous_mask, normal_mask]),
    )

    assert without_normal == pytest.approx(1.0)
    assert with_normal < without_normal


def test_order_and_monotonic_score_transform_do_not_change_aupro() -> None:
    masks = np.zeros((2, 3, 3), dtype=np.uint8)
    masks[0, 1, 1] = 1
    maps = np.arange(18, dtype=np.float64).reshape(2, 3, 3)

    original = aupro(maps, masks)
    reordered = aupro(maps[::-1], masks[::-1])
    transformed = aupro(3.0 * maps + 7.0, masks)

    assert reordered == pytest.approx(original)
    assert transformed == pytest.approx(original)


def test_draem_pooling_matches_torch_and_validates_inputs() -> None:
    torch = pytest.importorskip("torch")
    maps = np.zeros((2, 25, 25), dtype=np.float64)
    maps[0, 0, 0] = 1.0
    maps[1, 8:17, 8:17] = 0.5
    expected = (
        torch.nn.functional.avg_pool2d(
            torch.from_numpy(maps[:, None]), 21, stride=1, padding=10
        )
        .amax(dim=(1, 2, 3))
        .numpy()
    )

    assert draem_image_scores(maps) == pytest.approx(expected)
    with pytest.raises(ValueError, match="NaN"):
        aupro(np.array([[[np.nan]]]), np.array([[[0]]]))
    with pytest.raises(ValueError, match="binary"):
        aupro(np.zeros((1, 2, 2)), np.array([[[0, 2], [0, 1]]]))
