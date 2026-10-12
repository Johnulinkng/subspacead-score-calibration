from __future__ import annotations

import numpy as np
import pytest

from industrial_anomaly_lab.quick_features import (
    fit_statistics,
    load_statistics,
    prepare_scorer,
    render_maps,
    save_statistics,
    select_channels,
)


def test_streaming_covariance_matches_numpy_per_location() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(1e6, 0.4, size=(19, 4, 2, 3))
    counts: list[int] = []
    stats = fit_statistics(
        (sample for sample in values), batch_size=5,
        position_chunk_size=2, progress_callback=counts.append,
    )
    flat = values.reshape(19, 4, 6)
    np.testing.assert_allclose(stats.mean, flat.mean(axis=0).T, rtol=0, atol=8e-10)
    for location in range(6):
        np.testing.assert_allclose(
            stats.covariance[location], np.cov(flat[:, :, location], rowvar=False),
            rtol=2e-9, atol=2e-10,
        )
    assert stats.count == 19
    assert counts == [5, 10, 15, 19]


def test_raw_mahalanobis_and_shrinkage_match_direct_inverse() -> None:
    rng = np.random.default_rng(37)
    training = rng.normal(size=(17, 3, 2, 2)) * np.array([1, 2, 3])[None, :, None, None]
    stats = fit_statistics(training, batch_size=4, position_chunk_size=1)
    query = rng.normal(size=(3, 2, 2)).astype(np.float32)
    flat = query.reshape(3, 4).T
    for alpha in (0.0, 0.1, 1.0):
        scorer = prepare_scorer(
            stats, shrinkage_alpha=alpha, sigma=0,
            model_size=2, output_size=2, position_chunk_size=1,
        )
        expected = []
        for location in range(4):
            covariance = stats.covariance[location]
            covariance = (
                (1 - alpha) * covariance
                + alpha * np.trace(covariance) / 3 * np.eye(3)
                + 0.01 * np.eye(3)
            )
            difference = flat[location] - stats.mean[location]
            expected.append(np.sqrt(difference @ np.linalg.inv(covariance) @ difference))
        np.testing.assert_allclose(scorer(query)[0], np.array(expected).reshape(2, 2), rtol=1e-6)
        assert scorer.metadata["per_image_normalization"] is False


def test_render_preserves_raw_scale_and_never_smooths_across_images() -> None:
    raw = np.stack([
        np.full((8, 8), 2.5, dtype=np.float32),
        np.full((8, 8), 70.0, dtype=np.float32),
    ])
    rendered = render_maps(raw, sigma=4, output_size=16)
    np.testing.assert_array_equal(rendered[0], np.full((16, 16), 2.5, dtype=np.float32))
    np.testing.assert_array_equal(rendered[1], np.full((16, 16), 70, dtype=np.float32))


def test_seeded_dimensions_and_tensor_only_stats_roundtrip(tmp_path) -> None:
    indices = select_channels(5, 4)
    np.testing.assert_array_equal(indices, select_channels(5, 4))
    assert len(set(indices)) == 4
    assert not np.array_equal(indices, select_channels(6, 4))
    features = np.random.default_rng(5).normal(size=(7, 4, 2, 2)).astype(np.float32)
    stats = fit_statistics(features, channel_indices=indices, metadata={"split": "train"})
    target = tmp_path / "stats.npz"
    save_statistics(stats, target)
    restored = load_statistics(target)
    np.testing.assert_array_equal(stats.mean, restored.mean)
    np.testing.assert_array_equal(stats.covariance, restored.covariance)
    np.testing.assert_array_equal(indices, restored.channel_indices)
    assert restored.count == 7
    assert restored.metadata["split"] == "train"
    with pytest.raises(FileExistsError):
        save_statistics(stats, target)


def test_invalid_or_insufficient_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_statistics([np.zeros((3, 2, 2), dtype=np.float32)])
    with pytest.raises(ValueError, match="nonnegative"):
        select_channels(-1)
    with pytest.raises(ValueError, match="finite"):
        fit_statistics([np.full((3, 2, 2), np.nan, dtype=np.float32)] * 2)
    with pytest.raises(ValueError, match="geometry"):
        fit_statistics([
            np.zeros((3, 2, 2), dtype=np.float32),
            np.zeros((3, 2, 3), dtype=np.float32),
        ])
