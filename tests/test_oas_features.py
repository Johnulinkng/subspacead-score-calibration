from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.covariance import oas

from industrial_anomaly_lab.oas_features import prepare_ml_scorer, prepare_oas_scorer
from industrial_anomaly_lab.quick_features import fit_statistics


@pytest.mark.parametrize("sample_count,dimensions", [(2, 4), (7, 3), (31, 5)])
def test_oas_matches_sklearn_training_covariance_and_precision(
    sample_count: int, dimensions: int,
) -> None:
    rng = np.random.default_rng(400 + sample_count)
    samples = rng.normal(size=(sample_count, dimensions, 2, 3))
    samples *= np.linspace(0.2, 3, dimensions)[None, :, None, None]
    samples += np.linspace(-3, 4, dimensions)[None, :, None, None]
    stats = fit_statistics(samples, batch_size=3, position_chunk_size=2)
    covariance_before = stats.covariance.copy()
    scorer = prepare_oas_scorer(stats, position_chunk_size=2, model_size=2)
    values = samples.reshape(sample_count, dimensions, 6)
    for position in range(6):
        expected_covariance, expected_alpha = oas(values[:, :, position])
        np.testing.assert_allclose(scorer.alphas[position], expected_alpha, atol=2e-14)
        expected_precision = np.linalg.inv(
            expected_covariance + 0.01 * np.eye(dimensions)
        )
        np.testing.assert_allclose(
            scorer.precision[position].numpy(), expected_precision,
            rtol=2e-13, atol=2e-13,
        )
    np.testing.assert_array_equal(stats.covariance, covariance_before)
    assert scorer.alphas.flags.writeable is False
    assert scorer.metadata["per_image_normalization"] is False


@pytest.mark.parametrize("kind", ["zero", "isotropic", "single_feature"])
def test_zero_isotropic_and_single_feature_follow_sklearn(kind: str) -> None:
    if kind == "zero":
        samples = np.zeros((5, 3, 1, 1), dtype=np.float64)
    elif kind == "isotropic":
        # The centered rows yield exactly identity ML covariance.
        samples = np.concatenate([np.eye(3), -np.eye(3)]) * np.sqrt(3)
        samples = samples[:, :, None, None]
    else:
        samples = np.array([2, -1, 4, 1, -2], dtype=np.float64)[:, None, None, None]
    stats = fit_statistics(samples)
    scorer = prepare_oas_scorer(stats)
    expected_covariance, expected_alpha = oas(samples[:, :, 0, 0])
    np.testing.assert_allclose(scorer.alphas[0], expected_alpha, atol=1e-15)
    np.testing.assert_allclose(
        scorer.precision[0].numpy(),
        np.linalg.inv(expected_covariance + 0.01 * np.eye(samples.shape[1])),
        atol=2e-14,
    )
    assert expected_alpha == (0 if kind == "single_feature" else 1)


def test_oas_distance_matches_direct_inverse_and_regularization_once() -> None:
    rng = np.random.default_rng(150)
    samples = rng.normal(size=(13, 4, 2, 2))
    samples *= np.array([0.1, 1, 2, 7])[None, :, None, None]
    stats = fit_statistics(samples)
    query = rng.normal(size=(4, 2, 2)).astype(np.float32)
    scorer = prepare_oas_scorer(
        stats, sigma=0, model_size=2, output_size=2,
        regularization=0.01, position_chunk_size=1,
    )
    expected_distances = []
    for position in range(4):
        covariance, _ = oas(samples.reshape(13, 4, 4)[:, :, position])
        covariance += np.eye(4) * 0.01
        difference = query.reshape(4, 4).T[position] - stats.mean[position]
        expected_distances.append(
            np.sqrt(difference @ np.linalg.inv(covariance) @ difference)
        )
        # An unbiased-covariance OAS input or a second epsilon would differ.
        np.testing.assert_allclose(
            scorer.precision[position].numpy(), np.linalg.inv(covariance),
            atol=1e-13, rtol=1e-13,
        )
    np.testing.assert_allclose(
        scorer.distance_maps(query)[0], np.array(expected_distances).reshape(2, 2),
        atol=2e-6, rtol=1e-6,
    )
    np.testing.assert_array_equal(scorer(query), scorer.distance_maps(query))


def test_oas_chunk_size_preserves_coefficients_precision_and_distance() -> None:
    rng = np.random.default_rng(613)
    samples = rng.normal(size=(11, 5, 2, 3))
    stats = fit_statistics(samples)
    query = rng.normal(size=(5, 2, 3)).astype(np.float32)
    reference = prepare_oas_scorer(stats, position_chunk_size=1)
    for chunk in (2, 64, 128):
        scorer = prepare_oas_scorer(stats, position_chunk_size=chunk)
        np.testing.assert_array_equal(scorer.alphas, reference.alphas)
        np.testing.assert_array_equal(scorer.precision.numpy(), reference.precision.numpy())
        np.testing.assert_array_equal(scorer(query), reference(query))


def test_ml_control_uses_n_denominator_without_shrinkage() -> None:
    rng = np.random.default_rng(799)
    samples = rng.normal(size=(9, 4, 2, 2))
    samples *= np.array([0.1, 0.8, 2, 8])[None, :, None, None]
    stats = fit_statistics(samples)
    scorer = prepare_ml_scorer(
        stats, sigma=0, model_size=2, output_size=2, position_chunk_size=2,
    )
    np.testing.assert_array_equal(scorer.alphas, np.zeros(4))
    values = samples.reshape(9, 4, 4)
    query = rng.normal(size=(4, 2, 2)).astype(np.float32)
    expected_distances = []
    for position in range(4):
        covariance = np.cov(values[:, :, position], rowvar=False, bias=True)
        covariance += np.eye(4) * 0.01
        np.testing.assert_allclose(
            scorer.precision[position].numpy(), np.linalg.inv(covariance),
            rtol=2e-13, atol=2e-13,
        )
        difference = query.reshape(4, 4).T[position] - stats.mean[position]
        expected_distances.append(
            np.sqrt(difference @ np.linalg.inv(covariance) @ difference)
        )
    np.testing.assert_allclose(
        scorer(query)[0], np.array(expected_distances).reshape(2, 2),
        rtol=1e-6, atol=2e-6,
    )
    assert scorer.metadata["mode"] == "ml_no_shrinkage"


@pytest.mark.parametrize("count", [0, 1, 2.5, True])
def test_invalid_training_count_rejected(count: int | float) -> None:
    stats = fit_statistics(np.random.default_rng(50).normal(size=(3, 2, 1, 1)))
    with pytest.raises(ValueError, match="samples|sample count"):
        prepare_oas_scorer(replace(stats, count=count))


@pytest.mark.parametrize(
    "settings", [
        {"regularization": 0}, {"regularization": float("nan")},
        {"sigma": -1}, {"position_chunk_size": 0}, {"cpu_threads": 0},
    ],
)
def test_invalid_scorer_settings_rejected(settings: dict) -> None:
    stats = fit_statistics(np.random.default_rng(77).normal(size=(3, 2, 1, 1)))
    with pytest.raises(ValueError):
        prepare_oas_scorer(stats, **settings)
