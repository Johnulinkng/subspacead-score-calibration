import numpy as np
import pytest
from industrial_anomaly_lab.context_calibration import ContextCalibration, nearest_centers
from industrial_anomaly_lab.subspace_features import render_subspace_map


def test_cluster_permutation_preserves_scoring_when_tails_follow_centers():
    centers = np.array([[0., 0.], [4., 4.]], np.float32)
    z = np.tile(np.array([[0., 0.], [4., 4.]], np.float16), (512, 1))
    raw = np.linspace(.1, 3, 1024, dtype=np.float32).reshape(32, 32)
    a = ContextCalibration(centers, np.array([1., 2.]), 1., 0., 1.)
    b = ContextCalibration(centers[::-1].copy(), np.array([2., 1.]), 1., 0., 1.)
    np.testing.assert_array_equal(a.apply(raw, z, "conditional_tail"), b.apply(raw, z, "conditional_tail"))


def test_strong_residuals_remain_identical_after_rendering():
    model = ContextCalibration(np.zeros((1, 2), np.float32), np.array([1.]), 1., 0., 1.)
    raw = np.full((32, 32), 2., np.float32)
    z = np.zeros((1024, 2), np.float16)
    np.testing.assert_array_equal(model.apply(raw, z, "conditional_tail"), render_subspace_map(raw))


def test_no_amplification_and_chunked_assignment_matches_direct_distances():
    rng = np.random.default_rng(602)
    centers = rng.normal(size=(8, 16)).astype(np.float32)
    z = rng.normal(size=(1024, 16)).astype(np.float16)
    expected = np.sum((z.astype(np.float64)[:, None, :] - centers.astype(np.float64)[None]) ** 2, axis=-1).argmin(axis=1)
    np.testing.assert_array_equal(nearest_centers(z, centers, chunk_size=31), expected)
    model = ContextCalibration(centers, np.arange(1, 9, dtype=float), 1., 0., 1.)
    raw = rng.exponential(size=(32, 32)).astype(np.float32)
    assert np.all(model.apply(raw, z, "conditional_tail") <= model.apply(raw, z, "baseline") + 1e-7)


def test_one_center_reduces_exactly_to_conditioning_removed():
    z = np.zeros((1024, 2), np.float16)
    raw = np.linspace(0, 3, 1024, dtype=np.float32).reshape(32, 32)
    model = ContextCalibration(np.zeros((1, 2), np.float32), np.array([1.]), 1., 0., 1.)
    np.testing.assert_array_equal(model.apply(raw, z, "conditional_tail"), model.apply(raw, z, "conditioning_removed"))


def test_bad_geometry_is_rejected():
    with pytest.raises(ValueError):
        nearest_centers(np.zeros((3, 4)), np.zeros((2, 5)))
