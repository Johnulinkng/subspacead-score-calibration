import numpy as np
import pytest
from industrial_anomaly_lab.tail_calibration import TailCalibration, fit_tail_calibration


def test_never_amplifies_noise_and_preserves_all_above_envelope_values():
    rng = np.random.default_rng(319)
    model = fit_tail_calibration(rng.exponential(size=(32, 8, 8)), smoothing_sigma=1)
    weak = rng.uniform(size=(8, 8)) * model.envelope
    strong = model.envelope + rng.uniform(size=(8, 8))
    actual = model.apply(weak, "local_guarded")
    assert np.all(actual >= 0)
    assert np.all(actual <= weak)
    np.testing.assert_array_equal(model.apply(strong, "local_guarded"), strong)


def test_continuity_and_monotonicity_at_protection_boundary():
    model = TailCalibration(0, 1, 1, np.full((2, 2), 2.), 16)
    scores = [model.apply(np.full((2, 2), r), "local_guarded")[0, 0]
              for r in (0., 1., 1.999999, 2., 2.000001, 10.)]
    assert np.all(np.diff(scores) > 0)
    assert abs(scores[2] - 2) < 3e-6
    assert scores[3] == 2


def test_same_fixed_statistics_and_ablation_only_removes_protection():
    model = TailCalibration(0, 1, 1, np.full((2, 2), 2.), 16)
    high = np.full((2, 2), 10.)
    np.testing.assert_array_equal(model.apply(high, "local_guarded"), high)
    np.testing.assert_array_equal(model.apply(high, "protection_removed"), high / 2)


def test_positive_rescaling_is_equivariant():
    rng = np.random.default_rng(21)
    normal = rng.exponential(size=(16, 5, 5))
    sample = rng.exponential(size=(5, 5))
    a = fit_tail_calibration(normal, smoothing_sigma=1)
    b = fit_tail_calibration(normal * 7, smoothing_sigma=1)
    for variant in ("local_guarded", "protection_removed"):
        np.testing.assert_allclose(b.apply(sample * 7, variant), a.apply(sample, variant) * 7, atol=1e-12)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        fit_tail_calibration(np.full((8, 2, 2), np.nan))
    model = fit_tail_calibration(np.ones((8, 2, 2)))
    with pytest.raises(ValueError):
        model.apply(-np.ones((2, 2)), "local_guarded")
    with pytest.raises(ValueError):
        model.apply(np.ones((3, 3)), "local_guarded")
