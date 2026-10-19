import numpy as np
import pytest
from scipy.stats import rankdata
from industrial_anomaly_lab.score_calibration import fit_normal_calibration


def test_global_calibration_preserves_all_rankings_and_ties():
    rng = np.random.default_rng(74)
    model = fit_normal_calibration(rng.exponential(size=(12, 8, 8)))
    scores = np.array([[0., 2., 2.], [3., 1., 0.]])
    # Fit same geometry; ordering preservation is required, not a gain claim.
    model = fit_normal_calibration(rng.exponential(size=(12, 2, 3)))
    np.testing.assert_array_equal(rankdata(scores), rankdata(model.apply(scores, "global")))


def test_shift_and_positive_scale_do_not_change_calibrated_scores():
    rng = np.random.default_rng(13)
    normal = rng.exponential(size=(16, 9, 9)) + 2
    query = rng.exponential(size=(9, 9)) + 2
    original = fit_normal_calibration(normal)
    changed = fit_normal_calibration(normal * 4 + 10)
    for variant in ("global", "local_shrunk", "local_no_shrink"):
        np.testing.assert_allclose(original.apply(query, variant),
                                   changed.apply(query * 4 + 10, variant), atol=1e-12)


def test_constant_normal_regions_never_produce_nonfinite_scores():
    normal = np.ones((8, 4, 4))
    model = fit_normal_calibration(normal)
    for variant in ("global", "local_shrunk", "local_no_shrink"):
        assert np.isfinite(model.apply(normal[0] + 1, variant)).all()


def test_reject_nonfinite_inputs_and_wrong_geometry():
    with pytest.raises(ValueError):
        fit_normal_calibration(np.full((8, 4, 4), np.nan))
    model = fit_normal_calibration(np.ones((8, 4, 4)))
    with pytest.raises(ValueError):
        model.apply(np.ones((5, 5)), "local_shrunk")
