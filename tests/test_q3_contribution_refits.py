"""Checks that the new mechanism controls remove the intended information."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec=importlib.util.spec_from_file_location("q3_refits", SCRIPTS/"run_q3_contribution_refits.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_content_removal_equals_full_when_all_groups_have_same_tail():
    rng=np.random.default_rng(14)
    raw=rng.uniform(.1,2,(32,32)).astype(np.float32)
    z=rng.normal(size=(1024,2)).astype(np.float16)
    fit={"centers":np.array([[0.,0.],[1.,1.]],dtype=np.float32),
         "tails":np.array([1.,1.]),"global_tail":np.array(1.)}
    a=module.apply_model(raw,z,fit,"conditional_tail",rng)
    b=module.apply_model(raw,z,fit,"global_tail",rng)
    np.testing.assert_array_equal(a,b)


def test_guard_preserves_high_scores_and_has_effect_below_tail():
    from industrial_anomaly_lab.subspace_features import render_subspace_map
    rng=np.random.default_rng(15)
    fit={"centers":np.zeros((1,2),np.float32),"tails":np.array([1.]),"global_tail":np.array(1.)}
    z=np.zeros((1024,2),np.float16)
    for level, expected in ((2.,2.),(.5,.25)):
        actual=module.apply_model(np.full((32,32),level,np.float32),z,fit,"conditional_tail",rng)
        np.testing.assert_array_equal(actual,render_subspace_map(np.full((32,32),expected)))


def test_paired_statistics_use_category_units_and_exact_sign_test():
    result=module.paired_summary(np.ones(12)*.01)
    assert result["categories"]==12
    assert np.isclose(result["mean_delta_pp"],1)
    assert result["exact_two_sided_sign_flip_p"]==2/4096
    assert np.allclose(result["bootstrap_95_ci_pp"],[1.,1.])
