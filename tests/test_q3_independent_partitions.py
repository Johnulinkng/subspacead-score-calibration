import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("q3_independent",ROOT/"scripts/run_q3_independent_partitions.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CFG=json.loads((ROOT/"configs/q3_independent_partitions_20261016.json").read_text())


def test_normal_partitions_are_complete_disjoint_and_seed_dependent():
    p=module.partitions(60,1887537248,CFG)
    assert all(len(v)>=8 for v in p.values())
    assert sorted(np.concatenate(list(p.values())))==list(range(60))
    other=module.partitions(60,1337854654,CFG)
    assert set(p["pca_fit"])!=set(other["pca_fit"])


def test_duplicate_content_cannot_cross_partitions():
    hashes=[str(i//2) for i in range(120)]
    p=module.partitions(120,1887537248,CFG,hashes)
    sets=[{hashes[i] for i in ix} for ix in p.values()]
    assert sum(map(len,p.values()))==120
    assert all(not a.intersection(b) for i,a in enumerate(sets) for b in sets[i+1:])


def test_combined_remove_content_equals_density_with_identity_tail():
    from industrial_anomaly_lab.subspace_features import render_subspace_map
    rng=np.random.default_rng(3);raw=np.ones((32,32),np.float32)*2
    z=rng.normal(size=(1024,2)).astype(np.float16)
    fit={"centers":np.zeros((1,2),np.float32),"tails":np.array([1.]),"global_tail":np.array(1.),"density_q95":np.array(2.)}
    full=module.apply_adjusted(raw,z,np.zeros((32,32)),fit,"combined_density_tail",CFG,rng)
    removed=module.apply_adjusted(raw,z,np.zeros((32,32)),fit,"prototype_density",CFG,rng)
    np.testing.assert_array_equal(full,removed)
