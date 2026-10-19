import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("q3_development",ROOT/"scripts/run_q3_trainonly_development.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CFG=json.loads((ROOT/"configs/q3_trainonly_development_20261016.json").read_text())


def test_synthetic_generation_is_deterministic_and_mask_matches_changes(tmp_path):
    a,b=tmp_path/"a.png",tmp_path/"b.png"
    Image.fromarray(np.full((256,256,3),40,np.uint8)).save(a)
    Image.fromarray(np.full((256,256,3),180,np.uint8)).save(b)
    im,mask,recipe=module.synthetic_image(a,b,11,.05,CFG)
    im2,mask2,recipe2=module.synthetic_image(a,b,11,.05,CFG)
    np.testing.assert_array_equal(np.array(im),np.array(im2));np.testing.assert_array_equal(mask,mask2)
    assert recipe==recipe2
    changed=np.any(np.array(im)!=40,axis=2)
    np.testing.assert_array_equal(changed,mask)


def test_removing_content_alone_keeps_density_component():
    raw=np.ones((32,32),np.float32)*.5;z=np.ones((1024,2),np.float16)
    fit={"centers":np.zeros((1,2),np.float32),"tails":np.array([1.]),"global_tail":np.array(1.),"density_q95":np.array(.5)}
    cfg={"name":"density100","family":"density","weight":.1}
    full=module.candidate_map(raw,z,None,fit,cfg)
    removed=module.candidate_map(raw,z,None,fit,cfg,remove_condition=True)
    np.testing.assert_array_equal(full,removed)
    without_density=module.candidate_map(raw,z,None,fit,cfg,remove_adjustment=True)
    assert np.all(full>without_density)
