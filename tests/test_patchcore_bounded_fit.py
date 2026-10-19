from types import SimpleNamespace
import numpy as np
import torch
from industrial_anomaly_lab.patchcore_bounded_fit import disk_backed_fit


def test_disk_bank_matches_author_order_and_unchanged_sampler(tmp_path,monkeypatch):
    monkeypatch.setattr("industrial_anomaly_lab.patchcore_bounded_fit.shutil.disk_usage",lambda p:SimpleNamespace(free=20*1024**3))
    class Loader:
        max_steps=None
        loader=SimpleNamespace(dataset=list(range(5)),batch_size=2)
        def __iter__(self):
            for values in (torch.tensor([[0.,1.],[2.,3.]]),torch.tensor([[4.,5.],[6.,7.]]),torch.tensor([[8.,9.]])):
                yield {"image":values}
    class Sampler:
        def run(self,features):
            # The author sampler's numerical input must match concatenation.
            self.seen=np.array(features);return np.asarray(features)[[0,2,4]].copy()
    class Scorer:
        def fit(self,detection_features):self.features=detection_features[0]
    sampler=Sampler();scorer=Scorer()
    model=SimpleNamespace(forward_modules=torch.nn.Identity(),device="cpu",_embed=lambda x:[v.numpy() for v in x],featuresampler=sampler,anomaly_scorer=scorer)
    result=disk_backed_fit(model,Loader(),temporary_root=tmp_path)
    expected=np.arange(10,dtype=np.float32).reshape(5,2)
    np.testing.assert_array_equal(sampler.seen,expected)
    np.testing.assert_array_equal(scorer.features,expected[[0,2,4]])
    assert result["author_sampler_unchanged"]
    assert not list(tmp_path.rglob("*.npy"))


def test_actual_author_coreset_is_bit_identical_for_same_seed(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    sampler_path=Path(__file__).resolve().parents[1]/"third_party/PatchCore/src/patchcore/sampler.py"
    spec=importlib.util.spec_from_file_location("frozen_author_sampler",sampler_path)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    monkeypatch.setattr("industrial_anomaly_lab.patchcore_bounded_fit.shutil.disk_usage",lambda p:SimpleNamespace(free=20*1024**3))
    images=torch.arange(24,dtype=torch.float32).reshape(6,4)/24
    class Loader:
        max_steps=None
        loader=SimpleNamespace(dataset=list(range(6)),batch_size=2)
        def __iter__(self):
            for x in images.split(2):yield {"image":x}
    def make_sampler():return author.ApproximateGreedyCoresetSampler(.5,torch.device("cpu"),number_of_starting_points=2,dimension_to_project_features_to=2)
    np.random.seed(14);torch.manual_seed(14)
    expected=make_sampler().run(images.numpy().copy())
    class Scorer:
        def fit(self,detection_features):self.features=detection_features[0]
    scorer=Scorer();sampler=make_sampler()
    model=SimpleNamespace(forward_modules=torch.nn.Identity(),device="cpu",_embed=lambda x:[v.numpy() for v in x],featuresampler=sampler,anomaly_scorer=scorer)
    np.random.seed(14);torch.manual_seed(14)
    disk_backed_fit(model,Loader(),temporary_root=tmp_path)
    np.testing.assert_array_equal(scorer.features,expected)
