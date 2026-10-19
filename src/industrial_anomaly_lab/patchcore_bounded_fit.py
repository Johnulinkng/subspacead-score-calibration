"""Exact author feature ordering and sampler, with disk-backed concatenation."""
from __future__ import annotations
from pathlib import Path
import shutil
import tempfile
import types
import numpy as np


def disk_backed_fit(model, loader, *, temporary_root):
    """Avoid simultaneously retaining a feature list and its concatenated copy."""
    import torch
    root=Path(temporary_root).resolve();root.mkdir(parents=True,exist_ok=True)
    model.forward_modules.eval()
    dataset_count=len(loader.loader.dataset)
    if loader.max_steps is not None:dataset_count=min(dataset_count,loader.max_steps*loader.loader.batch_size)
    bank=None;offset=0
    with tempfile.TemporaryDirectory(prefix="owned-patchcore-bank-",dir=root) as directory:
        target=Path(directory).resolve()/"features.npy"
        if not target.is_relative_to(root):raise ValueError("Temporary feature path escaped owned root")
        try:
            for batch in loader:
                image=batch["image"] if isinstance(batch,dict) else batch
                with torch.no_grad():
                    values=np.asarray(model._embed(image.to(torch.float).to(model.device)),dtype=np.float32)
                if bank is None:
                    patches=values.shape[0]//len(image)
                    if patches*len(image)!=len(values):raise ValueError("Unexpected patch count")
                    shape=(dataset_count*patches,values.shape[1])
                    if shutil.disk_usage(root).free<np.prod(shape)*4+8*1024**3:raise RuntimeError("Feature bank plus disk reserve does not fit")
                    bank=np.lib.format.open_memmap(target,mode="w+",dtype=np.float32,shape=shape)
                bank[offset:offset+len(values)]=values;offset+=len(values)
            if bank is None or offset!=len(bank):raise ValueError("Incomplete exact feature bank")
            bank.flush()
            selected=model.featuresampler.run(bank)
            model.anomaly_scorer.fit(detection_features=[selected])
            return {"full_feature_shape":list(bank.shape),"full_feature_dtype":"float32","full_feature_bytes":bank.nbytes,
                    "sampled_shape":list(selected.shape),"author_sampler_unchanged":True,"temporary_feature_bank_removed":True}
        finally:
            if bank is not None:bank._mmap.close()


def model_builder(settings):
    from .patchcore_runner import _default_model_builder
    model,report=_default_model_builder(settings)
    def fit(self,loader):
        measured=disk_backed_fit(self,loader,temporary_root=Path(settings.output_dir)/"fit_tmp")
        from .run_artifacts import write_json
        write_json(Path(settings.output_dir)/"disk_backed_fit.json",measured)
    model.fit=types.MethodType(fit,model)
    return model,{**report,"host_feature_collection":"disk-backed exact author-order float32 matrix; unchanged coreset and scoring"}
