"""Full PatchCore using bounded host feature collection, unchanged sampling."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fit_patchcore import parse_args,_project_config
from industrial_anomaly_lab.patchcore_runner import PatchCoreFitSettings,inspect_fit_inputs,run_patchcore_fit
from industrial_anomaly_lab.patchcore_bounded_fit import model_builder


def main():
    a=parse_args();cfg=_project_config()
    settings=PatchCoreFitSettings(dataset=a.dataset,category=a.category,
        dataset_root=a.dataset_root or ROOT/cfg["paths"]["mvtec" if a.dataset=="mvtec_ad" else "visa"],
        output_dir=a.output_dir,backbone_checkpoint=a.backbone_checkpoint,backbone_sha256=a.backbone_sha256,
        seed=a.seed,execution_scope=a.execution_scope,batch_size=a.batch_size,workers=a.workers,
        device=a.device,coreset_device=a.coreset_device,faiss_workers=a.faiss_workers,max_steps=a.max_steps,run_purpose=a.run_purpose)
    result=inspect_fit_inputs(settings) if a.validate_only else run_patchcore_fit(settings,model_builder=model_builder)
    result.pop("train_manifest_rows",None);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
