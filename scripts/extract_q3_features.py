"""Authenticate and cache deterministic full-precision features for new fits."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
os.environ["HF_HUB_OFFLINE"]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,lines,write,sha,now
from run_route1_resource_trial import Monitor


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--resume",action="store_true");a=p.parse_args()
    run=a.output.resolve()
    if not run.is_relative_to(ROOT/"runs") or (run.exists() and not a.resume):raise ValueError("Fresh run or explicit resume required")
    run.mkdir(parents=True,exist_ok=True)
    src=ROOT/"runs/quick/route1-controls-20261014"
    config=read(ROOT/"configs/subspace_dinos_20261012.json")
    tasks=read(ROOT/"runs/quick/context-single-pass-20261015/frozen_protocol.json")["tasks"]
    frozen_paths=[Path(__file__),ROOT/"src/industrial_anomaly_lab/subspace_features.py",ROOT/"configs/subspace_dinos_20261012.json"]
    for d,c in tasks:frozen_paths += [src/"categories"/d/c/f for f in ("train_manifest.json","samples.jsonl")]
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in frozen_paths}
    if a.resume:
        if read(run/"frozen_protocol.json")["sha256"]!=frozen:raise ValueError("Resume inputs changed")
    else:write(run/"frozen_protocol.json",{"sha256":frozen,"tasks":tasks,"frozen_at":now(),"dtype":"float32","feature_shape":[1024,384],"historical_test_access":True,"purpose":"Deterministic feature cache, not repeated backbone training"})
    policy={"initial_free_ram_gib":3.25,"reserve_free_ram_gib":1.5,"reserve_free_commit_gib":2.,"max_working_set_gib":2.5}
    monitor=Monitor(policy);monitor.sample();monitor.check();monitor.thread.start();start=time.perf_counter()
    try:
        import numpy as np
        import torch
        from industrial_anomaly_lab.subspace_features import LocalDinoExtractor
        while True:
            monitor.check();free,_=torch.cuda.mem_get_info()
            if free>=1.75*1024**3:break
            write(run/"run_state.json",{"status":"waiting_for_gpu_headroom","free_gpu_bytes":free,"pid":os.getpid(),"updated_at":now()});time.sleep(5)
        extractor=LocalDinoExtractor(ROOT/config["weights_dir"],expected_checkpoint_sha256=config["checkpoint_sha256"],expected_artifact_sha256=config["weight_files_sha256"],device="cuda:0",cpu_threads=2,max_cuda_memory_gib=.9)
        total=0
        for d,c in tasks:
            cat=run/"categories"/d/c;cat.mkdir(parents=True,exist_ok=True)
            normal=read(src/"categories"/d/c/"train_manifest.json")
            tests=lines(src/"categories"/d/c/"samples.jsonl")
            for split,records in (("normal",normal),("test",tests)):
                manifest=cat/(split+"_manifest.json")
                if manifest.exists():
                    previous=read(manifest)
                    for row in previous:
                        if sha(cat/row["features"])!=row["features_sha256"]:raise ValueError("Feature cache identity drift")
                    total+=len(previous);continue
                dest=cat/split;dest.mkdir(exist_ok=True);out=[]
                for i,row in enumerate(records):
                    monitor.check()
                    if shutil.disk_usage(run).free<10*1024**3:raise RuntimeError("Feature cache disk reserve")
                    image=ROOT/row["path"] if split=="normal" else Path(row["image_path"])
                    expected=row["sha256"] if split=="normal" else row["image_sha256"]
                    if sha(image)!=expected:raise ValueError("Input image identity drift")
                    path=dest/f"{i:06d}.npy"
                    values=extractor(image).reshape(1024,384)
                    if values.dtype!=np.float32:raise ValueError("Unquantized float32 features required")
                    np.save(path,values,allow_pickle=False)
                    item={"path":image.relative_to(ROOT).as_posix(),"sha256":expected,"features":path.relative_to(cat).as_posix(),"features_sha256":sha(path)}
                    if split=="test":
                        item.update(sample_id=row["sample_id"],label=row["label"],mask_path=str(src/row["mask_path"]),mask_sha256=row["mask_sha256"])
                    out.append(item);total+=1
                    if i%20==0 or i+1==len(records):
                        write(run/"run_state.json",{"status":"running","pid":os.getpid(),"category":f"{d}/{c}","split":split,"images":i+1,"split_total":len(records),"total_cached":total,"elapsed_seconds":time.perf_counter()-start,"updated_at":now()})
                write(manifest,out)
                print(f"cached {d}/{c}/{split}: {len(out)} images",flush=True)
            write(cat/"completion.json",{"normal_manifest_sha256":sha(cat/"normal_manifest.json"),"test_manifest_sha256":sha(cat/"test_manifest.json"),"finished_at":now()})
        write(run/"run_state.json",{"status":"completed","images":total,"categories":27,"elapsed_seconds":time.perf_counter()-start,"finished_at":now()})
    except BaseException as e:
        write(run/"failure.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
    finally:
        receipt=monitor.close()
        if "torch" in sys.modules and torch.cuda.is_available():receipt.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(run/"resources.json",receipt)


if __name__=="__main__":main()
