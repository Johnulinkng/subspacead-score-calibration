"""Freeze a train-only selection, then evaluate all real categories and seeds."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import sys
import time
import traceback
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="2"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,sha,now
from run_route1_resource_trial import Monitor
from run_q3_trainonly_development import candidate_map
from analyze_context_contribution import paired_summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();run=ROOT/"runs/research-q3-20261016";dev=run/"trainonly-development";ind=run/"independent-partitions";features=run/"features"
    if out.exists() or not out.is_relative_to(ROOT/"runs"):raise ValueError("Fresh evaluation required")
    out.mkdir(parents=True)
    while not (dev/"selection.json").exists():
        if (dev/"failure.json").exists():raise RuntimeError("Development failed; no real-test candidate chosen")
        write(out/"run_state.json",{"status":"waiting_for_trainonly_selection","pid":os.getpid(),"updated_at":now()});time.sleep(10)
    selection=read(dev/"selection.json");spec=selection["selected"]
    if spec is None:
        write(out/"run_state.json",{"status":"no_supported_development_candidate","q3_goal_achieved":False,"reason":"All registered families lacked sufficient development effect or independent content contribution","finished_at":now()});return
    cfg=read(ROOT/"configs/q3_independent_partitions_20261016.json");tasks=read(ind/"frozen_protocol.json")["tasks"]
    sources=[Path(__file__),ROOT/"scripts/run_q3_trainonly_development.py",dev/"selection.json",ind/"frozen_protocol.json"]
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    write(out/"frozen_protocol.json",{"selected_spec":spec,"sha256":frozen,"tasks":tasks,"seeds":cfg["seeds"],"frozen_at":now(),"before_selected_real_test_access":True,"historical_test_access":True})
    monitor=Monitor(cfg["resource_policy"]);monitor.sample();monitor.check();monitor.thread.start();rows=[]
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.subspace_features import load_subspace_model,render_subspace_map
        from industrial_anomaly_lab.context_calibration import quantized_descriptor
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        threadpool_limits(2)
        for d,c in tasks:
            fc=features/"categories"/d/c;tests=read(fc/"test_manifest.json")
            for seed in cfg["seeds"]:
                source=ind/"categories"/d/c/str(seed);cat=out/"categories"/d/c/str(seed);cat.mkdir(parents=True)
                frame_rows=read(source/"frames.json");part=read(source/"partitions.json");pca=load_subspace_model(source/"pca.npz")
                with np.load(source/"calibration.npz",allow_pickle=False) as ar:fit={k:ar[k].copy() for k in ar.files}
                threshold_frames=[]
                for normal in part["threshold"]:
                    values=np.load(fc/normal["features"],allow_pickle=False);threshold_frames.append((pca.reconstruction_scores(values).reshape(32,32),quantized_descriptor(values,pca)))
                def apply(raw,z,variant):
                    if variant=="baseline":return render_subspace_map(raw)
                    if variant=="without_adjustment" and spec["family"]=="shrinkage":return candidate_map(raw,z,None,fit,{**spec,"alpha":1.})
                    return candidate_map(raw,z,None,fit,spec,remove_condition=variant=="without_conditioning",remove_adjustment=variant=="without_adjustment")
                for variant in ("baseline","full","without_conditioning","without_adjustment"):
                    monitor.set_stage(f"selected:{d}/{c}/{seed}/{variant}");dest=cat/variant;dest.mkdir();paths=[]
                    normal_maps=np.stack([apply(raw,z,variant) for raw,z in threshold_frames]);pixel_threshold=float(np.quantile(normal_maps,.99,method="higher"));image_threshold=float(np.quantile(normal_maps.max(axis=(1,2)),.95,method="higher"))
                    normal_pixels=normal_fp=anomaly_pixels=anomaly_tp=normal_images=normal_image_fp=0
                    for i,(fr,t) in enumerate(zip(frame_rows,tests,strict=True)):
                        monitor.check();path=source/fr["path"]
                        if __import__("shutil").disk_usage(out).free<8*1024**3:raise RuntimeError("Selected evaluation disk reserve")
                        if sha(path)!=fr["sha256"] or fr["sample_id"]!=t["sample_id"]:raise ValueError("Selected evaluation cache drift")
                        if sha(t["mask_path"])!=t["mask_sha256"]:raise ValueError("Selected evaluation GT drift")
                        with np.load(path,allow_pickle=False) as ar:scores=apply(ar["residual_grid"],ar["descriptor"],variant)
                        mask=np.load(t["mask_path"],allow_pickle=False).astype(bool);anomaly_pixels+=int(mask.sum());anomaly_tp+=int(np.count_nonzero(scores[mask]>pixel_threshold))
                        if t["label"]==0:normal_pixels+=scores.size;normal_fp+=int(np.count_nonzero(scores>pixel_threshold));normal_images+=1;normal_image_fp+=int(scores.max()>image_threshold)
                        fp=dest/f"{i:06d}.npy";np.save(fp,scores,allow_pickle=False);paths.append(fp)
                    metric=evaluate_from_files(paths,[t["mask_path"] for t in tests],[t["label"] for t in tests],temp_root=out/"metric_tmp")
                    row={"dataset":d,"category":c,"seed":seed,"variant":variant,**metric,"normal_pixel_fpr":normal_fp/normal_pixels,"normal_image_fpr":normal_image_fp/normal_images,"anomaly_pixel_recall":anomaly_tp/anomaly_pixels,"pixel_threshold":pixel_threshold,"image_threshold":image_threshold};rows.append(row);write(cat/(variant+".json"),row)
                    write(cat/(variant+"_maps.json"),{"sha256":{p.relative_to(cat).as_posix():sha(p) for p in paths}})
                write(out/"run_state.json",{"status":"running","dataset":d,"category":c,"seed":seed,"metric_rows":len(rows),"updated_at":now()})
        with (out/"category_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        contrasts=[];lookup={(r["dataset"],r["category"],r["seed"],r["variant"]):r for r in rows}
        for d in ("mvtec_ad","visa"):
            cats=[c for dd,c in tasks if dd==d]
            for reference in ("baseline","without_conditioning","without_adjustment"):
                for metric in ("aupro_0_30","pixel_ap","pixel_auroc","image_auroc","normal_pixel_fpr"):
                    delta=[np.mean([lookup[d,c,s,"full"][metric]-lookup[d,c,s,reference][metric] for s in cfg["seeds"]]) for c in cats]
                    contrasts.append({"dataset":d,"candidate":"full","reference":reference,"metric":metric,**paired_summary(delta)})
        write(out/"paired_statistics.json",contrasts)
        for rel,h in frozen.items():
            if sha(ROOT/rel)!=h:raise ValueError("Selected experiment source changed")
        write(out/"run_state.json",{"status":"completed","metric_rows":len(rows),"categories":27,"seeds":3,"historical_test_access":True,"q3_goal_achieved":False,"finished_at":now()})
    except BaseException as e:
        write(out/"failure.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
    finally:write(out/"resources.json",monitor.close())


if __name__=="__main__":main()
