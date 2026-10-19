"""Three genuine calibration refits and mechanism controls on sealed features."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "3"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from run_score_calibration import read, lines, sha, write, now
from run_route1_resource_trial import Monitor
from analyze_context_contribution import paired_summary


def fit_models(z, residual, cfg, seed, clusters=None, quantile=None):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from scipy.ndimage import gaussian_filter
    from industrial_anomaly_lab.context_calibration import nearest_centers
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(z))
    n = min(len(z)-8, max(8, int(len(z)*cfg["statistics_fraction"])))
    indices = order[:n]
    x, r = z[indices].reshape(-1, z.shape[-1]), residual[indices].ravel()
    k, q = clusters or cfg["clusters"], quantile or cfg["quantile"]
    centers = MiniBatchKMeans(n_clusters=k, random_state=seed, **cfg["kmeans"]).fit(x.astype(np.float32)).cluster_centers_
    labels = nearest_centers(x, centers)
    random_labels = rng.integers(0, k, len(r))
    global_tail = max(float(np.quantile(r, q, method="higher")), 1e-12)
    global_iqr = max(float(np.quantile(r,.75)-np.quantile(r,.25)),1e-12)
    tails, random_tails, centers50, scales = [], [], [], []
    for j in range(k):
        v, vr = r[labels == j], r[random_labels == j]
        tails.append(float(np.quantile(v,q,method="higher")) if len(v)>=cfg["minimum_group_tokens"] else global_tail)
        random_tails.append(float(np.quantile(vr,q,method="higher")) if len(vr)>=cfg["minimum_group_tokens"] else global_tail)
        centers50.append(float(np.quantile(v,q,method="higher")) if len(v) else global_tail)
        scales.append(max(float(np.quantile(v,.75)-np.quantile(v,.25)) if len(v) else global_iqr, global_iqr*.05))
    spatial = gaussian_filter(np.quantile(residual[indices],q,axis=0,method="higher"),cfg["spatial_sigma_grid"])
    return {"centers": centers, "tails": np.array(tails), "random_tails": np.maximum(random_tails,global_tail),
            "global_tail": np.array(global_tail), "quantile_centers": np.array(centers50), "scales": np.array(scales),
            "spatial_tail": np.maximum(spatial,global_tail), "statistics_indices": indices,
            "threshold_indices": order[n:], "seed": np.array(seed), "quantile": np.array(q)}


def apply_model(raw, z, fitted, variant, rng):
    import numpy as np
    from industrial_anomaly_lab.context_calibration import nearest_centers
    from industrial_anomaly_lab.subspace_features import render_subspace_map
    r = raw.astype(np.float64)
    g = float(fitted["global_tail"])
    if variant == "global_tail":
        bound = g
    elif variant == "spatial_tail":
        bound = fitted["spatial_tail"]
    elif variant == "random_group_tail":
        bound = fitted["random_tails"][rng.integers(0,len(fitted["centers"]),r.shape)]
    else:
        labels = nearest_centers(z,fitted["centers"]).reshape(r.shape)
        tails = fitted["tails"][labels]
        if variant == "prototype_quantile_control":
            return render_subspace_map((r-fitted["quantile_centers"][labels])/fitted["scales"][labels])
        bound = tails if variant == "conditional_no_floor" else np.maximum(tails,g)
        if variant == "conditional_no_guard":
            return render_subspace_map(r*g/bound)
    return render_subspace_map(r*np.minimum(1.,r/np.maximum(bound,1e-12)))


def summarize(run, rows, cfg):
    import numpy as np
    metrics=("aupro_0_30","pixel_ap","pixel_auroc","image_auroc")
    by = {(r["dataset"],r["category"],r["seed"],r["variant"]):r for r in rows}
    summary=[]
    all_variants=["baseline"]+cfg["variants"]
    for d in ("mvtec_ad","visa"):
        cats=sorted({r["category"] for r in rows if r["dataset"]==d})
        for v in all_variants:
            selected=[r for r in rows if r["dataset"]==d and r["variant"]==v]
            if len(selected)!=len(cats)*len(cfg["seeds"]):continue
            seed_macro={s:{m:float(np.mean([by[d,c,s,v][m] for c in cats])) for m in metrics} for s in cfg["seeds"]}
            summary.append({"dataset":d,"variant":v,"seed_macro":seed_macro,
                "mean":{m:float(np.mean([x[m] for x in seed_macro.values()])) for m in metrics},
                "std":{m:float(np.std([x[m] for x in seed_macro.values()],ddof=1)) for m in metrics}})
    contrasts=[]
    for d in ("mvtec_ad","visa"):
        cats=sorted({r["category"] for r in rows if r["dataset"]==d})
        for reference in ("baseline","global_tail","random_group_tail","spatial_tail","prototype_quantile_control"):
            for m in metrics:
                delta=[float(np.mean([by[d,c,s,"conditional_tail"][m]-by[d,c,s,reference][m] for s in cfg["seeds"]])) for c in cats]
                contrasts.append({"dataset":d,"candidate":"conditional_tail","reference":reference,"metric":m,**paired_summary(delta)})
    previous=0.
    for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]["exact_two_sided_sign_flip_p"])):
        previous=max(previous,min(1.,(len(contrasts)-rank)*contrasts[i]["exact_two_sided_sign_flip_p"]))
        contrasts[i]["holm_familywise_p"]=previous
    with (run/"category_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(run/"dataset_macro_metrics.json",summary)
    write(run/"paired_statistics.json",contrasts)
    primary=[r for r in contrasts if r["metric"]=="aupro_0_30" and r["reference"]=="baseline"]
    content=[r for r in contrasts if r["metric"]=="aupro_0_30" and r["reference"]=="global_tail"]
    passed=(all(r["mean_delta_pp"]>0 and r["positive_categories"]/r["categories"]>=.6 for r in primary)
        and np.mean([r["mean_delta_pp"] for r in primary])>=1.
        and any(r["bootstrap_95_ci_pp"][0]>0 for r in primary)
        and all(r["mean_delta_pp"]>0 for r in content))
    write(run/"contribution_decision.json",{"original_method_contribution_supported":bool(passed),"primary":primary,
        "independent_content_effect":content,"scope":cfg["scope"],"blind_test":False,
        "q3_goal_achieved":False,"reason":"Full strong-control and independent backbone/partition experiments remain required"})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/q3_contribution_followup_20261016.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args();cfg=read(a.config);run=a.output.resolve()
    if not run.is_relative_to(ROOT/"runs") or (run.exists() and not a.resume):raise ValueError("Fresh run or explicit resume required")
    run.mkdir(parents=True,exist_ok=True)
    normal,test,inventory=(ROOT/cfg[k] for k in ("normal_source","test_source","inventory_source"))
    tasks=read(test/"frozen_protocol.json")["tasks"]
    seal_paths=[a.config,Path(__file__),ROOT/"scripts/analyze_context_contribution.py",ROOT/"src/industrial_anomaly_lab/context_calibration.py",ROOT/"src/industrial_anomaly_lab/quick_metrics.py",ROOT/"src/industrial_anomaly_lab/subspace_features.py"]
    for d,c in tasks:
        seal_paths += [normal/"categories"/d/c/f for f in ("normal_context.npz","normal_records.json","completion.json")]
        seal_paths += [test/"categories"/d/c/f for f in ("frames.json","metrics.json","completion.json")]
        seal_paths += [inventory/"categories"/d/c/"samples.jsonl"]
    seal={str(x.relative_to(ROOT).as_posix()):sha(x) for x in seal_paths}
    if a.resume:
        if read(run/"frozen_protocol.json")["sha256"]!=seal:raise ValueError("Resume source drift")
    else:write(run/"frozen_protocol.json",{"protocol":cfg,"sha256":seal,"tasks":tasks,"frozen_at":now(),"before_new_refits":True})
    monitor=Monitor(cfg["resource_policy"]);monitor.sample();monitor.check();monitor.thread.start()
    started=time.perf_counter();rows=[]
    try:
        import cv2
        import numpy as np
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        cv2.setNumThreads(2);threadpool_limits(3)
        for d,c in tasks:
            cat=run/"categories"/d/c;cat.mkdir(parents=True,exist_ok=True)
            tc=test/"categories"/d/c
            frames=read(tc/"frames.json");samples=lines(inventory/"categories"/d/c/"samples.jsonl")
            masks=[inventory/s["mask_path"] for s in samples]
            for s,m in zip(samples,masks,strict=True):
                if sha(m)!=s["mask_sha256"]:raise ValueError("Mask identity drift")
            for fr in frames:
                if sha(tc/fr["cache"])!=fr["cache_sha256"]:raise ValueError("Feature cache drift")
            baseline=next(x for x in read(tc/"metrics.json") if x["variant"]=="baseline")
            with np.load(normal/"categories"/d/c/"normal_context.npz",allow_pickle=False) as ar:
                z,r=ar["latent"].copy(),ar["residual_grid"].copy()
            for si,seed in enumerate(cfg["seeds"]):
                sd=cat/str(seed);sd.mkdir(exist_ok=True)
                base={"dataset":d,"category":c,"seed":seed,"variant":"baseline",**{m:baseline[m] for m in ("image_auroc","pixel_auroc","pixel_ap","aupro_0_30")},"fit_seconds":0.,"apply_median_ms":0.,"parameter_bytes":0}
                rows.append(base)
                specs=[{"name":v} for v in cfg["variants"]]+(cfg["sensitivity_first_seed"] if si==0 else [])
                tick=time.perf_counter();fitted=fit_models(z,r,cfg,seed);fit_seconds=time.perf_counter()-tick
                np.savez_compressed(sd/"fit.npz",**fitted)
                for spec in specs:
                    v=spec["name"];result=sd/(v+".json")
                    if result.exists():rows.append(read(result));continue
                    monitor.set_stage(f"{d}/{c}/{seed}/{v}")
                    stage=sd/v;stage.mkdir(exist_ok=True);files=[];timings=[];rng=np.random.default_rng(seed ^ 428101)
                    model=fitted
                    if "clusters" in spec:
                        model=fit_models(z,r,cfg,seed,spec["clusters"],spec["quantile"])
                        np.savez_compressed(sd/(v+"_fit.npz"),**model)
                    for j,fr in enumerate(frames):
                        monitor.check()
                        with np.load(tc/fr["cache"],allow_pickle=False) as ar:
                            raw,desc=ar["residual_grid"],ar["descriptor"]
                        tick=time.perf_counter();scores=apply_model(raw,desc,model,"conditional_tail" if "clusters" in spec else v,rng)
                        timings.append(time.perf_counter()-tick)
                        path=stage/f"{j:06d}.npy";np.save(path,scores,allow_pickle=False);files.append(path)
                    metric=evaluate_from_files(files,masks,[s["label"] for s in samples],temp_root=run/"metric_tmp")
                    record={**base,"variant":v,**metric,"fit_seconds":fit_seconds,"apply_median_ms":float(np.median(timings)*1000),"parameter_bytes":sum(x.nbytes for x in model.values())}
                    write(result,record);write(sd/(v+"_map_manifest.json"),{"map_sha256":{p.relative_to(sd).as_posix():sha(p) for p in files},"samples":[s["sample_id"] for s in samples]})
                    rows.append(record)
                    write(run/"run_state.json",{"status":"running","pid":os.getpid(),"category":f"{d}/{c}","seed":seed,"variant":v,"metric_rows":len(rows),"updated_at":now()})
                    print(f"{d}/{c} seed={seed} {v}: AP={metric['pixel_ap']:.6f} PRO={metric['aupro_0_30']:.6f}",flush=True)
                gc.collect()
        summarize(run,rows,cfg)
        for rel,h in seal.items():
            if sha(ROOT/rel)!=h:raise ValueError("Source changed during run: "+rel)
        write(run/"run_state.json",{"status":"completed","metric_rows":len(rows),"categories":27,"calibration_refits":81,"elapsed_seconds":time.perf_counter()-started,"finished_at":now()})
    except BaseException as e:
        write(run/"failure.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
    finally:write(run/"resources.json",monitor.close())


if __name__=="__main__":main()
