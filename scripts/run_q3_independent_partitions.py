"""New PCA/calibration fits with genuinely disjoint normal partitions."""
from __future__ import annotations
import argparse
import csv
import gc
import os
from pathlib import Path
import sys
import time
import traceback
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="3"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,sha,now
from run_route1_resource_trial import Monitor
from run_q3_contribution_refits import apply_model
from analyze_context_contribution import paired_summary


def partitions(n,seed,cfg,content_hashes=None):
    import numpy as np
    groups=None
    if content_hashes is not None:
        groups={}
        for i,h in enumerate(content_hashes):groups.setdefault(h,[]).append(i)
        groups=[groups[h] for h in sorted(groups)]
        n=len(groups)
    p=cfg["partition"];order=np.random.default_rng(seed).permutation(n)
    ns=min(p["statistics_maximum"],max(p["statistics_minimum"],int(np.ceil(n*p["statistics_fraction"]))))
    nd=min(p["development_maximum"],max(p["development_minimum"],int(np.ceil(n*p["development_fraction"]))))
    if n-2*ns-nd<8:raise ValueError("Too few disjoint normal images")
    result={"statistics":order[:ns],"threshold":order[ns:2*ns],"development":order[2*ns:2*ns+nd],"pca_fit":order[2*ns+nd:]}
    if groups is not None:result={k:np.array([i for gi in ix for i in groups[gi]],dtype=np.int64) for k,ix in result.items()}
    return result


def hardness(path):
    import numpy as np
    import cv2
    from PIL import Image
    with Image.open(path) as im:x=np.asarray(im.convert("RGB").resize((448,448),Image.Resampling.BICUBIC),dtype=np.float32)/255
    pad=np.pad(x,((1,1),(1,1),(0,0)),mode="edge")
    mean=(pad[1:-1,:-2]+pad[1:-1,2:]+pad[:-2,1:-1]+pad[2:,1:-1])*.25
    return cv2.resize(np.mean(np.abs(x-mean),axis=2),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)


def fit_calibration(z,r,h,cfg,seed):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import QuantileRegressor
    from scipy.ndimage import gaussian_filter
    from industrial_anomaly_lab.context_calibration import nearest_centers
    x=z.reshape(-1,z.shape[-1]);flat=r.ravel();rng=np.random.default_rng(seed)
    centers=MiniBatchKMeans(n_clusters=cfg["clusters"],random_state=seed,**cfg["kmeans"]).fit(x.astype(np.float32)).cluster_centers_
    labels=nearest_centers(x,centers)
    global_tail=max(float(np.quantile(flat,cfg["quantile"],method="higher")),1e-12)
    random_labels=rng.integers(0,len(centers),len(flat));tails=[];rt=[];counts=[]
    for j in range(len(centers)):
        v,rv=flat[labels==j],flat[random_labels==j];counts.append(len(v))
        tails.append(float(np.quantile(v,cfg["quantile"],method="higher")) if len(v)>=cfg["minimum_group_tokens"] else global_tail)
        rt.append(float(np.quantile(rv,cfg["quantile"],method="higher")) if len(rv)>=cfg["minimum_group_tokens"] else global_tail)
    density=np.sum((x.astype(np.float64)-centers[labels])**2,axis=1)
    qd=max(float(np.quantile(density,.95,method="higher")),1e-12)
    selected=rng.choice(len(flat),min(len(flat),cfg["cadet"]["normal_patch_sample_maximum"]),replace=False)
    hf=h.ravel().astype(np.float64);hscale=max(float(hf.std()),1e-8);hn=(hf-hf.mean())/hscale
    regress=[]
    for q in cfg["cadet"]["quantiles"]:
        model=QuantileRegressor(quantile=q,alpha=cfg["cadet"]["alpha"],solver="highs").fit(hn[selected,None],flat[selected])
        regress.append([float(model.intercept_),float(model.coef_[0])])
    return {"centers":centers,"tails":np.array(tails),"random_tails":np.maximum(rt,global_tail),"global_tail":np.array(global_tail),
            "counts":np.array(counts),"spatial_tail":np.maximum(gaussian_filter(np.quantile(r,.95,axis=0,method="higher"),cfg["spatial_sigma_grid"]),global_tail),
            "density_q95":np.array(qd),"cadet_coefficients":np.array(regress),"hardness_mean":np.array(hf.mean()),"hardness_std":np.array(hscale)}


def apply_adjusted(raw,z,h,fit,variant,cfg,rng):
    import numpy as np
    from industrial_anomaly_lab.subspace_features import render_subspace_map
    from industrial_anomaly_lab.context_calibration import nearest_centers
    if variant=="baseline":return render_subspace_map(raw)
    if variant in ("global_tail","conditional_tail","random_group_tail","spatial_tail"):
        return apply_model(raw,z,fit,variant,rng)
    if variant=="conditional_shrunk":
        alpha=fit["counts"]/(fit["counts"]+cfg["tail_shrinkage_token_prior"])
        adjusted={**fit,"tails":float(fit["global_tail"])+alpha*(fit["tails"]-float(fit["global_tail"]))}
        return apply_model(raw,z,adjusted,"conditional_tail",rng)
    if variant=="cadet_linear_control":
        hv=(h.astype(np.float64)-float(fit["hardness_mean"]))/float(fit["hardness_std"])
        low,high=(a+b*hv for a,b in fit["cadet_coefficients"])
        prior=(h.astype(np.float64)-float(fit["hardness_mean"]))**2/float(fit["hardness_std"])
        scores=np.abs(raw-high)/(np.abs(high-low)+cfg["cadet"]["denominator_offset"])+cfg["cadet"]["hardness_prior_weight"]*prior
        return render_subspace_map(scores)
    labels=nearest_centers(z,fit["centers"])
    distances=np.sum((z.astype(np.float64)-fit["centers"][labels])**2,axis=1).reshape(raw.shape)
    density=np.maximum(0.,distances/float(fit["density_q95"])-1.)
    if variant=="prototype_density":scores=raw.astype(np.float64)
    elif variant=="combined_density_tail":
        bound=np.maximum(fit["tails"][labels].reshape(raw.shape),float(fit["global_tail"]))
        scores=raw.astype(np.float64)*np.minimum(1.,raw/bound)
    else:raise ValueError(variant)
    return render_subspace_map(scores+cfg["density_weight"]*float(fit["global_tail"])*density)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,default=ROOT/"configs/q3_independent_partitions_20261016.json");p.add_argument("--resume",action="store_true");a=p.parse_args()
    cfg=read(a.config);run=a.output.resolve();features=ROOT/cfg["feature_cache"]
    if not run.is_relative_to(ROOT/"runs") or (run.exists() and not a.resume):raise ValueError("Fresh run or explicit resume required")
    run.mkdir(parents=True,exist_ok=True)
    tasks=read(features/"frozen_protocol.json")["tasks"]
    files=[a.config,Path(__file__),ROOT/"scripts/run_q3_contribution_refits.py",ROOT/"scripts/analyze_context_contribution.py",ROOT/"src/industrial_anomaly_lab/subspace_features.py",ROOT/"src/industrial_anomaly_lab/quick_metrics.py",features/"frozen_protocol.json"]
    seal={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    if a.resume:
        if read(run/"frozen_protocol.json")["sha256"]!=seal:raise ValueError("Resume input drift")
    else:write(run/"frozen_protocol.json",{"protocol":cfg,"sha256":seal,"tasks":tasks,"frozen_at":now(),"before_new_fits_and_metrics":True})
    policy=cfg["resource_policy"];monitor=Monitor(policy);monitor.sample();monitor.check();monitor.thread.start();started=time.perf_counter();rows=[]
    try:
        import cv2
        import numpy as np
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.subspace_features import StreamingPCA,save_subspace_model
        from industrial_anomaly_lab.context_calibration import quantized_descriptor
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        cv2.setNumThreads(2);threadpool_limits(3)
        for d,c in tasks:
            fc=features/"categories"/d/c
            while not (fc/"completion.json").exists():
                monitor.check();write(run/"run_state.json",{"status":"waiting_for_feature_category","category":f"{d}/{c}","pid":os.getpid(),"updated_at":now()});time.sleep(5)
            normals,tests=read(fc/"normal_manifest.json"),read(fc/"test_manifest.json")
            if {r["sha256"] for r in normals}.intersection(r["sha256"] for r in tests):raise ValueError("Normal/Test identical content overlap")
            for row in normals+tests:
                if sha(fc/row["features"])!=row["features_sha256"]:raise ValueError("Feature identity drift")
            for t in tests:
                if sha(t["mask_path"])!=t["mask_sha256"]:raise ValueError("Mask identity drift")
            for seed in cfg["seeds"]:
                out=run/"categories"/d/c/str(seed);out.mkdir(parents=True,exist_ok=True)
                if (out/"metrics.json").exists():rows.extend(read(out/"metrics.json"));continue
                part=partitions(len(normals),seed,cfg,[r["sha256"] for r in normals])
                partition_rows={k:[normals[i] for i in ix] for k,ix in part.items()}
                hashes=[{r["sha256"] for r in rr} for rr in partition_rows.values()]
                for i,hs in enumerate(hashes):
                    if any(hs.intersection(other) for other in hashes[i+1:]):raise ValueError("Content overlap between normal partitions")
                write(out/"partitions.json",partition_rows)
                monitor.set_stage(f"PCA-fit:{d}/{c}/{seed}");tick=time.perf_counter();acc=StreamingPCA(384)
                for i in part["pca_fit"]:
                    monitor.check();acc.update(np.load(fc/normals[i]["features"],allow_pickle=False))
                model=acc.finalize(cfg["explained_variance"]);save_subspace_model(model,out/"pca.npz");pca_seconds=time.perf_counter()-tick
                z,r,h=[],[],[]
                for i in part["statistics"]:
                    values=np.load(fc/normals[i]["features"],allow_pickle=False)
                    z.append(quantized_descriptor(values,model));r.append(model.reconstruction_scores(values).reshape(32,32));h.append(hardness(ROOT/normals[i]["path"]))
                tick=time.perf_counter();fit=fit_calibration(np.stack(z),np.stack(r),np.stack(h),cfg,seed);fit_seconds=time.perf_counter()-tick
                np.savez_compressed(out/"calibration.npz",**fit)
                frames=out/"frames";frames.mkdir(exist_ok=True);frame_rows=[]
                for i,t in enumerate(tests):
                    monitor.check();values=np.load(fc/t["features"],allow_pickle=False)
                    path=frames/f"{i:06d}.npz";np.savez_compressed(path,residual_grid=model.reconstruction_scores(values).reshape(32,32),descriptor=quantized_descriptor(values,model),hardness=hardness(ROOT/t["path"]))
                    frame_rows.append({"path":path.relative_to(out).as_posix(),"sha256":sha(path),"sample_id":t["sample_id"]})
                write(out/"frames.json",frame_rows);category=[]
                for variant in cfg["variants"]:
                    result=out/(variant+".json")
                    if result.exists():category.append(read(result));continue
                    monitor.set_stage(f"metrics:{d}/{c}/{seed}/{variant}");dest=out/variant;dest.mkdir(exist_ok=True);paths=[];timings=[];rng=np.random.default_rng(seed ^ 428101)
                    for i,fr in enumerate(frame_rows):
                        monitor.check()
                        if __import__("shutil").disk_usage(run).free<policy["minimum_disk_gib"]*1024**3:raise RuntimeError("Disk reserve")
                        with np.load(out/fr["path"],allow_pickle=False) as ar:
                            tick=time.perf_counter();scores=apply_adjusted(ar["residual_grid"],ar["descriptor"],ar["hardness"],fit,variant,cfg,rng);timings.append(time.perf_counter()-tick)
                        path=dest/f"{i:06d}.npy";np.save(path,scores,allow_pickle=False);paths.append(path)
                    m=evaluate_from_files(paths,[t["mask_path"] for t in tests],[t["label"] for t in tests],temp_root=run/"metric_tmp")
                    row={"dataset":d,"category":c,"seed":seed,"variant":variant,**m,"normal_pca_images":len(part["pca_fit"]),"normal_statistics_images":len(part["statistics"]),"pca_fit_seconds":pca_seconds,"calibrator_fit_seconds":fit_seconds,"apply_median_ms":float(np.median(timings)*1000)}
                    write(result,row);write(out/(variant+"_map_manifest.json"),{"sha256":{p.relative_to(out).as_posix():sha(p) for p in paths}});category.append(row)
                    print(f"independent {d}/{c}/{seed}/{variant}: AP={m['pixel_ap']:.6f} PRO={m['aupro_0_30']:.6f}",flush=True)
                write(out/"metrics.json",category);rows.extend(category)
                write(run/"run_state.json",{"status":"running","category":f"{d}/{c}","seed":seed,"metric_rows":len(rows),"updated_at":now(),"pid":os.getpid()});gc.collect()
        with (run/"category_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        contrasts=[]
        lookup={(r["dataset"],r["category"],r["seed"],r["variant"]):r for r in rows}
        for d in ("mvtec_ad","visa"):
            cats=sorted({r["category"] for r in rows if r["dataset"]==d})
            for candidate,reference in (("conditional_tail","baseline"),("conditional_tail","global_tail"),("conditional_tail","random_group_tail"),("conditional_tail","spatial_tail"),("conditional_tail","cadet_linear_control"),("conditional_shrunk","conditional_tail"),("combined_density_tail","prototype_density"),("prototype_density","baseline")):
                for metric in ("aupro_0_30","pixel_ap","pixel_auroc","image_auroc"):
                    diffs=[np.mean([lookup[d,c,s,candidate][metric]-lookup[d,c,s,reference][metric] for s in cfg["seeds"]]) for c in cats]
                    contrasts.append({"dataset":d,"candidate":candidate,"reference":reference,"metric":metric,**paired_summary(diffs)})
        prior=0.
        for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]["exact_two_sided_sign_flip_p"])):
            prior=max(prior,min(1.,(len(contrasts)-rank)*contrasts[i]["exact_two_sided_sign_flip_p"]));contrasts[i]["holm_familywise_p"]=prior
        write(run/"paired_statistics.json",contrasts)
        for rel,hsh in seal.items():
            if sha(ROOT/rel)!=hsh:raise ValueError("Source drift during independent experiment")
        write(run/"run_state.json",{"status":"completed","categories":27,"pca_fits":81,"calibration_fits":81,"metric_rows":len(rows),"finished_at":now(),"elapsed_seconds":time.perf_counter()-started,"blind_test":False,"q3_goal_achieved":False})
    except BaseException as e:
        write(run/"failure.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
    finally:write(run/"resources.json",monitor.close())


if __name__=="__main__":main()
