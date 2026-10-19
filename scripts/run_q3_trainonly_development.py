"""Bounded candidate adjustment driven only by held-out normal/DTD development."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="2"
os.environ["HF_HUB_OFFLINE"]="1"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,sha,now
from run_route1_resource_trial import Monitor
from analyze_context_contribution import paired_summary


def candidate_map(raw,z,h,fit,spec,*,remove_condition=False,remove_adjustment=False):
    import numpy as np
    from scipy.ndimage import uniform_filter
    from industrial_anomaly_lab.context_calibration import nearest_centers
    from industrial_anomaly_lab.subspace_features import render_subspace_map
    r=raw.astype(np.float64);g=float(fit["global_tail"])
    labels=nearest_centers(z,fit["centers"]);bound=np.maximum(fit["tails"][labels].reshape(r.shape),g)
    if remove_condition:bound=np.full(r.shape,g)
    family=spec["family"]
    if family=="shrinkage":bound=g+spec["alpha"]*(bound-g)
    corrected=r*np.minimum(1.,r/np.maximum(bound,1e-12))
    if remove_adjustment:return render_subspace_map(corrected)
    if family=="density":
        dist=np.sum((z.astype(np.float64)-fit["centers"][labels])**2,axis=1).reshape(r.shape)
        corrected += spec["weight"]*g*np.maximum(0.,dist/float(fit["density_q95"])-1.)
    elif family=="support":
        support=uniform_filter((r>g).astype(np.float64),size=spec["window"],mode="reflect")
        corrected=r+spec["strength"]*(1-support)*(corrected-r)
    return render_subspace_map(corrected)


def synthetic_image(image_path,texture_path,seed,scale,cfg):
    import numpy as np
    from PIL import Image
    rng=np.random.default_rng(seed)
    with Image.open(image_path) as im:x=np.asarray(im.convert("RGB").resize((256,256),Image.Resampling.BICUBIC),dtype=np.float32)/255
    with Image.open(texture_path) as im:t=np.asarray(im.convert("RGB").resize((256,256),Image.Resampling.BICUBIC),dtype=np.float32)/255
    aspect=float(np.exp(rng.uniform(np.log(.5),np.log(2.))))
    height=min(240,max(4,int(np.sqrt(scale*256*256/aspect))));width=min(240,max(4,int(height*aspect)))
    top=int(rng.integers(0,257-height));left=int(rng.integers(0,257-width));mask=np.zeros((256,256),bool);mask[top:top+height,left:left+width]=True
    rotation=int(rng.integers(0,4));t=np.rot90(t,rotation).copy();blend=float(rng.uniform(*cfg["blend_range"]))
    x[mask]=(1-blend)*x[mask]+blend*t[mask]
    return Image.fromarray(np.round(np.clip(x,0,1)*255).astype(np.uint8)),mask,{"seed":seed,"scale":scale,"height":height,"width":width,"top":top,"left":left,"blend":blend,"rotation90":rotation}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,default=ROOT/"configs/q3_trainonly_development_20261016.json");a=p.parse_args()
    cfg=read(a.config);out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/"runs"):raise ValueError("Fresh task output required")
    out.mkdir(parents=True);ind=ROOT/"runs/research-q3-20261016/independent-partitions";features=ROOT/"runs/research-q3-20261016/features"
    source=[a.config,Path(__file__),ROOT/"scripts/run_q3_independent_partitions.py",ind/"frozen_protocol.json",features/"frozen_protocol.json",ROOT/"src/industrial_anomaly_lab/subspace_features.py"]
    write(out/"frozen_protocol.json",{"config":cfg,"sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in source},"frozen_at":now(),"before_synthetic_selection":True})
    specs=[{**x,"family":family} for family,items in cfg["families"].items() for x in items]
    categories=read(ROOT/"configs/project.json")["datasets"]["visa"]["categories"]
    # GPU extraction is shared by configurations and starts only when the full original-image cache has finished.
    while not (features/"run_state.json").exists() or read(features/"run_state.json").get("status")!="completed":
        if (features/"failure.json").exists():raise RuntimeError("Feature extraction failed; development not started")
        write(out/"run_state.json",{"status":"waiting_for_feature_extraction","pid":os.getpid(),"updated_at":now()});time.sleep(10)
    import numpy as np
    import torch
    from PIL import Image
    from threadpoolctl import threadpool_limits
    from industrial_anomaly_lab.subspace_features import LocalDinoExtractor,load_subspace_model,render_subspace_map
    from industrial_anomaly_lab.context_calibration import quantized_descriptor
    from industrial_anomaly_lab.quick_metrics import evaluate_from_files
    from run_q3_independent_partitions import hardness
    threadpool_limits(2)
    policy={"reserve_free_ram_gib":1.5,"reserve_free_commit_gib":2.,"max_working_set_gib":2.5};monitor=Monitor(policy);monitor.sample();monitor.check();monitor.thread.start()
    while True:
        free,_=torch.cuda.mem_get_info()
        if free>=1.75*1024**3:break
        monitor.check();time.sleep(5)
    backbone=read(ROOT/"configs/subspace_dinos_20261012.json")
    extractor=LocalDinoExtractor(ROOT/backbone["weights_dir"],expected_checkpoint_sha256=backbone["checkpoint_sha256"],expected_artifact_sha256=backbone["weight_files_sha256"],device="cuda:0",cpu_threads=2,max_cuda_memory_gib=.9)
    textures=sorted((ROOT/"data/dtd/images").glob("*/*.jpg"));rows=[];winners=[];start=time.perf_counter()
    try:
        for seed_index,seed in enumerate(cfg["seed_order"]):
            active=specs if seed_index==0 else winners
            if seed_index>0 and not active:break
            for category in categories:
                model_dir=ind/"categories/visa"/category/str(seed)
                while not (model_dir/"metrics.json").exists():
                    monitor.check()
                    if (ind/"failure.json").exists():raise RuntimeError("Independent model fitting failed")
                    write(out/"run_state.json",{"status":"waiting_for_independent_fit","category":category,"seed":seed,"pid":os.getpid(),"updated_at":now()});time.sleep(10)
                model=load_subspace_model(model_dir/"pca.npz")
                write(out/(f"model_source_{category}_{seed}.json"),{"sha256":{p.name:sha(p) for p in [model_dir/"pca.npz",model_dir/"calibration.npz",model_dir/"partitions.json"]},"recorded_at":now()})
                with np.load(model_dir/"calibration.npz",allow_pickle=False) as ar:fit={k:ar[k].copy() for k in ar.files}
                partitions=read(model_dir/"partitions.json");targets=partitions["development"][:cfg["targets_per_category_and_seed"]]
                threshold_frames=[]
                for target in partitions["threshold"]:
                    values=np.load(features/"categories/visa"/category/target["features"],allow_pickle=False)
                    threshold_frames.append((model.reconstruction_scores(values).reshape(32,32),quantized_descriptor(values,model)))
                cat=out/"categories"/category/str(seed);cat.mkdir(parents=True);frames=[];masks=[];labels=[];recipes=[]
                for i,target in enumerate(targets):
                    image=ROOT/target["path"]
                    if sha(image)!=target["sha256"]:raise ValueError("Development input changed")
                    values=np.load(features/"categories/visa"/category/target["features"],allow_pickle=False)
                    records=[(values,np.zeros((256,256),bool),{"type":"normal","target_sha256":target["sha256"]})]
                    for j,scale in enumerate(cfg["synthetic_scales"]):
                        digest=hashlib.sha256(f"{seed}|{category}|{target['sha256']}|{j}".encode()).hexdigest();recipe_seed=int(digest[:16],16)
                        texture=textures[int(digest[16:32],16)%len(textures)]
                        pil,mask,recipe=synthetic_image(image,texture,recipe_seed,scale,cfg)
                        values=extractor.extract_batch([pil]).reshape(1024,384)
                        recipe.update(type="synthetic",target_sha256=target["sha256"],texture_path=texture.relative_to(ROOT).as_posix(),texture_sha256=sha(texture))
                        records.append((values,mask,recipe))
                    for values,mask,recipe in records:
                        index=len(frames);fp=cat/f"frame{index:04d}.npz";mp=cat/f"mask{index:04d}.npy"
                        # Hardness is unused by candidate families; store zero to keep this phase independent of real-test controls.
                        np.savez_compressed(fp,residual_grid=model.reconstruction_scores(values).reshape(32,32),descriptor=quantized_descriptor(values,model))
                        np.save(mp,mask,allow_pickle=False);frames.append(fp);masks.append(mp);labels.append(int(mask.any()));recipes.append(recipe)
                write(cat/"recipes.json",recipes)
                for spec in ([{"name":"baseline","family":"baseline"}]+active):
                    variants=[spec["name"]] if spec["family"]=="baseline" else [spec["name"],spec["name"]+"_no_condition"]
                    for variant in variants:
                        dest=cat/variant;dest.mkdir();paths=[]
                        threshold_maps=[]
                        for raw,z in threshold_frames:
                            threshold_maps.append(render_subspace_map(raw) if spec["family"]=="baseline" else candidate_map(raw,z,None,fit,spec,remove_condition=variant.endswith("_no_condition")))
                        threshold=float(np.quantile(np.stack(threshold_maps),.99,method="higher"))
                        false_positives=normal_pixels=0
                        for i,fp in enumerate(frames):
                            monitor.check()
                            with np.load(fp,allow_pickle=False) as ar:
                                scores=render_subspace_map(ar["residual_grid"]) if spec["family"]=="baseline" else candidate_map(ar["residual_grid"],ar["descriptor"],None,fit,spec,remove_condition=variant.endswith("_no_condition"))
                            path=dest/f"{i:04d}.npy";np.save(path,scores,allow_pickle=False);paths.append(path)
                            if labels[i]==0:
                                false_positives+=int(np.count_nonzero(scores>threshold));normal_pixels+=scores.size
                        metric=evaluate_from_files(paths,masks,labels,temp_root=out/"metric_tmp")
                        row={"category":category,"seed":seed,"variant":variant,**metric,"normal_pixel_fpr":false_positives/normal_pixels,"pixel_threshold":threshold};rows.append(row);write(cat/(variant+".json"),row)
                write(out/"development_metrics.json",rows)
                write(out/"run_state.json",{"status":"running_synthetic_development","category":category,"seed":seed,"metric_rows":len(rows),"pid":os.getpid(),"updated_at":now()})
                print(f"development {category} seed={seed}: {len(active)} candidate configurations",flush=True)
            if seed_index==0:
                stops=[]
                for family in cfg["families"]:
                    eligible=[s for s in specs if s["family"]==family]
                    winner=sorted(eligible,key=lambda s:(-np.mean([r["aupro_0_30"] for r in rows if r["variant"]==s["name"]]),s["name"]))[0]
                    idx={(r["category"],r["variant"]):r for r in rows}
                    delta=np.array([idx[c,winner["name"]]["aupro_0_30"]-idx[c,"baseline"]["aupro_0_30"] for c in categories])*100
                    if delta.mean()<=-cfg["screen_stopping"]["macro_aupro_loss_pp"] or np.count_nonzero(delta<-cfg["screen_stopping"]["category_aupro_loss_pp"])>cfg["screen_stopping"]["maximum_such_regressions"]:
                        stops.append({"family":family,"best_screen_spec":winner,"mean_delta_pp":float(delta.mean()),"reason":"registered negative-development stopping rule"})
                    else:winners.append(winner)
                write(out/"screen_winners.json",{"winners":winners,"stopped_families":stops,"selected_at":now(),"before_remaining_seed_confirmation":True})
        lookup={(r["category"],r["seed"],r["variant"]):r for r in rows};decisions=[]
        for spec in winners:
            gains=[np.mean([lookup[c,s,spec["name"]]["aupro_0_30"]-lookup[c,s,"baseline"]["aupro_0_30"] for s in cfg["seed_order"]]) for c in categories]
            component=[np.mean([lookup[c,s,spec["name"]]["aupro_0_30"]-lookup[c,s,spec["name"]+"_no_condition"]["aupro_0_30"] for s in cfg["seed_order"]]) for c in categories]
            gain,contribution=paired_summary(gains),paired_summary(component)
            fpr_delta=float(np.mean([lookup[c,s,spec["name"]]["normal_pixel_fpr"]-lookup[c,s,"baseline"]["normal_pixel_fpr"] for c in categories for s in cfg["seed_order"]])*100)
            nonnegative=int(np.count_nonzero(np.array(gains)>=0.))
            passed=gain["mean_delta_pp"]>=cfg["advancement"]["mean_aupro_gain_pp"] and nonnegative>=cfg["advancement"]["nonnegative_categories_minimum"] and contribution["bootstrap_95_ci_pp"][0]>cfg["advancement"]["component_ci_lower_minimum"] and fpr_delta<=cfg["advancement"]["normal_fpr_regression_pp_maximum"]
            decisions.append({"spec":spec,"gain":gain,"content_contribution":contribution,"normal_fpr_delta_pp":fpr_delta,"nonnegative_categories":nonnegative,"supported_on_synthetic_dev":bool(passed)})
        eligible=[d for d in decisions if d["supported_on_synthetic_dev"]]
        selected=max(eligible,key=lambda d:d["gain"]["mean_delta_pp"])["spec"] if eligible else None
        write(out/"selection.json",{"selected":selected,"decisions":decisions,"frozen_at":now(),"selection_used_real_test_metrics":False,"real_defect_effectiveness_proven":False,"q3_goal_achieved":False})
        write(out/"run_state.json",{"status":"completed","selected":selected,"metric_rows":len(rows),"elapsed_seconds":time.perf_counter()-start,"finished_at":now()})
    except BaseException as e:
        write(out/"failure.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
    finally:write(out/"resources.json",monitor.close())


if __name__=="__main__":main()
