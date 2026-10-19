"""Fixed-sample and mechanically selected failure plots from saved raw maps."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,lines,write,sha,now


def main():
    run=ROOT/"runs/research-q3-20261016";out=run/"prediction-comparisons";out.mkdir(exist_ok=True)
    selected=run/"selected-candidate";ind=run/"independent-partitions"
    protocol={"fixed_samples":"reuse selected=true from the pre-existing sealed sample inventory","failure_selection":"one anomalous image per category with smallest candidate-minus-baseline per-image pixel AP; ties by sample_id ascending","seed":1887537248,"heatmap_limit":"shared baseline normal-statistics Q99.5; never per-image min-max","failure_pool":"all anomalous test images in every category","frozen_at":now(),"script_sha256":sha(Path(__file__))}
    write(out/"visualization_protocol.json",protocol)
    while True:
        if (selected/"failure.json").exists() or (ind/"failure.json").exists():raise RuntimeError("Missing validated predictions")
        status=read(selected/"run_state.json").get("status") if (selected/"run_state.json").exists() else None
        if status=="completed":source=selected;candidate="full";scope="selection rules frozen before selected-candidate real inference";break
        if status=="no_supported_development_candidate" and read(ind/"run_state.json").get("status")=="completed":source=ind;candidate="conditional_tail";scope="post-hoc diagnostic plots of a rejected original candidate";break
        write(out/"run_state.json",{"status":"waiting_for_saved_predictions","updated_at":now()});time.sleep(10)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.metrics import average_precision_score
    from industrial_anomaly_lab.subspace_features import load_subspace_model,render_subspace_map
    tasks=read(ind/"frozen_protocol.json")["tasks"];evidence=[];failure_pool=[];seed=1887537248
    for d,c in tasks:
        samples=lines(ROOT/"runs/quick/route1-controls-20261014/categories"/d/c/"samples.jsonl")
        original=ind/"categories"/d/c/str(seed);current=source/"categories"/d/c/str(seed)
        model=load_subspace_model(original/"pca.npz");parts=read(original/"partitions.json")
        fc=run/"features/categories"/d/c
        normal_maps=[render_subspace_map(model.reconstruction_scores(np.load(fc/t["features"],allow_pickle=False)).reshape(32,32)) for t in parts["statistics"]]
        vmax=max(float(np.quantile(np.stack(normal_maps),.995)),1e-12)
        fixed=[i for i,s in enumerate(samples) if s["selected"]]
        candidates=[]
        for i,s in enumerate(samples):
            if not s["label"]:continue
            a=np.load(current/"baseline"/f"{i:06d}.npy",allow_pickle=False);b=np.load(current/candidate/f"{i:06d}.npy",allow_pickle=False)
            mask=np.load(ROOT/"runs/quick/route1-controls-20261014"/s["mask_path"],allow_pickle=False)
            ap0=average_precision_score(mask.ravel(),a.ravel());ap1=average_precision_score(mask.ravel(),b.ravel())
            item={"dataset":d,"category":c,"sample_id":s["sample_id"],"baseline_ap":float(ap0),"candidate_ap":float(ap1),"delta_ap":float(ap1-ap0),"index":i}
            failure_pool.append(item);candidates.append(item)
        worst=sorted(candidates,key=lambda x:(x["delta_ap"],x["sample_id"]))[0]["index"]
        for kind,indices in (("fixed",fixed),("failure",[worst])):
            for i in indices:
                s=samples[i];a_path=current/"baseline"/f"{i:06d}.npy";b_path=current/candidate/f"{i:06d}.npy"
                a=np.load(a_path,allow_pickle=False);b=np.load(b_path,allow_pickle=False)
                with Image.open(s["image_path"]) as im:image=np.asarray(im.convert("RGB").resize((256,256)))
                mask_path=ROOT/"runs/quick/route1-controls-20261014"/s["mask_path"];mask=np.load(mask_path,allow_pickle=False)
                fig,axs=plt.subplots(1,4,figsize=(10.5,2.8))
                for ax,data,title in zip(axs,(image,mask,a,b),("Image","GT","Baseline",candidate)):
                    if title in ("Image","GT"):ax.imshow(data,cmap="gray" if title=="GT" else None)
                    else:ax.imshow(data,cmap="inferno",vmin=0,vmax=vmax)
                    ax.set_title(title);ax.axis("off")
                fig.suptitle(f"{d}/{c} | {s['sample_id']} | {kind}",fontsize=10);fig.tight_layout()
                dest=out/f"{d}_{c}_{kind}_{i:04d}.png";fig.savefig(dest,dpi=150);plt.close(fig)
                evidence.append({"figure":dest.name,"kind":kind,"sample_id":s["sample_id"],"image_sha256":s["image_sha256"],"mask_sha256":sha(mask_path),"baseline_map_sha256":sha(a_path),"candidate_map_sha256":sha(b_path),"figure_sha256":sha(dest),"shared_vmax":vmax})
        write(out/"run_state.json",{"status":"running","category":f"{d}/{c}","figures":len(evidence),"updated_at":now()})
    with (out/"failure_candidate_pool.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(failure_pool[0]));w.writeheader();w.writerows(failure_pool)
    write(out/"figure_manifest.json",{"scope":scope,"candidate":candidate,"source":str(source),"figures":evidence,"protocol":protocol})
    write(out/"run_state.json",{"status":"completed","fixed_figures":sum(r["kind"]=="fixed" for r in evidence),"failure_figures":27,"finished_at":now()})


if __name__=="__main__":main()
