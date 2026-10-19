"""Check all generated map hashes and cross-check metrics with the dense evaluator."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,lines,write,sha,now


def main():
    import numpy as np
    from industrial_anomaly_lab.metrics import evaluate_category
    run=ROOT/"runs/research-q3-20261016/contribution-refits";started=time.perf_counter()
    frozen=read(run/"frozen_protocol.json")
    for rel,h in frozen["sha256"].items():
        if sha(ROOT/rel)!=h:raise ValueError("Frozen input drift: "+rel)
    with (run/"category_metrics.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    if len(rows)!=756 or len({(r['dataset'],r['category'],r['seed'],r['variant']) for r in rows})!=756:raise ValueError("Incomplete or duplicated metric rows")
    maps=0
    for path in sorted((run/"categories").rglob("*_map_manifest.json")):
        record=read(path)
        for rel,h in record["map_sha256"].items():
            if sha(path.parent/rel)!=h:raise ValueError("Generated raw map drift: "+rel)
            maps+=1
    if maps!=3887*25:raise ValueError("Unexpected generated map denominator")
    replay=[];max_error=0.
    inventory=ROOT/"runs/quick/route1-controls-20261014"
    for d,c in (("mvtec_ad","bottle"),("visa","candle")):
        samples=lines(inventory/"categories"/d/c/"samples.jsonl")
        masks=np.stack([np.load(inventory/s["mask_path"],allow_pickle=False) for s in samples]);labels=[s["label"] for s in samples]
        selected=[r for r in rows if r["dataset"]==d and r["category"]==c and int(r["seed"])==1887537248 and r["variant"]!="baseline"]
        for row in selected:
            folder=run/"categories"/d/c/row["seed"]/row["variant"]
            values=np.stack([np.load(folder/f"{i:06d}.npy",allow_pickle=False) for i in range(len(samples))])
            metrics=evaluate_category(values,masks,labels).to_dict()
            error=max(abs(metrics[k]-float(row[k])) for k in metrics);max_error=max(max_error,error)
            if error>1e-8:raise ValueError("Independent dense/streaming metric mismatch")
            replay.append({"dataset":d,"category":c,"seed":int(row["seed"]),"variant":row["variant"],"maximum_metric_error":error})
            print(f"independent replay {d}/{c}/{row['variant']}: {error:.3g}",flush=True)
    core=len(list((run/"categories").rglob("fit.npz")));sensitivity=len(list((run/"categories").rglob("*_fit.npz")))
    write(run/"new_audit_receipt.json",{"status":"passed","metric_rows":756,"full_map_hashes_verified":maps,"main_calibration_refits":core,"sensitivity_refits":sensitivity,"sampled_independent_dense_metric_rows":len(replay),"sampled_maximum_metric_error":max_error,"sampled_replays":replay,"scope":"All source and generated-map hashes verified; independent metric replay is sampled, not a full independent replay of 756 rows","elapsed_seconds":time.perf_counter()-started,"finished_at":now()})
    print(f"verified maps={maps}, main fits={core}, sensitivity fits={sensitivity}, sampled metric rows={len(replay)}",flush=True)


if __name__=="__main__":main()
