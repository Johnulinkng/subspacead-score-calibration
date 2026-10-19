"""Audit the full registered denominator before cross-method statistics."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,sha,now
from analyze_context_contribution import paired_summary


def main():
    run=ROOT/"runs/research-q3-20261016";out=run/"final-analysis";out.mkdir(exist_ok=True)
    active=read(run/"active_baseline_queue.json") if (run/"active_baseline_queue.json").exists() else None
    queue_path=Path(active["queue"]) if active else run/"baseline-queue-v2.json"
    baseline_state_path=Path(active["state"]) if active else run/"baseline_queue_state-v2.json"
    if active and sha(queue_path)!=active["queue_sha256"]:raise ValueError("Active comparator queue drift")
    queue=read(queue_path);ind=run/"independent-partitions";chosen=run/"selected-candidate"
    while True:
        failures=[str(p) for p in (ind/"failure.json",run/"trainonly-development/failure.json",chosen/"failure.json") if p.exists()]
        if failures:
            write(out/"run_state.json",{"status":"failed_prerequisite_reported","failures":failures,"q3_goal_achieved":False,"at":now()});return
        i=read(ind/"run_state.json") if (ind/"run_state.json").exists() else {}
        b=read(baseline_state_path) if baseline_state_path.exists() else {}
        c=read(chosen/"run_state.json") if (chosen/"run_state.json").exists() else {}
        ready=(i.get("status")=="completed" and b.get("status") in ("completed","completed_with_failures") and c.get("status") in ("completed","no_supported_development_candidate"))
        if ready:break
        write(out/"run_state.json",{"status":"waiting_for_full_registered_matrix","pid":os.getpid(),"independent_fits":i.get("status"),"baseline_queue":b.get("status"),"selected_candidate":c.get("status"),"updated_at":now()})
        time.sleep(20)
    import numpy as np
    metrics=("image_auroc","pixel_auroc","pixel_ap","aupro_0_30");rows=[];missing=[];evidence={}
    for job in queue["training_jobs"]:
        base=Path(job["receipt_dir"]).parent;receipt=base/"execution/receipt.json";inference=base/"inference";state=inference/"run_state.json"
        if not receipt.exists() or not state.exists() or read(receipt).get("status")!="completed" or read(state).get("status")!="completed":missing.append(job["id"]);continue
        st=read(state);path=inference/"metrics.json";payload=read(path)
        if st.get("execution_scope")!="full" or st.get("metrics_sha256")!=sha(path):raise ValueError("Unverified or incomplete baseline result")
        if payload["protocol"]!="industrial_anomaly_raw_map_max_256_v2":raise ValueError("Mixed evaluation protocol")
        rows.append({"dataset":job["dataset"],"category":job["category"],"seed":job["seed"],"method":job["method"],**{m:payload["metrics"][m] for m in metrics}})
        evidence[str(path)]=sha(path)
    for folder,prefix in ((ind,"Subspace"),(chosen,"Selected")):
        path=folder/"category_metrics.csv"
        if not path.exists():continue
        with path.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):rows.append({"dataset":row["dataset"],"category":row["category"],"seed":int(row["seed"]),"method":prefix+"/"+row["variant"],**{m:float(row[m]) for m in metrics}})
        evidence[str(path)]=sha(path)
    with (out/"category_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    seeds=queue["seeds"];macros=[];lookup={(r["dataset"],r["category"],r["seed"],r["method"]):r for r in rows}
    methods=sorted({r["method"] for r in rows})
    tasks=read(ind/"frozen_protocol.json")["tasks"]
    for d,expected in (("mvtec_ad",15),("visa",12)):
        cats=[c for dd,c in tasks if dd==d]
        for method in methods:
            required=[(d,c,s,method) for c in cats for s in seeds]
            count=sum(key in lookup for key in required)
            record={"dataset":d,"method":method,"completed_category_seed_rows":count,"expected_category_seed_rows":expected*3,"complete":count==expected*3}
            if record["complete"]:
                per_seed={s:{m:float(np.mean([lookup[d,c,s,method][m] for c in cats])) for m in metrics} for s in seeds}
                record.update(seed_macro=per_seed,mean={m:float(np.mean([v[m] for v in per_seed.values()])) for m in metrics},std={m:float(np.std([v[m] for v in per_seed.values()],ddof=1)) for m in metrics})
            macros.append(record)
    candidate="Selected/full" if c.get("status")=="completed" else "Subspace/conditional_tail"
    contrasts=[]
    for d in ("mvtec_ad","visa"):
        cats=[cc for dd,cc in tasks if dd==d]
        for reference in ("DRAEM","PatchCore","RD4AD","Subspace/baseline","Subspace/global_tail"):
            required=[(d,cc,s,method) for cc in cats for s in seeds for method in (candidate,reference)]
            if not all(key in lookup for key in required):continue
            for metric in metrics:
                diffs=[np.mean([lookup[d,cc,s,candidate][metric]-lookup[d,cc,s,reference][metric] for s in seeds]) for cc in cats]
                summary=paired_summary(diffs);summary["nonnegative_categories"]=int(np.count_nonzero(np.asarray(diffs)>=0))
                contrasts.append({"dataset":d,"candidate":candidate,"reference":reference,"metric":metric,**summary})
    primary=[r for r in contrasts if r["reference"]=="DRAEM" and r["metric"]=="aupro_0_30"]
    secondary=[r for r in contrasts if r["reference"]=="DRAEM" and r["metric"]!="aupro_0_30"]
    numerical=(len(primary)==2 and all(r["mean_delta_pp"]>0 and r["nonnegative_categories"]/r["categories"]>=.6 for r in primary)
        and np.mean([r["mean_delta_pp"] for r in primary])>=1.
        and any(r["bootstrap_95_ci_pp"][0]>0 for r in primary)
        and len(secondary)==6 and all(r["mean_delta_pp"]>=-.5 for r in secondary))
    component_source=chosen/"paired_statistics.json" if candidate.startswith("Selected") else ind/"paired_statistics.json"
    component_rows=read(component_source)
    component=[r for r in component_rows if r["metric"]=="aupro_0_30" and r["reference"] in ("without_conditioning","global_tail")]
    independent=len(component)==2 and all(r["mean_delta_pp"]>0 for r in component) and any(r["bootstrap_95_ci_pp"][0]>0 for r in component)
    write(out/"dataset_macro_metrics.json",macros);write(out/"paired_statistics.json",contrasts)
    write(out/"scientific_acceptance.json",{"candidate":candidate,"full_denominator":243,"completed_baseline_runs":243-len(missing),"missing":missing,"original_numerical_gates_passed":bool(numerical and not missing),"independent_content_contribution_supported":bool(independent),"original_blind_F0_round_eligible":False,"historical_test_access":True,"source_files":evidence,"strong_controls_included":not missing,"novelty_review_required":True,"q3_goal_achieved":False,"scope":"Complete retrospective experimental comparison when denominator is complete. Numerical success does not establish scientific originality or journal acceptance.","timing_notice":"Training occurred alongside other registered work; current end-to-end wall times are not isolated speed comparisons.","information_notice":"Different pretrained backbones and normal-data allocations are disclosed; total-system comparisons do not isolate the new component. Only same-backbone ablations support component attribution.","geometry_notice":"PatchCore retains author resize/center-crop geometry; other methods use full-frame geometry. Common metric definitions and 256 plane do not make their physical fields of view identical."})
    write(out/"run_state.json",{"status":"completed_analysis" if not missing else "incomplete_matrix_reported","finished_at":now(),"q3_goal_achieved":False})


if __name__=="__main__":main()
