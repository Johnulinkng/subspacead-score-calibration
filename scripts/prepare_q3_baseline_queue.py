"""Freeze a clean, separate source snapshot and full-budget comparator queue."""
from __future__ import annotations
import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,now,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--version",choices=("v1","v2"),default="v2");args=parser.parse_args()
    version=args.version;suffix="" if version=="v1" else "-v2"
    dest=ROOT/("runs/research-q3-20261016/baseline-source"+suffix)
    if dest.exists():raise ValueError("Fresh source snapshot required")
    dest.mkdir(parents=True)
    hashes={}
    for directory in ("src","scripts","configs","third_party","vendor"):
        for p in (ROOT/directory).rglob("*"):
            if not p.is_file() or any(x in p.parts for x in (".git","__pycache__")) or p.suffix in (".pyc",".pt",".pth",".png",".jpg",".npz",".npy"):
                continue
            q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ("pyproject.toml","uv.lock"):
        shutil.copy2(ROOT/name,dest/name);hashes[name]=sha(ROOT/name)
    (dest/".gitignore").write_text("__pycache__/\n*.pyc\n.cache/\ntmp/\nruns/\n.runtime/\n",encoding="utf-8")
    for args in (("init",),("config","user.name","Codex research snapshot"),("config","user.email","research-snapshot@localhost"),("add","."),("commit","-qm","Freeze full-budget Q3 follow-up baseline source")):
        subprocess.run(["git","-C",str(dest),*args],check=True,capture_output=True)
    commit=subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
    project=read(ROOT/"configs/project.json");seeds=read(ROOT/"configs/preregistered_experiments.json")["model_seeds"]
    run=ROOT/"runs/research-q3-20261016";queue=[];smokes=[]
    teacher=str(ROOT/"artifacts/weights/wide_resnet50_2-95faca4d.pth");teacher_sha="95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
    for method in ("RD4AD","PatchCore","DRAEM"):
        for seed in seeds:
            for d in ("mvtec_ad","visa"):
                # Fixed alphabetical order, unrelated to observed category results.
                for c in sorted(project["datasets"][d]["categories"]):
                    completed_first=method=="RD4AD" and d=="mvtec_ad" and c=="bottle" and seed==seeds[0]
                    base=run/("baselines" if version=="v1" or completed_first else "baselines-v2")/method/d/c/str(seed)
                    args=["--dataset",d,"--category",c,"--dataset-root",str(ROOT/project["paths"]["mvtec" if d=="mvtec_ad" else "visa"]),
                          "--seed",str(seed),"--device","cuda:0","--output-dir",str(base/"model"),"--run-purpose","author_reproduction"]
                    if method=="RD4AD":
                        script="train_rd4ad.py";args += ["--teacher-checkpoint",teacher,"--teacher-sha256",teacher_sha,"--batch-size","16","--epochs","200"]
                    elif method=="PatchCore":
                        script="fit_patchcore.py" if version=="v1" else "fit_patchcore_bounded.py";args += ["--backbone-checkpoint",teacher,"--backbone-sha256",teacher_sha,"--execution-scope","full","--batch-size","2","--coreset-device","cuda:0","--faiss-workers","3"]
                    else:
                        script="train_draem.py";args += ["--anomaly-source-root",str(ROOT/"data/dtd/images"),"--batch-size","8","--epochs","700","--activation-checkpointing","--no-pin-memory"]
                    job={"id":f"{method}/{d}/{c}/{seed}","method":method,"dataset":d,"category":c,"seed":seed,"kind":"full_author_reproduction_previously_accessed_benchmarks","script":script,"arguments":args,"receipt_dir":str(base/"execution")}
                    queue.append(job)
        # Two real domains must pass the same resource configuration before full runs.
        for d,c in (("mvtec_ad","bottle"),("visa","candle")):
            job=next(j for j in queue if j["method"]==method and j["dataset"]==d and j["category"]==c)
            item=json.loads(json.dumps(job));out=run/"resource-smoke"/(method.lower()+("-bounded" if method=="PatchCore" and version=="v2" else "")+"-measured-"+d)
            item["id"]=f"smoke/{method}/{d}";item["kind"]="debug_limited_resource_only";item["receipt_dir"]=str(out/"execution")
            args=item["arguments"];args[args.index("--output-dir")+1]=str(out/"model");args[args.index("--run-purpose")+1]="debug_limited"
            if "--execution-scope" in args:args[args.index("--execution-scope")+1]="debug_limited"
            args += ["--max-steps","2"];smokes.append(item)
    queue_file=run/("baseline-queue"+suffix+".json")
    write(queue_file,{"frozen_at":now(),"source_snapshot":str(dest),"source_commit":commit,"source_sha256":hashes,
        "seeds":seeds,"training_jobs":queue,"resource_smokes":smokes,"original_scientific_gates":read(ROOT/"configs/preregistered_experiments.json")["formal_claim_rules"],
        "scope":"Full author-compatible training budgets; retrospective follow-up. Original blind-F0 round is invalidated by historical Test access; no blind/generalization or Q3 acceptance claim.",
        "strict_original_round_eligible":False,"stop_policy":"Do not infer full performance from smoke, failed jobs, partial categories or one seed. No automatic batch or epoch reductions."})
    print(json.dumps({"source_snapshot_commit":commit,"training_jobs":len(queue),"resource_smokes":len(smokes),"queue":str(queue_file)}))


if __name__=="__main__":main()
