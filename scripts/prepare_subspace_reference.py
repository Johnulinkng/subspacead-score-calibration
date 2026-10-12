"""Fetch or seal the pinned public SubspaceAD source and local DINOv2-S weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
COMMIT = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
EXPECTED = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def download(url,path):
    if path.exists():
        return
    temporary=path.with_suffix(path.suffix+".part")
    with urllib.request.urlopen(url,timeout=60) as response, temporary.open("wb") as stream:
        while block:=response.read(1024*1024):
            stream.write(block)
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    vendor=ROOT/"vendor/SubspaceAD"
    weights=ROOT/"artifacts/weights/dinov2-small-ed25f3a"
    if args.download:
        if not vendor.exists():
            subprocess.run(["git","clone","--no-checkout","https://github.com/CLendering/SubspaceAD.git",str(vendor)],check=True)
            subprocess.run(["git","-C",str(vendor),"checkout","--detach",COMMIT],check=True)
        weights.mkdir(parents=True,exist_ok=True)
        for name in ("config.json","preprocessor_config.json","README.md","model.safetensors"):
            download(f"https://huggingface.co/facebook/dinov2-small/resolve/{REVISION}/{name}",weights/name)
        download("https://raw.githubusercontent.com/facebookresearch/dinov2/main/LICENSE",weights/"LICENSE")
    if subprocess.check_output(["git","-C",str(vendor),"rev-parse","HEAD"],text=True).strip()!=COMMIT:
        raise ValueError("Official reference must remain at the pinned commit")
    if subprocess.check_output(["git","-C",str(vendor),"status","--porcelain"],text=True).strip():
        raise ValueError("Official reference tree changed")
    if digest(weights/"model.safetensors")!=EXPECTED or (weights/"model.safetensors").stat().st_size!=88249960:
        raise ValueError("DINO checkpoint differs from published HF LFS SHA256/size")
    files=[p for p in weights.iterdir() if p.is_file() and p.name!="WEIGHTS-MANIFEST.json"]
    manifest={"schema_version":1,"status":"verified","repo":"facebook/dinov2-small","revision":REVISION,
        "source":"https://huggingface.co/facebook/dinov2-small/tree/"+REVISION,"license":"Apache-2.0",
        "published_checkpoint_sha256":EXPECTED,"tensor_only_safetensors":True,
        "files":[{"path":p.name,"size_bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(files)]}
    (weights/"WEIGHTS-MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    refs=[vendor/p for p in ("LICENSE","README.md","main.py","scripts/backbone_ablation.sh","scripts/benchmark_full_shot.sh")]
    refs+=sorted((vendor/"src").rglob("*.py"))
    provenance={"schema_version":1,"status":"verified","repo":"https://github.com/CLendering/SubspaceAD",
        "commit":COMMIT,"license":"Apache-2.0","checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_files_sha256":{p.relative_to(vendor).as_posix():digest(p) for p in refs},"weight_manifest":manifest}
    (ROOT/"docs/subspace_reference.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"verified","vendor_commit":COMMIT,"weight_revision":REVISION,"checkpoint_sha256":EXPECTED}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
