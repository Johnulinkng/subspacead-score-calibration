"""Run 36 registered fresh CPU resource trials, with bounded concurrent work."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_route1_controls import memory, read, write, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/route1_controls_20261014.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve();cfg=read(args.config)
    if not out.is_relative_to(ROOT/'runs/quick') or (out.exists() and not args.resume):raise ValueError('Fresh/resumed project output required')
    out.mkdir(parents=True,exist_ok=args.resume)
    plan=[]
    for repeat in range(cfg['resource_repeats']):
        modes=list(cfg['resource_modes']);random.Random(cfg['resource_order_seed']+repeat).shuffle(modes)
        for dataset,category in cfg['resource_categories']:
            plan.extend({'dataset':dataset,'category':category,'mode':mode,'repeat':repeat} for mode in modes)
    frozen={'protocol_sha256':sha(args.config),'worker_sha256':sha(ROOT/'scripts/run_route1_resource_trial.py'),'matrix_source_sha256':sha(Path(__file__)),
            'plan':plan,'cpu_threads':4,'gpu_used':False,'maximum_simultaneous_resource_workers':1}
    if args.resume:
        if read(out/'matrix_config.json')!=frozen:raise ValueError('Matrix resume identity changed')
    else:write(out/'matrix_config.json',frozen)
    completed=[];started=time.monotonic();child=None
    state={'status':'running','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'completed_trials':0,'expected_trials':36,'gpu_used':False}
    write(out/'matrix_state.json',state)
    try:
        for index,trial in enumerate(plan):
            trial_id=f"{index:03d}-{trial['dataset']}-{trial['category']}-{trial['mode']}-r{trial['repeat']}"
            receipt=out/'receipts'/(trial_id+'.json');receipt.parent.mkdir(exist_ok=True)
            if receipt.exists() and read(receipt).get('status')=='completed':
                completed.append(receipt);continue
            attempt=0
            while True:
                counters=memory()
                if counters['free_ram_gib']<cfg['resource_policy']['initial_free_ram_gib'] or counters['free_commit_gib']<cfg['resource_policy']['initial_free_commit_gib']:
                    state.update(status='waiting_resource',current_trial=trial,completed_trials=len(completed),**counters)
                    write(out/'matrix_state.json',state)
                    time.sleep(10);continue
                if receipt.exists():
                    old=read(receipt)
                    history=out/'failed_attempts'/(trial_id+f'-a{attempt:02d}.json');history.parent.mkdir(exist_ok=True)
                    if history.exists():attempt+=1;continue
                    os.replace(receipt,history);attempt+=1
                    if not ('Resource' in str(old.get('error')) or 'resource' in str(old.get('error')) or 'memory' in str(old.get('error')).lower()):
                        raise RuntimeError('Non-resource trial failure retained: '+str(history))
                log=out/'logs'/(trial_id+f'-a{attempt:02d}.log');log.parent.mkdir(exist_ok=True)
                command=[str(ROOT/'.venv/Scripts/python.exe'),'-B',str(ROOT/'scripts/run_route1_resource_trial.py'),
                         '--config',str(args.config.resolve()),'--output',str(receipt),'--dataset',trial['dataset'],'--category',trial['category'],
                         '--mode',trial['mode'],'--repeat',str(trial['repeat'])]
                env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
                with log.open('wb') as output:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=0x08000000 if os.name=='nt' else 0)
                    state.update(status='running',current_trial=trial,child_pid=child.pid,completed_trials=len(completed));write(out/'matrix_state.json',state)
                    while child.poll() is None:
                        time.sleep(2)
                        if sha(args.config)!=frozen['protocol_sha256'] or sha(ROOT/'scripts/run_route1_resource_trial.py')!=frozen['worker_sha256']:
                            child.terminate();child.wait();raise RuntimeError('Frozen protocol/worker drift')
                code=child.returncode;child=None
                if not receipt.exists():raise RuntimeError('Resource worker failed without receipt; inspect '+str(log))
                result=read(receipt)
                if code==0 and result.get('status')=='completed':
                    completed.append(receipt)
                    print(json.dumps({'completed_trials':len(completed),'expected_trials':36,**trial,'fit_seconds':result['fit_seconds']}),flush=True)
                    break
                time.sleep(10)
        rows=[]
        for path in completed:
            row=read(path);row.update(measurement_scope='owned_fresh_process',receipt_path=path.relative_to(out).as_posix(),receipt_sha256=sha(path))
            rows.append(row)
        with (out/'resource_trials.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if len(rows)!=36:raise ValueError('Resource matrix incomplete')
        state.update(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat(),completed_trials=36,current_trial=None,
                     elapsed_seconds=time.monotonic()-started,failed_attempt_count=len(list((out/'failed_attempts').glob('*.json'))))
        write(out/'matrix_state.json',state);print(json.dumps(state),flush=True);return 0
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',completed_trials=len(completed),elapsed_seconds=time.monotonic()-started)
        write(out/'matrix_state.json',state);raise
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait()


if __name__=='__main__':raise SystemExit(main())
