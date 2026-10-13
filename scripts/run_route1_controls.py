"""Same-feature full-data SubspaceAD controls; no gradient training or downloads."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)
def append(p,v):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')


def memory():
    class System(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    class Process(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ('peak_ws','ws','peak_pp','pp','peak_np','np','page','peak_page','private')]
    system=System();system.length=ctypes.sizeof(system)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system)):raise OSError('System memory counters unavailable')
    process=Process();process.cb=ctypes.sizeof(process)
    get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=ctypes.c_void_p
    info=ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes=[ctypes.c_void_p,ctypes.POINTER(Process),ctypes.c_ulong]
    if not info(get(),ctypes.byref(process),process.cb):raise OSError('Process memory counters unavailable')
    return dict(working_set_bytes=int(process.ws),peak_working_set_bytes=int(process.peak_ws),private_commit_bytes=int(process.private),
                free_ram_gib=system.avail_phys/1024**3,free_commit_gib=system.avail_page/1024**3)


def check_resources(protocol,initial=False):
    sample=memory();policy=protocol['resource_policy']
    minimum=policy['initial_free_ram_gib'] if initial else policy['reserve_free_ram_gib']
    commit=policy['initial_free_commit_gib'] if initial else policy['reserve_free_commit_gib']
    if sample['free_ram_gib']<minimum or sample['free_commit_gib']<commit or sample['working_set_bytes']>policy['max_working_set_gib']*1024**3:
        raise RuntimeError('Resource reserve boundary: '+json.dumps(sample))
    if shutil.disk_usage(ROOT).free<policy['minimum_disk_gib']*1024**3:raise RuntimeError('Disk reserve boundary')
    return sample


def authenticated(p,expected):
    p=Path(p).resolve(strict=True)
    if not p.is_relative_to(ROOT) or sha(p)!=expected:raise ValueError('Input identity drift: '+str(p))
    return p


def safe_remove_cache(p,run):
    p=p.resolve();run=run.resolve()
    if not p.is_relative_to(run/'token_cache') or p==run/'token_cache':raise ValueError('Unsafe cache target')
    if p.exists():shutil.rmtree(p)


def main():
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/route1_controls_20261014.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();run=args.output_dir.resolve();protocol=read(args.config)
    if not run.is_relative_to(ROOT/'runs/quick') or (run.exists() and not args.resume):raise ValueError('Fresh output or explicit resume required')
    if (protocol['device']!='cpu' or protocol['image_res']!=448 or protocol['layers']!=[-4,-5] or protocol['explained_variance']!=.99
        or protocol['output_size']!=256 or protocol['feature_dim']!=384 or protocol['eps']!=1e-6 or protocol['cpu_threads']!=4):
        raise ValueError('Frozen CPU same-backbone geometry changed')
    check_resources(protocol,initial=True)
    source=ROOT/protocol['source_run'];old=read(source/'run_config.json')
    if read(source/'run_state.json')['status']!='completed':raise ValueError('Historical source incomplete')
    tasks=[tuple(x) for x in old['expected_matrix']['categories']]
    methods=protocol['methods']
    if len(tasks)!=27 or methods!=['official_cpu_two_pass','numpy_cpu_two_pass','current_cpu_one_pass']:raise ValueError('Full matrix contract violated')
    files=['scripts/run_route1_controls.py','scripts/run_route1_resource_trial.py','scripts/run_route1_resource_matrix.py','src/industrial_anomaly_lab/subspace_controls.py',
           'src/industrial_anomaly_lab/subspace_features.py','src/industrial_anomaly_lab/quick_metrics.py','src/industrial_anomaly_lab/metrics.py']
    config={'schema_version':1,'protocol':protocol,'protocol_sha256':sha(args.config),'project_root':str(ROOT),'source_run':str(source),
            'source_sha256':{x:sha(ROOT/x) for x in files},'environment_sha256':{x:sha(ROOT/x) for x in ('pyproject.toml','uv.lock')},
            'vendor_source_sha256':old['vendor_source_sha256'],'backbone':old['backbone'],
            'expected_matrix':{'categories':tasks,'methods':methods,'variants':[m+'_'+s for m in methods for s in ('max','top1p')]},
            'input_metadata_sha256':{p.relative_to(source).as_posix():sha(p) for p in [source/'run_config.json',source/'run_state.json']+
              [source/'categories'/d/c/f for d,c in tasks for f in ('train_manifest.json','test_inventory.json','metrics.json','samples.jsonl','completion.json')]},
            'pilot':args.pilot,'created_at_utc':now(),'scientific_claim_eligible':False}
    for rel,h in config['vendor_source_sha256'].items():authenticated(ROOT/'vendor/SubspaceAD'/rel,h)
    for rel,h in protocol['weight_files_sha256'].items():authenticated(ROOT/protocol['weights_dir']/rel,h)
    completed=set();prior=0
    if args.resume:
        previous=read(run/'run_config.json');old_state=read(run/'run_state.json')
        if old_state['status']=='completed' or previous['pilot']:raise ValueError('Completed or pilot cannot resume')
        for name in ('protocol_sha256','source_sha256','environment_sha256','vendor_source_sha256','input_metadata_sha256','expected_matrix'):
            if previous[name]!=config[name]:raise ValueError('Resume drift: '+name)
        prior=float(old_state.get('elapsed_seconds',0));config=previous
        for d,c in tasks:
            cat=run/'categories'/d/c
            if (cat/'completion.json').exists():
                seal=read(cat/'completion.json')
                for rel,h in seal['sha256'].items():authenticated(run/rel,h)
                completed.add((d,c))
            elif cat.exists():
                dest=run/'execution_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')/d/c
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not cat.resolve().is_relative_to(run) or not dest.resolve().is_relative_to(run):raise ValueError('Unsafe archive paths')
                shutil.move(str(cat),str(dest))
                pred=run/'predictions'/d/c
                if pred.exists():shutil.move(str(pred),str(dest/'partial_predictions'))
                safe_remove_cache(run/'token_cache'/d/c,run)
    else:
        run.mkdir(parents=True);write(run/'run_config.json',config)
    state={'status':'running','started_at_utc':now(),'pid':os.getpid(),'completed_categories':len(completed),
           'test_content_accessed':bool(old_state.get('test_content_accessed',False)) if args.resume else False,
           'gradient_training_performed':False,'scientific_claim_eligible':False,'device':'cpu','gpu_used':False,'elapsed_seconds':prior}
    if args.resume:state['started_at_utc']=read(run/'run_state.json')['started_at_utc']
    start=time.monotonic()
    def progress(d,c,phase,images=None):
        state.update(current_task={'dataset':d,'category':c,'phase':phase,'images':images},updated_at_utc=now(),
                     elapsed_seconds=prior+time.monotonic()-start,**check_resources(protocol))
        write(run/'run_state.json',state)
    write(run/'run_state.json',state)
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from sklearn.metrics import roc_auc_score
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.subspace_features import LocalDinoExtractor,save_subspace_model,render_subspace_map,image_aggregates
        from industrial_anomaly_lab.subspace_controls import fit_official_cpu_two_pass,fit_numpy_cpu_two_pass,fit_current_cpu_one_pass,compare_subspace_models,EquivalenceTolerances
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        cv2.setNumThreads(4)
        torch.set_num_interop_threads(1)
        fitters=dict(zip(methods,(fit_official_cpu_two_pass,fit_numpy_cpu_two_pass,fit_current_cpu_one_pass)))
        extractor=LocalDinoExtractor(ROOT/protocol['weights_dir'],expected_checkpoint_sha256=protocol['checkpoint_sha256'],
                    expected_artifact_sha256=protocol['weight_files_sha256'],device='cpu',cpu_threads=4)
        with threadpool_limits(limits=4):
            for d,c in tasks[:1] if args.pilot else tasks:
                if (d,c) in completed:continue
                cat=run/'categories'/d/c;cat.mkdir(parents=True)
                old_cat=source/'categories'/d/c
                train=read(old_cat/'train_manifest.json');inventory=read(old_cat/'test_inventory.json')
                if args.pilot:train=train[:16]
                write(cat/'train_manifest.json',train);write(cat/'test_inventory.json',inventory)
                cache=run/'token_cache'/d/c;cache.mkdir(parents=True)
                token_records=[];extraction_start=time.monotonic()
                progress(d,c,'shared_normal_features',0)
                for i,record in enumerate(train):
                    image=authenticated(ROOT/record['path'],record['sha256'])
                    tokens=extractor(image).reshape(1024,384)
                    path=cache/f'{i:06d}.npy';np.save(path,tokens,allow_pickle=False)
                    token_records.append({'index':i,'source_path':record['path'],'source_sha256':record['sha256'],
                                          'sha256':sha(path),'size_bytes':path.stat().st_size})
                    if (i+1)%25==0 or i+1==len(train):progress(d,c,'shared_normal_features',i+1)
                extraction_seconds=time.monotonic()-extraction_start
                write(cat/'token_cache_manifest.json',{'shape_per_image':[1024,384],'dtype':'float32','temporary_cache_deleted_after_completion':True,'files':token_records})
                def batches():
                    for i in range(len(train)):yield np.load(cache/f'{i:06d}.npy',allow_pickle=False)
                fits={}
                for method in methods:
                    progress(d,c,'fit_'+method)
                    fits[method]=fitters[method](batches,feature_dim=384,total_tokens=len(train)*1024,num_batches=len(train),
                                                explained_variance=.99,eps=1e-6)
                    (cat/'models').mkdir(exist_ok=True)
                    save_subspace_model(fits[method].model,cat/'models'/f'{method}.npz',fits[method].metadata())
                if len({f.input_sha256 for f in fits.values()})!=1:raise ValueError('Fit controls did not consume identical token bytes')
                def queries():
                    for i in range(min(4,len(train))):yield np.load(cache/f'{i:06d}.npy',allow_pickle=False)
                diagnostics={'models':{m:f.metadata() for m,f in fits.items()},'shared_normal_extraction_seconds':extraction_seconds,
                             'comparisons':{m:compare_subspace_models(fits[methods[0]],fits[m],query_batches=queries,
                                              tolerances=EquivalenceTolerances(**protocol['model_tolerances'])) for m in methods[1:]}}
                write(cat/'diagnostics.json',diagnostics)
                if any(not v['passed'] for v in diagnostics['comparisons'].values()):raise ValueError('PCA equivalence tolerance exceeded')
                if args.pilot:
                    state.update(status='pilot_completed',finished_at_utc=now(),elapsed_seconds=time.monotonic()-start,**memory())
                    write(run/'run_state.json',state);safe_remove_cache(cache,run);print(json.dumps(state),flush=True);return 0
                old_samples={v['sample_id']:v for v in lines(old_cat/'samples.jsonl')}
                pred=run/'predictions'/d/c
                for method in methods+['masks']:(pred/method).mkdir(parents=True)
                labels=[];maps={m:[] for m in methods};masks=[];scores={m:[] for m in methods}
                errors={m:{'maximum_absolute_error':0.,'mean_absolute_error_sum':0.} for m in methods[1:]}
                history_error=0.
                progress(d,c,'shared_test_features',0);state['test_content_accessed']=True
                for i,record in enumerate(inventory):
                    image=authenticated(record['image_path'],record['image_sha256'])
                    tokens=extractor(image).reshape(1024,384)
                    raws={m:render_subspace_map(fits[m].model.reconstruction_scores(tokens).reshape(32,32)) for m in methods}
                    reference=raws[methods[0]]
                    for method in methods[1:]:
                        delta=np.abs(raws[method].astype(np.float64)-reference.astype(np.float64))
                        errors[method]['maximum_absolute_error']=max(errors[method]['maximum_absolute_error'],float(delta.max()))
                        errors[method]['mean_absolute_error_sum']+=float(delta.mean())
                        if not np.allclose(raws[method],reference,atol=protocol['map_atol'],rtol=protocol['map_rtol']):raise ValueError('Paired map tolerance exceeded')
                    old_sample=old_samples[record['sample_id']]
                    old_raw=np.load(authenticated(source/old_sample['raw_maps']['subspace'],old_sample['raw_map_sha256']['subspace']),allow_pickle=False)
                    history_error=max(history_error,float(np.max(np.abs(raws[methods[-1]].astype(np.float64)-old_raw.astype(np.float64)))))
                    mask=np.load(authenticated(source/old_sample['mask_path'],old_sample['mask_sha256']),allow_pickle=False)
                    if record['mask_path'] is not None:
                        mask_source=authenticated(record['mask_path'],record['mask_sha256'])
                        with Image.open(mask_source) as handle:actual=np.asarray(handle.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
                        if not np.array_equal(mask,actual):raise ValueError('Historical GT differs from authenticated original')
                    elif mask.any():raise ValueError('Normal GT not empty')
                    mask_path=pred/'masks'/f'{i:06d}.npy';np.save(mask_path,mask,allow_pickle=False);masks.append(mask_path);labels.append(record['label'])
                    paths={};hashes={};aggregates={}
                    for method in methods:
                        path=pred/method/f'{i:06d}.npy';np.save(path,raws[method],allow_pickle=False);maps[method].append(path)
                        values=image_aggregates(raws[method]);scores[method].append((values['max'],values['top1p']))
                        paths[method]=path.relative_to(run).as_posix();hashes[method]=sha(path);aggregates[method]={'max':values['max'],'top1p':values['top1p']}
                    append(cat/'samples.jsonl',{'dataset':d,'category':c,'seed':protocol['seed'],'sample_id':record['sample_id'],'label':record['label'],
                           'image_path':str(image),'image_sha256':record['image_sha256'],'raw_maps':paths,'raw_map_sha256':hashes,'image_scores':aggregates,
                           'mask_path':mask_path.relative_to(run).as_posix(),'mask_sha256':sha(mask_path),'selected':old_sample['selected']})
                    if (i+1)%25==0 or i+1==len(inventory):progress(d,c,'shared_test_features',i+1)
                rows=[]
                for method in methods:
                    progress(d,c,'metrics_'+method)
                    measured=evaluate_from_files(maps[method],masks,labels,temp_root=run/'metric_tmp')
                    for index,aggregation in enumerate(('max','top1p')):
                        rows.append({'dataset':d,'category':c,'variant':method+'_'+aggregation,'method':method+'_'+aggregation,
                                     'seed':protocol['seed'],'status':'completed','train_images':len(train),'test_images':len(inventory),
                                     'pca_rank':fits[method].model.k,**measured,'image_auroc':float(roc_auc_score(labels,np.array(scores[method])[:,index]))})
                reference_rows={r['variant'].rsplit('_',1)[1]:r for r in rows if r['variant'].startswith(methods[0]+'_')}
                metric_errors={}
                for row in rows:
                    reference_row=reference_rows[row['variant'].rsplit('_',1)[1]]
                    err={key:abs(row[key]-reference_row[key]) for key in ('image_auroc','pixel_auroc','pixel_ap','aupro_0_30')}
                    metric_errors[row['variant']]=err
                    if max(err.values())>protocol['metric_atol']:raise ValueError('Paired metric tolerance exceeded')
                for value in errors.values():value['mean_absolute_error']=value.pop('mean_absolute_error_sum')/len(inventory)
                diagnostics.update(raw_map_comparisons=errors,metric_absolute_errors=metric_errors,
                                   historical_gpu_map_maximum_absolute_error=history_error,historical_map_comparison_scope='diagnostic only; previous phase GPU versus current phase CPU features')
                write(cat/'diagnostics.json',diagnostics);write(cat/'metrics.json',rows)
                write(cat/'metadata.json',{'dataset':d,'category':c,'train_images':len(train),'test_images':len(inventory),
                      'device':'cpu','shared_feature_extraction':True,'normal_extraction_seconds':extraction_seconds,'models':{m:f.metadata() for m,f in fits.items()}})
                safe_remove_cache(cache,run)
                files=[p for directory in (cat,pred) for p in directory.rglob('*') if p.is_file()]
                write(cat/'completion.json',{'status':'completed','completed_at_utc':now(),'sha256':{p.relative_to(run).as_posix():sha(p) for p in files}})
                completed.add((d,c));state['completed_categories']=len(completed);progress(d,c,'completed')
                print(json.dumps({'completed_categories':len(completed),'dataset':d,'category':c,'model_comparison_passed':True,
                      'maximum_metric_error':max(e for v in metric_errors.values() for e in v.values()),'rows':len(rows)}),flush=True)
                del fits;gc.collect()
        result=run/'results';result.mkdir(exist_ok=True)
        for name in ('category_metrics','samples'):
            with (result/(name+'.jsonl')).open('w',encoding='utf-8') as f:
                for d,c in tasks:
                    records=read(run/'categories'/d/c/'metrics.json') if name=='category_metrics' else lines(run/'categories'/d/c/'samples.jsonl')
                    for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        counts=(len(lines(result/'category_metrics.jsonl')),len(lines(result/'samples.jsonl')))
        if counts!=(162,3887):raise ValueError('Final full matrix count differs')
        state.update(status='completed',finished_at_utc=now(),current_task=None,completed_categories=27,metric_rows=162,training_images=12288,test_images=3887,
                     elapsed_seconds=prior+time.monotonic()-start,**memory())
        write(run/'run_state.json',state);print(json.dumps(state),flush=True);return 0
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',updated_at_utc=now(),elapsed_seconds=prior+time.monotonic()-start,**memory())
        write(run/'run_state.json',state);raise


if __name__=='__main__':raise SystemExit(main())
