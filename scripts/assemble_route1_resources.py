"""Bind full-category cache-fit timings to completed independent resource trials."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_route1_controls import read, lines, sha, write


def main():
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--resource-dir',type=Path,required=True)
    args=ap.parse_args();run=args.run.resolve();resource=args.resource_dir.resolve()
    if not run.is_relative_to(ROOT/'runs/quick') or not resource.is_relative_to(ROOT/'runs/quick'):raise ValueError('Project outputs only')
    state=read(run/'run_state.json');matrix=read(resource/'matrix_state.json')
    if state['status']!='completed' or matrix['status']!='completed':raise ValueError('Both full precision and resource matrix must be completed')
    config=read(run/'run_config.json');tasks=config['expected_matrix']['categories'];methods=config['expected_matrix']['methods']
    target=resource/'resource_trials.jsonl';original=lines(target)
    if len(original)!=36 or any(r['experiment']=='fit_only' for r in original):raise ValueError('Expected untouched 36-trial source matrix')
    rows=list(original)
    for dataset,category in tasks:
        directory=run/'categories'/dataset/category;diagnostics=directory/'diagnostics.json'
        recorded=read(diagnostics)['models'];train=read(directory/'train_manifest.json')
        for method in methods:
            fit=recorded[method]
            rows.append({'trial_id':f'full-{dataset}-{category}-{method}','experiment':'fit_only','method':method,
                'dataset':dataset,'category':category,'repeat':0,'status':'completed','device':'cpu','gpu_used':False,
                'train_images':len(train),'elapsed_seconds':fit['elapsed_seconds'],'fit_seconds':fit['elapsed_seconds'],
                'input_sha256':fit['input_sha256'],'batch_partition_sha256':fit['batch_partition_sha256'],'pass_count':fit['pass_count'],
                'protocol_sha256':config['protocol_sha256'],'diagnostics_path':diagnostics.relative_to(run).as_posix(),
                'diagnostics_sha256':sha(diagnostics),'measurement_scope':'shared_process_not_per_method',
                'timing_scope':fit['timing_scope'],'timing_includes_content_digest':True,
                'timing_excludes_feature_extraction':True,'process_peak_working_set_bytes':None,'process_peak_commit_bytes':None,
                'cuda_peak_allocated_bytes':0,'cuda_peak_reserved_bytes':0})
    if len(rows)!=117:raise ValueError('Full timing coverage missing')
    backup=resource/'resource_trials_independent_only.jsonl'
    if backup.exists():raise ValueError('Assembly already performed')
    backup.write_bytes(target.read_bytes())
    with target.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(resource/'assembly_receipt.json',{'status':'completed','full_precision_run':str(run),'full_precision_config_sha256':sha(run/'run_config.json'),
        'independent_trials':36,'full_category_fit_timings':81,'total_rows':117,'original_independent_matrix_sha256':sha(backup),
        'combined_resource_trials_sha256':sha(target),'prior_resource_failed_attempt_count':len(list((resource/'failed_attempts').glob('*.json'))),
        'shared_fit_memory_comparisons_available':False,'normal_subset_resource_results_not_extrapolated_to_full_datasets':True})
    print(json.dumps({'status':'completed','rows':117,'full_fit_timings':81}))
    return 0


if __name__=='__main__':raise SystemExit(main())
