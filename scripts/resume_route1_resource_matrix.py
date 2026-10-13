"""Resume v1 resource orchestration using its explicit resource_violation field.

The original scheduler classified errors by message substrings and omitted
'RAM'. Successful measurements and failed receipt bytes remain unchanged.
This adapter uses the already-recorded resource_violation field solely to
classify retryable resource failures, keeping the worker and plan frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/route1_controls_20261014.json')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'runs/quick'):raise ValueError('Project output required')
    frozen=json.loads((out/'matrix_config.json').read_text(encoding='utf-8'))
    source=ROOT/'scripts/run_route1_resource_matrix.py'
    if digest(source)!=frozen['matrix_source_sha256'] or digest(args.config)!=frozen['protocol_sha256']:
        raise ValueError('Original matrix/protocol drift')
    if digest(ROOT/'scripts/run_route1_resource_trial.py')!=frozen['worker_sha256']:raise ValueError('Worker drift')
    full=json.loads((ROOT/'runs/quick/route1-controls-20261014/run_config.json').read_text(encoding='utf-8'))
    for relative,expected in full['source_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Frozen execution dependency drift: '+relative)
    spec=importlib.util.spec_from_file_location('frozen_route1_matrix',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_read=module.read
    def classified_read(path):
        value=original_read(path)
        if isinstance(value,dict) and value.get('status')=='failed' and value.get('resource_violation'):
            value={**value,'error':'ResourceViolation: '+str(value.get('error',value['resource_violation']))}
        return value
    module.read=classified_read
    context=out/'resume_context'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')/'resume_entry.json'
    context.parent.mkdir(parents=True,exist_ok=True)
    context.with_name('previous_matrix_state.json').write_bytes((out/'matrix_state.json').read_bytes())
    context.write_text(json.dumps({'adapter_sha256':digest(Path(__file__)),'original_matrix_source_sha256':digest(source),
         'worker_sha256':frozen['worker_sha256'],'repair':'classify retryable failures from existing resource_violation field; in-memory only',
         'measurement_or_receipt_bytes_changed':False,'plan_changed':False},indent=2)+'\n',encoding='utf-8')
    sys.argv=[str(source),'--config',str(args.config.resolve()),'--output-dir',str(out),'--resume']
    return module.main()


if __name__=='__main__':raise SystemExit(main())
