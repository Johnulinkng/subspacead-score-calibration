"""Resume the frozen v1 runner after canonicalizing its JSON matrix types.

No algorithm/configuration/output bytes are changed. JSON arrays and the
runner's Python tuples represent the same category pairs; v1 compared their
container types on resume. This entry normalizes only the saved in-memory
matrix after authenticating all frozen execution source files.
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
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/route1_controls_20261014.json')
    ap.add_argument('--resume',action='store_true',required=True)
    args=ap.parse_args();run=args.output_dir.resolve();saved=run/'run_config.json'
    if not run.is_relative_to(ROOT/'runs/quick'):raise ValueError('Existing project run required')
    previous=json.loads(saved.read_text(encoding='utf-8'))
    for relative,expected in previous['source_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Frozen source drift: '+relative)
    if digest(args.config)!=previous['protocol_sha256']:raise ValueError('Frozen protocol drift')
    source=ROOT/'scripts/run_route1_controls.py'
    spec=importlib.util.spec_from_file_location('frozen_route1_runner',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_read=module.read
    def canonical_read(path):
        value=original_read(path)
        if Path(path).resolve()==saved:
            original_serialized=json.dumps(value['expected_matrix'],sort_keys=True)
            value['expected_matrix']['categories']=[tuple(pair) for pair in value['expected_matrix']['categories']]
            if json.dumps(value['expected_matrix'],sort_keys=True)!=original_serialized:raise ValueError('Canonicalization changed matrix values')
        return value
    module.read=canonical_read
    context=run/'execution_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')/'resume_entry.json'
    context.parent.mkdir(parents=True,exist_ok=True)
    context.write_text(json.dumps({'resume_adapter_sha256':digest(Path(__file__)),'frozen_runner_sha256':digest(source),
        'saved_configuration_sha256_before_resume':digest(saved),'protocol_sha256':previous['protocol_sha256'],
        'repair':'normalize JSON arrays to Python tuple category pairs in memory only','algorithm_and_saved_results_changed':False,
        'frozen_source_hashes_verified':True},indent=2)+'\n',encoding='utf-8')
    sys.argv=[str(source),'--config',str(args.config.resolve()),'--output-dir',str(run),'--resume']
    return module.main()


if __name__=='__main__':raise SystemExit(main())
