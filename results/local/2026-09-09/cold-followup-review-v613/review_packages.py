#!/usr/bin/env python3
"""CPU-only package/raw-vector review; no solver calls or package mutations.

The retained Decimal audit source is compiled directly, never imported from
inherited bytecode. All KKT arithmetic starts from exact represented FP64.
"""
import argparse
from decimal import localcontext
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import types

BASE=Path('results/local/2026-09-09')
NAMES=('automatic-cold-v610','upstream-zero-quadratic-v613')
EXCLUDE={'x','x_solver','normal_dual_solver','y','z','s'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def records(data):
    out={}
    for line in data.decode().splitlines():
        if line.startswith(('PERSISTENT_REPLAY','UPSTREAM_REPLAY')) and ' {' in line:
            name,value=line.split(' ',1)
            assert name not in out
            out[name]=json.loads(value)
    return out


def package(name,math):
    base=BASE/name
    idx=json.loads((base/'sha256.json').read_text())
    actual={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file() and p.name!='sha256.json'}
    assert actual==set(idx)
    for path,item in idx.items():
        data=(base/path).read_bytes()
        assert (len(data),sha(data))==(item['bytes'],item['sha256'])
    assert not any('__pycache__' in p or p.endswith('.pyc') for p in actual)
    report=json.loads((base/'run/report.json').read_text())
    first=json.loads((base/'run/first-observation-failure.json').read_text())
    assert report['complete'] and len(report['cases'])==4 and not first['complete']
    upstream=name==NAMES[1]
    initial_count=3 if upstream else 1
    assert len(first['cases'])==initial_count
    assert sum(bool(r.get('existing_completed_call_reused_without_rerun')) for r in report['cases'])==initial_count
    assert report['maximum_native_calls' if upstream else 'maximum_solve_calls']==4
    assert report['native_calls' if upstream else 'solve_calls']==4
    runner=base/'run/run.py' if upstream else base/'run.py'
    resume=base/'run/resume.py' if upstream else base/'resume.py'
    assert report['runner_sha256']==sha(runner.read_bytes())
    assert report['resume_runner_sha256']==sha(resume.read_bytes())
    assert 'if previous is None:' in resume.read_text()
    assert 'existing_completed_call_reused_without_rerun' in resume.read_text()
    assert report['observation_failure']==first['failure']
    if upstream:
        manifest=json.loads((base/'manifest.json').read_text())
        assert report['manifest_sha256']==sha((base/'manifest.json').read_bytes())
        assert manifest['complete'] and manifest['gpu_calls']==0
        assert all(r['returncode']==r['expected'] for r in manifest['cpu_checks']) and len(manifest['cpu_checks'])==5
        for path,expected in manifest['source_files_sha256'].items():
            assert sha((base/'source'/path).read_bytes())==expected
        for path,expected in manifest['fixtures_sha256'].items():
            assert sha((base/'fixtures'/path).read_bytes())==expected
        assert sha(json.dumps(manifest['source_files_sha256'],sort_keys=True,separators=(',',':')).encode())==manifest['source_sha256']
        assert manifest['executable_sha256']==report['executable_sha256']
        prior=json.loads((BASE/'upstream-identical-capture-v608/manifest.json').read_text())
        assert manifest['upstream_commit']==prior['upstream_commit']
        assert manifest['reference_archive_sha256']==prior['reference_archive_sha256']
        changed=[p for p,h in manifest['source_files_sha256'].items() if prior['source_files_sha256'][p]!=h]
        assert changed==['cpp/cuda/tests/upstream_snapshot_replay.cu']
    members=json.loads((base/'raw-log-members.json').read_text())
    sources=json.loads((base/'source-files.json').read_text())
    raw={}
    with tarfile.open(base/'raw-logs.tar.gz') as archive:
        assert len(archive.getnames())==4 and set(archive.getnames())==set(members)
        for item in archive.getmembers():
            data=archive.extractfile(item).read()
            assert (len(data),sha(data))==(members[item.name]['bytes'],members[item.name]['sha256'])
            assert members[item.name]==sources[item.name]
            raw[item.name]=data
    for path,expected in sources.items():
        data=raw[path] if path in raw else (base/path).read_bytes()
        assert (len(data),sha(data))==(expected['bytes'],expected['sha256'])
    for old in first['cases']:
        new=next(r for r in report['cases'] if r['name']==old['name'])
        assert old['returncode']==new['returncode']==0 and old['command']==new['command']
        assert old['log_sha256']==new['log_sha256']==sha(raw['run/'+old['name']+'.log'])
        assert old['wall_seconds']==new['wall_seconds']
    result=[]
    vectors={}
    for case in report['cases']:
        data=raw['run/'+case['name']+'.log']
        assert case['returncode']==0 and case['log_sha256']==sha(data)
        rec=records(data)
        capture=case['name'].split('-')[0]
        path=base/('fixtures' if upstream else 'inputs')/(capture+'.txt')
        snap=math.Snapshot(path)
        meta=rec['UPSTREAM_REPLAY_META' if upstream else 'PERSISTENT_REPLAY_META']
        final=rec['UPSTREAM_REPLAY_RESULT' if upstream else 'PERSISTENT_REPLAY']
        assert {k:v for k,v in final.items() if k not in EXCLUDE}==case['final']
        assert meta['input_sha256']==sha(path.read_bytes())
        audit=snap.audit(final if upstream else dict(final,termination_code=final['termination']))
        assert audit['passes']==case['audit']['passes_common_kkt_gate']
        assert audit['qualified']==case['audit']['qualified']
        if upstream:
            assert 'UPSTREAM_REPLAY_DONE' in rec
            assert snap.q_all_zero and meta['quadratic_exactly_zero'] and meta['omit_zero_quadratic']
            assert meta['quadratic_descriptor']=='nullptr_exact_linear_objective'
            assert meta['executable_sha256']==manifest['executable_sha256']
            assert meta['source_sha256']==manifest['source_sha256']
            assert final['inner_iterations']==final['iterations']
            assert audit['passes']==final['passes_common_kkt_gate'] and audit['qualified']==final['qualified']
            assert final['iterations']==(0 if case['seeded'] else 100000)
            assert audit['qualified']==case['seeded']
            assert snap.mapping(final)
        else:
            assert 'PERSISTENT_REPLAY_BOOTSTRAP' not in rec and 'PERSISTENT_REPLAY_INITIAL_POINT' not in rec
            assert rec['PERSISTENT_REPLAY_SUMMARY']['repeats']==1
            assert meta['requested_execution_blocks'] is None and final['execution_blocks'] is None
            assert meta['library_sha256']==report['core_sha256']
            assert final['iterations']==100000 and final['termination_name']=='iteration_limit'
            assert not final['deadline_requested'] and final['recovery_iterations']==0
            assert not audit['passes'] and not final['qualified_original']
            if case['common']:
                assert final['gpu_common_kkt']['valid'] and not final['gpu_common_kkt']['passes']
            vectors[case['name']]={k:final[k] for k in ('x','y','z','s')}
        result.append(dict(name=case['name'],log_sha256=sha(data),iterations=final['iterations'],
                           inner_iterations=final.get('inner_iterations'),decimal65=audit,
                           native_wall_or_solve_seconds=final['solve_wall_seconds' if upstream else 'solve_seconds']))
    comparisons={}
    if not upstream:
        for capture in ('conditioning','difficult'):
            a,b=(vectors[f'{capture}-{mode}-auto'] for mode in ('natural','common'))
            comparisons[capture]={}
            for key in a:
                assert len(a[key])==len(b[key])
                comparisons[capture][key]=dict(maximum_absolute_delta=max(abs(x-y) for x,y in zip(a[key],b[key])),
                                               unequal_fp64_bits=sum(struct.pack('<d',x)!=struct.pack('<d',y) for x,y in zip(a[key],b[key])))
    return dict(package=name,index_sha256=sha((base/'sha256.json').read_bytes()),files=len(idx),
                indexed_bytes=sum(v['bytes'] for v in idx.values()),all_hashes_match=True,
                report_sha256=sha((base/'run/report.json').read_bytes()),native_calls=4,
                prior_completed_calls_reused=initial_count,new_calls_on_resume=4-initial_count,
                original_observation_failure=first['failure'],rows=result,automatic_endpoint_comparison=comparisons)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).with_name('decimal_audit_source.py')
    module=types.ModuleType('source_compiled_decimal_auditor')
    module.__file__=str(source)
    exec(compile(source.read_bytes(),str(source),'exec'),module.__dict__)
    with localcontext() as context:
        context.prec=65
        out=dict(scope='CPU package, source and saved FP64 vector audit; no new solver calls',
                 script_sha256=sha(Path(__file__).read_bytes()),decimal_auditor_sha256=sha(source.read_bytes()),
                 auditor_loading='exec(compile(exact_source_bytes)); inherited Python bytecode is not used',
                 decimal_precision=65,
                 pinned_dispatch_source_sha256={p:sha(Path(p).read_bytes()) for p in (
                     '_upstream/pdhcg/src/utils.cu','_upstream/pdhcg/src/preconditioner.c',
                     '_upstream/pdhcg/src/solver_state.cu','_upstream/pdhcg/src/pdhg_core_op.cu')},
                 packages=[package(n,module) for n in NAMES])
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
