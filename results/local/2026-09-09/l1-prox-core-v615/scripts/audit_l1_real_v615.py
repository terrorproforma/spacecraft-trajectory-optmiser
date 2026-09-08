"""Fresh CPU original-equation re-audit of the six saved GPU outputs."""
from pathlib import Path
import hashlib,importlib.util,json,math,struct
root=Path(__file__).resolve().parents[2];folder=root/'build/performance/l1-real-v615'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((folder/'run/report.json').read_text());assert report['complete']
manifest=json.loads((folder/'manifest.json').read_text())
spec=importlib.util.spec_from_file_location('l1_independent_reaudit',folder/'independent_auditor.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
assert sha(folder/'independent_auditor.py')=='0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
def invalid(value):raise ValueError(value)
def bits(values):return b''.join(struct.pack('d',float(x)) for x in values)
findings=[]
for case in report['cases']:
    path=folder/'run'/(case['name']+'.log');assert sha(path)==case['log_sha256']
    records={}
    for line in path.read_text().splitlines():
        prefix,encoded=line.split(' ',1);records.setdefault(prefix,[]).append(json.loads(encoded,parse_int=float,parse_constant=invalid))
    assert len(records['PERSISTENT_REPLAY'])==1
    final=records['PERSISTENT_REPLAY'][0];meta=records['PERSISTENT_REPLAY_META'][0]
    capture=folder/'inputs'/(case['capture']+'.txt');q=audit.load_snapshot(capture)
    assert meta['input_sha256']==sha(capture) and meta['source_commit']==manifest['frozen_commit']
    assert meta['source_sha256']==manifest['compiled_snapshot_source_sha256'] and meta['library_sha256']==manifest['library_sha256']
    fresh=audit.audit(q,final,backend='persistent',coordinates='original')
    assert fresh==case['audit'];assert fresh['qualified']==final['qualified_original']
    gate=final['gpu_common_kkt'];assert gate['valid'] and gate['passes']==fresh['passes_common_kkt_gate']
    assert final['execution_blocks']==128 and final['recovery_iterations']==final['recovery_seconds']==0
    for key,value in case['final'].items():assert final[key]==value
    if case['seeded']:
        initial=records['PERSISTENT_REPLAY_INITIAL_POINT'][0]
        assert final['termination']==1 and final['iterations']==0 and final['l1']['completions']==0
        assert initial['supplied_qualified'] and initial['mapped_reference_qualified']
        assert records['PERSISTENT_REPLAY_PRESTEP'][0]['native_seed_verified_unchanged']
        for key in ('x','y','z'):assert bits(initial[key])==bits(final[key])
    else:assert final['termination']==2 and final['iterations']==100000 and not fresh['qualified']
    findings.append({'case':case['name'],'raw_sha256':sha(path),'qualified':fresh['qualified'],
        'termination':int(final['termination']),'iterations':int(final['iterations']),
        'solve_seconds':final['solve_seconds'],'fresh_original_audit':fresh,'gpu_common_gate_agrees':True,
        'seed_bits_preserved':True if case['seeded'] else None,'l1':final.get('l1')})
out={'complete':True,'gpu_calls':0,'raw_executions':6,'solve_API_calls':8,'bootstrap_updates':2,
     'actual_cold_updates':400000,'qualified_seed_cases':2,'qualified_cold_cases':0,
     'report_sha256':sha(folder/'run/report.json'),'manifest_sha256':sha(folder/'manifest.json'),'cases':findings}
target=folder/'audit';target.mkdir(exist_ok=True)
(target/'findings.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if k!='cases'}))
