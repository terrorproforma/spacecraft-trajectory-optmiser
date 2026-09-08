"""Fresh saved-vector original-equation re-audit; no solver or GPU calls."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,struct
parser=argparse.ArgumentParser()
parser.add_argument('--folder',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
folder=args.folder or root/'build/performance/l1-weight-real-v618'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def invalid(value):raise ValueError(value)
def load(path):return json.loads(path.read_text(),parse_constant=invalid)
def bits(values):return b''.join(struct.pack('d',float(x)) for x in values)
report=load(folder/'run/report.json');assert report['complete'] and len(report['cases'])==6
manifest=load(folder/'manifest.json')
assert manifest['frozen_commit'] is None and manifest['compiled_source_commit']=='uncommitted'
assert sha(folder/'independent_auditor.py')=='0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
spec=importlib.util.spec_from_file_location('l1_weight_original_reaudit',folder/'independent_auditor.py')
audit=importlib.util.module_from_spec(spec)
exec(compile((folder/'independent_auditor.py').read_bytes(),str(folder/'independent_auditor.py'),'exec'),audit.__dict__)
findings=[];bootstrap_updates=0;cold_updates=0
for case in report['cases']:
    path=folder/'run'/(case['name']+'.log');assert sha(path)==case['log_sha256']
    records={}
    for line in path.read_text().splitlines():
        prefix,encoded=line.split(' ',1);records.setdefault(prefix,[]).append(json.loads(encoded,parse_int=float,parse_constant=invalid))
    assert len(records['PERSISTENT_REPLAY'])==1 and len(records['PERSISTENT_REPLAY_META'])==1
    final=records['PERSISTENT_REPLAY'][0];meta=records['PERSISTENT_REPLAY_META'][0]
    capture=folder/'inputs'/(case['capture']+'.txt');q=audit.load_snapshot(capture)
    assert meta['input_sha256']==sha(capture) and meta['source_commit']=='uncommitted'
    assert meta['source_tree_sha256']==manifest['source_tree_sha256'] and meta['source_dirty'] is True
    assert meta['source_commit_scope']=='uncommitted_frozen_source_tree' and meta['base_commit']==manifest['base_commit']
    assert meta['source_sha256']==manifest['compiled_snapshot_source_sha256'] and meta['library_sha256']==manifest['library_sha256']
    fresh=audit.audit(q,final,backend='persistent',coordinates='original')
    assert fresh==case['audit'] and fresh['qualified']==final['qualified_original']
    gate=final['gpu_common_kkt']
    if gate['valid']:assert gate['passes']==fresh['passes_common_kkt_gate']
    else:assert not gate['passes'] and final['termination'] in (3,4)
    assert final['execution_blocks']==128 and final['recovery_iterations']==final['recovery_seconds']==0
    for key,value in case['final'].items():assert final[key]==value
    l1=final['l1'];mode=case['mode']
    assert meta['l1_prox'] and l1['weight_mode']==(0 if mode=='unit' else 2)
    if l1['weight_valid']:
        omega=1.0 if mode=='unit' else l1['objective_scale']/l1['bound_scale']
        assert omega==l1['omega'] and l1['eta']/omega==l1['primal_base_step'] and l1['eta']*omega==l1['dual_base_step']
    if case['seeded']:
        initial=records['PERSISTENT_REPLAY_INITIAL_POINT'][0]
        assert final['termination']==1 and final['iterations']==0 and l1['completions']==0
        assert initial['supplied_qualified'] and initial['mapped_reference_qualified']
        assert records['PERSISTENT_REPLAY_PRESTEP'][0]['native_seed_verified_unchanged']
        for key in ('x','y','z'):assert bits(initial[key])==bits(final[key])
        point=folder/'inputs'/(case['capture']+'-initial.txt')
        assert meta['initial_point_sha256']==initial['point_sha256']==sha(point)
    else:cold_updates+=int(final['iterations'])
    bootstrap_updates+=sum(int(x['iterations']) for x in records.get('PERSISTENT_REPLAY_BOOTSTRAP',[]))
    findings.append({'case':case['name'],'raw_sha256':sha(path),'qualified':fresh['qualified'],
        'termination':int(final['termination']),'iterations':int(final['iterations']),
        'solve_seconds':final['solve_seconds'],'fresh_original_audit':fresh,
        'gpu_common_gate_agrees':bool(gate['valid']),'seed_bits_preserved':True if case['seeded'] else None,'l1':l1})
result={'complete':True,'gpu_calls':0,'raw_executions':6,'solve_API_calls':report['solve_API_calls'],
    'bootstrap_updates':bootstrap_updates,'actual_cold_updates':cold_updates,
    'qualified_seed_cases':sum(c['qualified'] for c,r in zip(findings,report['cases']) if r['seeded']),
    'qualified_cold_cases':sum(c['qualified'] for c,r in zip(findings,report['cases']) if not r['seeded']),
    'report_sha256':sha(folder/'run/report.json'),'manifest_sha256':sha(folder/'manifest.json'),'cases':findings}
assert result['solve_API_calls']<=8 and bootstrap_updates<=2
target=folder/'audit';target.mkdir(exist_ok=False)
(target/'findings.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
