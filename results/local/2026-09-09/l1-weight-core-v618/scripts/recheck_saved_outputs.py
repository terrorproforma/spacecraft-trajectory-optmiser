"""Re-audit packaged original equations on CPU (NumPy/SciPy required)."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,struct,sys,tarfile
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=Path(__file__).resolve().parents[1]
sha=lambda data:hashlib.sha256(data).hexdigest()
source=root/'real/independent_auditor.py'
assert sha(source.read_bytes())=='0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
spec=importlib.util.spec_from_file_location('saved_original_audit',source);auditor=importlib.util.module_from_spec(spec)
exec(compile(source.read_bytes(),str(source),'exec'),auditor.__dict__)
report=json.loads((root/'real/report.json').read_text());manifest=json.loads((root/'real/manifest.json').read_text())
members=json.loads((root/'real/raw-members.json').read_text())
def invalid(value):raise ValueError('nonfinite JSON constant '+value)
def bits(values):return b''.join(struct.pack('d',float(x)) for x in values)
findings=[];qualified_seeds=qualified_cold=0
with tarfile.open(root/'real/raw-logs.tar.gz','r:gz') as archive:
    assert sorted(m.name for m in archive.getmembers())==sorted(members)
    for case in report['cases']:
        name=case['name']+'.log';raw=archive.extractfile(name).read()
        assert len(raw)==members[name]['bytes'] and sha(raw)==members[name]['sha256']==case['log_sha256']
        records={}
        for line in raw.decode().splitlines():
            prefix,payload=line.split(' ',1);records.setdefault(prefix,[]).append(json.loads(payload,parse_int=float,parse_constant=invalid))
        assert len(records['PERSISTENT_REPLAY'])==1
        final=records['PERSISTENT_REPLAY'][0];meta=records['PERSISTENT_REPLAY_META'][0]
        snapshot=root/'real/inputs'/(case['capture']+'.txt')
        assert sha(snapshot.read_bytes())==meta['input_sha256']==report['inputs_sha256'][snapshot.name]
        assert meta['source_commit']=='uncommitted' and meta['source_tree_sha256']==manifest['source_tree_sha256']
        assert meta['library_sha256']==manifest['library_sha256'] and meta['source_sha256']==manifest['compiled_snapshot_source_sha256']
        result=auditor.audit(auditor.load_snapshot(snapshot),final,backend='persistent',coordinates='original')
        assert result==case['audit'] and result['qualified']==final['qualified_original']
        gate=final['gpu_common_kkt']
        if gate['valid']:assert gate['passes']==result['passes_common_kkt_gate']
        else:assert not gate['passes'] and final['termination'] in (3,4)
        l1=final['l1'];assert l1['weight_mode']==(0 if case['mode']=='unit' else 2)
        if l1['weight_valid']:
            expected=1.0 if case['mode']=='unit' else l1['objective_scale']/l1['bound_scale']
            assert l1['omega']==expected and l1['primal_base_step']==l1['eta']/expected and l1['dual_base_step']==l1['eta']*expected
        if case['seeded']:
            initial=records['PERSISTENT_REPLAY_INITIAL_POINT'][0]
            assert result['qualified'] and final['iterations']==0 and l1['completions']==0
            for key in ('x','y','z'):assert bits(initial[key])==bits(final[key])
            qualified_seeds+=int(result['qualified'])
        else:qualified_cold+=int(result['qualified'])
        findings.append({'case':case['name'],'raw_sha256':sha(raw),'audit':result})
output={'complete':True,'solver_calls':0,'report_sha256':sha((root/'real/report.json').read_bytes()),
    'raw_archive_sha256':sha((root/'real/raw-logs.tar.gz').read_bytes()),
    'qualified_seeds':qualified_seeds,'qualified_cold':qualified_cold,'cases':findings}
with args.output.open('x') as file:json.dump(output,file,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in output.items() if k!='cases'}))
