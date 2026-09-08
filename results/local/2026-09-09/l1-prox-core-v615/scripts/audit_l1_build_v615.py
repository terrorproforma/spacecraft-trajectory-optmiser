"""CPU-only identity, JSON, and compiled-resource check; launches no executable."""
from pathlib import Path
import hashlib,json,re,struct,tarfile
root=Path(__file__).resolve().parents[2]
bundle=root/'build/performance/l1-v615a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((bundle/'manifest.json').read_text());assert manifest['complete']
archive=tarfile.open(bundle/'source.tar.gz','r:gz')
identities={}
for name in manifest['owned_paths']:
    archived=archive.extractfile(name).read();expected=manifest['source_sha256'][name]
    assert hashlib.sha256(archived).hexdigest()==expected==sha(root/name)
    identities[name]=expected
resource_names=['cooperative_solve_kernelILb0','cooperative_solve_kernelILb1','solve_kernelILb0','solve_kernelILb1',
                'cooperative_halpern_kernel','cooperative_initialise_kernel','cooperative_residual_kernel',
                'initialise_control_kernel','recovery_kernel','cooperative_l1_kernel','cooperative_l1_initialise_kernel']
def resources(path):
    lines=path.read_text().splitlines();out={}
    for i,line in enumerate(lines):
        if 'Function ' not in line:continue
        for name in resource_names:
            if name in line and not(name.startswith('solve_kernel') and 'cooperative_solve_kernel' in line):
                assert name not in out
                out[name]={k:int(v) for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',lines[i+1])}
    return out
before=resources(root/'build/performance/halpern-v612d/resource-usage.log');after=resources(bundle/'resource-usage.log')
comparison={k:{'before':v,'after':after[k],'equal':v==after[k]} for k,v in before.items()}
assert all(v['equal'] for k,v in comparison.items() if k!='cooperative_halpern_kernel')
initial=[]
for name in ('conditioning','difficult'):
    logs={}
    for line in (bundle/f'validate-{name}-l1-seeded.log').read_text().splitlines():
        prefix,text=line.split(' ',1);logs[prefix]=json.loads(text,parse_int=float)
    point=root/f'build/performance/known-point-replay-v606/inputs/{name}-initial.txt'
    values={}
    for line in point.read_text().splitlines()[3:]:
        words=line.split();values[words[0]]=[float(s) for s in words[2:]]
    record=logs['PERSISTENT_REPLAY_INITIAL_POINT']
    def bits(v):return b''.join(struct.pack('d',x) for x in v)
    for key in ('x','y','z','s'):assert bits(record[key])==bits(values[key]),(name,key)
    assert record['supplied_qualified'] and record['mapped_reference_qualified']
    initial.append({'capture':name,'point_sha256':sha(point),'all_original_vector_bits_preserved_by_cpu_import':True})
out={'complete':True,'gpu_calls':0,'manifest_sha256':sha(bundle/'manifest.json'),'owned_source_identities':identities,
     'resource_comparison':comparison,'new_resources':{k:v for k,v in after.items() if k not in before},
     'initial_points':initial,'scope':'CPU/build only; GPU arithmetic, capacity, original gate and timing remain untested'}
(bundle/'cpu-independent-check.json').write_text(json.dumps(out,indent=2));print(json.dumps({'complete':True,'old_resource_changes':[k for k,v in comparison.items() if not v['equal']]}))
