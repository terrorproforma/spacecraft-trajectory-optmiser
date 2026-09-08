"""Independent saved-vector audit. Stdlib only; no NumPy, native or project imports."""
from pathlib import Path
import ast
import hashlib
import json
import math
import struct
import types
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
helper_path = ROOT/'build/performance/completion-integration-review-v621/review_native_readback.py'
assert sha(helper_path.read_bytes()) == '86d63e1be44bd2222e8bd7103b489a2bf05a049e82979a3c1c56b0fd661aa88a'
helper = types.ModuleType('independent_completion_formula')
helper.__file__ = str(helper_path)
exec(compile(helper_path.read_text(), str(helper_path), 'exec'), helper.__dict__)

def npy(data):
    assert data[:6] == b'\x93NUMPY'
    version = tuple(data[6:8]); assert version in ((1,0),(2,0))
    off = 10 if version == (1,0) else 12
    length = struct.unpack_from('<H' if off == 10 else '<I',data,8)[0]
    h = ast.literal_eval(data[off:off+length].decode('ascii'))
    assert not h['fortran_order'] and len(h['shape']) == 1
    count = h['shape'][0]; payload = data[off+length:]; d = h['descr']
    if d == '<f8':
        assert len(payload) == count*8
        return [r[0] for r in struct.iter_unpack('<d',payload)]
    formats = {'<i4':'i','<u8':'Q','<f8':'d'}
    fields=[];fmt='<'
    for item in d:
        name, kind = item[:2]; assert kind in formats
        size = 1 if len(item)==2 else math.prod(item[2])
        fields.append((name,size,len(item)==3));fmt += formats[kind]*size
    assert len(payload) == count*struct.calcsize(fmt)
    result=[]
    for row in struct.iter_unpack(fmt,payload):
        out={};offset=0
        for name,size,vector in fields:
            out[name]=list(row[offset:offset+size]) if vector else row[offset]
            offset+=size
        result.append(out)
    return result

def compare(actual,expected,*,strict=False,atol=2e-11,rtol=2e-14):
    if isinstance(actual,dict):
        assert set(actual)==set(expected)
        return max((compare(actual[k],expected[k],strict=strict,atol=atol,rtol=rtol) for k in actual),default=0.)
    if isinstance(actual,list):
        assert len(actual)==len(expected)
        return max((compare(a,b,strict=strict,atol=atol,rtol=rtol) for a,b in zip(actual,expected)),default=0.)
    if isinstance(actual,int) or strict:
        assert actual==expected,(actual,expected)
        return 0.
    if math.isnan(actual) and math.isnan(expected):return 0.
    assert math.isfinite(actual) and math.isfinite(expected),(actual,expected)
    error=abs(actual-expected)
    assert error<=atol+rtol*abs(expected),(actual,expected,error)
    return error

results={}
for folder_name,version in [('completion-model-gpu-v622','e'),('completion-model-gpu-v622b','f'),
                            ('completion-model-gpu-v622c','g'),('completion-model-gpu-v622d','g')]:
    directory=ROOT/'build/performance'/folder_name
    if not (directory/'report.json').exists():continue
    report=json.loads((directory/'report.json').read_text())
    if 'exit_code' not in report:continue
    frozen=ROOT/'build/performance'/('completion-model-v622'+version)
    manifest=json.loads((frozen/'report.json').read_text())
    assert sha((frozen/'report.json').read_bytes())==report['source_report_sha256']
    assert report['library_sha256']==manifest['library']['sha256']
    assert sha((directory/'pytest.log').read_bytes())==report['log_sha256']
    suites=list(ET.parse(directory/'pytest.xml').getroot().iter('testsuite'))
    junit={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert junit['tests']==(7 if version=='g' else 6) and junit['errors']==0 and junit['skipped']==0
    files={};full_count=0
    for name,identity in report['readbacks'].items():
        path=directory/'readbacks'/name
        assert path.stat().st_size==identity['bytes'] and sha(path.read_bytes())==identity['sha256']
        with zipfile.ZipFile(path) as archive:
            arrays={Path(n).stem:npy(archive.read(n)) for n in archive.namelist()}
        result_error=compare(arrays['actual_0'],arrays['expected_0'])
        leg_error=compare(arrays['actual_1'],arrays['expected_1'])
        compare(arrays['actual_2'],arrays['expected_2'],strict=True)
        for key in ('candidates','deploy_slots','leg_slots'):
            assert arrays['actual_3'][0][key]==arrays['expected_3'][0][key]
        mismatched_fields={};expanded_error=0.
        for actual,expected in zip(arrays['expanded'],arrays['full_3'],strict=True):
            for key in actual:
                try:expanded_error=max(expanded_error,compare(actual[key],expected[key],atol=1e-12))
                except AssertionError:
                    assert version=='e' and key in ('flat','floor','slope'),(name,key)
                    mismatched_fields[key]=mismatched_fields.get(key,0)+1
        policy=arrays['full_0'][0];formula_error=0.;counts={}
        for index,candidate in enumerate(arrays['full_1']):
            d0,nd,l0,nl=(candidate[k] for k in ('deploy_begin','deploy_count','leg_begin','leg_count'))
            case={'partial_mass':candidate['partial_mass'],'deploys':arrays['full_2'][d0:d0+nd],
                  'legs':arrays['full_3'][l0:l0+nl]}
            answer=helper.replay(case,policy)
            formula_error=max(formula_error,compare(arrays['actual_0'][index],answer['result']))
            formula_error=max(formula_error,compare(arrays['actual_1'][l0:l0+nl],answer['leg_results']))
            compare(arrays['actual_2'][d0:d0+nd],answer['collected_by_deploy'],strict=True)
            failure=arrays['actual_0'][index]['failure'];counts[str(failure)]=counts.get(str(failure),0)+1
        full_count+=len(arrays['full_1'])
        files[name]={'sha256':identity['sha256'],'candidate_pairs':len(arrays['full_1']),
                     'max_result_error':result_error,'max_leg_error':leg_error,
                     'max_independent_formula_error':formula_error,'max_supported_metadata_error':expanded_error,
                     'unsupported_unused_field_mismatches':mismatched_fields,'failure_counts':counts}
    findings={'report_sha256':sha((directory/'report.json').read_bytes()),'source_report_sha256':report['source_report_sha256'],
              'library_sha256':report['library_sha256'],'complete':report['complete'],'exit_code':report['exit_code'],
              'junit':junit,'saved_original_compact_pairs':full_count,'saved_pair_valid_calls':2*len(files),
              'saved_pair_candidate_evaluations':2*full_count,'files':files}
    if version=='e':
        assert len(files)==3 and full_count==777 and junit['failures']==3
        findings['observed_control_flow']='Six valid calls /1554 evaluations; metadata assertion stops each model before size3, malformed and production-capture calls.'
    elif version=='f':
        assert len(files)==6 and full_count==786 and junit['failures']==3
        findings['observed_control_flow']='Twelve valid calls /1572 evaluations plus three malformed calls; CollectDPSettings construction throws before second malformed call and all production captures.'
    else:
        assert len(files)==6 and full_count==786 and junit['failures']==0 and report['complete']
        findings['observed_control_flow']='Passed test control flow:18 valid calls/2358 evaluations, six malformed calls; production capture outcomes are asserted by tests but not part of these six saved comparison NPZs.'
    results[folder_name]=findings
out={'scope':'Saved NPZ and frozen test control-flow review only; no runtime work. Exact integer gates and cargo; tolerance applies only to ordinary arithmetic.',
     'independent_formula_sha256':sha(helper_path.read_bytes()),'runs':results,'reviewer_GPU_calls':0}
(OUT/'readback-findings.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'runs':len(results),'report_sha256':sha((OUT/'readback-findings.json').read_bytes())}))
