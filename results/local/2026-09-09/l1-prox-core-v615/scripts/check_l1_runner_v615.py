"""Extract and test JSON parsing only; never execute either runner's main body."""
from pathlib import Path
import ast,hashlib,json,math
root=Path(__file__).resolve().parents[2]
path=root/'build/performance/run_l1_real_v615.py';tree=ast.parse(path.read_text())
namespace={'json':json}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('invalid_constant','records')]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
parse=namespace['records']
assert math.copysign(1,parse('P {"x":-0}','P')[0]['x'])==-1
for bad in ('NaN','Infinity','-Infinity'):
    try:parse('P {"x":'+bad+'}','P')
    except ValueError:pass
    else:raise AssertionError('nonfinite JSON accepted')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tiny=root/'build/performance/l1-tiny-v615';report=json.loads((tiny/'report.json').read_text())
assert sha(tiny/'tiny.log')==report['cases'][0]['log_sha256']
records=[]
for line in (tiny/'tiny.log').read_text().splitlines():
    prefix=line.split(' ',1)[0];values=parse(line,prefix);assert len(values)==1
    records.append({'prefix':prefix,'record':values[0]})
assert records==report['cases'][0]['records']
calls=[r['record'] for r in records if r['prefix']=='L1_TEST']
assert len(calls)==9 and sum(r['iterations'] for r in calls)==10
assert report['complete'] and report['actual_solve_api_calls']==9
out={'complete':True,'gpu_calls':0,'real_runner_sha256':sha(path),'strict_json_rejections':['NaN','Infinity','-Infinity'],
     'signed_zero_preserved':True,'tiny_report_sha256':sha(tiny/'report.json'),'tiny_log_sha256':sha(tiny/'tiny.log'),
     'tiny_records_match_saved_report':True,'actual_tiny_solve_calls':9,'actual_tiny_updates':10}
(root/'build/performance/l1-v615c/runner-check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
