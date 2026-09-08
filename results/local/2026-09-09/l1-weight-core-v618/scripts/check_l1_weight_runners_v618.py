"""CPU-only syntax, finite budget and JSON parser checks; never execute runner main."""
from pathlib import Path
import ast,hashlib,json,struct
root=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance')
tiny=root/'run_l1_weight_tiny_v618.py';real=root/'run_l1_weight_real_v618.py'
for path in (tiny,real):compile(path.read_text(),str(path),'exec')
tree=ast.parse(real.read_text());ns={'json':json}
selected=[]
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('invalid_constant','records'):selected.append(node)
    if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='cases' for x in node.targets):selected.append(node)
    if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='cases':selected.append(node)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(real),'exec'),ns)
assert ns['cases']==[(c,'cancel-global',True) for c in ('conditioning','difficult')]+[(c,m,False) for c in ('conditioning','difficult') for m in ('unit','cancel-global')]
assert len(ns['cases'])==6 and sum(1+int(x[2]) for x in ns['cases'])==8
parser=ns['records'];assert struct.pack('d',parser('P {"x":-0}','P')[0]['x'])==struct.pack('d',-0.0)
for value in ('NaN','Infinity','-Infinity'):
    try:parser('P {"x":'+value+'}','P')
    except ValueError:pass
    else:raise AssertionError('nonfinite JSON accepted')
tiny_ast=ast.parse(tiny.read_text());tiny_ns={}
assignment=next(x for x in tiny_ast.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_case_modes' for t in x.targets))
exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(tiny),'exec'),tiny_ns)
assert len(tiny_ns['expected_case_modes']['unit'])==9 and len(tiny_ns['expected_case_modes']['weighted'])==13
result={'complete':True,'gpu_calls':0,'syntax_checked':True,'real_executions':6,'real_solve_api_cap':8,
    'tiny_expected_case_modes':tiny_ns['expected_case_modes'],'strict_nonfinite_json_rejected':True,'signed_zero_preserved':True,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tiny,real)}}
out=root/'l1-weight-runner-cpu-v618';out.mkdir(exist_ok=False)
(out/'report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
