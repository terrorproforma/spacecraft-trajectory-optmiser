from pathlib import Path
import ast,os,subprocess
root=Path('/home/ubuntu/spacepdhcg-qp-block-reg-v324');root.mkdir(exist_ok=False)
(root/'audit.py').write_text(Path('/home/ubuntu/spacepdhcg-qp-sweep-v308/audit.py').read_text())
runner=Path('/home/ubuntu/spacepdhcg-qp-scaling-v311/run.py').read_text().replace('spacepdhcg-qp-scaling-v311','spacepdhcg-qp-block-reg-v324')
runner=runner.replace("cases=[('ruiz'+str(k),k) for k in [0,1,2,4,8,12]]", "cases=[('base',1e-8,1e-8,1e-8),('a10',1e-8,1e-10,1e-8),('a12',1e-8,1e-12,1e-8),('g10',1e-8,1e-8,1e-10),('ag10',1e-8,1e-10,1e-10),('p6',1e-6,1e-8,1e-8),('p6_ag10',1e-6,1e-10,1e-10),('p6_ag12',1e-6,1e-12,1e-12)]")
runner=runner.replace('for name,ruiz in cases:', 'for name,regp,rega,regg in cases:')
start=runner.index('  lines=original');end=runner.index('  qp=root/',start)
runner=runner[:start]+"  lines=original.read_text().splitlines();settings=lines[3].split();settings[1:4]=[str(regp),str(rega),str(regg)];lines[3]=' '.join(settings)\n"+runner[end:]
runner=runner.replace('name=name,ruiz=ruiz,','name=name,regularization_p=regp,regularization_a=rega,regularization_g=regg,')
(root/'run.py').write_text(runner)
env=os.environ.copy()
for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
env['LD_LIBRARY_PATH']='/home/ubuntu/spacepdhcg-diagnose-v174/final:'+runtime+'/lib:/usr/local/cuda/lib64'
with (root/'runner.log').open('x') as log:p=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
