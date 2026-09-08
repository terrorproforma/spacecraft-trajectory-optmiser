from pathlib import Path
import json
s=Path('results/lambda/2026-09-08/gpu-soc-step-v359/lambda-final-v359/v360/run.py').read_text()
s=s.replace('/home/ubuntu/spacepdhcg-step-final-v359/v360','/home/ubuntu/spacepdhcg-fleet-search-v374').replace('gpu_step_360','gpu_fleet_374')
old="scope='One-ship full-catalogue-box search and top-three GPU refinements; four retiming/extension attempts with device pricing; not an incumbent fleet replacement.'"
new="scope='Fresh up-to-32-ship search with validated GPU graph backend, beam width 32, top-five refinement and eight retiming attempts. One-hour search budget. Incumbent replacement requires both independent and official mission checks plus a higher bonus-weighted score.'"
assert old in s;s=s.replace(old,new)
for old,new in [("'--ships','1'","'--ships','32'"),("'--beam-width','16'","'--beam-width','32'"),("'--refine-top','3'","'--refine-top','5'"),("'--budget-seconds','600'","'--budget-seconds','3600'"),("'--retime-attempts','4'","'--retime-attempts','8'"),("child.wait(timeout=1800)","child.wait(timeout=4500)"),("1800-second execution ceiling","4500-second execution ceiling"),("fcntl.flock(lock,fcntl.LOCK_EX)","fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)")]:
 assert old in s,old;s=s.replace(old,new)
anchor="report.update(command=cmd,runtime_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]})"
s=s.replace(anchor,anchor+"\nassert report['runtime_sha256']=={'libspacepdhcg_cuda.so':'79df81283d543862427f1d6235d5cfa5b1bc090fab7f5a2553d125a7963279f1','libqoco.so':'2317f607a403b141cb32a492260c2f30a63a02a94b54f08ff2afdaabd888c45a'}")
compile(s,'run-v374.py','exec');Path('build/performance/run_fleet_search_v374.py').write_text(s)
launcher='''from pathlib import Path
import subprocess,json,hashlib
root=Path('/home/ubuntu/spacepdhcg-fleet-search-v374')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),gpu.stdout
root.mkdir(exist_ok=False)
source=SOURCE
(root/'run.py').write_text(source)
with (root/'runner.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'launcher.json').write_text(json.dumps(dict(pid=p.pid,runner_sha256=hashlib.sha256((root/'run.py').read_bytes()).hexdigest()),indent=2))
print((root/'launcher.json').read_text())
'''.replace('SOURCE',repr(s))
compile(launcher,'launch-v374.py','exec');Path('build/performance/launch_fleet_search_v374.py').write_text(launcher)
