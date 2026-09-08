# Run only after the v387 process is terminal and its build has passed.
from pathlib import Path
import os,subprocess,fcntl,json,hashlib,time,shutil
root=Path('/home/ubuntu/spacepdhcg-warp-native-v390');root.mkdir(exist_ok=False)
core=Path('/home/ubuntu/spacepdhcg-warp-hops-v387/core-build/cuda/libspacepdhcg_cuda.so');qoco=Path('/home/ubuntu/spacepdhcg-step-final-v359/final/libqoco.so')
import ast
for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
env=dict(os.environ,LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64')
binary=root/'probe';source=Path('/home/ubuntu/spacepdhcg-warp-hops-v387/repo/cpp/cuda/tests/orbitweaver_hop_test.cu');shutil.copy2(source,root/source.name)
cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-I/home/ubuntu/spacepdhcg-warp-hops-v387/repo/cpp/include','-I/home/ubuntu/spacepdhcg-warp-hops-v387/repo/cpp/cuda/include',str(source),'-L'+str(core.parent),'-lspacepdhcg_cuda','-o',str(binary)]
report=dict(complete=False,stages=[],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
try:
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobs=[('compile',cmd),('probe',[str(binary)])]+[(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(binary)]) for tool in ['memcheck','synccheck','racecheck']]
  for name,command in jobs:
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
   report['stages'].append(dict(name=name,code=r.returncode,seconds=time.perf_counter()-start,command=command));assert r.returncode==0,(name,r.returncode)
 report['complete']=True
except Exception as e:report['error']=repr(e)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
