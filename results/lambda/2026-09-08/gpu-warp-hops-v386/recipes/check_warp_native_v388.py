from pathlib import Path
import os,subprocess,fcntl,json,hashlib,time,shutil
root=Path('build/performance/warp-native-v388');root.mkdir(exist_ok=False)
core=Path('/home/angus/build-spacepdhcg-warp-hops-v385/final/libspacepdhcg_cuda.so');qoco=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
env=dict(os.environ,LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
binary=Path('/home/angus/warp-native-probe-v388');source=Path('cpp/cuda/tests/orbitweaver_hop_test.cu');shutil.copy2(source,root/source.name)
cmd=['/usr/local/cuda-12.8/bin/nvcc','-std=c++17','-Icpp/include','-Icpp/cuda/include',str(source),'-L'+str(core.parent),'-lspacepdhcg_cuda','-o',str(binary)]
report=dict(complete=False,stages=[],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobs=[('compile',cmd),('probe',[str(binary)])]+[(tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(binary)]) for tool in ['memcheck','synccheck','racecheck']]
  for name,command in jobs:
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
   report['stages'].append(dict(name=name,code=r.returncode,seconds=time.perf_counter()-start,command=command));assert r.returncode==0,(name,r.returncode)
 report['complete']=True
except Exception as e:report['error']=repr(e)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
