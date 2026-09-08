from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
from analyse_qps_v169 import problem,audit
root=Path('build/performance/preserve-objective-v536');root.mkdir(exist_ok=False)
qoco=Path('/home/angus/build-qoco-preserve-objective-v534/final/libqoco.so')
source=qoco.parent.parent/'source'
core=Path('/home/angus/build-spacepdhcg-retained-replay-v512/final/libspacepdhcg_cuda.so')
py='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
flag='SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1')
report=dict(pid=os.getpid(),complete=False,stages=[],qp=[],source_sha256={},runtime_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]})
def save():
 t=root/'report.tmp';t.write_text(json.dumps(report,indent=2));t.replace(root/'report.json')
def run(name,cmd,timeout=1200):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save();rc=child.wait(timeout=timeout)
 report['stages'].append(dict(name=name,returncode=rc,seconds=time.perf_counter()-start,command=cmd));save();assert rc==0,(name,rc)
save()
try:
 files=['scripts/gpu/prepare_qoco_preserve_objective.py','scripts/gpu/prepare_qoco_gpu.py','cpp/cuda/tests/qoco_gpu_numeric_update_test.cu','cpp/cuda/tests/qoco_snapshot_replay.cu','build/performance/replay_conditioning.py','build/performance/test_preserve_objective_v536.py','build/performance/analyse_qps_v169.py','build/performance/grid-cache-fleet-v403/scvx-calls.json']
 for name in files:
  dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest);report['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 for name in ['src/equilibration.c','algebra/cuda/qoco_device_update.cuh']:
  dest=root/'vendor'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
 for test in ['qoco_gpu_numeric_update_test','qoco_snapshot_replay']:
  cmd=['/usr/local/cuda-12.8/bin/nvcc','--default-stream','per-thread','-arch=sm_120','-std=c++17']
  for folder in ['include','algebra/cuda','lib/qdldl/include','lib/amd']:cmd+=['-I',str(source/folder)]
  cmd+=['cpp/cuda/tests/'+test+'.cu','-L',str(qoco.parent),'-lqoco','-ldl','-o',str((root/test).resolve())]
  run('build-'+test,cmd)
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  qp=root/'ruiz2.txt';shutil.copy2('build/performance/conditioning-qp-reg-v527/ruiz2.txt',qp);data=problem(qp)
  for enabled in ['0','1']:
   env[flag]=enabled
   run('numeric-'+enabled,[str((root/'qoco_gpu_numeric_update_test').resolve())])
   run('qp-'+enabled,[str((root/'qoco_snapshot_replay').resolve()),str(qp),'3'])
   records=[json.loads(s[10:]) for s in (root/f'qp-{enabled}.log').read_text().splitlines() if s.startswith('QP_REPLAY ')]
   assert len(records)==3
   row=dict(enabled=enabled,iterations=[r['iterations'] for r in records],audits=[audit(data,r) for r in records]);report['qp'].append(row);save()
   if enabled=='1':assert all(a['qualified'] for a in row['audits']),row
  for sanitizer in ['memcheck','synccheck','racecheck']:
   run(sanitizer,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',sanitizer,'--error-exitcode','99',str((root/'qoco_gpu_numeric_update_test').resolve())])
  boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/replay_conditioning.py',run_name='__main__')"
  for label,ruiz,enabled in [('baseline','0','0'),('candidate','2','1')]:
   env[flag]=enabled;run(label,[py,'-c',boot,ruiz,str(root/label)])
 report['complete']=True
except Exception as error:report['error']=repr(error)
save();print(json.dumps(report,indent=2))
