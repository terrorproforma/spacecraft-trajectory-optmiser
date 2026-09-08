from pathlib import Path
import ast,fcntl,hashlib,json,os,shutil,subprocess,time,sys
root=Path('/home/ubuntu/spacepdhcg-nt-v350');overlay=root/'overlay';source=root/'qoco'
report=dict(pid=os.getpid(),complete=False,stages=[],replays=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 shutil.copytree('/home/ubuntu/spacepdhcg-nonfinite-ir-v337/qoco',source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))
 patch=overlay/'nt-normalization-v347'
 (source/'src/qoco_cone_arithmetic.cuh').write_bytes((patch/'arithmetic.cuh').read_bytes())
 p=source/'src/cone.cu';text=p.read_text()
 a=text.index('__global__ void compute_nt_scaling_kernel');b=text.index('__global__ void nt_multiply_kernel',a)
 text='#include "qoco_cone_arithmetic.cuh"\n'+text[:a]+(patch/'new_kernel.cuh').read_text()+text[b:]
 a=text.index('__device__ QOCOFloat soc_step_length_dev(');b=text.index('/**',a)
 text=text[:a]+(overlay/'new_step.cuh').read_text()+text[b:];p.write_text(text)
 for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
 env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
 env.update(LD_LIBRARY_PATH=str(root/'final')+':'+runtime+'/lib:/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 def run(name,cmd,timeout=180):
  start=time.perf_counter()
  with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
  report['stages'].append(dict(name=name,returncode=r.returncode,seconds=time.perf_counter()-start));save();print(name,r.returncode,flush=True);r.check_returncode()
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake'
 run('configure',[cmake,'-S',str(source),'-B',str(root/'build'),'-DQOCO_ALGEBRA_BACKEND=cuda','-DCMAKE_CUDA_ARCHITECTURES=90','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_BUILD_TYPE=Release','-DQOCO_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF','-DCUDSS_LIB='+runtime+'/lib/libcudss.so','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I'+runtime+'/include','-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration'])
 run('build',[cmake,'--build',str(root/'build'),'--target','qoco','-j','3'],600)
 libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
 (root/'final').mkdir();lib=root/'final/libqoco.so';shutil.copy2(libs[0],lib)
 report['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest();report['cone_source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();save()
 probe=root/'final/nt_probe'
 run('probe-build',['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_90','--default-stream','per-thread',str(patch/'probe.cu'),'-o',str(probe)])
 sys.path.insert(0,str(overlay));from audit import problem,audit
 qp=Path('/home/ubuntu/spacepdhcg-arc-snapshot-v307/snapshots/qp-153736-000000.txt');data=problem(qp)
 report['qp_sha256']=hashlib.sha256(qp.read_bytes()).hexdigest()
 binary=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/qoco_snapshot_replay')
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run('probe-live',[str(probe),str(overlay/'nt-snapshot-v348/inputs.bin'),str(root/'probe-live.bin')])
  run('probe-synthetic',[str(probe),str(patch/'inputs.bin'),str(root/'probe-synthetic.bin')])
  run('probe-memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(probe),str(overlay/'nt-snapshot-v348/inputs.bin'),str(root/'probe-memcheck.bin')])
  for name,candidate in [('guard_baseline',False),('nt_step',True),('nt_step_repeat',True),('guard_baseline_repeat',False)]:
   actual=lib if candidate else Path('/home/ubuntu/spacepdhcg-nonfinite-ir-v337/final/libqoco.so')
   env.update(LD_LIBRARY_PATH=str(actual.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64',SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1')
   assert str(actual) in subprocess.check_output(['ldd',str(binary)],env=env,text=True)
   run(name,[str(binary),str(qp),'16'])
   records=[json.loads(s[10:]) for s in (root/(name+'.log')).read_text().splitlines() if s.startswith('QP_REPLAY ')];assert len(records)==16
   audits=[dict(iterations=x['iterations'],ir_iterations=x['ir_iterations'],status=x['status'],**audit(data,x)) for x in records]
   report['replays'].append(dict(name=name,qualified=sum(x['qualified'] for x in audits),audits=audits,library_sha256=hashlib.sha256(actual.read_bytes()).hexdigest()));save();print(name,report['replays'][-1]['qualified'],flush=True)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
