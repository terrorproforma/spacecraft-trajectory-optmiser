from pathlib import Path
import os,subprocess,shutil,tarfile,json,hashlib,ast,time,fcntl,sys
root=Path('/home/ubuntu/spacepdhcg-nonfinite-ir-v337');source=root/'qoco'
report=dict(pid=os.getpid(),complete=False,stages=[],replays=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 with tarfile.open('/tmp/nonfinite-ir-v336.tar.gz') as t:
  for m in t.getmembers():
   p=(root/'overlay'/m.name).resolve();assert p.is_relative_to((root/'overlay').resolve()) and m.isfile()
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
 shutil.copytree('/home/ubuntu/spacepdhcg-diagnose-v168/qoco',source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))
 runtime_header=source/'algebra/cuda/qoco_ir_runtime.cuh';text=runtime_header.read_text()
 for old,new in [('cudaGraphSetConditional(handle, maximum > 0 && !(*norm < tolerance));','cudaGraphSetConditional(handle, maximum > 0 && isfinite(*norm) && isfinite(tolerance) && !(*norm < tolerance));'),('if (*norm >= state->best) {','if (!isfinite(*norm) || *norm >= state->best) {')]:
  assert text.count(old)==1;text=text.replace(old,new)
 runtime_header.write_text(text)
 backend=source/'algebra/cuda/cudss_backend.cu';text=backend.read_text()
 for old,new in [('if (res < ir_tol) {','if (!isfinite(res) || !isfinite(ir_tol) || res < ir_tol) {'),('if (new_res >= best_res) {','if (!isfinite(new_res) || new_res >= best_res) {')]:
  assert text.count(old)==1;text=text.replace(old,new)
 backend.write_text(text)
 report['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [runtime_header,backend,*[p for p in (root/'overlay').rglob('*') if p.is_file()]]};save()
 for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake';python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
 env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
 env.update(LD_LIBRARY_PATH=str(root/'final')+':'+runtime+'/lib:/usr/local/cuda/lib64',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 def run(name,cmd,timeout=180,cwd=root):
  start=time.perf_counter()
  with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
  report['stages'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));save();print(name,r.returncode,flush=True);r.check_returncode()
 run('configure',[cmake,'-S',str(source),'-B',str(root/'build'),'-DQOCO_ALGEBRA_BACKEND=cuda','-DCMAKE_CUDA_ARCHITECTURES=90','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_BUILD_TYPE=Release','-DQOCO_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF','-DCUDSS_LIB='+runtime+'/lib/libcudss.so','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I'+runtime+'/include','-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration'])
 run('build',[cmake,'--build',str(root/'build'),'--target','qoco','-j','3'],600)
 libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
 (root/'final').mkdir();lib=root/'final/libqoco.so';shutil.copy2(libs[0],lib)
 report['library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest();save()
 exe=root/'final/qoco_ir_control_probe'
 run('control-build',['/usr/local/cuda/bin/nvcc','-O2','-std=c++17','-arch=sm_90','--default-stream','per-thread',str(root/'overlay/cpp/cuda/tests/qoco_ir_control_probe.cu'),'-I'+str(root/'overlay/cpp/cuda/patches'),*['-I'+str(source/p) for p in ['algebra/cuda','include','lib/qdldl/include']],'-I'+runtime+'/include','-L'+str(lib.parent),'-lqoco','-ldl','-o',str(exe)])
 repo=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/repo')
 env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_QOCO_LIBRARY=str(lib),SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/ubuntu/spacepdhcg-conic-retry-v314/core-build/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data')
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run('control',[str(exe)])
  for tool in ['memcheck','initcheck','racecheck','synccheck']:run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(exe)])
  boot="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))"
  run('pytest',[python,'-c',boot,'tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','-q'],180,repo)
  # Audit via the pinned venv; no scientific dependencies assumed in system Python.
  replay_source=Path('/home/ubuntu/spacepdhcg-qp-graph-v325/run.py')
  if replay_source.exists():report['previous_replay_recipe']=str(replay_source)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
