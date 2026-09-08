from pathlib import Path
import os,subprocess,json,time,fcntl,ast,hashlib,shutil
root=Path('/home/ubuntu/spacepdhcg-step-v353/v356');integrated=Path('/home/ubuntu/spacepdhcg-conic-retry-v314');repo=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/repo')
report=dict(start=time.time(),pid=os.getpid(),source_base_commit='f07e82c5',source_changes=['resident float32 collection tables and device gather into DP', 'bounded cold retries before conic trust shrink'],complete=False,scope='One-ship full-catalogue-box search and top-three GPU refinements; four retiming/extension attempts with device pricing; not an incumbent fleet replacement.')
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
core=integrated/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-nonfinite-ir-v337/final/libqoco.so')
env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/usr/local/cuda/lib64')
for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):env['LD_LIBRARY_PATH']+=':'+ast.literal_eval(node.value)+'/lib'
python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('spacepdhcg',run_name='__main__')"
cmd=[python,'-c',boot,'gtoc12','run','--run-id','gpu_step_356','--output',str(root/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda']
cmd += ['--gpu-execution','graph']
report['execution_source_sha256']=json.loads((repo/'execution-source-sha256.json').read_text())
report['nonfinite_candidate']=True
report['qoco_guard_source_sha256']={'qoco/algebra/cuda/qoco_ir_runtime.cuh': '88ca3ab6430c7e5fad140101dd61419332661dad2eead7f4b6f09df61e6726e0', 'qoco/algebra/cuda/cudss_backend.cu': 'be30b4f3f600a4a184dd052d75c0cc3a85c9edc4211b0828375b1b55b1953e2a', 'overlay/audit.py': '7adb8ed1094f97f447cfcb689eeaccd00b8db760931b6c7d4322a5ab860151f4', 'overlay/scripts/gpu/prepare_qoco_device_ir.py': '9013a41f6592e0d25a569c6214dff0775b6d0ea604832cfeac49185445165ebe', 'overlay/cpp/cuda/tests/qoco_ir_control_probe.cu': '3a5f2572cd9272d5da3ee21c404a6d5425a7a297fa0ca9ceacb17349f62f4fd8', 'overlay/cpp/cuda/patches/qoco_ir_runtime.cuh': '54d7c5870e17f29dde341c3408f2a116969daeed652290700e2e75a4213116a1'}
report['step_candidate']=False
report['step_cone_source_sha256']=None
report['kernel_source_sha256']=json.loads((repo/'source-sha256.json').read_text())
if (repo/'retry-source-sha256.json').exists():report['kernel_source_sha256'].update(json.loads((repo/'retry-source-sha256.json').read_text()))
report['source_sha256']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ['src/spacepdhcg/gtoc12/collectdp.py','src/spacepdhcg/gtoc12/gpu_collect_dp.py']}
report.update(command=cmd,runtime_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]})
try:
 lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 assert json.loads((integrated/'report.json').read_text())['complete']
 preflight="import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table;from spacepdhcg.gtoc12.official import official_verifier_available;print(load_catalogue().source_sha256);load_bonus_table();assert official_verifier_available(),'official verifier unavailable';print('Pinned catalogue, bonus table and official checker ready')"
 r=subprocess.run([python,'-c',preflight],cwd=repo,env=env,text=True,capture_output=True,timeout=60)
 (root/'preflight.log').write_text(r.stdout+r.stderr);r.check_returncode()
 (root/'report.json').write_text(json.dumps(report,indent=2))
 with (root/'campaign.log').open('x') as log:
  child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  report['child_pid']=child.pid;(root/'report.json').write_text(json.dumps(report,indent=2))
  try:report['returncode']=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   child.kill();child.wait();report['returncode']=124;report['error']='1800-second execution ceiling reached; partial artifacts retained'
 report['complete']=True
except Exception as e:report['error']=str(e)
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
