from pathlib import Path
import subprocess,os,fcntl,json,time,hashlib
root=Path('/home/ubuntu/spacepdhcg-preserve-campaign-v538')
core='/home/ubuntu/spacepdhcg-retained-replay-v514/core-build/cuda/libspacepdhcg_cuda.so';qoco='/home/ubuntu/spacepdhcg-preserve-objective-v535/final/libqoco.so'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=core,SPACEPDHCG_QOCO_LIBRARY=qoco,SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(Path(core).parent)+':'+str(Path(qoco).parent)+':/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib:/usr/local/cuda/lib64')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')"
cmd=['/home/ubuntu/spacepdhcg/v1/.venv/bin/python','-c',boot,'gtoc12','run','--run-id','preserve537','--output',str(root/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','900','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
r=dict(source_commit='71fb2b51b798ff0cb8b2b9002f5ed9a056f1de20',source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/spacepdhcg/gtoc12/search.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/gpu_beam.py','src/spacepdhcg/gtoc12/gpu_options.py','cpp/cuda/src/native_qoco_adapter.cpp','build/performance/solver_phase_details.py']},pid=os.getpid(),complete=False,command=cmd,core_sha256=hashlib.sha256(Path(core).read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(Path(qoco).read_bytes()).hexdigest())
def save():
 temp=root/'report.tmp';temp.write_text(json.dumps(r,indent=2));temp.replace(root/'report.json')
save()
try:
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'
  env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1'
  env['SPACEPDHCG_GTOC12_GPU_TESTS']='1'
  test_command=['/home/ubuntu/spacepdhcg/v1/.venv/bin/python', '-c', "import sys;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];import pytest;sys.exit(pytest.main(sys.argv[1:]))", 'tests/test_gtoc12_gpu_retained_replay.py', 'tests/test_gtoc12_gpu_workspace_pool.py', 'tests/test_gtoc12_verifier_knots.py', 'tests/test_gtoc12_verifier.py', 'tests/test_gtoc12_gpu_verifier.py', 'tests/test_gtoc12_gpu_resident_options.py', 'tests/test_gtoc12_gpu_collection.py', 'tests/test_gtoc12_gpu_elements.py', 'tests/test_gtoc12_gpu_scvx.py', 'tests/test_gtoc12_gpu_cli.py', 'tests/test_gtoc12_run_final_verification.py', '-x', '-s', '-q']
  with (root/'pytest.log').open('x') as log:
   child=subprocess.Popen(test_command,env=env,stdout=log,stderr=subprocess.STDOUT);r.update(stage='pytest',child_pid=child.pid);save();code=child.wait(timeout=1200)
  r['pytest']=dict(command=test_command,returncode=code);save();assert code==0
  r['campaigns']=[]
  for name,enabled in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:
   env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='0'
   env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1' if enabled else '0'
   env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'
   env['SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE']='1'
   (root/name).mkdir()
   env['SPACEPDHCG_PHASE_OUTPUT']=str(root/name)
   command=cmd.copy();command[command.index('--output')+1]=str(root/name/'output');command+=['--qoco-ruiz-iterations','2' if enabled else '0']
   start=time.perf_counter()
   with (root/(name+'.log')).open('x') as log:
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);r.update(child_pid=child.pid,stage=name);save();code=child.wait(timeout=900)
   seconds=time.perf_counter()-start
   assert code==0,(name,code)
   result=json.loads((root/name/'output/run_report.json').read_text());assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
   screening=result['screening'];assert screening.get('resident_option_builds',0)>0
   if enabled:assert screening.get('resident_option_read_bytes',0)==0 and screening['collection_option_upload_bytes']==0
   r['campaigns'].append(dict(name=name,candidate=enabled,process_seconds=seconds,cli_seconds=result['wall_seconds_total'],score=result['best']['independent']['weighted_score_fixed_bonus_kg'],screening=screening));save()
  r['complete']=True
except Exception as e:r['error']=repr(e)
save();print(json.dumps(r,indent=2))
