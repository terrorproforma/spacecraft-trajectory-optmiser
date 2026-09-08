from pathlib import Path
import subprocess,os,fcntl,json,time
root=Path('build/performance/pipeline-timers-v383');root.mkdir(exist_ok=False)
core='/home/angus/build-spacepdhcg-recovery-v380/final/libspacepdhcg_cuda.so';qoco='/home/angus/build-qoco-soc-step-v358/final/libqoco.so'
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(Path('src').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=core,SPACEPDHCG_QOCO_LIBRARY=qoco,SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',LD_LIBRARY_PATH=str(Path(core).parent)+':'+str(Path(qoco).parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/time_pipeline_v383.py',run_name='__main__')"
cmd=['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot,'gtoc12','run','--run-id','pipeline_timers_383','--output',str(root/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.perf_counter()
 with (root/'campaign.log').open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 (root/'report.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.perf_counter()-start,command=cmd),indent=2));r.check_returncode()
 print((root/'timers.json').read_text()[:2400])
