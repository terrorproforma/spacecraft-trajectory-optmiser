from pathlib import Path
import os,subprocess,json,fcntl,time
base=Path('/home/angus/build-spacepdhcg-conic-retry-v313/final')
root=Path('build/performance/conic-retry-v313');root.mkdir(exist_ok=False)
env=dict(os.environ,SPACEPDHCG_QOCO_LIBRARY='/home/angus/build-qoco-gpu-device-ir-v137/final/libqoco.so',LD_LIBRARY_PATH=str(base)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',PYTHONPATH='src',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(base/'libspacepdhcg_cuda.so'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
report=dict(pid=os.getpid(),complete=False,checks=[])
checks=[(n,[str(base/n)]) for n in ['gtoc12_scvx_test','gtoc12_outer_graph_test','gtoc12_graph_deadline_test','gtoc12_qoco_guard_test']]
checks+=[('pytest',['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-m','pytest','tests/test_gtoc12_gpu_scvx.py','-q'])]
with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,cmd in checks:
  start=time.monotonic();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
  (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr)
  report['checks'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.monotonic()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
  print(name,r.returncode,r.stdout[-700:],r.stderr[-500:],flush=True);r.check_returncode()
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
