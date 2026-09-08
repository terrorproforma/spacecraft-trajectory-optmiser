from pathlib import Path
import fcntl,json,os,subprocess,time
root=Path('/home/angus/spacepdhcg-return-replay-v672');build=Path('/home/angus/spacepdhcg-joint-mesh-v662')
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(PYTHONPATH=str(build/'repo/src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(build/'build/cuda/libspacepdhcg_cuda.so'),SPACEPDHCG_QOCO_LIBRARY='/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=str(build/'build/cuda')+':/home/angus/build-qoco-scaled-pool-v540/final:/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
env.update(SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE='1',SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL='1')
cmd=['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(root/'ruiz5-run.py'),'--repo',str(build/'repo'),'--output',str(root/'ruiz5'),'--fixture',str(root/'ruiz5-fixture.json'),'--execute','--max-solves','3','--wall-seconds','600','--lock',str(root/'child.lock')]
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (root/'ruiz5.log').open('x') as log:r=subprocess.run(cmd,env=env,cwd=build/'repo',stdout=log,stderr=subprocess.STDOUT)
(root/'ruiz5-worker-report.json').write_text(json.dumps(dict(complete=True,returncode=r.returncode,command=cmd),indent=2))
