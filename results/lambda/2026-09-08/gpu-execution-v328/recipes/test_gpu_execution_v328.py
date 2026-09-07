from pathlib import Path
import os,subprocess,json,fcntl,time
root=Path('build/performance/gpu-execution-v328');root.mkdir(exist_ok=False)
base=Path('/home/angus/build-spacepdhcg-conic-retry-v313/final')
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(SPACEPDHCG_QOCO_LIBRARY='/home/angus/build-qoco-gpu-device-ir-v137/final/libqoco.so',LD_LIBRARY_PATH=str(base)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',PYTHONPATH='src',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(base/'libspacepdhcg_cuda.so'),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
cmd=['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-m','pytest','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_gpu_scvx.py','-q']
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.perf_counter();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
 (root/'pytest.log').write_text(r.stdout+r.stderr)
 (root/'report.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start),indent=2))
 print(r.stdout,r.stderr);r.check_returncode()
