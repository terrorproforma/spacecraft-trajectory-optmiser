from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
root=Path('/home/angus/spacepdhcg-joint-geometry-v651');source=root/'repo';out=root/'benchmark-v653';out.mkdir(exist_ok=False)
helper=Path('build/performance/joint-benchmark-v593/benchmark_joint.py')
(source/helper).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(helper,source/helper)
script=Path('build/performance/benchmark_joint_selection_v634.py').read_text().replace('SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION','SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY')
script=script.replace('Matched full-download / device-winner timing','Matched staged / resident geometry timing')
(out/'benchmark.py').write_text(script)
library=root/'build/cuda/libspacepdhcg_cuda.so'
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(PYTHONPATH=str(source/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv="+repr([str(out/'benchmark.py'),str(out/'benchmark.json')])+";runpy.run_path(sys.argv[0],run_name='__main__')"
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    start=time.perf_counter()
    with (out/'benchmark.log').open('x') as log:r=subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-c',boot],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    report=dict(complete=True,returncode=r.returncode,seconds=time.perf_counter()-start,core_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2));print((out/'benchmark.log').read_text());raise SystemExit(r.returncode)
