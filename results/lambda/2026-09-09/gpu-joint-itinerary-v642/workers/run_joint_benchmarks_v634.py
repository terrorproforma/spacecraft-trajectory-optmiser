from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
build=Path('/home/angus/spacepdhcg-joint-selection-v630');source=build/'repo'
root=build/'benchmark-v634';root.mkdir(exist_ok=False)
python='/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
library=build/'build/cuda/libspacepdhcg_cuda.so'
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(library),SPACEPDHCG_GTOC12_DATA='/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data',PYTHONPATH=str(source/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
script=Path('build/performance/benchmark_joint_selection_v634.py')
shutil.copy2(script,source/script)
report=dict(complete=False,core_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),checks=[])
commands=[('scalar-vs-batched',str(source/'build/performance/joint-benchmark-v593/benchmark_joint.py'),[str(root/'scalar-vs-batched.json'),'--source-root',str(source),'--archive-dir',str(source/'results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources'),'--repeats','3']),('selection',str(source/script),[str(root/'selection.json')])]
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for name,path,args in commands:
        boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.argv="+repr([path,*args])+";runpy.run_path(sys.argv[0],run_name='__main__')"
        start=time.perf_counter()
        with (root/(name+'.log')).open('x') as log:result=subprocess.run([python,'-c',boot],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['checks'].append(dict(name=name,returncode=result.returncode,seconds=time.perf_counter()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
        print(name,result.returncode,flush=True)
        if result.returncode:print((root/(name+'.log')).read_text());raise SystemExit(result.returncode)
report['complete']=True
(root/'report.json').write_text(json.dumps(report,indent=2))
