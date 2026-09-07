from pathlib import Path
import os,subprocess,json,time,tarfile,shutil,hashlib,fcntl
root=Path('/home/ubuntu/spacepdhcg-retime-graph-v238');repo=root/'repo'
report=dict(pid=os.getpid(),start=time.time(),steps=[],complete=False)
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
python='/home/ubuntu/spacepdhcg/v1/.venv/bin/python';cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake'
def run(name,cmd):
    start=time.time()
    with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    report['steps'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.time()-start));(root/'report.json').write_text(json.dumps(report,indent=2));r.check_returncode()
try:
    shutil.copytree('/home/ubuntu/spacepdhcg-sweeps-v234/repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'),symlinks=True)
    with tarfile.open('/tmp/retime-graph-v237.tar.gz') as t:t.extractall(repo,filter='data')
    for name,digest in json.loads((repo/'source-sha256.json').read_text()).items():assert hashlib.sha256((repo/name).read_text().encode()).hexdigest()==digest,name
    run('git-init',['git','init','-q'])
    run('git-add',['git','add','-f','cpp','src','tests','scripts','pyproject.toml'])
    run('git-snapshot',['git','-c','user.name=GPU snapshot','-c','user.email=snapshot@localhost','commit','-qm','CUDA graph replay validation source'])
    run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'core-build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=90','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/ubuntu/spacepdhcg/v1/_upstream/pdhcg'])
    run('build',[cmake,'--build',str(root/'core-build'),'--target','spacepdhcg_cuda','-j','3'])
    core=root/'core-build/cuda/libspacepdhcg_cuda.so'
    env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data')
    run('benchmark',[python,'benchmark_retime_graph_v236.py',str(root/'measurement'),'/home/ubuntu/spacepdhcg-native-campaign-v209/output/ship_01/refinements.json','input-return.json'])
    lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_module('pytest',run_name='__main__')"
    run('tests',[python,'-c',boot,'tests/test_gtoc12_retime_graph.py','tests/test_gtoc12_gpu_return_sweep.py','tests/test_gtoc12_gpu_retime.py','tests/test_gtoc12_resident_retime.py','tests/test_gtoc12_certified_objective.py','tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_gpu_lambert.py','tests/test_gtoc12_gpu_neighbours.py','tests/test_gtoc12_gpu_collection.py','tests/test_gtoc12_jointopt.py','tests/test_gtoc12_returns.py','tests/test_gtoc12_cooperative.py','-q'])
    for tool in ['memcheck','initcheck']:
        run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',python,'-c',boot,'tests/test_gtoc12_retime_graph.py','tests/test_gtoc12_gpu_return_sweep.py','tests/test_gtoc12_gpu_retime.py','tests/test_gtoc12_resident_retime.py','tests/test_gtoc12_certified_objective.py','-q'])
    report['runtime_sha256']=hashlib.sha256(core.read_bytes()).hexdigest();report['complete']=True
except Exception as e:report['error']=str(e)
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2))
