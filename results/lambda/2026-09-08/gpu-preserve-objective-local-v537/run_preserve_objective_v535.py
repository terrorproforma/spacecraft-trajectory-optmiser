
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,time,shutil,sys
root=Path('/home/ubuntu/spacepdhcg-preserve-objective-v535');repo=root/'repo';source=root/'source'
report=dict(pid=os.getpid(),complete=False,stages=[])
def save():
 t=root/'report.tmp';t.write_text(json.dumps(report,indent=2));t.replace(root/'report.json')
def run(name,cmd):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=repo);report.update(stage=name,child_pid=child.pid);save();rc=child.wait(timeout=1800)
 report['stages'].append(dict(name=name,command=cmd,returncode=rc,seconds=time.perf_counter()-start));save();assert rc==0,(name,rc)
save()
try:
 shutil.copytree('/home/ubuntu/spacepdhcg-conditioning-v520/repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
 with tarfile.open('/tmp/preserve-objective-v535.tar.gz') as t:
  for m in t.getmembers():
   path=(repo/m.name).resolve();assert path.is_relative_to(repo) and m.isfile()
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(t.extractfile(m).read())
 manifest=json.loads((root/'source-sha256.json').read_text());assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
 shutil.copytree('/home/ubuntu/spacepdhcg-step-final-v359/qoco',source,ignore=shutil.ignore_patterns('.git','build','__pycache__'))
 sys.path.insert(0,str(repo/'scripts/gpu'))
 from prepare_qoco_preserve_objective import prepare
 report['patch']=prepare(source);save()
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake';runtime='/home/ubuntu/spacepdhcg-recovery-v152/cudss'
 flags=['-DQOCO_ALGEBRA_BACKEND=cuda','-DCMAKE_CUDA_ARCHITECTURES=90','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_BUILD_TYPE=Release','-DQOCO_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF','-DCUDSS_LIB='+runtime+'/lib/libcudss.so','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I'+runtime+'/include','-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration']
 run('configure',[cmake,'-S',str(source),'-B',str(root/'build'),*flags]);run('build',[cmake,'--build',str(root/'build'),'--target','qoco','-j','8'])
 libs=list((root/'build').rglob('libqoco.so'));assert len(libs)==1
 (root/'final').mkdir();shutil.copy2(libs[0],root/'final/libqoco.so')
 path=repo/'build/performance/test_preserve_objective_v536.py';s=path.read_text()
 s=s.replace('/home/angus/build-qoco-preserve-objective-v534','/home/ubuntu/spacepdhcg-preserve-objective-v535')
 s=s.replace('/home/angus/build-spacepdhcg-retained-replay-v512/final','/home/ubuntu/spacepdhcg-retained-replay-v514/core-build/cuda')
 s=s.replace('/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','/home/ubuntu/spacepdhcg/v1/.venv/bin/python')
 s=s.replace('/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data','/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data')
 s=s.replace('/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib',runtime+'/lib').replace('/usr/local/cuda-12.8','/usr/local/cuda').replace('sm_120','sm_90')
 s=s.replace('/home/angus/.spacepdhcg-gpu.lock','/home/ubuntu/.spacepdhcg-gpu.lock');path.write_text(s)
 report['executed_worker_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();save()
 run('validation',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(path)])
 r=json.loads((repo/'build/performance/preserve-objective-v536/report.json').read_text());assert r['complete'] and not r.get('error'),r
 report['complete']=True
except Exception as error:report['error']=repr(error)
save()
