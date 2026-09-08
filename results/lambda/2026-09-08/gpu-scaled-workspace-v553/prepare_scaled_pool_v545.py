from pathlib import Path
import hashlib,json,os,subprocess,tarfile
p=Path('build/performance');tag='scaled-pool-v545'
files=['scripts/gpu/prepare_qoco_preserve_objective.py','scripts/gpu/prepare_qoco_gpu.py','cpp/cuda/src/native_qoco_adapter.cpp','cpp/cuda/src/gtoc12_qoco.cu','cpp/cuda/src/gtoc12_scvx.cu','tests/test_gtoc12_gpu_scaled_workspace_pool.py']+['build/performance/'+n for n in ['validate_scaled_pool_v542.py','replay_scaled_pool.py','grid-cache-fleet-v403/scvx-calls.json','solver_phase_details.py']]
archive=Path('/tmp/'+tag+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in files:t.add(name,arcname=name)
manifest={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files}
s=(p/'run_preserve_objective_v535.py').read_text().replace('preserve-objective-v535','scaled-pool-v545')
s=s.replace("shutil.copytree('/home/ubuntu/spacepdhcg-conditioning-v520/repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))", "shutil.copytree('/home/ubuntu/spacepdhcg-preserve-objective-v535/repo',repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','build'))")
start=s.index(" path=repo/'build/performance/test_preserve_objective_v536.py'")
end=s.index(" report['complete']=True",start)
replacement=r'''
 run('git-init',['git','init'])
 run('git-add',['git','add','.'])
 run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze scaled workspace validation source'])
 report['frozen_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();save()
 core_flags=['-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=90','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/ubuntu/spacepdhcg/v1/_upstream/pdhcg']
 run('core-configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'core-build'),*core_flags])
 run('core-build',[cmake,'--build',str(root/'core-build'),'--target','spacepdhcg_cuda','-j','3'])
 path=repo/'build/performance/validate_scaled_pool_v542.py';worker=path.read_text()
 replacements={
 '/home/angus/build-spacepdhcg-scaled-pool-v539/final':str(root/'core-build/cuda'),
 '/home/angus/build-qoco-scaled-pool-v540/final':str(root/'final'),
 '/home/angus/build-qoco-preserve-objective-v534/final':'/home/ubuntu/spacepdhcg-preserve-objective-v535/final',
 '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python',
 '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',
 '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib':runtime+'/lib',
 '/usr/local/cuda-12.8':'/usr/local/cuda','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock'}
 for old,new in replacements.items():worker=worker.replace(old,new)
 path.write_text(worker);report['executed_worker_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();save()
 run('validation',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(path)])
 r=json.loads((repo/'build/performance/scaled-pool-v542/report.json').read_text());assert r['complete'] and not r.get('error'),r
'''
s=s[:start]+replacement+s[end:]
(p/'run_scaled_pool_v545.py').write_text(s)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-scaled-pool-v545');root.mkdir(exist_ok=False)\n"
launch+="(root/'source-sha256.json').write_text("+repr(json.dumps(manifest,indent=2))+")\n(root/'run.py').write_text("+repr(s)+")\n"
launch+="with (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_scaled_pool_v545.py').write_text(launch)
