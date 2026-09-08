from pathlib import Path
import hashlib,json,os,subprocess,tarfile
root=Path('/home/angus/spacepdhcg-joint-mesh-v662');p=Path('build/performance')
archive=Path('/tmp/joint-mesh-v666.tar.gz')
manifest={}
with tarfile.open(archive,'w:gz') as tar:
 for file in sorted((root/'repo').rglob('*')):
  if not file.is_file() or any(x in file.parts for x in ('.git','__pycache__','.pytest_cache')):continue
  name=file.relative_to(root/'repo').as_posix();manifest[name]=hashlib.sha256(file.read_bytes()).hexdigest();tar.add(file,arcname=name)
mapping={'/home/angus/spacepdhcg-joint-mesh-v662':'/home/ubuntu/spacepdhcg-joint-mesh-v666','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock','/usr/local/cuda-12.8':'/usr/local/cuda'}
validation=(p/'check_joint_mesh_v665.py').read_text()
benchmark=(p/'run_joint_mesh_benchmark_v664.py').read_text()
# The helper is already in the frozen archive; avoid copying it onto itself.
benchmark=benchmark.replace("(source/helper).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(helper,source/helper)","assert (source/helper).is_file()")
for old,new in mapping.items():validation=validation.replace(old,new);benchmark=benchmark.replace(old,new)
worker='''from pathlib import Path
import hashlib,json,subprocess,time,traceback
root=Path('/home/ubuntu/spacepdhcg-joint-mesh-v666');repo=root/'repo'
report=dict(complete=False,success=False,stages=[])
def run(name,cmd):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
 report['stages'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));(root/'report.json').write_text(json.dumps(report,indent=2))
 if r.returncode:raise RuntimeError(name)
try:
 run('git-init',['git','init']);run('git-add',['git','add','-f','.'])
 run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze resident geometry H100'])
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake'
 run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=90','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/ubuntu/spacepdhcg/v1/_upstream/pdhcg'])
 run('build',[cmake,'--build',str(root/'build'),'--target','spacepdhcg_cuda','-j3'])
 report['core_sha256']=hashlib.sha256((root/'build/cuda/libspacepdhcg_cuda.so').read_bytes()).hexdigest()
 run('validation',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'validation.py')])
 run('benchmark',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'benchmark.py')])
 report['success']=True
except BaseException:report['exception']=traceback.format_exc()
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
'''
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport hashlib,json,subprocess,tarfile\nroot=Path('/home/ubuntu/spacepdhcg-joint-mesh-v666');root.mkdir(exist_ok=False);repo=root/'repo';repo.mkdir()\n"
launch+='manifest='+repr(manifest)+'\n'
launch+="with tarfile.open('/tmp/joint-mesh-v666.tar.gz') as tar:\n for m in tar.getmembers():assert m.isfile() and m.name in manifest\n tar.extractall(repo,filter='data')\nfor name,digest in manifest.items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest\n(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))\n"
launch+="(repo/'build/performance/benchmark_joint_mesh_v664.py').write_text("+repr((p/'benchmark_joint_mesh_v664.py').read_text())+")\n"
for name,body in [('worker.py',worker),('validation.py',validation),('benchmark.py',benchmark)]:launch+="(root/"+repr(name)+").write_text("+repr(body)+")\n"
launch+="with (root/'worker.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'worker.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,source_files=len(manifest))))\n"
(p/'launch_mesh_lambda_v666.py').write_text(launch)
print(json.dumps(dict(archive_bytes=archive.stat().st_size,files=len(manifest))))
