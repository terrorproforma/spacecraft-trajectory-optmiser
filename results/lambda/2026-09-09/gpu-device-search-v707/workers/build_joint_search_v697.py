from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import traceback

root=Path('/home/angus/spacepdhcg-joint-search-v697');root.mkdir(exist_ok=False)
repo=root/'repo';repo.mkdir()
base='3175fb4497922bb4ada2f236c8f0d562fc22a4e4'
paths=['cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party','results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,*paths]))) as archive:
    archive.extractall(repo,filter='data')
names=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','cpp/cuda/src/gtoc12_joint.cu','src/spacepdhcg/gtoc12/gpu_joint.py','src/spacepdhcg/gtoc12/jointopt.py','tests/test_gtoc12_gpu_joint_search.py']
for name in names:shutil.copyfile(name,repo/name)
report=dict(complete=False,success=False,base=base,source_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in names},stages=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
def run(name,cmd,cwd=repo):
    started=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    report['stages'].append(dict(name=name,returncode=r.returncode,seconds=time.perf_counter()-started,command=cmd));save();r.check_returncode()
save()
try:
    run('git-init',['git','init']);run('git-add',['git','add','-f','.']);run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze device joint search'])
    cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
    run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg'])
    run('build',[cmake,'--build',str(root/'build'),'--target','spacepdhcg_cuda','-j3'])
    final=root/'final';final.mkdir();shutil.copyfile(root/'build/cuda/libspacepdhcg_cuda.so',final/'libspacepdhcg_cuda.so')
    report['libraries']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in final.iterdir()};report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save();print(json.dumps(report),flush=True)
