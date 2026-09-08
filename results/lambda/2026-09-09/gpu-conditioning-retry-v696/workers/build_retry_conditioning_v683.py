from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

root=Path('/home/angus/spacepdhcg-retry-conditioning-v683'); root.mkdir(exist_ok=False)
repo=root/'repo';repo.mkdir()
base='e7da7d9e7a25aa4a8309f5c80e516f0d4852dd8a'
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,'cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party']))) as archive:
    archive.extractall(repo,filter='data')
names=['cpp/cuda/internal/native_qoco_adapter.h','cpp/cuda/internal/gtoc12_qoco_graph.h','cpp/cuda/src/native_qoco_adapter.cpp','cpp/cuda/src/gtoc12_qoco.cu','cpp/cuda/src/gtoc12_scvx.cu','scripts/gpu/prepare_qoco_retry_conditioning.py']
for name in names:shutil.copyfile(name,repo/name)
source=root/'qoco'
shutil.copytree('/home/angus/build-qoco-scaled-pool-v540/source',source)
report=dict(complete=False,success=False,base_commit=base,source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names},stages=[])
def save(): (root/'report.json').write_text(json.dumps(report,indent=2))
def run(name,cmd,cwd=repo):
    started=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        r=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    report['stages'].append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.perf_counter()-started));save();r.check_returncode()
save()
try:
    run('prepare',[sys.executable,str(repo/'scripts/gpu/prepare_qoco_retry_conditioning.py'),'--destination',str(source)])
    run('git-init',['git','init']);run('git-add',['git','add','-f','.']);run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze conditional GPU conditioning retry'])
    cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
    cudss='/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12'
    run('qoco-configure',[cmake,'-S',str(source),'-B',str(root/'qoco-build'),'-G','Ninja','-DQOCO_ALGEBRA_BACKEND=cuda','-DQOCO_BUILD_TYPE=Release','-DCMAKE_BUILD_TYPE=Release','-DBUILD_QOCO_DEMO=OFF','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_CUDA_FLAGS=--default-stream per-thread -I'+cudss+'/include','-DCUDSS_LIB='+cudss+'/lib/libcudss.so'])
    run('qoco-build',[cmake,'--build',str(root/'qoco-build'),'-j3'])
    run('core-configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'core-build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg'])
    run('core-build',[cmake,'--build',str(root/'core-build'),'--target','spacepdhcg_cuda','-j3'])
    final=root/'final';final.mkdir()
    for p in [root/'qoco-build/libqoco.so',root/'core-build/cuda/libspacepdhcg_cuda.so']:
        shutil.copyfile(p,final/p.name)
    report['libraries']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in final.iterdir()};report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save();print(json.dumps(report))
