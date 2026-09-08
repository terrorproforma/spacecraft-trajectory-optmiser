from pathlib import Path
import hashlib,io,json,shutil,subprocess,tarfile,time,traceback
root=Path('/home/angus/spacepdhcg-joint-mesh-v662');root.mkdir(exist_ok=False)
repo=root/'repo';repo.mkdir()
base='2cbdd4255382f63deb598fc3b02095064b495799'
paths=['cpp','src','tests','benchmarks','pyproject.toml','third_party','results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources','results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,*paths]))) as tar:tar.extractall(repo,filter='data')
names=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','cpp/cuda/internal/gtoc12_joint_geometry.h','cpp/cuda/src/gtoc12_joint.cu','cpp/cuda/src/orbitweaver_gpu.cu','src/spacepdhcg/gtoc12/gpu_joint.py','tests/test_gtoc12_gpu_joint_geometry.py','src/spacepdhcg/gtoc12/jointopt.py','tests/test_gtoc12_gpu_joint_mesh.py']
fixture='results/gtoc12/runs/return_sweep_v2/ships/cluster_fleet_v7_clusters_family_0007/ship_01/route_summary.json'
for name in [*names,fixture]:
    (repo/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,repo/name)
r=dict(complete=False,success=False,base_commit=base,source_sha256={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in [*names,fixture]},stages=[])
def run(name,cmd):
    start=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:result=subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    r['stages'].append(dict(name=name,command=cmd,returncode=result.returncode,seconds=time.perf_counter()-start));assert result.returncode==0,name
try:
    run('git-init',['git','init']);run('git-add',['git','add','-f','.'])
    run('git-freeze',['git','-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze joint resident geometry'])
    cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
    run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg'])
    run('build',[cmake,'--build',str(root/'build'),'--target','spacepdhcg_cuda','-j3'])
    r['core_sha256']=hashlib.sha256((root/'build/cuda/libspacepdhcg_cuda.so').read_bytes()).hexdigest();r['success']=True
except BaseException:r['exception']=traceback.format_exc()
r['complete']=True;(root/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
