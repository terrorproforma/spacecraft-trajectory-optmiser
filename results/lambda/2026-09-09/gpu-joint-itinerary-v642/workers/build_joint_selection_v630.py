from pathlib import Path
import io,json,shutil,subprocess,tarfile,time,hashlib
root=Path('/home/angus/spacepdhcg-joint-selection-v630');root.mkdir(exist_ok=False)
repo=root/'repo';repo.mkdir()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'cpp','src','tests','benchmarks','pyproject.toml','third_party','results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json','results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources']))) as t:t.extractall(repo,filter='data')
names=['cpp/cuda/CMakeLists.txt','cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','cpp/cuda/src/gtoc12_joint.cu','cpp/cuda/tests/gtoc12_joint_smoke.cu','cpp/cuda/tests/gtoc12_joint_selection_test.cu','src/spacepdhcg/gtoc12/gpu_joint.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/jointopt.py','tests/test_gtoc12_gpu_joint.py','tests/test_gtoc12_gpu_joint_selection.py']
for name in names:shutil.copy2(name,repo/name)
report=dict(complete=False,source_commit=commit,source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names},stages=[])
def run(name,cmd):
 start=time.perf_counter()
 with (root/(name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 report['stages'].append(dict(name=name,code=r.returncode,seconds=time.perf_counter()-start,command=cmd));assert r.returncode==0,name
cmake='/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake'
try:
 run('git-init',['git','-C',str(repo),'init'])
 run('git-add',['git','-C',str(repo),'add','.'])
 run('git-freeze',['git','-C',str(repo),'-c','user.name=GPU validation','-c','user.email=gpu-validation@localhost','commit','-m','Freeze concurrent Lambert directions'])
 run('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(root/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DSPACEPDHCG_BUILD_CUDA=ON','-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120','-DSPACEPDHCG_PDHCG_SOURCE_ROOT=/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg'])
 run('build',[cmake,'--build',str(root/'build'),'--target','spacepdhcg_cuda','-j3'])
 report['core_sha256']=hashlib.sha256((root/'build/cuda/libspacepdhcg_cuda.so').read_bytes()).hexdigest()
 report['complete']=True
except Exception as e:report['error']=repr(e)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))





