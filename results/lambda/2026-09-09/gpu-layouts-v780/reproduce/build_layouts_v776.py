from pathlib import Path
import hashlib,io,json,shutil,subprocess,tarfile
root=Path('/home/angus/spacepdhcg-layouts-v776');root.mkdir(exist_ok=False)
repo=root/'repo';repo.mkdir()
base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
paths=['cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,*paths]))) as tar:tar.extractall(repo,filter='data')
owned=['cpp/cuda/src/gtoc12_joint.cu', 'cpp/cuda/src/gtoc12_joint_insertion.cuh', 'cpp/cuda/src/orbitweaver_gpu.cu', 'cpp/cuda/internal/gtoc12_joint_geometry.h', 'cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h', 'src/spacepdhcg/gtoc12/gpu_joint.py', 'src/spacepdhcg/gtoc12/gpu_joint_insertions.py', 'src/spacepdhcg/gtoc12/jointopt.py', 'tests/test_gtoc12_gpu_joint_insertions.py','cpp/cuda/src/gtoc12_joint_layout.cuh','src/spacepdhcg/gtoc12/gpu_joint_layouts.py','tests/test_gtoc12_gpu_joint_layouts.py']
for name in owned:
    target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
(root/'source-manifest.json').write_text(json.dumps(dict(base=base,files={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}),indent=2))
shutil.copyfile('build/performance/worker_layouts_v776.py',root/'worker.py')
shutil.copytree('results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources',root/'incumbents')
with (root/'worker.log').open('x') as log:
    child=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid,base=base)))
