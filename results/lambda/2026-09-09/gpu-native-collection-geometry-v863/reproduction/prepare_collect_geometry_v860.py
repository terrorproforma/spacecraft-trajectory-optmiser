from pathlib import Path
import hashlib, io, json, shutil, subprocess, tarfile

home = Path.home()
root = home / 'spacepdhcg-collect-geometry-v860'
root.mkdir()
repo = root / 'repo'
repo.mkdir()
base = 'bd664c279a90e30c914b38aef0b8f9df630be56e'
owned = ['cpp/cuda/include/spacepdhcg/cuda/gtoc12_collect_dp_c_api.h', 'cpp/cuda/src/gtoc12_collect_dp.cu', 'src/spacepdhcg/gtoc12/gpu_collect_dp.py', 'tests/test_gtoc12_gpu_collect_geometry.py']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', base, '.gitignore', 'cpp', 'src', 'tests', 'scripts', 'benchmarks', 'pyproject.toml', 'third_party']))) as tar:
    tar.extractall(repo, filter='data')
for name in owned: shutil.copyfile(name, repo / name)
fixture = 'results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json'
(repo / fixture).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(fixture, repo / fixture)
(root / 'source-manifest.json').write_text(json.dumps(dict(base=base, owned=owned, files={p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}), indent=2))
worker = (home / 'spacepdhcg-native-collect-plan-v856/worker.py').read_text()
worker = worker.replace('Freeze native mining and collection plan runtime', 'Freeze native collection geometry runtime')
worker = worker.replace("'tests/test_gtoc12_gpu_collect_plan.py'", "'tests/test_gtoc12_gpu_collect_plan.py','tests/test_gtoc12_gpu_collect_geometry.py'")
(root / 'worker.py').write_text(worker)
with (root / 'worker.log').open('x') as log:
    child = subprocess.Popen(['python3', str(root / 'worker.py')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(root=str(root), pid=child.pid)))
