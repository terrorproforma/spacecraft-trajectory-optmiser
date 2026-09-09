from pathlib import Path
import hashlib, io, json, os, shutil, subprocess, tarfile

root = Path.home() / 'spacepdhcg-resident-catalogue-v823'
root.mkdir(); repo = root / 'repo'; repo.mkdir()
base = '6c686cce48763a5ab34687a74d810f5ce693589b'
owned = ['cpp/cuda/src/gtoc12_completion_model.cuh',
         'cpp/cuda/include/spacepdhcg/cuda/gtoc12_completion_c_api.h',
         'src/spacepdhcg/gtoc12/gpu_completion_model.py',
         'tests/test_gtoc12_gpu_completion_model.py']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output([
    'git', 'archive', base, '.gitignore', 'cpp', 'src', 'tests', 'scripts',
    'benchmarks', 'pyproject.toml', 'third_party']))) as tar:
    tar.extractall(repo, filter='data')
for name in owned: shutil.copyfile(name, repo / name)
fixture = 'results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json'
(repo / fixture).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(fixture, repo / fixture)
manifest = dict(base=base, owned=owned, files={p.relative_to(repo).as_posix():
    hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()})
(root / 'source-manifest.json').write_text(json.dumps(manifest, indent=2))
worker = (Path.home() / 'spacepdhcg-catalogue-v808/worker.py').read_text()
worker = worker.replace('build/cuda/gtoc12_scvx_test', 'build/cuda-tests/gtoc12_scvx_test')
worker = worker.replace('Freeze integrated endpoint and catalogue fingerprint runtime',
                        'Freeze resident catalogue ownership runtime')
(root / 'worker.py').write_text(worker)
archive = Path('/tmp/resident-catalogue-v823.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(repo, arcname='repo')
    for name in ('worker.py', 'source-manifest.json'): tar.add(root / name, arcname=name)
key = Path('/tmp/traj-key.pem')
if not key.exists():
    fd = os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as out: out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp', '-q', '-i', str(key), str(archive),
               'ubuntu@192.222.55.229:/tmp/resident-catalogue-v823.tar.gz'], check=True, timeout=55)
remote = """from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-resident-catalogue-v823';root.mkdir()
with tarfile.open('/tmp/resident-catalogue-v823.tar.gz') as tar:tar.extractall(root,filter='data')
for p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
"""
Path('build/performance/launch_resident_catalogue_h100_v823.py').write_text(remote)
r = subprocess.run(['ssh', '-i', str(key), '-o', 'BatchMode=yes', 'ubuntu@192.222.55.229',
                    'python3 -'], input=remote, text=True, capture_output=True, check=True, timeout=40)
with (root / 'worker.log').open('x') as log:
    pid = subprocess.Popen(['python3', str(root / 'worker.py')], stdout=log,
                           stderr=subprocess.STDOUT, start_new_session=True).pid
print(json.dumps(dict(base=base, local_pid=pid, h100=r.stdout, stderr=r.stderr)))
