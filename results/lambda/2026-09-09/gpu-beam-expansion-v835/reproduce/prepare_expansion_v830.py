from pathlib import Path
import hashlib,io,json,shutil,subprocess,tarfile
home=Path.home();root=home/'spacepdhcg-expansion-v830';root.mkdir();repo=root/'repo';repo.mkdir()
base='3404e926355beb3383f391c3a4bb80ce3d618afa'
owned=['cpp/cuda/CMakeLists.txt','cpp/cuda/include/spacepdhcg/cuda/gtoc12_expansion_c_api.h','cpp/cuda/src/gtoc12_expansion.cu','src/spacepdhcg/gtoc12/gpu_expansion.py','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_expansion.py']
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,'.gitignore','cpp','src','tests','scripts','benchmarks','pyproject.toml','third_party']))) as tar:tar.extractall(repo,filter='data')
for name in owned:shutil.copyfile(name,repo/name)
fixture='results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json';(repo/fixture).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(fixture,repo/fixture)
(root/'source-manifest.json').write_text(json.dumps(dict(base=base,owned=owned,files={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file()}),indent=2))
worker=(home/'spacepdhcg-resident-catalogue-v823/worker.py').read_text()
worker=worker.replace("        run('pytest',[py,'-c',boot,'-q',*tests])", "        tests += ['tests/test_gtoc12_gpu_expansion.py']\n        run('pytest',[py,'-c',boot,'-q',*tests])")
worker=worker.replace("'tests/test_gtoc12_gpu_completion_model.py'])", "'tests/test_gtoc12_gpu_completion_model.py','tests/test_gtoc12_gpu_expansion.py'])")
(root/'worker.py').write_text(worker)
archive=Path('/tmp/expansion-source-v830.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    tar.add(repo,arcname='repo')
    for name in ('worker.py','source-manifest.json'):tar.add(root/name,arcname=name)
subprocess.run(['scp','-q','-i','/tmp/traj-key.pem',str(archive),'ubuntu@192.222.55.229:/tmp/expansion-source-v830.tar.gz'],check=True,timeout=55)
remote="""from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-expansion-v830';root.mkdir()
with tarfile.open('/tmp/expansion-source-v830.tar.gz') as tar:tar.extractall(root,filter='data')
for p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
"""
Path('build/performance/launch_expansion_h100_v830.py').write_text(remote)
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
