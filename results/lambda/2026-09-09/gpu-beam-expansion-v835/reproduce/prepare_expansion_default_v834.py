from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
home=Path.home();build=home/'spacepdhcg-expansion-v830';root=home/'spacepdhcg-expansion-default-v834';root.mkdir()
assert json.loads((build/'report.json').read_text())['complete']
shutil.copytree(build/'repo',root/'repo',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
changed=['src/spacepdhcg/gtoc12/gpu_expansion.py','tests/test_gtoc12_gpu_expansion.py']
for name in changed:shutil.copyfile(name,root/'repo'/name)
manifest=dict(base='3404e926355beb3383f391c3a4bb80ce3d618afa',native_origin='v830; exact C++ unchanged',files={p.relative_to(root/'repo').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'repo').rglob('*') if p.is_file()})
old=json.loads((build/'source-manifest.json').read_text())['files']
assert all(manifest['files'][p]==h for p,h in old.items() if p.startswith('cpp/'))
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'final').mkdir();shutil.copyfile(build/'final/libspacepdhcg_cuda.so',root/'final/libspacepdhcg_cuda.so')
worker=(build/'worker.py').read_text()
begin=worker.index('    for name,command in ');end=worker.index('    report[\'core_sha256\']',begin)
worker=worker[:begin]+worker[end:]
worker=worker.replace("root/'build/cuda-tests/gtoc12_scvx_test'", "home/'spacepdhcg-expansion-v830/build/cuda-tests/gtoc12_scvx_test'")
(root/'worker.py').write_text(worker)
archive=root/'overlay.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for name in changed:tar.add(root/'repo'/name,arcname='repo/'+name)
    for name in ('worker.py','source-manifest.json'):tar.add(root/name,arcname=name)
subprocess.run(['scp','-q','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem',str(archive),'ubuntu@192.222.55.229:/tmp/expansion-default-v834.tar.gz'],check=True,timeout=55)
remote="""from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
home=Path.home();build=home/'spacepdhcg-expansion-v830';root=home/'spacepdhcg-expansion-default-v834';root.mkdir()
assert json.loads((build/'report.json').read_text())['complete']
shutil.copytree(build/'repo',root/'repo',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
with tarfile.open('/tmp/expansion-default-v834.tar.gz') as tar:tar.extractall(root,filter='data')
current=json.loads((root/'source-manifest.json').read_text())['files'];old=json.loads((build/'source-manifest.json').read_text())['files']
for p,h in current.items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
assert all(current[p]==h for p,h in old.items() if p.startswith('cpp/'))
(root/'final').mkdir();shutil.copyfile(build/'final/libspacepdhcg_cuda.so',root/'final/libspacepdhcg_cuda.so')
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
"""
Path('build/performance/launch_expansion_default_h100_v834.py').write_text(remote)
r=subprocess.run(['ssh','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
