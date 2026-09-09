from pathlib import Path
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
