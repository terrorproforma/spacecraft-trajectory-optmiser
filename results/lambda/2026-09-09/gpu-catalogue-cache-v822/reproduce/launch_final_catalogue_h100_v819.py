from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
home=Path.home();root=home/'spacepdhcg-catalogue-final-v819';root.mkdir()
shutil.copytree(home/'spacepdhcg-catalogue-v808/repo',root/'repo',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
with tarfile.open('/tmp/catalogue-final-overlay-v819.tar.gz') as tar:tar.extractall(root,filter='data')
for p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
(root/'final').mkdir();shutil.copyfile(home/'spacepdhcg-catalogue-v808/final/libspacepdhcg_cuda.so',root/'final/libspacepdhcg_cuda.so')
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)