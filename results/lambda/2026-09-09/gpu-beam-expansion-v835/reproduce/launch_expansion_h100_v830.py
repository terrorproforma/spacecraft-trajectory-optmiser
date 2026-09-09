from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-expansion-v830';root.mkdir()
with tarfile.open('/tmp/expansion-source-v830.tar.gz') as tar:tar.extractall(root,filter='data')
for p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
