from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-collect-geometry-v860';root.mkdir()
with tarfile.open('/tmp/collect-geometry-source-v860.tar.gz') as tar:tar.extractall(root,filter='data')
for name,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/name).read_bytes()).hexdigest()==h,name
with (root/'worker.log').open('x') as log:
    child=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid)))
