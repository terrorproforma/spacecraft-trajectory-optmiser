from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-native-collect-plan-v856';root.mkdir()
with tarfile.open('/tmp/native-collect-plan-source-v856.tar.gz') as tar:tar.extractall(root,filter='data')
for p,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/p).read_bytes()).hexdigest()==h,p
with (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
