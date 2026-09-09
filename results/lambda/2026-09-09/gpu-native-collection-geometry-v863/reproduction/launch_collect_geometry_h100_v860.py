from pathlib import Path
import json, subprocess, tarfile

root = Path.home() / 'spacepdhcg-collect-geometry-v860'
archive = root / 'source.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(root / 'repo', arcname='repo', filter=lambda info: None if '/.git' in info.name and '/.gitignore' not in info.name else info)
    for name in ('worker.py', 'source-manifest.json'): tar.add(root / name, arcname=name)
key = '/home/angus/.ssh/spacepdhcg-expansion-key.pem'
host = 'ubuntu@192.222.55.229'
subprocess.run(['scp', '-q', '-i', key, str(archive), host + ':/tmp/collect-geometry-source-v860.tar.gz'], check=True, timeout=55)
code = """from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.home()/'spacepdhcg-collect-geometry-v860';root.mkdir()
with tarfile.open('/tmp/collect-geometry-source-v860.tar.gz') as tar:tar.extractall(root,filter='data')
for name,h in json.loads((root/'source-manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'repo'/name).read_bytes()).hexdigest()==h,name
with (root/'worker.log').open('x') as log:
    child=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid)))
"""
(Path('build/performance') / 'launch_collect_geometry_remote_v860.py').write_text(code)
result = subprocess.run(['ssh', '-i', key, '-o', 'BatchMode=yes', host, 'python3 -'], input=code, text=True, capture_output=True, check=True, timeout=40)
print(result.stdout, result.stderr)
