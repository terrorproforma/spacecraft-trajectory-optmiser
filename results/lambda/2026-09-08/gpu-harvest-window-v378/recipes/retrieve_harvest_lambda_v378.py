from pathlib import Path
import os,subprocess,tarfile,json,hashlib
root=Path('results/lambda/2026-09-08/gpu-harvest-window-v378');root.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
archive=Path('build/performance/harvest-window-v378.tar.gz')
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-harvest-window-v378.tar.gz',str(archive)],check=True,timeout=180)
remote=root/'lambda';remote.mkdir()
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():
  target=(remote/m.name).resolve();assert target.is_relative_to(remote.resolve()) and m.isfile()
 t.extractall(remote)
manifest=json.loads((remote/'evidence-files.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((remote/name).read_bytes()).hexdigest()==sha,name
(root/'retrieval.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,verified_files=len(manifest)),indent=2));print((root/'retrieval.json').read_text())
