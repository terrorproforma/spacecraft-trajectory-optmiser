from pathlib import Path
import hashlib,json,os,subprocess,tarfile
root=Path('results/lambda/2026-09-08/gpu-solver-phase-v459')
root.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
archive=root/'raw.tar.gz'
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-solver-phase-v459.tar.gz',str(archive)],check=True,timeout=180)
expected=json.loads(Path('build/performance/archive-solver-phase-v459.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['sha256']
local=Path('build/performance/retrieved-solver-phase-v459');local.mkdir(exist_ok=False)
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():
  assert (local/m.name).resolve().is_relative_to(local.resolve()) and m.isfile()
 t.extractall(local)
manifest=json.loads((local/'files-sha256.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((local/name).read_bytes()).hexdigest()==sha,name
(root/'archive-manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'retrieval.json').write_text(json.dumps(dict(**expected,verified_files=len(manifest)),indent=2))
print(json.dumps(dict(verified_files=len(manifest),bytes=archive.stat().st_size)))
