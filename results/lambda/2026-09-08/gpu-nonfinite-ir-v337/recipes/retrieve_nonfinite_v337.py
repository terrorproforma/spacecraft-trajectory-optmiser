from pathlib import Path
import os,subprocess,tarfile,json,hashlib
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
root=Path('results/lambda/2026-09-08/gpu-nonfinite-ir-v337');out=root/'remote';out.mkdir(exist_ok=False)
archive=Path('build/performance/nonfinite-ir-v336/lambda-results.tar.gz')
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/tmp/nonfinite-ir-v337-results.tar.gz',str(archive)],check=True)
h=hashlib.sha256(archive.read_bytes()).hexdigest();assert h=='dbb73437b54aa8eab17037ad27dc4e346f6cb99f3b2936749606cabca719ae70'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  p=(out/m.name).resolve();assert p.is_relative_to(out.resolve()) and m.isfile()
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
manifest=json.loads((out/'remote-sha256.json').read_text());assert all(hashlib.sha256((out/p).read_bytes()).hexdigest()==v for p,v in manifest.items())
(root/'retrieval.json').write_text(json.dumps(dict(archive_sha256=h,verified_files=len(manifest)),indent=2));print('verified',len(manifest),'remote files')
