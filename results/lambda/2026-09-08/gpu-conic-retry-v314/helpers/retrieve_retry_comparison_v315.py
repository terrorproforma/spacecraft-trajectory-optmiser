from pathlib import Path
import subprocess,tarfile,hashlib,os
out=Path('results/lambda/2026-09-08/gpu-conic-retry-v314')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
archive=out/'campaign-comparison.tar.gz'
assert not archive.exists()
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-retry-comparison-v315/evidence.tar.gz',str(archive)],check=True,timeout=55)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='baa27bf92665c8130dca709ce96b878906d6766b0b6fcb7a8eb2fe689980b05d'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  if m.isdir():continue
  path=(out/m.name).resolve();assert path.is_relative_to(out.resolve()) and m.isfile()
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(t.extractfile(m).read())
print('Downloaded and hash-verified all eight complete campaigns')
