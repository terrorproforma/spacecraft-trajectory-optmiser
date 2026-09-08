from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

out=Path('build/performance/retrieved-retry-v696');out.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as file:file.write(Path('traj-key.pem').read_bytes())
for name in ['retry-conditioning-v696-h100.tar.gz','retry-conditioning-v696-h100.tar.manifest.json','retry-conditioning-v696-h100.tar.record.json']:
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/'+name,str(out/name)],check=True,timeout=55)
archive=out/'retry-conditioning-v696-h100.tar.gz'
record=json.loads(archive.with_suffix('.record.json').read_text());manifest=json.loads(archive.with_suffix('.manifest.json').read_text())
assert archive.stat().st_size==record['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==record['sha256']
with tarfile.open(archive) as tar:
    for member in tar:
        assert member.isfile() and member.name in manifest
        raw=tar.extractfile(member).read();expected=manifest[member.name]
        assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],member.name
print(json.dumps(dict(verified_h100_archive=record)))
