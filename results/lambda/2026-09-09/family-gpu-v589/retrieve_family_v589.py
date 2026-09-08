from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

root = Path('results/lambda/2026-09-09/family-gpu-v589')
root.mkdir(parents=True, exist_ok=False)
metadata = json.loads(Path('build/performance/archive-family-v589.json').read_text())
archive = root / 'raw.tar.gz'
subprocess.run(['scp', '-i', '/tmp/traj-key.pem', '-o', 'BatchMode=yes',
                'ubuntu@192.222.55.229:' + metadata['path'], str(archive)],
               check=True, timeout=180)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata['sha256']
with tarfile.open(archive, 'r:gz') as stream:
    for member in stream.getmembers():
        assert member.isfile() and (root / member.name).resolve().is_relative_to(root.resolve())
    stream.extractall(root)
manifest = json.loads((root / 'files-sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
(root / 'retrieval.json').write_text(json.dumps(dict(**metadata, verified_files=len(manifest)), indent=2))
local = Path('results/local/2026-09-09/family-gpu-v588')
local.mkdir(parents=True, exist_ok=False)
source = Path('build/performance/family-gpu-v588')
for path in source.rglob('*'):
    if not path.is_file() or path.is_relative_to(source / 'source'):
        continue
    destination = local / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
local_manifest = {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(local.rglob('*')) if p.is_file()}
(local / 'files-sha256.json').write_text(json.dumps(local_manifest, indent=2))
print(json.dumps(dict(remote_files=len(manifest), local_files=len(local_manifest), remote=str(root), local=str(local))))
