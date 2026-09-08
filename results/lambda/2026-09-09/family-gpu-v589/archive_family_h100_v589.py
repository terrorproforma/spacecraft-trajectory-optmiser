from pathlib import Path
import hashlib
import json
import tarfile

root = Path('/home/ubuntu/spacepdhcg-family-v589')
report = json.loads((root / 'report.json').read_text())
assert report['complete'] and report['returncode'] == 0
files = [p for p in root.rglob('*') if p.is_file() and not p.is_relative_to(root / 'source')]
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root / 'files-sha256.json').write_text(json.dumps(manifest, indent=2))
archive = root.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    for path in files + [root / 'files-sha256.json']:
        stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
print(json.dumps(dict(path=str(archive), bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=len(files))))
