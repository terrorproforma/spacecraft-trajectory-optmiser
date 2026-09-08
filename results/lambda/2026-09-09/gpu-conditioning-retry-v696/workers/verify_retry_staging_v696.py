from pathlib import Path
import hashlib
import json
import subprocess

root = Path('results/lambda/2026-09-09/gpu-conditioning-retry-v696')
count = 0
for package in (root, Path('results/local/2026-09-09/return-qp-v680')):
    manifest = json.loads((package / 'sha256.json').read_text())
    for name, record in manifest.items():
        blob = subprocess.check_output(['git', 'show', ':' + (package / name).as_posix()])
        assert len(blob) == record['bytes'], name
        assert hashlib.sha256(blob).hexdigest() == record['sha256'], name
        count += 1
viewer = Path('results/lambda/2026-09-06/visualiser/data/gtoc12-retry-v696')
for name in ('fleet.json', 'manifest.json', 'compute.json'):
    assert subprocess.check_output(['git', 'show', ':' + (viewer / name).as_posix()]) == (root / 'viewer-dataset' / name).read_bytes()
sources = json.loads((root / 'published-source.json').read_text())['files']
for name, digest in sources.items():
    live = Path(name).read_bytes()
    assert hashlib.sha256(live).hexdigest() == digest, name
    blob = subprocess.check_output(['git', 'show', ':' + name])
    assert blob.replace(b'\r\n', b'\n') == live.replace(b'\r\n', b'\n'), name
print(json.dumps({'verified_evidence_blobs': count, 'verified_viewer_blobs': 3, 'verified_tested_source_files': len(sources)}))
