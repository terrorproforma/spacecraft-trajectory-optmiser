from pathlib import Path
import hashlib
import json
import subprocess
root=Path('results/lambda/2026-09-09/gpu-device-search-v707')
manifest=json.loads((root/'sha256.json').read_text())
for name,record in manifest.items():
    blob=subprocess.check_output(['git','show',':'+(root/name).as_posix()])
    assert len(blob)==record['bytes'] and hashlib.sha256(blob).hexdigest()==record['sha256'],name
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-search-v707')
for name in ('fleet.json','manifest.json','compute.json'):
    assert subprocess.check_output(['git','show',':'+(viewer/name).as_posix()])==(root/'viewer-dataset'/name).read_bytes()
sources=json.loads((root/'published-source.json').read_text())['files']
for name,digest in sources.items():
    live=Path(name).read_bytes();assert hashlib.sha256(live).hexdigest()==digest,name
    staged=subprocess.check_output(['git','show',':'+name])
    assert staged.replace(b'\r\n',b'\n')==live.replace(b'\r\n',b'\n'),name
print(json.dumps(dict(evidence_blobs=len(manifest),viewer_blobs=3,source_files=len(sources))))
