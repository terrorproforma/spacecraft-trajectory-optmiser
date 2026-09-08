from pathlib import Path
import hashlib
import json
import tarfile

root=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686')
for name in ['report.json','broader-v689/report.json','reporting-v692/report.json','campaign-v695/report.json']:
    r=json.loads((root/name).read_text());assert r['complete'],name
assert json.loads((root/'campaign-v695/report.json').read_text())['success']
archive=Path('/home/ubuntu/retry-conditioning-v696-h100.tar.gz')
assert not archive.exists()
manifest={}
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(v in p.relative_to(root).parts for v in ('.git','__pycache__','.pytest_cache','core-build','qoco-build')) or p.name.endswith('.lock'):continue
        name=p.relative_to(root).as_posix();raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());tar.add(p,arcname=name)
record=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest))
archive.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
archive.with_suffix('.record.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
