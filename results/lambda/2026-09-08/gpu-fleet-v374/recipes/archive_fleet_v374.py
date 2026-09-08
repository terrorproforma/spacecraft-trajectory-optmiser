from pathlib import Path
import json,hashlib,tarfile
root=Path('/home/ubuntu/spacepdhcg-fleet-search-v374');r=json.loads((root/'report.json').read_text());assert r['complete'] and r['returncode']==0
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='files-sha256.json'}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(root,arcname=root.name)
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest))))
