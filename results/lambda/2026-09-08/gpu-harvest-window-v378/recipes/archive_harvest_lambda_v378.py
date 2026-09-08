from pathlib import Path
import json,tarfile,hashlib
root=Path('/home/ubuntu/spacepdhcg-harvest-window-v378');r=json.loads((root/'report.json').read_text());assert r['complete'] and len(r['campaigns'])==4
files=[p for p in root.iterdir() if p.is_file()]
for name in ['baseline0','candidate0','candidate1','baseline1']:files.extend(p for p in (root/name).rglob('*') if p.is_file())
files.extend(root/'repo'/name for name in r['source_sha256'])
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/'evidence-files.json').write_text(json.dumps(manifest,indent=2));files.append(root/'evidence-files.json')
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),complete=r['complete'])))
