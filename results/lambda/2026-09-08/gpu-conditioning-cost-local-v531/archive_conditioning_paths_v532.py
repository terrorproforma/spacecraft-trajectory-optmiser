from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/ubuntu/spacepdhcg-conditioning-paths-v532')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
for name,sha in r['executed_sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
for tag in ['conditioning-paths-v529','conditioning-matrices-v530','conditioning-cost-v531']:
 r=json.loads((root/'build/performance'/tag/'report.json').read_text());assert r['complete'] and not r.get('error')
paths=[p for p in root.rglob('*') if p.is_file() and p.name!='files-sha256.json']
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
