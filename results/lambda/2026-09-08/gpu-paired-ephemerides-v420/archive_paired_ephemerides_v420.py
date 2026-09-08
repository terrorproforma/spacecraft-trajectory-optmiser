from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/ubuntu/spacepdhcg-paired-ephemerides-v420')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
for name,sha in r['source_sha256'].items():
 p=root/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for c in r['campaigns']:
 f=json.loads((root/c['name']/'output/run_report.json').read_text());assert f['best']['accepted'] and f['best']['official']['ok'] and f['best']['independent']['ok']
paths=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ['repo','core-build','stationary-probe'] and p.name!='files-sha256.json']
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
