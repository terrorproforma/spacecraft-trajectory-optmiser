from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path('/home/ubuntu/spacepdhcg-fleet-recovery-v381')
r=json.loads((root/'report.json').read_text())
assert r['complete'] and not r.get('error')
f=json.loads((root/'output/run_report.json').read_text())
assert f['best']['official']['ok'] and f['best']['independent']['ok']
source=root/'frozen-source.tar.gz'
subprocess.run(['git','-C',str(root/'repo'),'archive','--format=tar.gz','--output',str(source),'HEAD'],check=True)
paths=[p for p in root.rglob('*') if p.is_file() and
       p.relative_to(root).parts[0] not in ['repo','core-build'] and p.name!='files-sha256.json']
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:
  t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,
 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest),
 ships=f['best']['independent']['ships'],score=f['best']['independent']['weighted_score_fixed_bonus_kg'],
 largest=sorted([(p.stat().st_size,p.relative_to(root).as_posix()) for p in paths],reverse=True)[:4])))
