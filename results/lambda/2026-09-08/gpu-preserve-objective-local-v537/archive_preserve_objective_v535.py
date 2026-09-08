from pathlib import Path
import hashlib,json,shutil,tarfile
root=Path('/home/ubuntu/spacepdhcg-preserve-objective-v535')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
work=root/'repo/build/performance/preserve-objective-v536'
q=json.loads((work/'report.json').read_text());assert q['complete'] and not q.get('error')
for label in ['baseline','candidate']:
 rows=json.loads((work/label/'results.json').read_text());assert len(rows)==225
 assert all(row.get('certified') for row in rows if row['status']=='converged')
paths=[path for path in root.rglob('*') if path.is_file() and path.relative_to(root).parts[0] not in ['build','repo']]
for path in work.rglob('*'):
 if path.is_file():
  dest=root/'validation'/path.relative_to(work);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);paths.append(dest)
for path in (root/'repo/src/spacepdhcg/gtoc12').glob('*.py'):
 dest=root/'source-overlay'/path.relative_to(root/'repo');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);paths.append(dest)
paths=sorted(set(path for path in paths if path.name!='files-sha256.json'))
manifest={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2));archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
