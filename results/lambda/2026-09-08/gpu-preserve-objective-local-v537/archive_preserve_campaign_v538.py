from pathlib import Path
import hashlib,json,tarfile,shutil
root=Path('/home/ubuntu/spacepdhcg-preserve-campaign-v538');repo=Path('/home/ubuntu/spacepdhcg-preserve-objective-v535/repo')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
assert '107 passed' in (root/'pytest.log').read_text()
for c in r['campaigns']:
 q=json.loads((root/c['name']/'output/run_report.json').read_text());assert q['best']['accepted'] and q['best']['official']['ok'] and q['best']['independent']['ok']
for name in r['source_sha256']:
 path=repo/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==r['source_sha256'][name]
 dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
for name in r['pytest']['command']:
 if name.startswith('tests/'):
  dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,dest)
paths=[path for path in root.rglob('*') if path.is_file() and path.name!='files-sha256.json']
manifest={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2));archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
