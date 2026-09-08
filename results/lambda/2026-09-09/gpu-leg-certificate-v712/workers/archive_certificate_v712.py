from pathlib import Path
import hashlib,json,tarfile
home=Path('/home/ubuntu' if Path('/home/ubuntu').exists() else '/home/angus')
roots={'profile':home/'spacepdhcg-campaign-profile-v708','certificate':home/'spacepdhcg-certificate-v709'}
for root,report in [(roots['profile'],'profile.json'),(roots['certificate'],'report.json'),(roots['certificate']/'followup-v710','report.json')]:
    r=json.loads((root/report).read_text());assert r['complete'] and r['success']
destination=home/('certificate-v712-'+('h100' if home.name=='ubuntu' else 'local')+'.tar.gz')
manifest={}
with tarfile.open(destination,'x:gz') as tar:
    for prefix,root in roots.items():
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or any(x in ('.git','__pycache__','.pytest_cache','.ruff_cache') for x in rel.parts):continue
            if rel.parts[:2] in (('repo','results'),('repo','benchmarks')):continue
            raw=p.read_bytes();name=prefix+'/'+rel.as_posix()
            manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());tar.add(p,arcname=name)
destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
record=dict(bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),members=len(manifest))
destination.with_suffix('.record.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(path=str(destination),**record)))
