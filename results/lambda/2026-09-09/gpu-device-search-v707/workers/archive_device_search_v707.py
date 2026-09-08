from pathlib import Path
import hashlib
import json
import tarfile

remote=Path('/home/ubuntu').exists();home=Path('/home/ubuntu' if remote else '/home/angus')
campaign=home/'spacepdhcg-search-campaign-v703'
report=json.loads((campaign/'report.json').read_text());assert report['complete'] and report['success']
roots={'first-controller':home/'spacepdhcg-joint-search-v700','final-controller':home/'spacepdhcg-joint-search-v702','campaign':campaign}
if not remote:roots.update({'initial-build':home/'spacepdhcg-joint-search-v697','initial-validation':home/'spacepdhcg-joint-search-v698'})
destination=home/('device-search-v707-'+('h100' if remote else 'local')+'.tar.gz')
manifest={}
with tarfile.open(destination,'x:gz') as tar:
    for prefix,root in roots.items():
        for p in sorted(root.rglob('*')):
            relative=p.relative_to(root)
            if not p.is_file() or any(x in ('.git','__pycache__','.pytest_cache','.ruff_cache') for x in relative.parts):continue
            if relative.parts[0]=='build':continue
            name=prefix+'/'+relative.as_posix();raw=p.read_bytes()
            manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());tar.add(p,arcname=name)
    qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if remote else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so')
    raw=qoco.read_bytes();assert hashlib.sha256(raw).hexdigest()==report['qoco_sha256']
    manifest['qoco/libqoco.so']=dict(bytes=len(raw),sha256=report['qoco_sha256']);tar.add(qoco,arcname='qoco/libqoco.so')
destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
record=dict(bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),members=len(manifest))
destination.with_suffix('.record.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(path=str(destination),**record)),flush=True)
