from pathlib import Path
import hashlib
import json
import tarfile

build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
assert json.loads((build/'campaign-v694/report.json').read_text())['success']
folders={'build':build,'reporting':Path('/home/angus/spacepdhcg-retry-report-v690'),'screening':Path('/home/angus/spacepdhcg-return-conditioning-v681'),'fixed-normalization':Path('/home/angus/spacepdhcg-return-normalized-v682'),'return-replay':Path('/home/angus/spacepdhcg-return-conditioning-retry-v685'),'leg-replay':Path('/home/angus/spacepdhcg-retry-fleet-legs-v688')}
archive=Path('/home/angus/retry-conditioning-v696-local.tar.gz');assert not archive.exists()
manifest={}
with tarfile.open(archive,'w:gz') as tar:
    for prefix,root in folders.items():
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(v in p.relative_to(root).parts for v in ('.git','__pycache__','.pytest_cache','core-build','qoco-build')) or p.name.endswith('.lock'):continue
            name=prefix+'/'+p.relative_to(root).as_posix();raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());tar.add(p,arcname=name)
record=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest))
archive.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));archive.with_suffix('.record.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
