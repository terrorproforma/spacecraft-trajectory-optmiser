"""Preserve the exact resident-geometry build, fixtures, tests and every replay."""
from pathlib import Path
import argparse,hashlib,json,tarfile
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('archive',type=Path);p.add_argument('qoco',type=Path);args=p.parse_args();root=args.root
assert json.loads((root/'validation-v654/report.json').read_text())['success']
assert json.loads((root/'campaign-v656/report.json').read_text())['success']
assert json.loads((root/'benchmark-v653/report.json').read_text())['returncode']==0
assert not args.archive.exists()
files={}
for path in root.rglob('*'):
 if not path.is_file():continue
 rel=path.relative_to(root)
 if any(x in ('.git','__pycache__','.pytest_cache') for x in rel.parts):continue
 if rel.parts[0]=='build' and rel.as_posix()!='build/cuda/libspacepdhcg_cuda.so':continue
 if path.suffix=='.lock':continue
 files[rel.as_posix()]=path
files['qoco/libqoco.so']=args.qoco
manifest={n:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for n,f in sorted(files.items())}
args.archive.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(args.archive,'w:gz') as tar:
 for n,f in sorted(files.items()):tar.add(f,arcname=n,recursive=False)
with tarfile.open(args.archive) as tar:
 members=tar.getmembers();assert len(members)==len(manifest)
 for m in members:assert m.isfile() and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==manifest[m.name]['sha256']
record=dict(bytes=args.archive.stat().st_size,sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),verified_members=len(manifest))
args.archive.with_suffix('.record.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
