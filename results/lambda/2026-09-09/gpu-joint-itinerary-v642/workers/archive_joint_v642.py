"""Archive frozen sources, binaries, all validation and all campaign attempts."""
from pathlib import Path
import argparse,hashlib,json,tarfile
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('archive',type=Path);parser.add_argument('qoco',type=Path);args=parser.parse_args()
root=args.root
assert json.loads((root/'compatibility-validation-v640/report.json').read_text())['returncode']==0
assert json.loads((root/'campaign-v636/report.json').read_text())['complete']
assert not args.archive.exists()
files={}
for path in root.rglob('*'):
    if not path.is_file():continue
    relative=path.relative_to(root)
    if any(p in ('.git','__pycache__','.pytest_cache') for p in relative.parts):continue
    if relative.parts[0]=='build' and relative.as_posix()!='build/cuda/libspacepdhcg_cuda.so':continue
    if path.suffix=='.lock':continue
    files[relative.as_posix()]=path
files['qoco/libqoco.so']=args.qoco
manifest={name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in sorted(files.items())}
manifest_path=args.archive.with_suffix('.manifest.json')
manifest_path.write_text(json.dumps(manifest,indent=2))
with tarfile.open(args.archive,'w:gz') as tar:
    for name,path in sorted(files.items()):tar.add(path,arcname=name,recursive=False)
with tarfile.open(args.archive,'r:gz') as tar:
    members=tar.getmembers();assert len(members)==len(manifest)
    for m in members:assert m.isfile() and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==manifest[m.name]['sha256'],m.name
record=dict(bytes=args.archive.stat().st_size,sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),verified_members=len(files))
args.archive.with_suffix('.record.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
