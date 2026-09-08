"""Add the stable independent review and verify a flat publication index."""
from pathlib import Path
import argparse,hashlib,json,shutil
parser=argparse.ArgumentParser()
parser.add_argument('--review-dir',type=Path,required=True)
parser.add_argument('--review-hashes',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2];package=root/'results/local/2026-09-09/l1-weight-core-v618'
assert not (package/'sha256.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads(args.review_hashes.read_text())
assert set(expected)=={'review_real.py','decimal_audit_source.py','findings.json','REPORT.md'}
target=package/'real/independent-review';target.mkdir(exist_ok=False)
for name,value in expected.items():
    assert sha(args.review_dir/name)==value,name
    shutil.copy2(args.review_dir/name,target/name)
for p in (root/'build/performance/l1-weight-review-v618').iterdir():
    if not p.is_file():continue
    q=package/'analysis'/p.name
    if q.exists():assert sha(p)==sha(q),'stable analysis file changed: '+p.name
    else:shutil.copy2(p,q)
for name in ('package_l1_weight_v618.py','finalize_l1_weight_v618.py'):
    shutil.copy2(root/'build/performance'/name,package/'scripts'/name)
shutil.copy2(args.review_hashes,package/'real/independent-review-sha256.json')
readme=package/'README.md';text=readme.read_text()
text=text.replace('After the independent Decimal65 review is copied, the package-local\n',
    'The independent [Decimal65 review](real/independent-review/REPORT.md) agrees\nwith all six original-equation verdicts and both untouched seeds. The package-local\n')
readme.write_text(text)
files={p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
    for p in sorted(package.rglob('*')) if p.is_file() and p!=package/'sha256.json'}
assert not any('__pycache__' in name or Path(name).suffix in ('.o','.so','.dll','.exe','.pyc','.pdb','.a') for name in files)
for name in files:
    with (package/name).open('rb') as f:magic=f.read(4)
    assert magic!=b'\x7fELF' and magic[:2]!=b'MZ',name
(package/'sha256.json').write_text(json.dumps(files,indent=2)+'\n')
for name,value in files.items():assert (package/name).stat().st_size==value['bytes'] and sha(package/name)==value['sha256']
print(json.dumps({'package':str(package),'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'index_sha256':sha(package/'sha256.json')}))
