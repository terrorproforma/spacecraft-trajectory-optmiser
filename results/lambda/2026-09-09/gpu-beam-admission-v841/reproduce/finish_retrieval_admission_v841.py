from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-beam-admission-v841')
expected={'h100':'c0215336686a1cd86b91781c2c0c38eaa4c69850fbe88dfcff16c11362ab7aab','local':'5713c86bd76d7f94e5f51be8aa0359a378b4c5f194fa0d45c7d6b00e5dc4a6ed'}
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']==expected[side]
    files={};names=[]
    with tarfile.open(archive,'r|gz') as tar:
        for item in tar:
            assert item.isfile();names.append(item.name);files[item.name]=tar.extractfile(item).read()
    manifest=json.loads(files.pop('FILES.json'));assert len(names)==len(set(names)) and set(files)==set(manifest)
    for name,data in files.items():
        row=manifest[name];assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
        if name in ('hardware.txt','runtime/source-manifest.json','validation/report.json','benchmark/report.json','leaks/report.json','leaks/leakcheck.log','profile/report.json','profile/summary.json') or name.startswith('validation/') and name.endswith('.log') and name not in ('validation/build.log','validation/configure.log'):
            target=dest/side/name;assert target.resolve().is_relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2)+'\n')
(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
(dest/'reproduce').mkdir(exist_ok=True)
for p in Path('build/performance').glob('*admission_v84*.py'):
    if p.name.startswith(('launch_','package_')):(dest/'reproduce'/p.name).write_text(p.read_text().rstrip()+'\n')
for name in ('prepare_admission_v839.py','launch_admission_h100_v839.py'):
    p=Path('build/performance')/name;(dest/'reproduce'/name).write_text(p.read_text().rstrip()+'\n')
shutil.copyfile('build/performance/audit_admission_v841.py',dest/'audit_saved.py')
result=subprocess.check_output(['python3',str(dest/'audit_saved.py')],text=True)
(dest/'saved-audit.json').write_text(result)
r=json.loads(result);print(json.dumps({side:r[side]['comparison'] for side in r},indent=2))
