from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-catalogue-cache-v822');dest.mkdir()
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-catalogue-evidence-v822/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):shutil.copyfile(Path.home()/'spacepdhcg-catalogue-evidence-v822'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,entry in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],name
            extract=name in ('audit.json','hardware.txt','runtime/source-manifest.json') or name.endswith('/report.json') and name.count('/')==1 or name.startswith('v819/') and name.endswith('.log')
            if side=='h100' and name.startswith('v817/fleet/'):extract=True
            if extract:
                target=dest/side/name;assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2))
shutil.copytree(dest/'h100/v817/fleet',dest/'h100-best');shutil.copyfile(dest/'h100/v817/report.json',dest/'h100-best/campaign-report.json')
(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
(dest/'reproduce').mkdir()
for pattern in ('*catalogue*v808.py','*catalogue*v809.py','*integrated_refine_v810.py','*catalogue*v811.py','*catalogue*v814.py','*catalogue*v815.py','*catalogue*v817.py','*catalogue*v819.py','*catalogue*v820.py','*catalogue*v822.py'):
    for path in Path('build/performance').glob(pattern):shutil.copyfile(path,dest/'reproduce'/path.name)
shutil.copyfile('build/performance/audit_catalogue_v822.py',dest/'audit_saved.py')
print(json.dumps({side:json.loads((dest/(side+'-receipt.json')).read_text()) for side in ('local','h100')},indent=2))
