from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-resident-catalogue-v827');dest.mkdir()
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-resident-evidence-v827/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):
    shutil.copyfile(Path.home()/'spacepdhcg-resident-evidence-v827'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,row in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
            if name=='hardware.txt' or name in ('runtime/source-manifest.json','v823/report.json','v826/report.json','v824/report.json','v828/report.json','v828/leakcheck.log') or name.startswith('v826/') and name.endswith('.log'):
                target=dest/side/name;assert target.resolve().is_relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2))
(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
(dest/'reproduce').mkdir()
for name in ('prepare_resident_catalogue_v823.py','launch_resident_catalogue_h100_v823.py','prepare_resident_final_v826.py','launch_resident_final_h100_v826.py','launch_resident_catalogue_bench_v824.py','launch_resident_catalogue_both_v824.py','package_resident_catalogue_v827.py','retrieve_resident_catalogue_v827.py','launch_resident_leaks_v828.py','launch_resident_leaks_both_v828.py'):
    shutil.copyfile(Path('build/performance')/name,dest/'reproduce'/name)
shutil.copyfile('build/performance/audit_resident_catalogue_v827.py',dest/'audit_saved.py')
result=subprocess.check_output(['python3',str(dest/'audit_saved.py')],text=True)
(dest/'saved-audit.json').write_text(result)
print(result)
