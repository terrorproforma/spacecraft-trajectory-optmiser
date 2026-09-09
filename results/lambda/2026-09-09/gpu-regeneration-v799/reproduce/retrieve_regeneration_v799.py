from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-regeneration-v799');dest.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-regeneration-evidence-v799/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):shutil.copyfile(Path.home()/'spacepdhcg-regeneration-evidence-v799'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
summary={}
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,entry in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],name
            # Retain all bytes in the archives; extract concise reports plus the final replay.
            extract=name in ('audit.json','hardware.txt') or name.endswith('/run.py') or name.endswith('/report.json') and name.count('/')==1 or name.startswith('v798/fleet/') or name=='runtime/source-manifest.json'
            if extract:
                target=dest/side/name;assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2))
    summary[side]=json.loads((dest/side/'audit.json').read_text())
shutil.copytree(dest/'h100/v798/fleet',dest/'h100-best');shutil.copyfile(dest/'h100/v798/report.json',dest/'h100-best/campaign-report.json')
(dest/'summary.json').write_text(json.dumps(summary,indent=2));(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
repro=dest/'reproduce';repro.mkdir()
for name in ('regenerate_fleet_v792.py','launch_regeneration_v792.py','refine_regeneration_v793.py','launch_refine_regeneration_v793.py','prepare_raw_regeneration_v794.py','prepare_refine_raw_v795.py','master_regeneration_v796.py','launch_master_v796.py','package_regeneration_v799.py','retrieve_regeneration_v799.py','prepare_master_deep_v798.py','analyse_regeneration_v794.py','launch_grid_probe_v787.py','launch_routes_v766.py'):
    shutil.copyfile(Path('build/performance')/name,repro/name)
print(json.dumps({side:{v:{k:val for k,val in r.items() if k!='telemetry_totals'} for v,r in rows.items()} for side,rows in summary.items()},indent=2))
