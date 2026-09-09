from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-collect-reuse-v806');dest.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-collect-evidence-v806/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):shutil.copyfile(Path.home()/'spacepdhcg-collect-evidence-v806'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
summary={}
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,entry in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],name
            extract=name in ('audit.json','hardware.txt') or name.endswith('/report.json') and name.count('/')==1 or name.endswith('/run.py') and name.count('/')==1 or name.startswith('v804/fleet/') or name=='runtime/source-manifest.json' or name.startswith('v805/') and name.endswith('.log')
            if extract:
                target=dest/side/name;assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2))
    summary[side]=json.loads((dest/side/'audit.json').read_text())
shutil.copytree(dest/'h100/v804/fleet',dest/'h100-best');shutil.copyfile(dest/'h100/v804/report.json',dest/'h100-best/campaign-report.json')
(dest/'summary.json').write_text(json.dumps(summary,indent=2));(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
repro=dest/'reproduce';repro.mkdir()
for name in ('prepare_collect_v800.py','build_collect_v800.py','worker_collect_v800.py','send_collect_v800.py','bench_collect_v801.py','launch_collect_bench_v801.py','launch_collect_fleet_v802.py','launch_collect_refine_v803.py','launch_collect_master_v804.py','final_collect_tests_v805.py','launch_final_collect_v805.py','package_collect_v806.py','retrieve_collect_v806.py'):
    shutil.copyfile(Path('build/performance')/name,repro/name)
print(json.dumps({side:{v:{k:value for k,value in row.items() if k not in ('telemetry_totals','budget_sweep')} for v,row in r.items()} for side,r in summary.items()},indent=2))
