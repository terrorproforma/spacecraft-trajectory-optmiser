from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-fleet-routes-v767');dest.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-fleet-routes-evidence-v767/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):
    shutil.copyfile(Path.home()/'spacepdhcg-fleet-routes-evidence-v767'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
summary={}
for side in ('local','h100'):
    archive=dest/(side+'.tar.gz');receipt=json.loads((dest/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,e in manifest.items():
            b=tar.extractfile(name).read();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
            if '/repo/' not in name and '/input/' not in name and not name.endswith('.so'):
                target=dest/side/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        (dest/(side+'-files.json')).write_text(json.dumps(manifest,indent=2))
    r=json.loads((dest/side/'v766/report.json').read_text());audit=json.loads((dest/side/'audit.json').read_text())
    summary[side]=dict(score_kg=r['score_kg'],gain_weighted_kg=r['score_kg']-r['baseline_score_kg'],raw_kg=r['total_mass_kg'],raw_gain_kg=r['total_mass_kg']-14043.750855578397,raw_kg_per_ship=r['total_mass_kg']/23,ships=23,asteroids=195,campaign_seconds=r['seconds'],route_search_seconds=sum(x['wall_seconds'] for x in r['ships']),independent_seconds=r['independent_seconds'],official_seconds=r['official']['wall_seconds'],native_solves=r['native_solves'],native_seconds=audit['native_seconds'],native_status_counts=audit['native_status_counts'],gpu_telemetry=r['gpu_telemetry'],solution_sha256=r['solution_sha256'],archive_sha256=receipt['sha256'],verified_members=len(manifest))
shutil.copytree(dest/'h100/v766/fleet',dest/'h100-best');shutil.copyfile(dest/'h100/v766/report.json',dest/'h100-best/campaign-report.json')
(dest/'summary.json').write_text(json.dumps(summary,indent=2));(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
repro=dest/'reproduce';repro.mkdir()
for name in ('search_fleet_routes_v766.py','launch_routes_v766.py','package_routes_v767.py','retrieve_routes_v767.py'):shutil.copyfile(Path('build/performance')/name,repro/name)
print(json.dumps(summary,indent=2))
