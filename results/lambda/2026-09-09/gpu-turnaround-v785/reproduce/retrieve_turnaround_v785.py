from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-turnaround-v785');dest.mkdir()
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
for name in ('h100.tar.gz','receipt.json'):
    subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-turnaround-evidence-v785/'+name,str(dest/(name if name.endswith('gz') else 'h100-receipt.json'))],check=True,timeout=55)
for name in ('local.tar.gz','receipt.json'):
    shutil.copyfile(Path.home()/'spacepdhcg-turnaround-evidence-v785'/name,dest/(name if name.endswith('gz') else 'local-receipt.json'))
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
    r=json.loads((dest/side/'v784/report.json').read_text());audit=json.loads((dest/side/'audit.json').read_text())
    probe=json.loads((dest/side/'v783/report.json').read_text())
    summary[side]=dict(score_kg=r['score_kg'],raw_kg=r['total_mass_kg'],score_change_from_v780=audit['score_change_from_v780'],ships=23,asteroids=195,campaign_seconds=r['seconds'],route_search_seconds=sum(x['wall_seconds'] for x in r['ships']),independent_seconds=r['independent_seconds'],official_seconds=r['official']['wall_seconds'],native_solves=r['native_solves'],native_seconds=audit['native_seconds'],native_status_counts=audit['native_status_counts'],gpu_telemetry=r['gpu_telemetry'],solution_sha256=r['solution_sha256'],archive_sha256=receipt['sha256'],verified_members=len(manifest),probes=[{k:v for k,v in t.items() if k not in ('top','telemetry')} for t in probe['trials']])
assert [[{k:v for k,v in t.items() if k!='seconds'} for t in summary[side]['probes']] for side in ('local','h100')][0]==[{k:v for k,v in t.items() if k!='seconds'} for t in summary['h100']['probes']]
(dest/'summary.json').write_text(json.dumps(summary,indent=2));(dest/'.gitattributes').write_text('* -text\n*.tar.gz -diff\n')
repro=dest/'reproduce';repro.mkdir()
for name in ('prepare_turnaround_v782.py','send_turnaround_v782.py','package_turnaround_v785.py','retrieve_turnaround_v785.py','probe_wide_insertions_v781.py','probe_wide_insertions_v783.py','launch_wide_insertions_v781.py','launch_wide_insertions_v783.py','launch_insertions_bench_v769.py','launch_routes_v784.py','launch_routes_v766.py','search_fleet_routes_v784.py'):
    shutil.copyfile(Path('build/performance')/name,repro/name)
source=json.loads((dest/'local/v782/source-manifest.json').read_text())['overlays']
for side in ('local','h100'):
    manifest=json.loads((dest/side/'v782/source-manifest.json').read_text())['files']
    for name in source:assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==manifest[name],(side,name)
hashes={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
(dest/'sha256.json').write_text(json.dumps(hashes,indent=2))
Path('build/performance/owned_turnaround_v785.json').write_text(json.dumps(source+['README.md','docs/GPU_INSERTION_LAYOUTS.md','docs/GPU_INSERTION_TURNAROUNDS.md',str(dest)]))
print(json.dumps({side:{k:v for k,v in r.items() if k not in ('probes','gpu_telemetry')} for side,r in summary.items()},indent=2))
