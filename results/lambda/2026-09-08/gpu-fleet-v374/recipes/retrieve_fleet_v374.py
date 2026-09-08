from pathlib import Path
import os,subprocess,tarfile,json,hashlib
root=Path('results/lambda/2026-09-08/gpu-fleet-v374');root.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
archive=Path('build/performance/fleet-v374.tar.gz')
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/spacepdhcg-fleet-search-v374.tar.gz',str(archive)],check=True,timeout=180)
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():
  target=(root/m.name).resolve();assert target.is_relative_to(root.resolve()) and not m.issym() and not m.islnk()
 t.extractall(root)
folder=root/'spacepdhcg-fleet-search-v374';manifest=json.loads((folder/'files-sha256.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,name
(root/'retrieval.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,verified_files=len(manifest)),indent=2))
print((root/'retrieval.json').read_text())
r=json.loads((folder/'output/run_report.json').read_text());assert r['best']['official']['ok'] and r['best']['independent']['ok']
print(json.dumps(dict(seconds=r['wall_seconds_total'],fleet=r['fleet']['ships'],weighted=r['best']['independent']['weighted_score_fixed_bonus_kg'],ships=[dict(ship=s['ship'],status=s.get('status'),candidates=s['search']['candidates'],refinements=[dict(rank=a['rank'],certified=a['refined']['certified']) for a in s.get('refinements',[])]) for s in r['ships']]),indent=2))
