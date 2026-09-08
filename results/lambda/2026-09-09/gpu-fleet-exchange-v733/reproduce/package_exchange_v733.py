from pathlib import Path
import hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-fleet-exchange-evidence-v733';dest.mkdir();members={}
def add(p,n):
    assert n not in members;members[n]=p
base=home/'spacepdhcg-fleet-v727';r=json.loads((base/'report.json').read_text());assert r['complete'] and r['success']
manifest=json.loads((base/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'v727/repo/'+name)
for p in base.iterdir():
    if p.is_file() and p.suffix in ('.json','.py','.log'):add(p,'v727/'+p.name)
add(base/'final/libspacepdhcg_cuda.so','v727/final/libspacepdhcg_cuda.so')
for version in ('v728','v730'):
    root=home/('spacepdhcg-fleet-'+version);r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success']
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.json','.py','.log'):add(p,version+'/'+p.name)
for version in ('v729','v732'):
    root=home/('spacepdhcg-fleet-refine-'+version);r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success'] and r['independent']['ok'] and r['official']['ok']
    assert abs(r['score_kg']-12842.970672270907)<1e-6
    solves=[json.loads(x) for x in (root/'native-solves.jsonl').read_text().splitlines()]
    assert len(solves)==33 and all(x['status']=='converged' for x in solves)
    for row in r['refinements']:
        assert row['certified'] and all(leg['certification_backend']=='cuda' for leg in row['summary']['legs'])
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix=='.gz':continue
        if p.parent.name=='official' and p.name in ('GTOC12_Verify','GTOC12_Asteroids_Data.txt','Result.txt'):continue
        add(p,version+'/'+p.relative_to(root).as_posix())
add(home/'spacepdhcg-fleet-pool-v716/pool.json','input/pool.json')
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
(dest/'hardware.txt').write_text(hardware);add(dest/'hardware.txt','hardware.txt')
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(members.items())}
(dest/'FILES.json').write_text(json.dumps(manifest,indent=2))
archive=dest/('h100.tar.gz' if home.name=='ubuntu' else 'local.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for n,p in sorted(members.items()):tar.add(p,arcname=n)
    tar.add(dest/'FILES.json',arcname='FILES.json')
with tarfile.open(archive) as tar:
    names=[p.name for p in tar.getmembers()];assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
    for n,e in manifest.items():
        b=tar.extractfile(n).read();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest))
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
