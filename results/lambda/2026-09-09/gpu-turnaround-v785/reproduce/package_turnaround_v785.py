from pathlib import Path
import collections,hashlib,json,math,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-turnaround-evidence-v785';dest.mkdir();members={}
def add(p,n):
    assert n not in members;members[n]=p
base=home/'spacepdhcg-turnaround-v782';source=json.loads((base/'source-manifest.json').read_text())
for name,digest in source['files'].items():
    p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'v782/repo/'+name)
add(base/'final/libspacepdhcg_cuda.so','v782/final/libspacepdhcg_cuda.so')
for version,dirname in [('v781','spacepdhcg-wide-insertions-v781'),('v782','spacepdhcg-turnaround-v782'),('v783','spacepdhcg-wide-insertions-v783')]:
    root=home/dirname;r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success']
    for p in root.iterdir():
        if p.is_file():add(p,version+'/'+p.name)
root=home/'spacepdhcg-fleet-routes-v784';r=json.loads((root/'report.json').read_text())
assert r['complete'] and r['success'] and r['independent']['ok'] and r['official']['ok']
weights=json.loads((root/'input/pool.json').read_text())['weights']
lines=(root/'fleet/official/ScoreData.txt').read_text().splitlines();data=[line.split() for line in lines[1:]]
assert len(data)==int(lines[0])==195
weighted=math.fsum(float(m)*weights[i] for i,m in data);raw=math.fsum(float(m) for i,m in data)
assert abs(weighted-r['score_kg'])<1e-8 and abs(raw-r['total_mass_kg'])<1e-8
rows=[json.loads(x) for x in (root/'native-solves.jsonl').read_text().splitlines()]
for identifier in (1786,2297):
    summary=json.loads((root/'routes'/str(identifier)/'route_summary.json').read_text())
    assert summary['certified'] and all(x['certification_backend']=='cuda' and x['certified'] and x['status']=='feasible' for x in summary['legs'])
    flown=[x for x in rows if x['column']==identifier][-len(summary['legs']):]
    assert len(flown)==len(summary['legs']) and all(x['status']=='converged' for x in flown)
audit=dict(official_weighted_kg=weighted,independent_weighted_kg=r['score_kg'],official_raw_kg=raw,native_status_counts=dict(collections.Counter(x.get('status','error') for x in rows)),native_seconds=sum(x['seconds'] for x in rows),score_change_from_v780=r['score_kg']-12843.555695585388)
(dest/'audit.json').write_text(json.dumps(audit,indent=2));add(dest/'audit.json','audit.json')
for p in root.rglob('*'):
    if not p.is_file() or p.suffix=='.gz':continue
    if p.parent.name=='official' and p.name in ('GTOC12_Verify','GTOC12_Asteroids_Data.txt','Result.txt'):continue
    add(p,'v784/'+p.relative_to(root).as_posix())
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if home.name=='ubuntu' else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so')
assert hashlib.sha256(qoco.read_bytes()).hexdigest()==r['qoco_sha256'];add(qoco,'qoco/libqoco.so')
(dest/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True));add(dest/'hardware.txt','hardware.txt')
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
