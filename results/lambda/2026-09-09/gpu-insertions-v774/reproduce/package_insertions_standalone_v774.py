from pathlib import Path
import collections,hashlib,json,math,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-insertions-evidence-v774';dest.mkdir(exist_ok=True);members={}
assert not list(dest.glob('*.tar.gz'))
def add(p,n):
    assert n not in members;members[n]=p
base=home/'spacepdhcg-insertions-v772';manifest=json.loads((base/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if '.pytest_cache/' in name or '.ruff_cache/' in name:continue
    p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'v772/repo/'+name)
add(base/'source-manifest.json','v772/source-manifest.json');add(base/'final/libspacepdhcg_cuda.so','v772/final/libspacepdhcg_cuda.so')
root=home/'spacepdhcg-fleet-routes-v773';r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success'] and r['improved'] and r['independent']['ok'] and r['official']['ok']
rows=[json.loads(x) for x in (root/'native-solves.jsonl').read_text().splitlines()];assert len(rows)==36
weights=json.loads((root/'input/pool.json').read_text())['weights']
lines=(root/'fleet/official/ScoreData.txt').read_text().splitlines();data=[line.split() for line in lines[1:]];assert len(data)==int(lines[0])==195
weighted=math.fsum(float(m)*weights[i] for i,m in data);raw=math.fsum(float(m) for i,m in data)
assert abs(weighted-r['score_kg'])<1e-8 and abs(raw-r['total_mass_kg'])<1e-8
audit=dict(official_scoredata_weighted_kg=weighted,independent_weighted_kg=r['score_kg'],weighted_difference_kg=weighted-r['score_kg'],official_scoredata_raw_kg=raw,native_status_counts=dict(collections.Counter(x.get('status','error') for x in rows)),native_seconds=sum(x['seconds'] for x in rows),gpu_telemetry=r['gpu_telemetry'])
for identifier in (1786,2297):
    summary=json.loads((root/'routes'/str(identifier)/'route_summary.json').read_text())
    assert summary['certified'] and all(x['certification_backend']=='cuda' for x in summary['legs'])
    assert all(x['certified'] and x['status']=='feasible' for x in summary['legs'])
    flown=[x for x in rows if x['column']==identifier][-len(summary['legs']):]
    assert len(flown)==len(summary['legs']) and all(x['status']=='converged' for x in flown)
(dest/'audit.json').write_text(json.dumps(audit,indent=2));add(dest/'audit.json','audit.json')
for p in root.rglob('*'):
    if not p.is_file() or p.suffix=='.gz':continue
    if p.parent.name=='official' and p.name in ('GTOC12_Verify','GTOC12_Asteroids_Data.txt','Result.txt'):continue
    add(p,'v773/'+p.relative_to(root).as_posix())

for version in ('v768','v770','v772'):
    run=home/('spacepdhcg-insertions-'+version)
    state=json.loads((run/'report.json').read_text());assert state['complete']
    if version!='v768':assert state['success']
    for p in run.iterdir():
        if p.is_file() and p.suffix in ('.json','.log','.py'):add(p,version+'/'+p.name) if version+'/'+p.name not in members else None
    if version!='v772':
        add(run/'repo/tests/test_gtoc12_gpu_joint_insertions.py',version+'/test_gtoc12_gpu_joint_insertions.py')
bench=home/'spacepdhcg-insertions-v769';b=json.loads((bench/'report.json').read_text());assert b['complete'] and b['success']
assert b['core_sha256']==r['core_sha256']
for p in bench.iterdir():
    if p.is_file():add(p,'v769/'+p.name)
assert r['gpu_telemetry']['completed_joint_insertion_batches']==13
assert r['gpu_telemetry']['completed_joint_evaluations']==15748
assert abs(r['score_kg']-12843.555695585188)<1e-8

qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if home.name=='ubuntu' else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so')
assert hashlib.sha256(qoco.read_bytes()).hexdigest()==r['qoco_sha256'];add(qoco,'qoco/libqoco.so')
(dest/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True));add(dest/'hardware.txt','hardware.txt')
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(members.items())};(dest/'FILES.json').write_text(json.dumps(manifest,indent=2))
archive=dest/('h100.tar.gz' if home.name=='ubuntu' else 'local.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for n,p in sorted(members.items()):tar.add(p,arcname=n)
    tar.add(dest/'FILES.json',arcname='FILES.json')
with tarfile.open(archive) as tar:
    names=[p.name for p in tar.getmembers()];assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
    for n,e in manifest.items():
        b=tar.extractfile(n).read();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest));(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
