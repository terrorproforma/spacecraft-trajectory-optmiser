from pathlib import Path
import collections,hashlib,json,math,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-layouts-evidence-v780';dest.mkdir(exist_ok=True);members={}
assert not list(dest.glob('*.tar.gz'))
def add(p,n):
    assert n not in members;members[n]=p
base=home/'spacepdhcg-layouts-v778';manifest=json.loads((base/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'v778/repo/'+name)
add(base/'source-manifest.json','v778/source-manifest.json');add(base/'final/libspacepdhcg_cuda.so','v778/final/libspacepdhcg_cuda.so')
root=home/'spacepdhcg-fleet-routes-v779';r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success'] and r['improved'] and r['independent']['ok'] and r['official']['ok']
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
    add(p,'v779/'+p.relative_to(root).as_posix())

for version in ('v776','v778'):
    run=home/('spacepdhcg-layouts-'+version)
    state=json.loads((run/'report.json').read_text());assert state['complete'] and state['success']
    for p in run.iterdir():
        if p.is_file() and p.suffix in ('.json','.log','.py'):
            if version+'/'+p.name not in members:add(p,version+'/'+p.name)
    if version!='v778':
        for name in ('src/spacepdhcg/gtoc12/gpu_joint_layouts.py','tests/test_gtoc12_gpu_joint_layouts.py'):
            add(run/'repo'/name,version+'/original/'+name)
for dirname,prefix in [('spacepdhcg-insertions-v775','v775'),('spacepdhcg-layouts-v777','v777')]:
    run=home/dirname;state=json.loads((run/'report.json').read_text());assert state['complete'] and state['success']
    for p in run.iterdir():
        if p.is_file():add(p,prefix+'/'+p.name)
b=json.loads((home/'spacepdhcg-layouts-v777/report.json').read_text());assert b['core_sha256']==r['core_sha256']
assert r['gpu_telemetry']['completed_joint_insertion_batches']==1
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
