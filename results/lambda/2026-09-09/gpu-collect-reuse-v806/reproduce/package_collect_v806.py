from pathlib import Path
import collections,hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-collect-evidence-v806';dest.mkdir();members={};audit={}
def add(path,name):
    assert name not in members;members[name]=path
final=home/'spacepdhcg-collect-final-v805';build=home/'spacepdhcg-collect-v800'
source=json.loads((final/'source-manifest.json').read_text())
for name,digest in source['files'].items():
    p=final/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'runtime/repo/'+name)
add(final/'source-manifest.json','runtime/source-manifest.json');add(build/'final/libspacepdhcg_cuda.so','runtime/libspacepdhcg_cuda.so')
add(build/'repo/tests/test_gtoc12_gpu_collect_workspace.py','v800/original/tests/test_gtoc12_gpu_collect_workspace.py')
data=home/('spacepdhcg/gtoc12/benchmarks/gtoc12/data' if home.name=='ubuntu' else 'worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
add(data/'bonus_coefficients.txt','runtime/bonus_coefficients.txt')
for version,dirname in [('v800','spacepdhcg-collect-v800'),('v805','spacepdhcg-collect-final-v805'),('v801','spacepdhcg-collect-bench-v801'),('v802','spacepdhcg-collect-fleet-v802'),('v803','spacepdhcg-collect-refine-v803'),('v804','spacepdhcg-collect-master-v804')]:
    root=home/dirname;r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success'],(version,r.get('error'))
    row={k:r[k] for k in ('seconds','score_kg','qualified','replacements','native_solves','solution_sha256','medians','budget_sweep') if k in r}
    if version=='v802':
        records=[json.loads(p.read_text()) for p in root.glob('ship-*/report.json')]
        telemetry=collections.Counter()
        for x in records:
            for k,v in x['gpu_telemetry'].items():
                if isinstance(v,(int,float)):telemetry[k]+=v
        row.update(ships=len(records),candidates=sum(x['candidates'] for x in records),expansions=sum(x['expansions'] for x in records),search_seconds=sum(x['search_seconds'] for x in records),lambert_evaluations=sum(x['lambert_evaluations'] for x in records),telemetry_totals=dict(telemetry))
        assert hashlib.sha256((root/'fleet.txt').read_bytes()).hexdigest()=='97d1f351bf6ad4907ddce887aa47d1fa974f587ab270374f4bb46a5898491d48'
    if (root/'solves.jsonl').exists():
        solves=[json.loads(x) for x in (root/'solves.jsonl').read_text().splitlines()];assert len(solves)==r['native_solves']
        row['native_status_counts']=dict(collections.Counter(x.get('status','error') for x in solves));row['native_seconds']=sum(x['seconds'] for x in solves)
    if 'solution_sha256' in r:
        path=root/'fleet/Result.txt';assert hashlib.sha256(path.read_bytes()).hexdigest()==r['solution_sha256']
        official=root/'fleet/official/Result.txt'
        if official.exists():assert official.read_bytes()==path.read_bytes()
        assert r['qualified'] and r['independent']['ok'] and r['official']['ok']
    audit[version]=row
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix=='.gz' or any(x in ('repo','build','final','incumbents') for x in p.relative_to(root).parts):continue
        if p.parent.name=='official' and p.name in ('GTOC12_Verify','GTOC12_Asteroids_Data.txt','Result.txt'):continue
        add(p,version+'/'+p.relative_to(root).as_posix())
# Certified historical columns used by the final master are retained explicitly.
for version,dirname in [('v793','spacepdhcg-regeneration-refine-v793'),('v795','spacepdhcg-raw-refine-v795')]:
    root=home/dirname;add(root/'report.json','prior/'+version+'/report.json')
    for p in root.glob('ship-*/Result.txt'):
        add(p,'prior/'+version+'/'+p.relative_to(root).as_posix())
        add(p.parent/'route_summary.json','prior/'+version+'/'+p.parent.name+'/route_summary.json')
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if home.name=='ubuntu' else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so');add(qoco,'runtime/libqoco.so')
assert hashlib.sha256(qoco.read_bytes()).hexdigest()==json.loads((home/'spacepdhcg-collect-refine-v803/report.json').read_text())['qoco_sha256']
(dest/'audit.json').write_text(json.dumps(audit,indent=2));add(dest/'audit.json','audit.json')
(dest/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True));add(dest/'hardware.txt','hardware.txt')
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(members.items())}
(dest/'FILES.json').write_text(json.dumps(manifest,indent=2))
archive=dest/('h100.tar.gz' if home.name=='ubuntu' else 'local.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for n,p in sorted(members.items()):tar.add(p,arcname=n)
    tar.add(dest/'FILES.json',arcname='FILES.json')
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest))
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
