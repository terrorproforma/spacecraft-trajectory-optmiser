from pathlib import Path
import collections,hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-regeneration-evidence-v799';dest.mkdir();members={};audit={}
def add(path,name):
    assert name not in members;members[name]=path
base=home/'spacepdhcg-grid-v788';source=json.loads((base/'source-manifest.json').read_text())
for name,digest in source['files'].items():
    p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'runtime/repo/'+name)
add(base/'source-manifest.json','runtime/source-manifest.json');add(base/'final/libspacepdhcg_cuda.so','runtime/libspacepdhcg_cuda.so')
data=home/('spacepdhcg/gtoc12/benchmarks/gtoc12/data' if home.name=='ubuntu' else 'worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
add(data/'bonus_coefficients.txt','runtime/bonus_coefficients.txt')
for version,dirname in [('v792','spacepdhcg-regeneration-v792'),('v793','spacepdhcg-regeneration-refine-v793'),('v794','spacepdhcg-raw-regeneration-v794'),('v795','spacepdhcg-raw-refine-v795'),('v796','spacepdhcg-regeneration-master-v796'),('v798','spacepdhcg-regeneration-master-v798')]:
    root=home/dirname;report=json.loads((root/'report.json').read_text());assert report['complete'] and report['success'],(version,report.get('error'))
    row={k:report[k] for k in ('seconds','score_kg','qualified','replacements','native_solves','solution_sha256') if k in report}
    if version in ('v792','v794'):
        records=[json.loads(p.read_text()) for p in root.glob('ship-*/report.json')]
        row.update(ships=len(records),candidates=sum(x['candidates'] for x in records),expansions=sum(x['expansions'] for x in records),search_seconds=sum(x['search_seconds'] for x in records),lambert_evaluations=sum(x['lambert_evaluations'] for x in records),profiled=version=='v792')
        telemetry=collections.Counter()
        for r in records:
            for k,v in r['gpu_telemetry'].items():
                if isinstance(v,(int,float)):telemetry[k]+=v
        row['telemetry_totals']=dict(telemetry)
    if (root/'solves.jsonl').exists():
        solves=[json.loads(x) for x in (root/'solves.jsonl').read_text().splitlines()]
        assert len(solves)==report['native_solves']
        row['native_status_counts']=dict(collections.Counter(x.get('status','error') for x in solves));row['native_seconds']=sum(x['seconds'] for x in solves)
    if 'solution_sha256' in report:
        assert hashlib.sha256((root/'fleet/Result.txt').read_bytes()).hexdigest()==report['solution_sha256']
        official_copy=root/'fleet/official/Result.txt'
        if official_copy.exists():assert official_copy.read_bytes()==(root/'fleet/Result.txt').read_bytes()
        assert report['qualified']==(report['independent']['ok'] and report['official']['ok'])
    if version=='v793':assert report['qualified'] is False
    if version in ('v795','v796','v798'):assert report['qualified'] is True
    audit[version]=row
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix=='.gz':continue
        if p.parent.name=='official' and p.name in ('GTOC12_Verify','GTOC12_Asteroids_Data.txt','Result.txt'):continue
        add(p,version+'/'+p.relative_to(root).as_posix())
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if home.name=='ubuntu' else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so');add(qoco,'runtime/libqoco.so')
assert hashlib.sha256(qoco.read_bytes()).hexdigest()==json.loads((home/'spacepdhcg-raw-refine-v795/report.json').read_text())['qoco_sha256']
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
