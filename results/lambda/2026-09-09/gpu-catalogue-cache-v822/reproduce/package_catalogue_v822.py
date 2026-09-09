from pathlib import Path
import collections,hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-catalogue-evidence-v822';dest.mkdir();members={};audit={}
def add(path,name):
    assert name not in members;members[name]=path
build=home/'spacepdhcg-catalogue-v808';final=home/'spacepdhcg-catalogue-final-v819';source=json.loads((final/'source-manifest.json').read_text())
for name,digest in source['files'].items():
    p=final/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'runtime/repo/'+name)
for name,digest in json.loads((build/'source-manifest.json').read_text())['files'].items():
    if source['files'].get(name)!=digest:add(build/'repo'/name,'v808/original/'+name)
add(final/'source-manifest.json','runtime/source-manifest.json');add(build/'final/libspacepdhcg_cuda.so','runtime/libspacepdhcg_cuda.so');add(build/'build/cuda-tests/gtoc12_scvx_test','runtime/gtoc12_scvx_test')
data=home/('spacepdhcg/gtoc12/benchmarks/gtoc12/data' if home.name=='ubuntu' else 'worktrees/spacepdhcg-release/benchmarks/gtoc12/data');add(data/'bonus_coefficients.txt','runtime/bonus_coefficients.txt')
versions=[('v808','spacepdhcg-catalogue-v808'),('v809','spacepdhcg-catalogue-bench-v809'),('v810','spacepdhcg-integrated-refine-v810'),('v811','spacepdhcg-catalogue-tests-v811'),('v812','spacepdhcg-catalogue-bench-v812'),('v813','spacepdhcg-integrated-refine-v813'),('v814','spacepdhcg-catalogue-tests-v814'),('v815','spacepdhcg-integrated-refine-v815'),('v816','spacepdhcg-catalogue-bench-v816'),('v817','spacepdhcg-catalogue-master-v817')]
versions += [('v819','spacepdhcg-catalogue-final-v819'),('v820','spacepdhcg-catalogue-bench-v820')]
for version,dirname in versions:
    root=home/dirname;r=json.loads((root/'report.json').read_text());assert r['complete']
    assert r['success']==(version in ('v814','v815','v816','v817','v819','v820'))
    row={k:r[k] for k in ('success','seconds','score_kg','qualified','replacements','native_solves','solution_sha256','medians','budget_sweep') if k in r}
    if (root/'solves.jsonl').exists():
        solves=[json.loads(x) for x in (root/'solves.jsonl').read_text().splitlines()];assert len(solves)==r['native_solves']
        row.update(native_status_counts=dict(collections.Counter(x.get('status','error') for x in solves)),native_seconds=sum(x['seconds'] for x in solves))
    if 'solution_sha256' in r:
        path=root/'fleet/Result.txt';assert hashlib.sha256(path.read_bytes()).hexdigest()==r['solution_sha256']
        assert (root/'fleet/official/Result.txt').read_bytes()==path.read_bytes()
        assert r['qualified'] and r['independent']['ok'] and r['official']['ok']
    audit[version]=row
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix=='.gz' or any(x in ('repo','build','final','incumbents') for x in p.relative_to(root).parts):continue
        if p.parent.name=='official' and p.name in ('GTOC12_Verify','GTOC12_Asteroids_Data.txt','Result.txt'):continue
        add(p,version+'/'+p.relative_to(root).as_posix())
for version,dirname in [('v793','spacepdhcg-regeneration-refine-v793'),('v795','spacepdhcg-raw-refine-v795')]:
    root=home/dirname;add(root/'report.json','prior/'+version+'/report.json')
    for p in root.glob('ship-*/Result.txt'):
        add(p,'prior/'+version+'/'+p.relative_to(root).as_posix());add(p.parent/'route_summary.json','prior/'+version+'/'+p.parent.name+'/route_summary.json')
qoco=home/('spacepdhcg-retry-conditioning-v686/final/libqoco.so' if home.name=='ubuntu' else 'spacepdhcg-retry-conditioning-v683/final/libqoco.so');add(qoco,'runtime/libqoco.so')
assert hashlib.sha256(qoco.read_bytes()).hexdigest()==json.loads((home/'spacepdhcg-integrated-refine-v815/report.json').read_text())['qoco_sha256']
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
