from pathlib import Path
import hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-fleet-seed-evidence-v755';dest.mkdir(exist_ok=False);members={}
def add(p,n):
    assert n not in members;members[n]=p
for version in ('v747','v750'):
    base=home/('spacepdhcg-fleet-'+version)
    r=json.loads((base/'report.json').read_text());assert r['complete'] and r['success']
    manifest=json.loads((base/'source-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,version+'/repo/'+name)
    for p in base.iterdir():
        if p.is_file() and p.suffix in ('.json','.py','.log'):add(p,version+'/'+p.name)
    add(base/'final/libspacepdhcg_cuda.so',version+'/final/libspacepdhcg_cuda.so')
for version,category in [('v749','workspace'),('v752','workspace'),('v753','workspace'),('v754','differential'),('v746','profile'),('v748','profile'),('v751','profile')]:
    base=home/f'spacepdhcg-fleet-{category}-{version}'
    r=json.loads((base/'report.json').read_text());assert r['complete'] and r['success']
    for p in base.iterdir():
        if p.is_file() and p.suffix in ('.json','.py','.log','.cu','.so'):add(p,version+'/'+p.name)
for name in ('cpp/cuda/include/spacepdhcg/cuda/gtoc12_fleet_c_api.h','cpp/cuda/src/gtoc12_fleet.cuh','cpp/cuda/src/gtoc12_fleet_topology.cuh'):
    add(home/'spacepdhcg-fleet-v742/repo'/name,'v742/repo/'+name)
add(home/'spacepdhcg-fleet-pool-v716/pool.json','input/pool.json')
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
