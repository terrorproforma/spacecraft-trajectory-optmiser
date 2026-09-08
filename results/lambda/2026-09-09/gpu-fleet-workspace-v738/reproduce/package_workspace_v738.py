from pathlib import Path
import hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-fleet-workspace-evidence-v738';dest.mkdir(exist_ok=False);members={}
def add(p,n):
    assert n not in members;members[n]=p
base=home/'spacepdhcg-fleet-v735'
manifest=json.loads((base/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    p=base/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'v735/repo/'+name)
for version,root in [('v735',base),('v736',home/'spacepdhcg-fleet-workspace-v736'),('v737',home/'spacepdhcg-fleet-workspace-v737')]:
    r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success'],(version,r)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.json','.py','.log','.nsys-rep'):add(p,version+'/'+p.name)
add(base/'final/libspacepdhcg_cuda.so','v735/final/libspacepdhcg_cuda.so')
add(home/'spacepdhcg-fleet-pool-v716/pool.json','input/pool.json')
if home.name=='angus':
    root=home/'spacepdhcg-fleet-workspace-v737'
    r=subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(root/'retained.nsys-rep')],text=True,capture_output=True)
    (dest/'nsys-kernel-summary.log').write_text(r.stdout+r.stderr);add(dest/'nsys-kernel-summary.log','profiling/nsys-kernel-summary.log')
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
