from pathlib import Path
import hashlib,importlib.metadata,json,os,platform,subprocess,tarfile
home=Path.home();remote=home.name=='ubuntu';dest=home/'spacepdhcg-fleet-evidence-v724';dest.mkdir()
members={}
def add(path,name):
    if name in members:raise ValueError(name)
    members[name]=path
for version in ('v713','v720'):
    root=home/('spacepdhcg-fleet-'+version)
    r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success']
    m=json.loads((root/'source-manifest.json').read_text())
    for name,digest in m['files'].items():
        path=root/'repo'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        add(path,version+'/repo/'+name)
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.py'):add(path,version+'/'+path.name)
    add(root/'final/libspacepdhcg_cuda.so',version+'/final/libspacepdhcg_cuda.so')
for version in ('v717','v721','v723'):
    root=home/('spacepdhcg-fleet-'+version)
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.py'):
            if path.suffix=='.json':
                r=json.loads(path.read_text());assert r['complete'] and r['success'],(version,path.name)
            add(path,version+'/'+path.name)
if not remote:
    root=home/'spacepdhcg-fleet-v718';assert json.loads((root/'report-retry.json').read_text())['success']
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.py'):add(path,'v718/'+path.name)
    add(root/'repo/cpp/cuda/src/gtoc12_fleet.cuh','v718/gtoc12_fleet.cuh')
    add(root/'final/libspacepdhcg_cuda.so','v718/final/libspacepdhcg_cuda.so')
add(home/'spacepdhcg-fleet-pool-v716/pool.json','input/pool.json')
hardware=dict(system=platform.platform(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
(dest/'hardware.json').write_text(json.dumps(hardware,indent=2));add(dest/'hardware.json','hardware.json')
manifest={name:dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size) for name,path in sorted(members.items())}
(dest/'FILES.json').write_text(json.dumps(manifest,indent=2))
archive=dest/('h100.tar.gz' if remote else 'local.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for name,path in sorted(members.items()):tar.add(path,arcname=name,recursive=False)
    tar.add(dest/'FILES.json',arcname='FILES.json')
with tarfile.open(archive) as tar:
    names=[x.name for x in tar.getmembers()];assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
    for name,entry in manifest.items():
        data=tar.extractfile(name).read();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],name
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest))
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
