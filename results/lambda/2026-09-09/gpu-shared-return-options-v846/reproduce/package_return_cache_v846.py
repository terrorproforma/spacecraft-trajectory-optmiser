from pathlib import Path
import hashlib,io,json,pstats,subprocess,tarfile
home=Path.home();build=home/'spacepdhcg-return-cache-v844';bench=home/'spacepdhcg-return-cache-bench-v845';leak=home/'spacepdhcg-return-cache-leaks-v847'
profile=home/'spacepdhcg-return-cache-profile-v848'
for p in (build,bench,leak,profile):
    r=json.loads((p/'report.json').read_text());assert r['complete'] and r['success'],p
root=home/'spacepdhcg-return-cache-evidence-v846';root.mkdir()
files={}
def add(name,path):
    assert name not in files;files[name]=path.read_bytes()
source=json.loads((build/'source-manifest.json').read_text())
for name,digest in source['files'].items():
    p=build/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name;add('runtime/repo/'+name,p)
for name,path in [('runtime/source-manifest.json',build/'source-manifest.json'),('runtime/libspacepdhcg_cuda.so',build/'final/libspacepdhcg_cuda.so'),('runtime/gtoc12_scvx_test',build/'build/cuda-tests/gtoc12_scvx_test')]:add(name,path)
for label,base in [('validation',build),('leaks',leak)]:
    for p in base.iterdir():
        if p.is_file() and p.suffix in ('.json','.log','.py') and p.name!='source-manifest.json':add(label+'/'+p.name,p)
for p in bench.rglob('*'):
    if p.is_file() and p.name in ('report.json','plans.json','run.py','launch.py','worker.log','fleet.txt','fit.json'):
        add('benchmark/'+p.relative_to(bench).as_posix(),p)
add('prior-v840-report.json',home/'spacepdhcg-admission-bench-v840/report.json')
profiles={}
for p in profile.rglob('*'):
    if p.is_file() and p.name in ('report.json','plans.json','run.py','launch.py','worker.log','fleet.txt','fit.json','ship-10.pstats','ship-21.pstats'):
        add('profile/'+p.relative_to(profile).as_posix(),p)
for ship in (10,21):
    s=pstats.Stats(str(profile/f'ship-{ship:02d}.pstats'))
    profiles[ship]=dict(total_calls=s.total_calls,total_seconds=s.total_tt,top_cumulative=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],total_calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in sorted(s.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:35]])
files['profile/summary.json']=json.dumps(profiles,indent=2).encode()
files['hardware.txt']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'])
manifest={n:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for n,data in files.items()}
archive=root/('h100.tar.gz' if home.name=='ubuntu' else 'local.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for name,data in sorted(files.items()):
        item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
    data=json.dumps(manifest,indent=2).encode();item=tarfile.TarInfo('FILES.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
receipt=dict(file=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),payload_files=len(files))
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
