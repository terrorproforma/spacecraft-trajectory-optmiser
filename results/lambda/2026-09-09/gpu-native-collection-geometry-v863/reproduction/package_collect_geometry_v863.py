from pathlib import Path
import hashlib,json,pstats,subprocess,tarfile
home=Path.home();remote=home.name=='ubuntu'
runtime=home/'spacepdhcg-collect-geometry-v860'
bench=home/'spacepdhcg-collect-geometry-bench-v861'
profile=home/'spacepdhcg-collect-geometry-profile-v862'
root=home/'spacepdhcg-collect-geometry-evidence-v863';root.mkdir()
reports={}
for key,path in [('runtime',runtime),('benchmark',bench)]+([('profile',profile)] if remote else []):
    r=json.loads((path/'report.json').read_text());assert r['complete'] and r['success'],key
    for field in ('pid','child_pid'):
        if field in r:assert not Path('/proc/'+str(r[field])).exists(),(key,field)
    reports[key]=r
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
(root/'terminal.json').write_text(json.dumps(dict(all_owned_workers_exited=True,gpu=gpu.stdout,compute_processes=apps.stdout),indent=2))
if remote:
    rows={}
    for ship in (10,21):
        stats=pstats.Stats(str(profile/f'ship-{ship:02d}.pstats'))
        rows[ship]=[dict(file=key[0],line=key[1],function=key[2],primitive_calls=value[0],calls=value[1],self_seconds=value[2],cumulative_seconds=value[3]) for key,value in sorted(stats.stats.items(),key=lambda x:-x[1][3])[:40]]
    (root/'profiles.json').write_text(json.dumps(rows,indent=2))
archive=root/('h100.tar.gz' if remote else 'local.tar.gz');files={}
with tarfile.open(archive,'w:gz') as tar:
    def add(path,name):
        assert path.is_file() and not path.is_symlink()
        raw=path.read_bytes();files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        tar.add(path,arcname=name,recursive=False)
    source=json.loads((runtime/'source-manifest.json').read_text())
    for name,expected in source['files'].items():
        p=runtime/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
        add(p,'runtime/repo/'+name)
    for p in sorted(runtime.iterdir()):
        if p.is_file() and p.name!='source.tar.gz':add(p,'runtime/'+p.name)
    add(runtime/'final/libspacepdhcg_cuda.so','runtime/libspacepdhcg_cuda.so')
    add(runtime/'build/cuda-tests/gtoc12_scvx_test','runtime/gtoc12_scvx_test')
    for key,path in [('benchmark',bench)]+([('profile',profile)] if remote else []):
        for p in sorted(path.rglob('*')):
            if p.is_file():add(p,key+'/'+str(p.relative_to(path)))
    for name in ('terminal.json','profiles.json'):
        if (root/name).exists():add(root/name,name)
manifest=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,files=files)
(root/('h100-manifest.json' if remote else 'local-manifest.json')).write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
