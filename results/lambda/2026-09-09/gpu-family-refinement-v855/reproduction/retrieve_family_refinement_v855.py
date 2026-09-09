from pathlib import Path
import hashlib,json,subprocess
repo=Path(__file__).resolve().parents[2]
out=repo/'results/lambda/2026-09-09/gpu-family-refinement-v855';out.mkdir(exist_ok=True)
code='''from pathlib import Path
import hashlib,json,subprocess,tarfile
home=Path.home()
names=['spacepdhcg-family-refinement-v851','spacepdhcg-family-penalty-v852','spacepdhcg-family-retiming-v853','spacepdhcg-family-retiming-v854','spacepdhcg-family-returns-v855']
reports={name:json.loads((home/name/'report.json').read_text()) for name in names}
for name,r in reports.items():
    assert r['complete'] and not Path('/proc/'+str(r['pid'])).exists(),name
    if name.endswith('v853'):assert not r['success'] and r['native_solves']==0
    else:assert r['success'],name
root=home/'spacepdhcg-family-refinement-evidence-v855';assert not root.exists();root.mkdir()
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
(root/'terminal.json').write_text(json.dumps(dict(all_owned_workers_exited=True,gpu=gpu.stdout,compute_processes=apps.stdout),indent=2))
qoco=home/'spacepdhcg-retry-conditioning-v686'
library=qoco/'final/libqoco.so'
ldd=subprocess.run(['ldd',str(library)],capture_output=True,text=True,check=True)
deps={}
for line in ldd.stdout.splitlines():
    parts=line.split()
    if '=>' in parts:
        p=Path(parts[parts.index('=>')+1])
        if p.is_file():deps[str(p)]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(root/'runtime-dependencies.json').write_text(json.dumps(dict(ldd=ldd.stdout,files=deps),indent=2))
archive=home/'spacepdhcg-family-refinement-evidence-v855.tar.gz';assert not archive.exists()
files={}
with tarfile.open(archive,'w:gz') as tar:
    def add(p,name):
        assert p.is_file() and not p.is_symlink()
        raw=p.read_bytes();files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        tar.add(p,arcname=name,recursive=False)
    for name in names:
        for p in sorted((home/name).rglob('*')):
            if p.is_file():add(p,name+'/'+str(p.relative_to(home/name)))
    for p in root.iterdir():add(p,'terminal/'+p.name)
    add(library,'qoco-runtime/libqoco.so')
    for name in ['source-manifest.json','worker.py','qoco-configure.log','qoco-build.log','report.json']:
        add(qoco/name,'qoco-runtime/'+name)
    for p in sorted((qoco/'qoco').rglob('*')):
        if p.is_file() and '.git' not in p.parts:add(p,'qoco-runtime/source/'+str(p.relative_to(qoco/'qoco')))
manifest=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,files=files)
(home/'spacepdhcg-family-refinement-evidence-v855-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(archive_bytes=manifest['archive_bytes'],archive_sha256=manifest['archive_sha256'],files=len(files),native_solves=sum(r['native_solves'] for r in reports.values()),gains=[a for r in reports.values() for a in r.get('attempts',[]) if a.get('verified_gain')])))
'''
ssh=['ssh.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@192.222.55.229']
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,timeout=60,check=True);print(r.stdout,r.stderr)
for source,target in [('spacepdhcg-family-refinement-evidence-v855.tar.gz','evidence.tar.gz'),('spacepdhcg-family-refinement-evidence-v855-manifest.json','manifest.json')]:
    assert not (out/target).exists()
    subprocess.run(['scp.exe','-i','C:/Users/Angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/'+source,str(out/target)],check=True,timeout=100)
manifest=json.loads((out/'manifest.json').read_text())
assert hashlib.sha256((out/'evidence.tar.gz').read_bytes()).hexdigest()==manifest['archive_sha256']
print('Retrieved and verified',manifest['archive_bytes'],'bytes')
