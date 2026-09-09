from pathlib import Path
import hashlib,json,subprocess,tarfile
home=Path.home();dest=home/'spacepdhcg-resident-evidence-v827';dest.mkdir();members={}
def add(p,name):
    assert name not in members;members[name]=p
build=home/'spacepdhcg-resident-catalogue-v823';final=home/'spacepdhcg-resident-final-v826'
source=json.loads((final/'source-manifest.json').read_text())
for name,digest in source['files'].items():
    p=final/'repo'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p,'runtime/repo/'+name)
old=json.loads((build/'source-manifest.json').read_text())
for name,digest in old['files'].items():
    if source['files'][name]!=digest:add(build/'repo'/name,'v823/original/'+name)
    if name.startswith('cpp/'):assert source['files'][name]==digest
add(final/'source-manifest.json','runtime/source-manifest.json')
add(final/'final/libspacepdhcg_cuda.so','runtime/libspacepdhcg_cuda.so')
add(build/'build/cuda-tests/gtoc12_scvx_test','runtime/gtoc12_scvx_test')
for version,dirname in (('v823','spacepdhcg-resident-catalogue-v823'),('v826','spacepdhcg-resident-final-v826'),('v824','spacepdhcg-resident-catalogue-bench-v824'),('v828','spacepdhcg-resident-leaks-v828')):
    root=home/dirname;r=json.loads((root/'report.json').read_text());assert r['complete'] and r['success']==(version!='v823')
    for p in root.rglob('*'):
        if not p.is_file() or any(x in ('repo','build','final','incumbents') for x in p.relative_to(root).parts):continue
        add(p,version+'/'+p.relative_to(root).as_posix())
(dest/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True));add(dest/'hardware.txt','hardware.txt')
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(members.items())}
(dest/'FILES.json').write_text(json.dumps(manifest,indent=2))
archive=dest/('h100.tar.gz' if home.name=='ubuntu' else 'local.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for name,p in sorted(members.items()):tar.add(p,arcname=name)
    tar.add(dest/'FILES.json',arcname='FILES.json')
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(manifest))
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
