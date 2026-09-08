from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
dest=Path('results/lambda/2026-09-09/gpu-joint-geometry-v659');dest.mkdir(parents=True,exist_ok=False)
retrieved=Path('build/performance/retrieved-geometry-v659');retrieved.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
for suffix in ('tar.record.json','tar.manifest.json','tar.gz'):
 subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:/home/ubuntu/joint-geometry-v659-h100.'+suffix,str(dest/('h100-raw.'+suffix))],check=True,timeout=55)
 shutil.copy2('/home/angus/joint-geometry-v659-local.'+suffix,dest/('local-raw.'+suffix))
for gpu in ('local','h100'):
 path=dest/(gpu+'-raw.tar.gz');r=json.loads((dest/(gpu+'-raw.tar.record.json')).read_text());manifest=json.loads((dest/(gpu+'-raw.tar.manifest.json')).read_text())
 assert path.stat().st_size==r['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
 with tarfile.open(path) as tar:
  assert len(tar.getmembers())==len(manifest)==r['verified_members']
  for m in tar.getmembers():assert m.isfile() and m.name in manifest and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==manifest[m.name]['sha256']
  if gpu=='h100':
   for m in tar.getmembers():assert (retrieved/m.name).resolve().is_relative_to(retrieved.resolve())
   tar.extractall(retrieved,filter='data')
 print(gpu,r,flush=True)
shutil.copytree(retrieved/'campaign-v656/candidate0/best',dest/'h100-best')
shutil.copy2(retrieved/'campaign-v656/candidate0/report.json',dest/'h100-best/campaign-report.json')
(dest/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n*.patch -whitespace\n')
