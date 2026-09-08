from pathlib import Path
import os,subprocess,json,hashlib,tarfile,sys
version=int(sys.argv[1]);assert version in [353,359]
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
remote_root='/home/ubuntu/spacepdhcg-step-v353' if version==353 else '/home/ubuntu/spacepdhcg-step-final-v359'
remote='''from pathlib import Path
import tarfile,hashlib,json
root=Path(ROOT)
report=json.loads((root/'report.json').read_text());assert report['complete']
files=[p for p in root.iterdir() if p.is_file()]+list((root/'overlay').rglob('*'))+[root/'qoco/src/cone.cu']
files+=list((root/'qoco/src').glob('qoco*step*.cuh'))+list((root/'qoco/src').glob('qoco_cone_arithmetic.cuh'))
for row in report.get('campaigns',[]):files+=list((root/('v'+str(row['version']))).rglob('*'))
files=[p for p in files if p.is_file() and '__pycache__' not in str(p) and p.name!='hashes.json']
files=sorted(set(files))
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'hashes.json').write_text(json.dumps(hashes,indent=2));files.append(root/'hashes.json')
archive=Path(ARCHIVE)
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))))
'''.replace('ROOT',repr(remote_root)).replace('ARCHIVE',repr(f'/tmp/step-v{version}-results.tar.gz'))
ssh=['ssh','-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@192.222.55.229','python3 -']
r=subprocess.run(ssh,input=remote,text=True,capture_output=True,timeout=55,check=True);print(r.stdout);meta=json.loads(r.stdout)
root=Path(f'build/performance/step-lambda-v{version}');root.mkdir(exist_ok=False)
archive=root/'remote.tar.gz';subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes',f'ubuntu@192.222.55.229:/tmp/step-v{version}-results.tar.gz',str(archive)],check=True,timeout=55)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
with tarfile.open(archive) as t:
 for m in t.getmembers():
  p=(root/'remote'/m.name).resolve();assert m.isfile() and p.is_relative_to((root/'remote').resolve())
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
for name,sha in json.loads((root/'remote/hashes.json').read_text()).items():assert hashlib.sha256((root/'remote'/name).read_bytes()).hexdigest()==sha
(root/'retrieval.json').write_text(json.dumps(meta,indent=2));print('verified all remote files')
