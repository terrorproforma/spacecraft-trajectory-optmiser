from pathlib import Path
import hashlib,json,os,subprocess,tarfile
p=Path('build/performance');dest=Path('results/lambda/2026-09-08/gpu-scaled-workspace-v553')
dest.mkdir(exist_ok=False)
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
cmd=['ssh','-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@192.222.55.229','python3 -']
r=subprocess.run(cmd,input=(p/'archive_scaled_remote_v553.py').read_text(),text=True,capture_output=True,timeout=55)
assert r.returncode==0,r.stderr
expected=json.loads(r.stdout);(dest/'retrieval.json').write_text(json.dumps(expected,indent=2))
archive=dest/'lambda-raw.tar.gz'
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229:'+expected['path'],str(archive)],check=True,timeout=180)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['sha256']
local=p/'retrieved-scaled-v553';local.mkdir(exist_ok=False)
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():assert m.isfile() and (local/m.name).resolve().is_relative_to(local.resolve())
 t.extractall(local)
manifest=json.loads((local/'archive-manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((local/name).read_bytes()).hexdigest()==sha,name
(dest/'lambda-archive-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(**expected,verified_files=len(manifest))))
