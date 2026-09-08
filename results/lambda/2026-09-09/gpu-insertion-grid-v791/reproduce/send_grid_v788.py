from pathlib import Path
import hashlib,json,os,subprocess,tarfile
root=Path('/home/angus/spacepdhcg-grid-v788')
manifest=json.loads((root/'source-manifest.json').read_text())['files']
archive=Path('/tmp/grid-source-v788.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for name,digest in manifest.items():
        path=root/'repo'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        tar.add(path,arcname='repo/'+name)
    for name in ('worker.py','source-manifest.json','incumbents'):tar.add(root/name,arcname=name)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as file:file.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
code='''from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
root=Path('/home/ubuntu/spacepdhcg-grid-v788');root.mkdir(exist_ok=False)
with tarfile.open('/tmp/grid-source-v788.tar.gz') as tar:tar.extractall(root,filter='data')
manifest=json.loads((root/'source-manifest.json').read_text())['files']
for name,digest in manifest.items():assert hashlib.sha256((root/'repo'/name).read_bytes()).hexdigest()==digest,name
shutil.copytree(root.with_name('spacepdhcg-grid-v786')/'final',root/'final')
with (root/'worker.log').open('x') as log:
    child=subprocess.Popen(['python3',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(root=str(root),pid=child.pid,files=len(manifest))))
'''
Path('build/performance/launch_grid_h100_v788.py').write_text(code)
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=55)
print(r.stdout);print(r.stderr);r.check_returncode()
