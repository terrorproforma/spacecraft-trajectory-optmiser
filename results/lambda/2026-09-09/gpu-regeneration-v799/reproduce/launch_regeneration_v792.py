from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile
root=Path.home()/'spacepdhcg-regeneration-v792';root.mkdir()
shutil.copyfile('build/performance/regenerate_fleet_v792.py',root/'run.py')
shutil.copyfile('results/lambda/2026-09-09/gpu-layouts-v780/h100-best/Result.txt',root/'fleet.txt')
shutil.copyfile(Path.home()/'spacepdhcg-fleet-routes-v766/input/fit.json',root/'fit.json')
(root/'input-hashes.json').write_text(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('fleet.txt','fit.json','run.py')},indent=2))
template=Path('build/performance/launch_grid_probe_v787.py').read_text();a=template.index("launch='''")+len("launch='''");b=template.index("'''",a)
launch=template[a:b].replace('spacepdhcg-grid-probe-v787','spacepdhcg-regeneration-v792').replace('spacepdhcg-grid-v786/final','spacepdhcg-grid-v788/final')
(root/'launch.py').write_text(launch)
archive=Path('/tmp/regeneration-v792.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in root.iterdir():tar.add(p,arcname=p.name)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
code="from pathlib import Path\nimport tarfile\nroot=Path.home()/'spacepdhcg-regeneration-v792';root.mkdir()\nwith tarfile.open('/tmp/regeneration-v792.tar.gz') as tar:tar.extractall(root,filter='data')\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
