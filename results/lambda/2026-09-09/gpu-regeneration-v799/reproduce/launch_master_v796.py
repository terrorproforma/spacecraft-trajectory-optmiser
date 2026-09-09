from pathlib import Path
import os,subprocess
root=Path.home()/'spacepdhcg-regeneration-master-v796';root.mkdir()
script=Path('build/performance/master_regeneration_v796.py').read_text();(root/'run.py').write_text(script)
launch=(Path.home()/'spacepdhcg-regeneration-refine-v793/launch.py').read_text().replace('spacepdhcg-regeneration-refine-v793','spacepdhcg-regeneration-master-v796');(root/'launch.py').write_text(launch)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-regeneration-master-v796';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
