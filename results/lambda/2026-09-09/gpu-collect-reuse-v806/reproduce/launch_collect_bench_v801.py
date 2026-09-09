from pathlib import Path
import os,subprocess
root=Path.home()/'spacepdhcg-collect-bench-v801';root.mkdir()
script=Path('build/performance/bench_collect_v801.py').read_text();(root/'run.py').write_text(script)
launch=(Path.home()/'spacepdhcg-raw-regeneration-v794/launch.py').read_text().replace('spacepdhcg-raw-regeneration-v794','spacepdhcg-collect-bench-v801').replace('spacepdhcg-grid-v788','spacepdhcg-collect-v800');(root/'launch.py').write_text(launch)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-collect-bench-v801';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
