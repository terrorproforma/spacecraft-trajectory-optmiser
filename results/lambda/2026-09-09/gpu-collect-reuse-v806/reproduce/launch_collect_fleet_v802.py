from pathlib import Path
import os,subprocess
home=Path.home();root=home/'spacepdhcg-collect-fleet-v802';root.mkdir()
script=(home/'spacepdhcg-raw-regeneration-v794/run.py').read_text().replace('spacepdhcg-grid-v788','spacepdhcg-collect-v800')
script=script.replace('    pass # Inputs copied byte-for-byte from the hashed v792 run', "    while not json.loads((home/'spacepdhcg-collect-bench-v801/report.json').read_text())['complete']:time.sleep(2)\n    assert json.loads((home/'spacepdhcg-collect-bench-v801/report.json').read_text())['success']")
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-raw-regeneration-v794/launch.py').read_text().replace('spacepdhcg-raw-regeneration-v794','spacepdhcg-collect-fleet-v802').replace('spacepdhcg-grid-v788','spacepdhcg-collect-v800');(root/'launch.py').write_text(launch)
(root/'fleet.txt').write_bytes(Path('results/lambda/2026-09-09/gpu-regeneration-v799/h100-best/Result.txt').read_bytes())
(root/'fit.json').write_bytes((home/'spacepdhcg-regeneration-v792/fit.json').read_bytes())
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-collect-fleet-v802';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\n(root/'fleet.txt').write_bytes((Path.home()/'spacepdhcg-regeneration-master-v798/fleet/Result.txt').read_bytes())\n(root/'fit.json').write_bytes((Path.home()/'spacepdhcg-regeneration-v792/fit.json').read_bytes())\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
