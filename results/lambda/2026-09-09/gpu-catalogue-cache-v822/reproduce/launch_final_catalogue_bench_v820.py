from pathlib import Path
import subprocess
code="from pathlib import Path\nimport json\nhome=Path.home();r=json.loads((home/'spacepdhcg-catalogue-final-v819/report.json').read_text());assert r['complete'] and r['success']\nroot=home/'spacepdhcg-catalogue-bench-v820';root.mkdir()\nscript=(home/'spacepdhcg-catalogue-bench-v816/run.py').read_text().replace('spacepdhcg-catalogue-v808','spacepdhcg-catalogue-final-v819').replace('spacepdhcg-catalogue-tests-v814','spacepdhcg-catalogue-final-v819')\n(root/'run.py').write_text(script)\nlaunch=(home/'spacepdhcg-catalogue-bench-v816/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v816','spacepdhcg-catalogue-bench-v820').replace('spacepdhcg-catalogue-v808','spacepdhcg-catalogue-final-v819')\n(root/'launch.py').write_text(launch)\nexec(launch)\n"
Path('build/performance/launch_final_catalogue_both_v820.py').write_text(code)
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
exec(code)
