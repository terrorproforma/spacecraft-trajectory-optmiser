from pathlib import Path
import subprocess
code="from pathlib import Path\nimport json,subprocess\nhome=Path.home()\nvalidation=home/'spacepdhcg-catalogue-tests-v814/report.json';r=json.loads(validation.read_text());assert r['complete'] and r['success']\n"
code+="root=home/'spacepdhcg-integrated-refine-v815';root.mkdir()\nscript=(home/'spacepdhcg-integrated-refine-v810/run.py').read_text().replace(\"(build/'report.json')\",\"(home/'spacepdhcg-catalogue-tests-v814/report.json')\")\n(root/'run.py').write_text(script)\nlaunch=(home/'spacepdhcg-integrated-refine-v810/launch.py').read_text().replace('spacepdhcg-integrated-refine-v810','spacepdhcg-integrated-refine-v815')\n(root/'launch.py').write_text(launch)\nexec(launch)\n"
code+="root=home/'spacepdhcg-catalogue-bench-v816';root.mkdir()\nscript=(home/'spacepdhcg-catalogue-bench-v809/run.py').read_text().replace(\"(build/'report.json')\",\"(home/'spacepdhcg-catalogue-tests-v814/report.json')\").replace('spacepdhcg-integrated-refine-v810','spacepdhcg-integrated-refine-v815')\n(root/'run.py').write_text(script)\nlaunch=(home/'spacepdhcg-catalogue-bench-v809/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v809','spacepdhcg-catalogue-bench-v816')\n(root/'launch.py').write_text(launch)\nexec(launch)\n"
Path('build/performance/launch_catalogue_measured_both_v815.py').write_text(code)
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
exec(code)
