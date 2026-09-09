from pathlib import Path
import subprocess
home=Path.home();tests=Path('build/performance/validate_catalogue_v811.py').read_text()
code="from pathlib import Path\nimport json,subprocess\nhome=Path.home()\n"
code+="for dirname in ('spacepdhcg-catalogue-v808','spacepdhcg-catalogue-bench-v809','spacepdhcg-integrated-refine-v810'):\n r=json.loads((home/dirname/'report.json').read_text());assert r['complete'] and not Path('/proc',str(r['pid'])).exists()\n"
code+="root=home/'spacepdhcg-catalogue-tests-v811';root.mkdir()\n(root/'run.py').write_text("+repr(tests)+")\nwith (root/'worker.log').open('x') as log:print('tests',subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)\n"
code+="root=home/'spacepdhcg-integrated-refine-v813';root.mkdir()\nscript=(home/'spacepdhcg-integrated-refine-v810/run.py').read_text().replace(\"(build/'report.json')\",\"(home/'spacepdhcg-catalogue-tests-v811/report.json')\")\n(root/'run.py').write_text(script)\nlaunch=(home/'spacepdhcg-integrated-refine-v810/launch.py').read_text().replace('spacepdhcg-integrated-refine-v810','spacepdhcg-integrated-refine-v813')\n(root/'launch.py').write_text(launch)\nexec(launch)\n"
code+="root=home/'spacepdhcg-catalogue-bench-v812';root.mkdir()\nscript=(home/'spacepdhcg-catalogue-bench-v809/run.py').read_text().replace(\"(build/'report.json')\",\"(home/'spacepdhcg-catalogue-tests-v811/report.json')\").replace('spacepdhcg-integrated-refine-v810','spacepdhcg-integrated-refine-v813')\n(root/'run.py').write_text(script)\nlaunch=(home/'spacepdhcg-catalogue-bench-v809/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v809','spacepdhcg-catalogue-bench-v812')\n(root/'launch.py').write_text(launch)\nexec(launch)\n"
Path('build/performance/resume_catalogue_both_v811.py').write_text(code)
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
exec(code)
