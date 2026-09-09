from pathlib import Path
import hashlib,subprocess
ignore=subprocess.check_output(['git','show','5f6987314292b8484c788ffd94d330fa9015f26a:.gitignore'])
script=Path('build/performance/validate_catalogue_v811.py').read_text()
script=script.replace("'runner_correction'", "'runner_correction'")
script=script.replace("def save():", "report['additional_test_fixture']={'.gitignore':hashlib.sha256((repo/'.gitignore').read_bytes()).hexdigest()}\nassert report['additional_test_fixture']['.gitignore']=="+repr(hashlib.sha256(ignore).hexdigest())+"\ndef save():",1)
code="from pathlib import Path\nimport json,subprocess\nhome=Path.home()\nfor dirname in ('spacepdhcg-catalogue-tests-v811','spacepdhcg-catalogue-bench-v812','spacepdhcg-integrated-refine-v813'):\n r=json.loads((home/dirname/'report.json').read_text());assert r['complete'] and not Path('/proc',str(r['pid'])).exists()\n"
code+="(home/'spacepdhcg-catalogue-v808/repo/.gitignore').write_bytes("+repr(ignore)+")\nroot=home/'spacepdhcg-catalogue-tests-v814';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\nwith (root/'worker.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)\n"
Path('build/performance/launch_catalogue_both_v814.py').write_text(code)
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
exec(code)
