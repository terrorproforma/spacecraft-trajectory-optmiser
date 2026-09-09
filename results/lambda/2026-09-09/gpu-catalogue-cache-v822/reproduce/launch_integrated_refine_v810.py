from pathlib import Path
import subprocess
home=Path.home();root=home/'spacepdhcg-integrated-refine-v810';root.mkdir()
script=(home/'spacepdhcg-collect-refine-v803/run.py').read_text().replace('spacepdhcg-collect-v800','spacepdhcg-catalogue-v808')
script=script.replace("try:\n", "try:\n    build=home/'spacepdhcg-catalogue-v808'\n    while not json.loads((build/'report.json').read_text())['complete']:time.sleep(2)\n    assert json.loads((build/'report.json').read_text())['success']\n",1)
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-collect-refine-v803/launch.py').read_text().replace('spacepdhcg-collect-refine-v803','spacepdhcg-integrated-refine-v810').replace('spacepdhcg-collect-v800','spacepdhcg-catalogue-v808')
(root/'launch.py').write_text(launch)
remote="from pathlib import Path\nroot=Path.home()/'spacepdhcg-integrated-refine-v810';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
exec(launch)
