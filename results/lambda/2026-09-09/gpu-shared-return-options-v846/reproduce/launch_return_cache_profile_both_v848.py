from pathlib import Path
import subprocess
home=Path.home();root=home/'spacepdhcg-return-cache-profile-v848';root.mkdir()
runner='''from pathlib import Path
import json,os,subprocess,time
root=Path(__file__).resolve().parent;home=Path.home()
(root/'waiting.json').write_text(json.dumps(dict(pid=os.getpid())))
for name in ('spacepdhcg-return-cache-bench-v845','spacepdhcg-return-cache-leaks-v847'):
    while not (home/name/'report.json').exists() or not json.loads((home/name/'report.json').read_text())['complete']:time.sleep(2)
    assert json.loads((home/name/'report.json').read_text())['success']
template=home/'spacepdhcg-return-cache-bench-v845/warm-reuse'
script=(template/'run.py').read_text().replace('profile=cProfile.Profile();result=search.run()', 'profile=cProfile.Profile();profile.enable();result=search.run();profile.disable();profile.dump_stats(str(root/f"ship-{ship.ship_id:02d}.pstats"))')
(root/'run.py').write_text(script)
for name in ('fleet.txt','fit.json'):(root/name).write_bytes((template/name).read_bytes())
launch=(home/'spacepdhcg-return-cache-bench-v845/launch.py').read_text().replace('spacepdhcg-return-cache-bench-v845','spacepdhcg-return-cache-profile-v848')
(root/'launch.py').write_text(launch);exec(launch)
'''
(root/'wait.py').write_text(runner)
with (root/'wait.log').open('x') as log:print(subprocess.Popen(['python3',str(root/'wait.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
