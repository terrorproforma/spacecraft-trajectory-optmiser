from pathlib import Path
import json,subprocess
home=Path.home();root=home/'spacepdhcg-native-collect-plan-profile-v858';root.mkdir()
state=json.loads((home/'spacepdhcg-native-collect-plan-bench-v857/report.json').read_text())
assert state['complete'] and state['success']
template=home/'spacepdhcg-native-collect-plan-bench-v857/warm-reuse'
script=(template/'run.py').read_text().replace('profile=cProfile.Profile();result=search.run()', 'profile=cProfile.Profile();profile.enable();result=search.run();profile.disable();profile.dump_stats(str(root/f"ship-{ship.ship_id:02d}.pstats"))')
assert 'profile.enable()' in script
(root/'run.py').write_text(script)
for name in ('fleet.txt','fit.json'):(root/name).write_bytes((template/name).read_bytes())
launch=(home/'spacepdhcg-native-collect-plan-bench-v857/launch.py').read_text().replace('spacepdhcg-native-collect-plan-bench-v857','spacepdhcg-native-collect-plan-profile-v858')
(root/'launch.py').write_text(launch);exec(launch)
