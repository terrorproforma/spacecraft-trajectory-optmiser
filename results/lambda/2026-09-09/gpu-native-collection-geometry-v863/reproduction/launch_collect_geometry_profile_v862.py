from pathlib import Path
import json,subprocess
home=Path.home();root=home/'spacepdhcg-collect-geometry-profile-v862';root.mkdir()
state=json.loads((home/'spacepdhcg-collect-geometry-bench-v861/report.json').read_text())
assert state['complete'] and state['success']
template=home/'spacepdhcg-collect-geometry-bench-v861/warm-reuse'
script=(template/'run.py').read_text().replace('profile=cProfile.Profile();result=search.run()', 'profile=cProfile.Profile();profile.enable();result=search.run();profile.disable();profile.dump_stats(str(root/f"ship-{ship.ship_id:02d}.pstats"))')
assert 'profile.enable()' in script
(root/'run.py').write_text(script)
for name in ('fleet.txt','fit.json'):(root/name).write_bytes((template/name).read_bytes())
launch=(home/'spacepdhcg-collect-geometry-bench-v861/launch.py').read_text().replace('spacepdhcg-collect-geometry-bench-v861','spacepdhcg-collect-geometry-profile-v862')
(root/'launch.py').write_text(launch);exec(launch)
