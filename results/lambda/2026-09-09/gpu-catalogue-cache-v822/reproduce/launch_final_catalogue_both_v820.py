from pathlib import Path
import json
home=Path.home();r=json.loads((home/'spacepdhcg-catalogue-final-v819/report.json').read_text());assert r['complete'] and r['success']
root=home/'spacepdhcg-catalogue-bench-v820';root.mkdir()
script=(home/'spacepdhcg-catalogue-bench-v816/run.py').read_text().replace('spacepdhcg-catalogue-v808','spacepdhcg-catalogue-final-v819').replace('spacepdhcg-catalogue-tests-v814','spacepdhcg-catalogue-final-v819')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-catalogue-bench-v816/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v816','spacepdhcg-catalogue-bench-v820').replace('spacepdhcg-catalogue-v808','spacepdhcg-catalogue-final-v819')
(root/'launch.py').write_text(launch)
exec(launch)
