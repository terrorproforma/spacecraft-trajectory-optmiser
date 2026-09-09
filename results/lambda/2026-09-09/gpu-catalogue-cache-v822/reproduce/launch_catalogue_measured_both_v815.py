from pathlib import Path
import json,subprocess
home=Path.home()
validation=home/'spacepdhcg-catalogue-tests-v814/report.json';r=json.loads(validation.read_text());assert r['complete'] and r['success']
root=home/'spacepdhcg-integrated-refine-v815';root.mkdir()
script=(home/'spacepdhcg-integrated-refine-v810/run.py').read_text().replace("(build/'report.json')","(home/'spacepdhcg-catalogue-tests-v814/report.json')")
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-integrated-refine-v810/launch.py').read_text().replace('spacepdhcg-integrated-refine-v810','spacepdhcg-integrated-refine-v815')
(root/'launch.py').write_text(launch)
exec(launch)
root=home/'spacepdhcg-catalogue-bench-v816';root.mkdir()
script=(home/'spacepdhcg-catalogue-bench-v809/run.py').read_text().replace("(build/'report.json')","(home/'spacepdhcg-catalogue-tests-v814/report.json')").replace('spacepdhcg-integrated-refine-v810','spacepdhcg-integrated-refine-v815')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-catalogue-bench-v809/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v809','spacepdhcg-catalogue-bench-v816')
(root/'launch.py').write_text(launch)
exec(launch)
