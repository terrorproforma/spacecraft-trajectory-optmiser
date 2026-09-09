from pathlib import Path
import json
home=Path.home();r=json.loads((home/'spacepdhcg-resident-final-v826/report.json').read_text());assert r['complete'] and r['success']
root=home/'spacepdhcg-resident-catalogue-bench-v824';root.mkdir()
script=(home/'spacepdhcg-catalogue-bench-v820/run.py').read_text().replace('spacepdhcg-catalogue-final-v819','spacepdhcg-resident-final-v826').replace('SPACEPDHCG_TEST_GTOC12_CACHE_CATALOGUE_DIGEST','SPACEPDHCG_TEST_GTOC12_RETAIN_COMPLETION_CATALOGUE').replace('Retained immutable catalogue digest; collection DP reuse enabled in both modes','Shared CUDA catalogue; catalogue digest and collection DP reuse enabled in both modes')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-catalogue-bench-v820/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v820','spacepdhcg-resident-catalogue-bench-v824').replace('spacepdhcg-catalogue-final-v819','spacepdhcg-resident-final-v826')
(root/'launch.py').write_text(launch)
exec(launch)
