from pathlib import Path
import subprocess
code="""from pathlib import Path
import json
home=Path.home();r=json.loads((home/'spacepdhcg-resident-final-v826/report.json').read_text());assert r['complete'] and r['success']
root=home/'spacepdhcg-resident-catalogue-bench-v824';root.mkdir()
script=(home/'spacepdhcg-catalogue-bench-v820/run.py').read_text().replace('spacepdhcg-catalogue-final-v819','spacepdhcg-resident-final-v826').replace('SPACEPDHCG_TEST_GTOC12_CACHE_CATALOGUE_DIGEST','SPACEPDHCG_TEST_GTOC12_RETAIN_COMPLETION_CATALOGUE').replace('Retained immutable catalogue digest; collection DP reuse enabled in both modes','Shared CUDA catalogue; catalogue digest and collection DP reuse enabled in both modes')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-catalogue-bench-v820/launch.py').read_text().replace('spacepdhcg-catalogue-bench-v820','spacepdhcg-resident-catalogue-bench-v824').replace('spacepdhcg-catalogue-final-v819','spacepdhcg-resident-final-v826')
(root/'launch.py').write_text(launch)
exec(launch)
"""
Path('build/performance/launch_resident_catalogue_both_v824.py').write_text(code)
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40)
print(r.stdout,r.stderr)
exec(code)

