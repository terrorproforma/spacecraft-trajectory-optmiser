from pathlib import Path
import subprocess
home=Path.home();root=home/'spacepdhcg-catalogue-bench-v809';root.mkdir()
script=Path('build/performance/bench_collect_v801.py').read_text().replace('spacepdhcg-collect-v800','spacepdhcg-catalogue-v808').replace('SPACEPDHCG_TEST_GTOC12_REUSE_COLLECT_DP','SPACEPDHCG_TEST_GTOC12_CACHE_CATALOGUE_DIGEST')
script=script.replace('runs=[])',"runs=[],comparison='Retained immutable catalogue digest; collection DP reuse enabled in both modes')")
script=script.replace("    script=(home/", "    refinement=home/'spacepdhcg-integrated-refine-v810/report.json'\n    while not refinement.exists() or not json.loads(refinement.read_text())['complete']:time.sleep(2)\n    assert json.loads(refinement.read_text())['success']\n    script=(home/",1)
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-collect-bench-v801/launch.py').read_text().replace('spacepdhcg-collect-bench-v801','spacepdhcg-catalogue-bench-v809').replace('spacepdhcg-collect-v800','spacepdhcg-catalogue-v808')
(root/'launch.py').write_text(launch)
remote="from pathlib import Path\nroot=Path.home()/'spacepdhcg-catalogue-bench-v809';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=40);print(r.stdout,r.stderr)
exec(launch)
