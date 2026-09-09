from pathlib import Path
import subprocess
code="""from pathlib import Path
import json,subprocess
home=Path.home();root=home/'spacepdhcg-native-collect-plan-bench-v857';root.mkdir()
validation=json.loads((home/'spacepdhcg-native-collect-plan-v856/report.json').read_text())
assert validation['complete'] and validation['success']
script=(home/'spacepdhcg-return-cache-bench-v845/run.py').read_text()
script=script.replace('spacepdhcg-return-cache-v844','spacepdhcg-native-collect-plan-v856')
script=script.replace('SPACEPDHCG_TEST_GTOC12_RETAIN_RETURN_OPTIONS','SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN')
script=script.replace('Shared resident return options; CUDA admission, expansion, catalogue and DP reuse enabled in both modes','Native mining and burn-policy selection versus host preparation; both use retained CUDA tables and DP operators')
script=script.replace('spacepdhcg-admission-bench-v840/warm-reuse','spacepdhcg-return-cache-bench-v845/warm-reuse')
script=script.replace('assert abs(a-b)<=2e-9,(a,b)','assert a==b,(a,b)')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-return-cache-bench-v845/launch.py').read_text().replace('spacepdhcg-return-cache-bench-v845','spacepdhcg-native-collect-plan-bench-v857').replace('spacepdhcg-return-cache-v844','spacepdhcg-native-collect-plan-v856')
(root/'launch.py').write_text(launch);exec(launch)
"""
Path('build/performance/launch_collect_plan_bench_host_v857.py').write_text(code)
r=subprocess.run(['ssh','-i','/home/angus/.ssh/spacepdhcg-expansion-key.pem','-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=40)
print('H100',r.stdout,r.stderr)
