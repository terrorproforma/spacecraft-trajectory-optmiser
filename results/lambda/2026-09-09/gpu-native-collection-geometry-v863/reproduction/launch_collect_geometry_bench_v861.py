from pathlib import Path
import json, subprocess

home = Path.home()
root = home / 'spacepdhcg-collect-geometry-bench-v861'
validation = json.loads((home / 'spacepdhcg-collect-geometry-v860/report.json').read_text())
assert validation['complete'] and validation['success'], validation
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
root.mkdir()
script = (home / 'spacepdhcg-native-collect-plan-bench-v857/run.py').read_text()
script = script.replace('spacepdhcg-native-collect-plan-v856', 'spacepdhcg-collect-geometry-v860')
script = script.replace('SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN', 'SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_GEOMETRY')
script = script.replace('Native mining and burn-policy selection versus host preparation; both use retained CUDA tables and DP operators', 'Native pair geometry and harvest-phase grid construction versus host preparation; native mining and burn scheduling enabled in both modes')
script = script.replace('spacepdhcg-return-cache-bench-v845/warm-reuse', 'spacepdhcg-native-collect-plan-bench-v857/warm-reuse')
(root / 'run.py').write_text(script)
launch = (home / 'spacepdhcg-native-collect-plan-bench-v857/launch.py').read_text().replace('spacepdhcg-native-collect-plan-bench-v857', 'spacepdhcg-collect-geometry-bench-v861').replace('spacepdhcg-native-collect-plan-v856', 'spacepdhcg-collect-geometry-v860')
(root / 'launch.py').write_text(launch)
exec(launch)
