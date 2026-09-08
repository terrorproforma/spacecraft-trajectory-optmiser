from pathlib import Path
import subprocess
for path in [Path('/home/ubuntu/bound-types-v574-runner.log'),Path('/home/ubuntu/spacepdhcg-bound-types-v569/repo/build/performance/bound-types-native-v574/report.json')]:
 if path.exists():print(path,path.stat().st_mtime,path.read_text()[-2500:])
print(subprocess.run(['pgrep','-af','bound_types_native_v574'],capture_output=True,text=True).stdout)
