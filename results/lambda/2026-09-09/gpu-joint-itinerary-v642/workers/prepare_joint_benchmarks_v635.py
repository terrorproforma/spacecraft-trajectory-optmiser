from pathlib import Path
import json
p=Path('build/performance')
runner=(p/'run_joint_benchmarks_v634.py').read_text().replace('shutil.copy2(script,source/script)','')
mapping={'/home/angus/spacepdhcg-joint-selection-v630':'/home/ubuntu/spacepdhcg-joint-selection-v632','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock'}
for old,new in mapping.items():runner=runner.replace(old,new)
code="from pathlib import Path\nimport json,subprocess\nroot=Path('/home/ubuntu/spacepdhcg-joint-selection-v632');repo=root/'repo'\nassert json.loads((root/'report.json').read_text())['success']\n"
code+="(repo/'build/performance/benchmark_joint_selection_v634.py').write_text("+repr((p/'benchmark_joint_selection_v634.py').read_text())+")\n"
code+="(root/'benchmark-runner.py').write_text("+repr(runner)+")\n"
code+="with (root/'benchmark-runner.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'benchmark-runner.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_joint_benchmarks_v635.py').write_text(code)
