from pathlib import Path
import json
import subprocess

root=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686')
old=json.loads((root/'campaign-v693/report.json').read_text())
assert old['complete'] and not old['success'] and not old['runs']
assert not (root/'campaign-v693/candidate0').exists()
runner=(root/'campaign-runner-v693.py').read_text().replace('campaign-v693','campaign-v695')
(root/'campaign-runner-v695.py').write_text(runner)
with (root/'campaign-runner-v695.log').open('x') as log:
    child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'campaign-runner-v695.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(root/'campaign-v695'),previous_error=old.get('exception'))))
