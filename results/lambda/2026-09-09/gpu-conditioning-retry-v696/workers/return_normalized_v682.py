from pathlib import Path
import hashlib
import json
import subprocess

root = Path('/home/angus/spacepdhcg-return-normalized-v682')
root.mkdir(exist_ok=False)
prior = Path('/home/angus/spacepdhcg-return-replay-v672')
fixture_bytes = (prior / 'ruiz5-fixture.json').read_bytes()
fixture = json.loads(fixture_bytes)
fixture['repeat_order'] *= 2
(root / 'fixture.json').write_text(json.dumps(fixture, indent=2))
driver = (prior / 'ruiz5-run.py').read_text().replace(hashlib.sha256(fixture_bytes).hexdigest(), hashlib.sha256((root / 'fixture.json').read_bytes()).hexdigest())
(root / 'run.py').write_text(driver)
worker = (prior / 'worker.py').read_text().replace(str(prior), str(root)).replace("'--max-solves','3'", "'--max-solves','6'")
(root / 'worker.py').write_text(worker)
with (root / 'worker.log').open('x') as log:
    child = subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python', str(root / 'worker.py')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=child.pid, root=str(root), repeats=6, ruiz=5, preserve_objective=False, pool='disabled by existing safety gate for normalized objectives')))
