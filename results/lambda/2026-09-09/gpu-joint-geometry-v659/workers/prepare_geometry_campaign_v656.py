from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/angus/spacepdhcg-joint-geometry-v651');source=root/'repo'
names=['build/performance/orphan-recovery-v595/run.py','results/gtoc12/hop_inflation_fit.json','results/lambda/2026-09-06/fleet_master_v11/fleet/Result.txt']
for name in names:
 target=source/name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
(root/'campaign-fixtures.json').write_text(json.dumps({n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names},indent=2))
runner=Path('build/performance/retry_joint_campaign_v639.py').read_text().replace('spacepdhcg-joint-selection-v630','spacepdhcg-joint-geometry-v651').replace('campaign-v639','campaign-v656')
runner=runner.replace("(('baseline-retry',0),)","(('baseline0',0),('candidate0',1),('candidate1',1),('baseline1',0))")
runner=runner.replace('env=dict(environment,SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=str(mode))','env=dict(environment,SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION="1",SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY=str(mode))').replace('selection=mode','resident_geometry=mode')
(root/'campaign-runner-v656.py').write_text(runner)
assert json.loads((root/'validation-v654/report.json').read_text())['success']
with (root/'campaign-runner-v656.log').open('x') as log:child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(root/'campaign-runner-v656.py')],cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid)))
