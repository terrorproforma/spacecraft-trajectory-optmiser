from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/angus/spacepdhcg-joint-mesh-v662');source=root/'repo'
previous=Path('/home/angus/spacepdhcg-joint-geometry-v651')
manifest=json.loads((previous/'campaign-fixtures.json').read_text())
for name,digest in manifest.items():
 data=(previous/'repo'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 target=source/name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
(root/'campaign-fixtures.json').write_text(json.dumps(manifest,indent=2))
runner=(previous/'campaign-runner-v656.py').read_text().replace('spacepdhcg-joint-geometry-v651','spacepdhcg-joint-mesh-v662').replace('campaign-v656','campaign-v667')
runner=runner.replace('SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY=str(mode)','SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY="1",SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH=str(mode)').replace('resident_geometry=mode','device_mesh=mode')
(root/'campaign-runner-v667.py').write_text(runner)
assert json.loads((root/'validation-v665/report.json').read_text())['success']
with (root/'campaign-runner-v667.log').open('x') as log:
 child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(root/'campaign-runner-v667.py')],cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid)))
