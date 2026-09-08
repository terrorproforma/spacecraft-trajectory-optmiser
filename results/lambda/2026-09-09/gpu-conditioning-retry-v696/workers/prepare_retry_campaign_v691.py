from pathlib import Path
import hashlib
import json
import subprocess

build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
source=Path('/home/angus/spacepdhcg-retry-report-v690/repo')
prior=Path('/home/angus/spacepdhcg-joint-mesh-v662')
manifest=json.loads((prior/'campaign-fixtures.json').read_text())
for name,digest in manifest.items():
    raw=(prior/'repo'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
    path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
(build/'campaign-fixtures.json').write_text(json.dumps(manifest,indent=2))
runner=(prior/'campaign-runner-v667.py').read_text().replace(str(prior),str(build)).replace("source=build/'repo'",'source=Path('+repr(str(source))+')').replace('campaign-v667','campaign-v691').replace("build/'build/cuda/libspacepdhcg_cuda.so'","build/'final/libspacepdhcg_cuda.so'").replace('/home/angus/build-qoco-scaled-pool-v540/final',str(build/'final')).replace('SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH=str(mode)','SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH="1",SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY=str(mode)').replace('device_mesh=mode','conditioning_retry=mode')
(build/'campaign-runner-v691.py').write_text(runner)
with (build/'campaign-runner-v691.log').open('x') as log:child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(build/'campaign-runner-v691.py')],cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(build/'campaign-v691'))))
