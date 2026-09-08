from pathlib import Path
import hashlib
import json
import subprocess

root=Path('/home/angus/spacepdhcg-return-conditioning-retry-v685');root.mkdir(exist_ok=False)
prior=Path('/home/angus/spacepdhcg-return-replay-v672')
build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
report=json.loads((build/'report.json').read_text());assert report['success']
original=(prior/'fixture.json').read_bytes();fixture=json.loads(original);fixture['repeat_order']*=2
(root/'fixture.json').write_text(json.dumps(fixture,indent=2))
driver=(prior/'run.py').read_text().replace(hashlib.sha256(original).hexdigest(),hashlib.sha256((root/'fixture.json').read_bytes()).hexdigest()).replace('6fd02e87f3e148c0e62aac2bee731bc58397605a1d29ef3aebec95b0b4303c6c',report['libraries']['libspacepdhcg_cuda.so']).replace('0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315',report['libraries']['libqoco.so'])
(root/'run.py').write_text(driver)
worker=(prior/'worker.py').read_text().replace(str(prior),str(root)).replace('/home/angus/spacepdhcg-joint-mesh-v662',str(build)).replace("build/'build/cuda/libspacepdhcg_cuda.so'","build/'final/libspacepdhcg_cuda.so'").replace("build/'build/cuda'","build/'final'").replace('/home/angus/build-qoco-scaled-pool-v540/final',str(build/'final')).replace("'--max-solves','3'","'--max-solves','6'")
worker=worker.replace("cmd=['/home", "env['SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY']='1'\ncmd=['/home")
(root/'worker.py').write_text(worker)
with (root/'worker.log').open('x') as log:child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(root),repeats=6,libraries=report['libraries'])))
