from pathlib import Path
import hashlib
import json
import subprocess

remote=Path('/home/ubuntu').exists();home=Path('/home/ubuntu' if remote else '/home/angus')
root=home/'spacepdhcg-search-campaign-v703';build=home/'spacepdhcg-joint-search-v702'
report=json.loads((build/'report.json').read_text());assert report['complete']
stages={r['name']:r['returncode'] for r in report['stages']}
assert stages['build']==0 and stages['pytest']==0
assert hashlib.sha256((build/'final/libspacepdhcg_cuda.so').read_bytes()).hexdigest()==report['core_sha256']
if remote:
    assert all(stages[name]==0 for name in ('memcheck','racecheck','synccheck'))
    follow=json.loads((build/'followup-v705/report.json').read_text());assert follow['complete'] and follow['success']
    limitations=[]
else:
    follow=json.loads((build/'followup-v704/report.json').read_text());assert follow['complete'] and follow['success']
    assert stages['memcheck']==86 and 'cudaErrorUnknown (error 999)' in (build/'memcheck.log').read_text()
    limitations=['Local active-loop memcheck returns CUDA error 999; empty-loop memcheck and active-loop race/sync pass. H100 active-loop memcheck passes.']
worker=(root/'worker.py').read_text().replace("assert json.loads((build/'report.json').read_text())['success']","# Validation gate is recorded by the launch driver, including the local instrumentation limitation.")
target=root/'worker-v706.py';target.write_text(worker)
(root/'launch-v706.json').write_text(json.dumps(dict(core_sha256=report['core_sha256'],worker_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),validation_stages=stages,limitations=limitations),indent=2))
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (root/'worker-v706.log').open('x') as log:child=subprocess.Popen([py,str(target)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,root=str(root))))
