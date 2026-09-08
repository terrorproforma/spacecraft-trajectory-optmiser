from pathlib import Path
import json,os,subprocess,tarfile
source=Path('/home/angus/spacepdhcg-joint-selection-v630/repo');p=Path('build/performance')
manifest=json.loads((source.parent/'campaign-fixtures.json').read_text())
archive=Path('/tmp/joint-campaign-v637.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for name in manifest:tar.add(source/name,arcname=name)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
runner=(p/'run_joint_campaign_v636.py').read_text()
mapping={'/home/angus/spacepdhcg-joint-selection-v630':'/home/ubuntu/spacepdhcg-joint-selection-v632','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/build-qoco-scaled-pool-v540/final':'/home/ubuntu/spacepdhcg-scaled-pool-v545/final','/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib':'/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib','/usr/local/cuda-12.8':'/usr/local/cuda'}
for old,new in mapping.items():runner=runner.replace(old,new)
code="from pathlib import Path\nimport json,subprocess,tarfile,hashlib\nroot=Path('/home/ubuntu/spacepdhcg-joint-selection-v632');repo=root/'repo'\nassert json.loads((root/'benchmark-v634/report.json').read_text())['complete']\n"
code+="manifest="+repr(manifest)+"\nwith tarfile.open('/tmp/joint-campaign-v637.tar.gz') as tar:\n for m in tar.getmembers():assert m.isfile() and m.name in manifest and not (repo/m.name).exists()\n tar.extractall(repo,filter='data')\nfor name,digest in manifest.items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest\n"
code+="(root/'campaign-fixtures.json').write_text(json.dumps(manifest,indent=2))\n(root/'campaign-runner.py').write_text("+repr(runner)+")\n"
code+="with (root/'campaign-runner.log').open('x') as log:child=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'campaign-runner.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_joint_campaign_v637.py').write_text(code)
