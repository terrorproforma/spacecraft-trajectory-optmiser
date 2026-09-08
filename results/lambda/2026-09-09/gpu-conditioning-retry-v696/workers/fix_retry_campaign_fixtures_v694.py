from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

prior=Path('/home/angus/spacepdhcg-joint-mesh-v662/repo')
source=Path('/home/angus/spacepdhcg-retry-report-v690/repo')
build=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
names=[]
for folder in ['results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources','results/lambda/2026-09-08/gpu-collect-profile-v272/v272']:
    names.extend(p.relative_to(prior).as_posix() for p in (prior/folder).rglob('*') if p.is_file())
manifest={}
archive=Path('/tmp/retry-fixtures-v694.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for name in names:
        raw=(prior/name).read_bytes();path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==raw
        else:path.write_bytes(raw)
        manifest[name]=hashlib.sha256(raw).hexdigest();tar.add(path,arcname=name)
(build/'fixture-correction-v694.json').write_text(json.dumps(dict(files=manifest,reason='The reporting overlay omitted two fixture directories present in the original frozen mesh repository. The first local campaign exited before any GPU work.'),indent=2))
old=json.loads((build/'campaign-v691/report.json').read_text());assert old['complete'] and not old['success'] and not old['runs']
runner=(build/'campaign-runner-v691.py').read_text().replace('campaign-v691','campaign-v694')
(build/'campaign-runner-v694.py').write_text(runner)
with (build/'campaign-runner-v694.log').open('x') as log:child=subprocess.Popen(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python',str(build/'campaign-runner-v694.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(local_pid=child.pid,files=len(manifest))))
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as file:file.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
remote="from pathlib import Path\nimport hashlib,json,tarfile\nbuild=Path('/home/ubuntu/spacepdhcg-retry-conditioning-v686');repo=build/'reporting-v692/repo'\nmanifest="+repr(manifest)+"\n"
remote+="with tarfile.open('/tmp/retry-fixtures-v694.tar.gz') as tar:\n for member in tar:\n  assert member.isfile() and member.name in manifest\n  raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==manifest[member.name]\n  path=repo/member.name;path.parent.mkdir(parents=True,exist_ok=True)\n  if path.exists():assert path.read_bytes()==raw\n  else:path.write_bytes(raw)\n(build/'fixture-correction-v694.json').write_text(json.dumps(manifest,indent=2))\nprint((build/'reporting-v692/report.json').read_text())\n"
Path('build/performance/correct_retry_h100_fixtures_v694.py').write_text(remote)
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=remote,text=True,capture_output=True,timeout=55);print(r.stdout);print(r.stderr);r.check_returncode()
