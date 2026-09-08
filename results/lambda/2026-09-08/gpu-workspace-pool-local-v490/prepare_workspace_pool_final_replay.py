from pathlib import Path
import hashlib,json,tarfile,subprocess
p=Path('build/performance')
local=(p/'run_workspace_pool_replay_v479.py').read_text().replace('workspace-pool-replay-v479','workspace-pool-replay-v485').replace('build-spacepdhcg-workspace-pool-v476','build-spacepdhcg-workspace-pool-v482')
(p/'run_workspace_pool_replay_v485.py').write_text(local)
remote=(p/'run_workspace_pool_replay_v480.py').read_text().replace('workspace-pool-replay-v480','workspace-pool-replay-v486').replace('workspace-pool-v478','workspace-pool-v484')
(p/'run_workspace_pool_replay_v486.py').write_text(remote)
manifest=json.loads((p/'workspace-pool-source-sha256-v484.json').read_text());manifest['build/performance/replay_workspace_pool.py']=hashlib.sha256((p/'replay_workspace_pool.py').read_bytes()).hexdigest()
mp=p/'workspace-pool-source-sha256-v486.json';mp.write_text(json.dumps(manifest,indent=2))
archive=Path('/tmp/workspace-pool-replay-v486.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 t.add(p/'replay_workspace_pool.py',arcname='build/performance/replay_workspace_pool.py')
 t.add(mp,arcname='fused-tables-source-sha256.json')
subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-workspace-pool-replay-v486');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_workspace_pool_replay_v486.py').write_text(launch)
(p/'status_workspace_pool_replay_v486.py').write_text((p/'status_workspace_pool_replay_v480.py').read_text().replace('workspace-pool-replay-v480','workspace-pool-replay-v486'))
