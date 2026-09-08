from pathlib import Path
import json,hashlib,os,tarfile,subprocess
p=Path('build/performance');tag='retained-replay-legs-v517'
r=(p/'run_workspace_pool_replay_v486.py').read_text().replace('workspace-pool-replay-v486',tag).replace('workspace-pool-v484','retained-replay-v514')
r=r.replace("run(name,[py,'-c',replay,mode,str(root/name)],1200)", "env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']=mode\n  run(name,[py,'-c',replay,'1',str(root/name)],1200)")
(p/'run_retained_replay_legs_v517.py').write_text(r)
sources=['build/performance/replay_workspace_pool.py','build/performance/grid-cache-fleet-v403/scvx-calls.json']
mp=p/(tag+'-source-sha256.json');mp.write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},indent=2))
archive=Path('/tmp/'+tag+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in sources:t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-"+tag+"');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(r)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_retained_replay_legs_v517.py').write_text(launch)
