from pathlib import Path
import hashlib,json,tarfile,subprocess
p=Path('build/performance')
files=list(json.loads((p/'workspace-pool-v476/report.json').read_text())['source_sha256'])+['build/performance/solver_phase_details.py','cpp/cuda/tests/gtoc12_workspace_reuse_test.cu']
local=(p/'run_workspace_pool_v477.py').read_text().replace('workspace-pool-v477','workspace-pool-v483').replace('workspace_pool477_','workspace_pool483_').replace('build-spacepdhcg-workspace-pool-v476','build-spacepdhcg-workspace-pool-v482')
a=local.index('names=');b=local.index('\nreport=',a);local=local[:a]+'names='+repr(files)+local[b:]
local=local.replace('fcntl.LOCK_EX|fcntl.LOCK_NB','fcntl.LOCK_EX')
(p/'run_workspace_pool_v483.py').write_text(local)
remote=(p/'run_workspace_pool_v478.py').read_text().replace('workspace-pool-v478','workspace-pool-v484').replace('workspace_pool478_','workspace_pool484_')
extra=""" binary=str(root/'workspace-pool-probe')
 run('compile-probe',['/usr/local/cuda/bin/nvcc','-std=c++17','--fmad=false','-arch=sm_90','-I'+str(repo/'cpp/include'),'-I'+str(repo/'cpp/cuda/include'),str(repo/'cpp/cuda/tests/gtoc12_workspace_reuse_test.cu'),'-L'+str(core.parent),'-lspacepdhcg_cuda','-o',binary])
 for tool in ['memcheck','synccheck','racecheck']:
  run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',binary],300)
"""
remote=remote.replace(" env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'",extra+" env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'")
(p/'run_workspace_pool_v484.py').write_text(remote)
mp=p/'workspace-pool-source-sha256-v484.json';mp.write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2))
archive=Path('/tmp/workspace-pool-v484.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in files:t.add(f,arcname=f)
 t.add(mp,arcname='fused-tables-source-sha256.json')
subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-workspace-pool-v484');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(remote)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_workspace_pool_v484.py').write_text(launch)
(p/'status_workspace_pool_v484.py').write_text((p/'status_workspace_pool_v478.py').read_text().replace('workspace-pool-v478','workspace-pool-v484'))
