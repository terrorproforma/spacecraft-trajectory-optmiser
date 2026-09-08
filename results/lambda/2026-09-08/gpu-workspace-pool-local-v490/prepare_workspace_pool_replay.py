from pathlib import Path
import json,hashlib,tarfile,subprocess
p=Path('build/performance')
replay=(p/'replay_early_graph.py').read_text().replace('SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH','SPACEPDHCG_TEST_GTOC12_QOCO_POOL')
replay=replay.replace('result=dict(index=index,', 'result=dict(workspace_creations=solution.solver_reports[-1]["workspace_creations"] if solution.solver_reports else 0,index=index,')
(p/'replay_workspace_pool.py').write_text(replay)
run=(p/'run_early_graph_replay_v470.py').read_text().replace('early-graph-replay-v470','workspace-pool-replay-v479').replace('build-spacepdhcg-early-graph-v465','build-spacepdhcg-workspace-pool-v476').replace('replay_early_graph.py','replay_workspace_pool.py')
(p/'run_workspace_pool_replay_v479.py').write_text(run)
run=(p/'run_early_graph_v471.py').read_text().replace('early-graph-v471','workspace-pool-replay-v480').replace('/home/ubuntu/spacepdhcg-early-graph-v469/repo','/home/ubuntu/spacepdhcg-workspace-pool-v478/repo').replace('/home/ubuntu/spacepdhcg-early-graph-v466/core-build','/home/ubuntu/spacepdhcg-workspace-pool-v478/core-build').replace(" env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'", " env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'").replace('replay_early_graph.py','replay_workspace_pool.py')
run=run.replace('try:\n shutil.copytree', "try:\n previous=json.loads(Path('/home/ubuntu/spacepdhcg-workspace-pool-v478/report.json').read_text());assert previous['complete'] and not previous.get('error')\n shutil.copytree")
(p/'run_workspace_pool_replay_v480.py').write_text(run)
manifest=json.loads((p/'workspace-pool-source-sha256-v478.json').read_text())
manifest['build/performance/replay_workspace_pool.py']=hashlib.sha256((p/'replay_workspace_pool.py').read_bytes()).hexdigest()
mp=p/'workspace-pool-source-sha256-v480.json';mp.write_text(json.dumps(manifest,indent=2))
archive=Path('/tmp/workspace-pool-replay-v480.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 t.add(p/'replay_workspace_pool.py',arcname='build/performance/replay_workspace_pool.py')
 t.add(mp,arcname='fused-tables-source-sha256.json')
subprocess.run(['scp','-q','-i','/tmp/traj-key.pem','-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:'+str(archive)],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-workspace-pool-replay-v480');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(run)+")\nwith (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
(p/'launch_workspace_pool_replay_v480.py').write_text(launch)
(p/'status_workspace_pool_replay_v480.py').write_text((p/'progress_early_graph_v471.py').read_text().replace('early-graph-v471','workspace-pool-replay-v480'))
