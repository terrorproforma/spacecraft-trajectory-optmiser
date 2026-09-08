from pathlib import Path
import hashlib,json,tarfile,subprocess,os
p=Path('build/performance')
files=list(json.loads((p/'workspace-pool-v476/report.json').read_text())['source_sha256'])+['build/performance/solver_phase_details.py']
base=(p/'run_early_graph_v467.py').read_text().replace('early-graph-v467','workspace-pool-v477').replace('early_graph467_','workspace_pool477_').replace('build-spacepdhcg-early-graph-v465','build-spacepdhcg-workspace-pool-v476').replace('SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH','SPACEPDHCG_TEST_GTOC12_QOCO_POOL')
a=base.index('names=');b=base.index('\nreport=',a);base=base[:a]+'names='+repr(files)+base[b:]
(p/'run_workspace_pool_v477.py').write_text(base)
base=(p/'run_early_graph_v466.py').read_text().replace('early-graph-v466','workspace-pool-v478').replace('early_graph466_','workspace_pool478_').replace('/home/ubuntu/spacepdhcg-solver-phase-v462/repo','/home/ubuntu/spacepdhcg-early-graph-v471/repo').replace(" env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1'", " env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'")
base=base.replace("  env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='1' if candidate else '0'", "  env['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']='1' if candidate else '0'")
base=base.replace("tests=['tests/test_gtoc12_gpu_scvx.py'", "tests=['tests/test_gtoc12_gpu_workspace_pool.py','tests/test_gtoc12_gpu_scvx.py'").replace("*tests,'-q'", "*tests,'-s','-q'")
(p/'run_workspace_pool_v478.py').write_text(base)
manifest=p/'workspace-pool-source-sha256-v478.json';manifest.write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2))
archive=Path('/tmp/workspace-pool-v478.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in files:t.add(f,arcname=f)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/workspace-pool-v478.tar.gz'],check=True,timeout=55)
launch="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-workspace-pool-v478');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
(p/'launch_workspace_pool_v478.py').write_text(launch)
(p/'status_workspace_pool_v478.py').write_text((p/'status_early_graph_v466.py').read_text().replace('early-graph-v466','workspace-pool-v478'))
