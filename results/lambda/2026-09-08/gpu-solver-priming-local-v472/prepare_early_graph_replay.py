from pathlib import Path
import hashlib,json,tarfile,os,subprocess
base=Path('build/performance/replay_stationary_v405.py').read_text().replace("os.environ['SPACEPDHCG_TEST_GTOC12_STATIONARY_FAILURE']=mode", "os.environ['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']=mode")
base=base.replace('history=solution.history,reports=solution.solver_reports)', 'history=solution.history,reports=solution.solver_reports,outer_transfer_bytes=solution.outer_transfer_bytes,priming=sum(p["solve_seconds"] is not None for p in solution.solver_reports))')
Path('build/performance/replay_early_graph.py').write_text(base)
base=Path('build/performance/run_stationary_v407.py').read_text().replace('stationary-v407','early-graph-replay-v470').replace('build-spacepdhcg-stationary-v406','build-spacepdhcg-early-graph-v465').replace('replay_stationary_v405.py','replay_early_graph.py')
base=base.replace("[('smoke','1','0,1,12,44,60,129,134,176,224'),('baseline','0',None),('candidate','1',None)]", "[('baseline','0',None),('candidate','1',None)]")
Path('build/performance/run_early_graph_replay_v470.py').write_text(base)
base=Path('build/performance/run_early_graph_v469.py').read_text().replace('spacepdhcg-early-graph-v469','spacepdhcg-early-graph-v471').replace('early-graph-v469.tar.gz','early-graph-v471.tar.gz').replace('/home/ubuntu/spacepdhcg-early-graph-v466/repo','/home/ubuntu/spacepdhcg-early-graph-v469/repo')
start=base.index(' tests=');end=base.index(" report['complete']=True",start)
replay="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/replay_early_graph.py',run_name='__main__')"
base=base[:start]+" replay="+repr(replay)+"\n for name,mode in [('baseline','0'),('candidate','1')]:\n  run(name,[py,'-c',replay,mode,str(root/name)],1200)\n"+base[end:]
Path('build/performance/run_early_graph_v471.py').write_text(base)
files=['build/performance/replay_early_graph.py','build/performance/grid-cache-fleet-v403/scvx-calls.json']
manifest=Path('build/performance/early-graph-source-sha256-v471.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/early-graph-v471.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/early-graph-v471.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-early-graph-v471');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_early_graph_v471.py').write_text(program)
Path('build/performance/status_early_graph_v471.py').write_text(Path('build/performance/status_early_graph_v469.py').read_text().replace('early-graph-v469','early-graph-v471'))
