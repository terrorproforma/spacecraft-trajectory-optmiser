from pathlib import Path
import hashlib,json,tarfile,os,subprocess
files=['cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/src/native_qoco_adapter.cpp','tests/test_gtoc12_gpu_scvx.py','build/performance/solver_phase_details.py']
base=Path('build/performance/run_fused_tables_v449.py').read_text().replace('spacepdhcg-fused-tables-v449','spacepdhcg-early-graph-v466').replace('fused-tables-v449.tar.gz','early-graph-v466.tar.gz').replace('fused449_','early_graph466_').replace('/home/ubuntu/spacepdhcg-earth-beam-v444/repo','/home/ubuntu/spacepdhcg-solver-phase-v462/repo').replace('SPACEPDHCG_TEST_GTOC12_FUSED_TABLES','SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH')
start=base.index(' tests=');end=base.index(' cli=',start)
base=base[:start]+" tests=['tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py']\n run('pytest',[py,'-c',boot,*tests,'-q'],300)\n env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'\n"+base[end:]
base=base.replace("runpy.run_module('spacepdhcg',run_name='__main__')", "runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')")
base=base.replace("  cmd=[py,'-c',cli,", "  env['SPACEPDHCG_PHASE_OUTPUT']=str(root/name)\n  cmd=[py,'-c',cli,")
Path('build/performance/run_early_graph_v466.py').write_text(base)
manifest=Path('build/performance/early-graph-source-sha256-v466.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/early-graph-v466.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/early-graph-v466.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-early-graph-v466');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_early_graph_v466.py').write_text(program)
Path('build/performance/status_early_graph_v466.py').write_text(Path('build/performance/status_solver_phase_v462.py').read_text().replace('solver-phase-v462','early-graph-v466'))

base=Path('build/performance/run_fused_tables_v448.py').read_text().replace('fused-tables-v448','early-graph-v467').replace('fused448_','early_graph467_').replace('build-spacepdhcg-fused-tables-v447','build-spacepdhcg-early-graph-v465').replace('SPACEPDHCG_TEST_GTOC12_FUSED_TABLES','SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH')
start=base.index('names=');end=base.index('\nreport=',start)
base=base[:start]+'names='+repr(files)+base[end:]
base=base.replace("runpy.run_module('spacepdhcg',run_name='__main__')", "runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')")
base=base.replace("   cmd=[py,'-c',cli,", "   env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'\n   env['SPACEPDHCG_PHASE_OUTPUT']=str(root/name)\n   cmd=[py,'-c',cli,")
Path('build/performance/run_early_graph_v467.py').write_text(base)
