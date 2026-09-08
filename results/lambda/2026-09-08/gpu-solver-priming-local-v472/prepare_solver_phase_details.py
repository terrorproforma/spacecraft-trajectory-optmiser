from pathlib import Path
import hashlib,json,tarfile,os,subprocess
base=Path('build/performance/run_fused_tables_v456.py').read_text().replace('spacepdhcg-fused-tables-v456','spacepdhcg-solver-phase-v459').replace('fused-tables-v456.tar.gz','solver-phase-v459.tar.gz').replace('fused456_','solver_phase459_').replace('/home/ubuntu/spacepdhcg-fused-tables-v449/repo','/home/ubuntu/spacepdhcg-fused-tables-v456/repo')
start=base.index(' tests=');end=base.index(' cli=',start)
base=base[:start]+" env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'\n env['SPACEPDHCG_PHASE_OUTPUT']=str(root)\n"+base[end:]
base=base.replace("runpy.run_module('spacepdhcg',run_name='__main__')", "runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')")
Path('build/performance/run_solver_phase_v459.py').write_text(base)
files=['cpp/cuda/src/gtoc12_scvx.cu','build/performance/solver_phase_details.py']
manifest=Path('build/performance/solver-phase-source-sha256-v459.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/solver-phase-v459.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/solver-phase-v459.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-solver-phase-v459');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_solver_phase_v459.py').write_text(program)
Path('build/performance/status_solver_phase_v459.py').write_text(Path('build/performance/status_fused_tables_v456.py').read_text().replace('fused-tables-v456','solver-phase-v459'))
base=Path('build/performance/run_solver_phase_v458.py').read_text().replace('solver-phase-v458','solver-phase-v460').replace('solver_phase458','solver_phase460')
start=base.index('from shutil import copy2');end=base.index('r=dict(',start)
base=base[:start]+"env['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'\nenv['SPACEPDHCG_PHASE_OUTPUT']=str(root)\n"+base[end:]
base=base.replace('build-spacepdhcg-solver-phase-v460','build-spacepdhcg-solver-phase-v458').replace("runpy.run_module('spacepdhcg',run_name='__main__')", "runpy.run_path('build/performance/solver_phase_details.py',run_name='__main__')")
base=base.replace("'cpp/cuda/src/gtoc12_scvx.cu']", "'cpp/cuda/src/gtoc12_scvx.cu','build/performance/solver_phase_details.py']")
Path('build/performance/run_solver_phase_v460.py').write_text(base)
