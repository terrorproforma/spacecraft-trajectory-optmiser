from pathlib import Path
import hashlib,json,tarfile,os,subprocess
base=Path('build/performance/run_solver_phase_v460.py').read_text().replace('solver-phase-v460','solver-phase-v461').replace('solver_phase460','solver_phase461')
Path('build/performance/run_solver_phase_v461.py').write_text(base)
base=Path('build/performance/run_solver_phase_v459.py').read_text().replace('spacepdhcg-solver-phase-v459','spacepdhcg-solver-phase-v462').replace('solver-phase-v459.tar.gz','solver-phase-v462.tar.gz').replace('solver_phase459_','solver_phase462_').replace('/home/ubuntu/spacepdhcg-fused-tables-v456/repo','/home/ubuntu/spacepdhcg-solver-phase-v459/repo')
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start)
base=base[:start]+base[end:]
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-solver-phase-v459/core-build/cuda/libspacepdhcg_cuda.so')")
Path('build/performance/run_solver_phase_v462.py').write_text(base)
files=['build/performance/solver_phase_details.py']
manifest=Path('build/performance/solver-phase-source-sha256-v462.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/solver-phase-v462.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/solver-phase-v462.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-solver-phase-v462');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_solver_phase_v462.py').write_text(program)
Path('build/performance/status_solver_phase_v462.py').write_text(Path('build/performance/status_solver_phase_v459.py').read_text().replace('solver-phase-v459','solver-phase-v462'))
