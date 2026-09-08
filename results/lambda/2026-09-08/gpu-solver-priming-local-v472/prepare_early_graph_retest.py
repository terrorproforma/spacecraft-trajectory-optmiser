from pathlib import Path
import hashlib,json,tarfile,os,subprocess
base=Path('build/performance/check_early_graph_v464.py').read_text().replace("root=Path('build/performance/early-graph-v464')", "root=Path('build/performance/early-graph-v468')")
start=base.index('jobs=');end=base.index('\nr=dict(',start)
base=base[:start]+"jobs=[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py','-q'])]"+base[end:]
Path('build/performance/check_early_graph_v468.py').write_text(base)

base=Path('build/performance/run_early_graph_v466.py').read_text().replace('spacepdhcg-early-graph-v466','spacepdhcg-early-graph-v469').replace('early-graph-v466.tar.gz','early-graph-v469.tar.gz').replace('early_graph466_','early_graph469_').replace('/home/ubuntu/spacepdhcg-solver-phase-v462/repo','/home/ubuntu/spacepdhcg-early-graph-v466/repo')
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start)
base=base[:start]+base[end:]
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-early-graph-v466/core-build/cuda/libspacepdhcg_cuda.so')")
Path('build/performance/run_early_graph_v469.py').write_text(base)
files=['tests/test_gtoc12_gpu_scvx.py']
manifest=Path('build/performance/early-graph-source-sha256-v469.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/early-graph-v469.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/early-graph-v469.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-early-graph-v469');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_early_graph_v469.py').write_text(program)
Path('build/performance/status_early_graph_v469.py').write_text(Path('build/performance/status_early_graph_v466.py').read_text().replace('early-graph-v466','early-graph-v469'))
