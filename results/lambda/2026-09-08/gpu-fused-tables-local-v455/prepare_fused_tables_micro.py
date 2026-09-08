from pathlib import Path
import json, hashlib, tarfile, os, subprocess
trace='build/performance/fused-tables-profile-v451/collect-grids.json'
assert json.loads(Path('build/performance/fused-tables-profile-v451/report.json').read_text())['complete']
base=Path('build/performance/check_fused_tables_v447.py').read_text().replace('fused-tables-v447','fused-tables-micro-v452')
start=base.index('frozen=');end=base.index("core=frozen/'libspacepdhcg_cuda.so'",start)
base=base[:start]+"frozen=Path('/home/angus/build-spacepdhcg-fused-tables-v447/final')\n"+base[end:]
base=base.replace("sources=[", "sources=['build/performance/fused_tables_micro.py',"+repr(trace)+",")
start=base.index('jobs=');end=base.index('\nr=dict(',start)
boot="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/fused_tables_micro.py',run_name='__main__')"
base=base[:start]+"jobs=[('micro',[py,'-c',"+repr(boot)+","+repr(trace)+",str(root/'measurement.json')])]"+base[end:]
Path('build/performance/run_fused_tables_micro_v452.py').write_text(base)

base=Path('build/performance/run_fused_tables_v449.py').read_text().replace('spacepdhcg-fused-tables-v449','spacepdhcg-fused-tables-v453').replace('fused-tables-v449.tar.gz','fused-tables-v453.tar.gz').replace('/home/ubuntu/spacepdhcg-earth-beam-v444/repo','/home/ubuntu/spacepdhcg-fused-tables-v449/repo')
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start)
base=base[:start]+base[end:]
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-fused-tables-v449/core-build/cuda/libspacepdhcg_cuda.so')")
start=base.index(' tests=');end=base.index(" report['complete']=True",start)
base=base[:start]+" run('micro',[py,'-c',"+repr(boot)+","+repr(trace)+",str(root/'measurement.json')],900)\n assert json.loads((root/'measurement.json').read_text())['complete']\n"+base[end:]
Path('build/performance/run_fused_tables_v453.py').write_text(base)
files=['build/performance/fused_tables_micro.py',trace]
manifest=Path('build/performance/fused-tables-source-sha256-v453.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/fused-tables-v453.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/fused-tables-v453.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-fused-tables-v453');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_fused_tables_v453.py').write_text(program)
Path('build/performance/status_fused_tables_v453.py').write_text(Path('build/performance/status_fused_tables_v449.py').read_text().replace('fused-tables-v449','fused-tables-v453'))
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
