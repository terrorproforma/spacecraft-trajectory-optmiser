from pathlib import Path
import json, hashlib, tarfile, os, subprocess
base=Path('build/performance/check_fused_tables_v447.py').read_text().replace('fused-tables-v447','fused-tables-v454').replace("env['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1'\n",'')
Path('build/performance/check_fused_tables_v454.py').write_text(base)
base=Path('build/performance/run_earth_beam_fleet_v438.py').read_text().replace('earth-beam-fleet-v438','fused-tables-fleet-v455').replace('earth_fleet438','fused_fleet455').replace('build-spacepdhcg-earth-beam-v431','build-spacepdhcg-fused-tables-v454').replace("SPACEPDHCG_TEST_GTOC12_EARTH_BEAM='1',",'')
base=base.replace("'cpp/cuda/src/orbitweaver_beam.cuh']", "'cpp/cuda/src/orbitweaver_beam.cuh','cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h']")
Path('build/performance/run_fused_tables_fleet_v455.py').write_text(base)

base=Path('build/performance/run_fused_tables_v449.py').read_text().replace('spacepdhcg-fused-tables-v449','spacepdhcg-fused-tables-v456').replace('fused-tables-v449.tar.gz','fused-tables-v456.tar.gz').replace('fused449_','fused456_').replace('/home/ubuntu/spacepdhcg-earth-beam-v444/repo','/home/ubuntu/spacepdhcg-fused-tables-v449/repo')
base=base.replace(" env['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1'\n",'')
base=base.replace("for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1' if candidate else '0'", "for name,candidate in [('default',True)]:")
Path('build/performance/run_fused_tables_v456.py').write_text(base)
files=list(json.loads(Path('build/performance/fused-tables-source-sha256-v449.json').read_text()))
manifest=Path('build/performance/fused-tables-source-sha256-v456.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/fused-tables-v456.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/fused-tables-v456.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-fused-tables-v456');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_fused_tables_v456.py').write_text(program)
Path('build/performance/status_fused_tables_v456.py').write_text(Path('build/performance/status_fused_tables_v449.py').read_text().replace('fused-tables-v449','fused-tables-v456'))
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
