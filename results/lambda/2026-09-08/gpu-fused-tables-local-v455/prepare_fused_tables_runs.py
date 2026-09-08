from pathlib import Path
import json, hashlib, tarfile, os, subprocess

files=['cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/src/orbitweaver_beam.cuh','cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','src/spacepdhcg/gtoc12/gpu_collect_tables.py','src/spacepdhcg/gtoc12/gpu_lambert.py','tests/test_gtoc12_gpu_fused_tables.py']
tests=['tests/test_gtoc12_gpu_fused_tables.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_harvest_window.py']
base=Path('build/performance/run_earth_beam_v440.py').read_text().replace('earth-beam-v440','fused-tables-v448').replace('earth440_', 'fused448_').replace('build-spacepdhcg-earth-beam-v431','build-spacepdhcg-fused-tables-v447').replace('SPACEPDHCG_TEST_GTOC12_EARTH_BEAM','SPACEPDHCG_TEST_GTOC12_FUSED_TABLES')
start=base.index('names=');end=base.index('\nreport=',start);base=base[:start]+'names='+repr(files)+base[end:]
Path('build/performance/run_fused_tables_v448.py').write_text(base)

base=Path('build/performance/run_earth_beam_v442.py').read_text().replace('spacepdhcg-earth-beam-v442','spacepdhcg-fused-tables-v449').replace('earth-beam-v442.tar.gz','fused-tables-v449.tar.gz').replace('earth_default442_','fused449_').replace('/home/ubuntu/spacepdhcg-compact-options-v430/repo','/home/ubuntu/spacepdhcg-earth-beam-v444/repo').replace('earth-beam-source-sha256.json','fused-tables-source-sha256.json')
base=base.replace(" env.update(PYTHONPATH=", " env['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1'\n env.update(PYTHONPATH=")
start=base.index(' tests=');end=base.index(' cli=',start)
base=base[:start]+" tests="+repr(tests)+"\n run('pytest',[py,'-c',boot,*tests,'-q'],300)\n for tool in ['memcheck','synccheck','racecheck']:\n  run(tool,['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_fused_tables.py','-q'],900)\n"+base[end:]
base=base.replace("for name,candidate in [('default',True)]:", "for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1' if candidate else '0'")
Path('build/performance/run_fused_tables_v449.py').write_text(base)
manifest=Path('build/performance/fused-tables-source-sha256-v449.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/fused-tables-v449.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='fused-tables-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/fused-tables-v449.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-fused-tables-v449');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_fused_tables_v449.py').write_text(program)
Path('build/performance/status_fused_tables_v449.py').write_text(Path('build/performance/status_earth_beam_v434.py').read_text().replace('earth-beam-v434','fused-tables-v449'))
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
