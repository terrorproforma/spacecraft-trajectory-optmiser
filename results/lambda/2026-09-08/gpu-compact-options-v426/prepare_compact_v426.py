from pathlib import Path
import json, hashlib, tarfile, os, subprocess
base=Path('build/performance/run_stationary_v412.py').read_text()
base=base.replace('spacepdhcg-stationary-v412','spacepdhcg-compact-options-v426').replace('spacepdhcg-stationary-v408/repo','spacepdhcg-paired-ephemerides-v420/repo').replace('stationary-v412.tar.gz','compact-options-v426.tar.gz').replace('stationary-source-sha256.json','compact-source-sha256.json')
base=base.replace("['tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py']", "['tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py']")
base=base.replace('stationary-probe','compact-options-probe').replace('cpp/cuda/tests/gtoc12_scvx_test.cu','cpp/cuda/tests/orbitweaver_options_test.cu')
base=base.replace("for name,candidate in [('candidate0',True)]:", "for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS']='1' if candidate else '0'")
base=base.replace('stationary_default412_', 'compact426_')
Path('build/performance/run_compact_v426.py').write_text(base)
files=['cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','cpp/cuda/tests/orbitweaver_options_test.cu','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_elements.py']
manifest=Path('build/performance/compact-source-sha256-v426.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/compact-options-v426.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='compact-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/compact-options-v426.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-compact-options-v426');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_compact_v426.py').write_text(program)
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
