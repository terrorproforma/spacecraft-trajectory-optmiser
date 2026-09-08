from pathlib import Path
import json,hashlib,tarfile,subprocess,os
base=Path('build/performance/run_stationary_v408.py').read_text()
prefix=base[:base.index(' tests=')].replace('spacepdhcg-stationary-v408','spacepdhcg-stationary-v412').replace('spacepdhcg-grid-cache-v402/repo','spacepdhcg-stationary-v408/repo').replace('/tmp/stationary-v408.tar.gz','/tmp/stationary-v412.tar.gz')
suffix=''' tests=['tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py']
 run('pytest',[py,'-c',boot,*tests,'-q'],300)
 binary=str(root/'stationary-probe')
 run('compile-probe',['/usr/local/cuda/bin/nvcc','-std=c++17','--fmad=false','-arch=sm_90','-I'+str(repo/'cpp/include'),'-I'+str(repo/'cpp/cuda/include'),str(repo/'cpp/cuda/tests/gtoc12_scvx_test.cu'),'-L'+str(core.parent),'-lspacepdhcg_cuda','-o',binary])
 run('probe',[binary])
 for name in ['memcheck','synccheck','racecheck']:
  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',binary])
'''
prior=Path('build/performance/run_grid_cache_v402.py').read_text();suffix+=prior[prior.index(' cli='):].replace('grid_cache402_','stationary_default412_')
runner=prefix+suffix;Path('build/performance/run_stationary_v412.py').write_text(runner)
files=['cpp/cuda/src/gtoc12_scvx.cu','cpp/cuda/include/spacepdhcg/cuda/gtoc12_scvx_c_api.h','cpp/cuda/tests/gtoc12_scvx_test.cu','src/spacepdhcg/gtoc12/gpu_scvx.py']
manifest=Path('build/performance/stationary-source-sha256-v412.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/stationary-v412.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='stationary-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/stationary-v412.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-stationary-v412');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(runner)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_stationary_v412.py').write_text(program)
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
