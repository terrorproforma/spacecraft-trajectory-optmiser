from pathlib import Path
import json,hashlib,tarfile,os,subprocess
base=Path('build/performance/run_earth_beam_v434.py').read_text().replace('spacepdhcg-earth-beam-v434','spacepdhcg-earth-beam-v437').replace('spacepdhcg-compact-options-v430/repo','spacepdhcg-earth-beam-v434/repo').replace('earth-beam-v434.tar.gz','earth-beam-v437.tar.gz')
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start)
base=base[:start]+base[end:]
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-earth-beam-v434/core-build/cuda/libspacepdhcg_cuda.so')")
start=base.index(' tests=');end=base.index(" report['complete']=True",start)
suffix=""" binary=str(root/'earth-beam-probe')
 run('compile-probe',['/usr/local/cuda/bin/nvcc','-std=c++17','--fmad=false','-arch=sm_90','-I'+str(repo/'cpp/include'),'-I'+str(repo/'cpp/cuda/include'),str(repo/'cpp/cuda/tests/orbitweaver_beam_test.cu'),'-L'+str(core.parent),'-lspacepdhcg_cuda','-o',binary])
 run('probe',[binary])
 for name in ['memcheck','synccheck','racecheck']:
  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',binary],300)
 env['SPACEPDHCG_BEAM_MICRO_OUTPUT']=str(root)
 cli="import sys,runpy;sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];runpy.run_path('build/performance/earth_beam_micro_v436.py',run_name='__main__')"
 cmd=[py,'-c',cli,'gtoc12','run','--run-id','earth_micro437','--output',str(root/'output'),'--full-catalogue','--ships','1','--beam-width','16','--max-deploys','10','--neighbours','48','--refine-top','3','--search-budget-seconds','120','--budget-seconds','600','--retime-attempts','4','--retime-budget-seconds','300','--no-cooperative','--node-days','2','--scvx-iterations','40','--screening-backend','cuda','--seed-backend','cuda','--discretisation-backend','cuda','--assembly-backend','cuda','--convex-solver','qoco','--outer-loop-backend','cuda','--gpu-execution','graph']
 run('micro',cmd,300)
 assert (root/'measurement.json').exists()
"""
base=base[:start]+suffix+base[end:];Path('build/performance/run_earth_beam_v437.py').write_text(base)
files=['cpp/cuda/tests/orbitweaver_beam_test.cu','build/performance/earth_beam_micro_v436.py']
manifest=Path('build/performance/earth-beam-source-sha256-v437.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/earth-beam-v437.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='earth-beam-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/earth-beam-v437.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-earth-beam-v437');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_earth_beam_v437.py').write_text(program)
Path('build/performance/status_earth_beam_v437.py').write_text(Path('build/performance/status_earth_beam_v434.py').read_text().replace('earth-beam-v434','earth-beam-v437'))

base=Path('build/performance/run_compact_fleet_v428.py').read_text().replace('compact-fleet-v428','earth-beam-fleet-v438').replace('compact_fleet428','earth_fleet438').replace('build-spacepdhcg-compact-options-v423','build-spacepdhcg-earth-beam-v431').replace('SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS','SPACEPDHCG_TEST_GTOC12_EARTH_BEAM')
base=base.replace("'src/spacepdhcg/gtoc12/lambert.py']", "'src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/gpu_beam.py','cpp/cuda/src/orbitweaver_beam.cuh']")
Path('build/performance/run_earth_beam_fleet_v438.py').write_text(base)
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
