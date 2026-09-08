from pathlib import Path
import json,hashlib,tarfile,os,subprocess
files=['cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/src/orbitweaver_beam.cuh','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','cpp/cuda/tests/orbitweaver_beam_test.cu','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/gpu_beam.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_beam.py']
base=Path('build/performance/check_earth_beam_v435.py').read_text().replace("root=Path('build/performance/earth-beam-v435')", "root=Path('build/performance/earth-beam-v441')").replace('build-spacepdhcg-earth-beam-v431','build-spacepdhcg-earth-beam-v441').replace('earth-beam-probe-v435','earth-beam-probe-v441')
base=base.replace("core=frozen/'libspacepdhcg_cuda.so'", "frozen.mkdir(parents=True,exist_ok=False)\ncore=frozen/'libspacepdhcg_cuda.so'\nshutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)")
end=base.index('\nr=dict')
base=base[:end]+"\njobs += [('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py','tests/test_gtoc12_gpu_beam.py','-q']),('api-memcheck',['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_beam.py','-q'])]\n"+base[end:]
Path('build/performance/check_earth_beam_v441.py').write_text(base)
base=Path('build/performance/run_earth_beam_v440.py').read_text().replace('earth-beam-v440','earth-beam-default-v441').replace('earth440_', 'earth_default441_').replace('build-spacepdhcg-earth-beam-v431','build-spacepdhcg-earth-beam-v441')
base=base.replace("for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:", "for name,candidate in [('default',True)]:").replace("   env['SPACEPDHCG_TEST_GTOC12_EARTH_BEAM']='1' if candidate else '0'\n", "")
Path('build/performance/run_earth_beam_default_v441.py').write_text(base)

base=Path('build/performance/run_earth_beam_v434.py').read_text().replace('spacepdhcg-earth-beam-v434','spacepdhcg-earth-beam-v442').replace('earth-beam-v434.tar.gz','earth-beam-v442.tar.gz').replace('earth434_', 'earth_default442_')
start=base.index(" for name in ['memcheck'");end=base.index(' cli=',start)
block=""" binary=str(root/'earth-beam-probe')
 run('compile-probe',['/usr/local/cuda/bin/nvcc','-std=c++17','--fmad=false','-arch=sm_90','-I'+str(repo/'cpp/include'),'-I'+str(repo/'cpp/cuda/include'),str(repo/'cpp/cuda/tests/orbitweaver_beam_test.cu'),'-L'+str(core.parent),'-lspacepdhcg_cuda','-o',binary])
 run('probe',[binary])
 for name in ['memcheck','synccheck','racecheck']:
  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',binary],300)
 run('api-memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_beam.py','-q'],300)
"""
base=base[:start]+block+base[end:]
base=base.replace("for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_EARTH_BEAM']='1' if candidate else '0'", "for name,candidate in [('default',True)]:")
Path('build/performance/run_earth_beam_v442.py').write_text(base)
manifest=Path('build/performance/earth-beam-source-sha256-v442.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/earth-beam-v442.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='earth-beam-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/earth-beam-v442.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-earth-beam-v442');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_earth_beam_v442.py').write_text(program)
Path('build/performance/status_earth_beam_v442.py').write_text(Path('build/performance/status_earth_beam_v434.py').read_text().replace('earth-beam-v434','earth-beam-v442'))
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
