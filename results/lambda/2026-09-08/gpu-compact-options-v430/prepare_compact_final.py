from pathlib import Path
import json, hashlib, tarfile, os, subprocess
# Final local default-on validation and a complete verified campaign.
base=Path('build/performance/run_compact_v425.py').read_text().replace('compact-options-v425','compact-options-v429').replace('compact425_','compact429_')
base=base.replace("for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:", "run('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py','-q'],300)\n  run('memcheck',['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','-q'],300)\n  for name,candidate in [('default',True)]:")
base=base.replace("   env['SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS']='1' if candidate else '0'\n", "")
Path('build/performance/run_compact_v429.py').write_text(base)

# Remote final Python source on the previously built and tested CUDA library.
base=Path('build/performance/run_compact_v426.py').read_text()
base=base.replace('spacepdhcg-compact-options-v426','spacepdhcg-compact-options-v430').replace('spacepdhcg-paired-ephemerides-v420/repo','spacepdhcg-compact-options-v426/repo').replace('compact-options-v426.tar.gz','compact-options-v430.tar.gz').replace('compact426_','compact430_')
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start)
base=base[:start]+base[end:]
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-compact-options-v426/core-build/cuda/libspacepdhcg_cuda.so')")
start=base.index(" binary=str(root/");end=base.index(' cli=',start)
base=base[:start]+" for name in ['memcheck','synccheck']:\n  run(name,['/usr/local/cuda/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','-q'],300)\n"+base[end:]
base=base.replace("for name,candidate in [('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]:\n  env['SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS']='1' if candidate else '0'", "for name,candidate in [('default',True)]:")
Path('build/performance/run_compact_v430.py').write_text(base)
files=['cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','cpp/cuda/tests/orbitweaver_options_test.cu','src/spacepdhcg/gtoc12/gpu_lambert.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/search.py','tests/test_gtoc12_gpu_elements.py']
manifest=Path('build/performance/compact-source-sha256-v430.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/compact-options-v430.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='compact-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/compact-options-v430.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-compact-options-v430');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_compact_v430.py').write_text(program)
Path('build/performance/status_compact_v430.py').write_text(Path('build/performance/status_compact_v426.py').read_text().replace('compact-options-v426','compact-options-v430'))
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
