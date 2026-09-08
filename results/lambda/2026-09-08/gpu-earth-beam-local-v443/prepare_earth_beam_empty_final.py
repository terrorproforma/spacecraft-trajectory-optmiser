from pathlib import Path
import json,hashlib,tarfile,os,subprocess
base=Path('build/performance/check_earth_beam_v441.py').read_text().replace("root=Path('build/performance/earth-beam-v441')", "root=Path('build/performance/earth-beam-v443')")
base=base.replace('frozen.mkdir(parents=True,exist_ok=False)','').replace("shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)",'')
start=base.index('jobs=[');end=base.index('\nr=dict',start);base=base[:start]+"jobs=[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py','tests/test_gtoc12_gpu_beam.py','-q'])]"+base[end:]
Path('build/performance/check_earth_beam_v443.py').write_text(base)
base=Path('build/performance/run_earth_beam_default_v441.py').read_text().replace('earth-beam-default-v441','earth-beam-default-v443').replace('earth_default441_', 'earth_default443_')
Path('build/performance/run_earth_beam_default_v443.py').write_text(base)
base=Path('build/performance/run_earth_beam_v442.py').read_text().replace('spacepdhcg-earth-beam-v442','spacepdhcg-earth-beam-v444').replace('spacepdhcg-compact-options-v430/repo','spacepdhcg-earth-beam-v442/repo').replace('earth-beam-v442.tar.gz','earth-beam-v444.tar.gz').replace('earth_default442_','earth_default444_')
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-earth-beam-v442/core-build/cuda/libspacepdhcg_cuda.so')")
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start);base=base[:start]+base[end:]
start=base.index(" binary=str(root/");end=base.index(' cli=',start);base=base[:start]+base[end:]
Path('build/performance/run_earth_beam_v444.py').write_text(base)
files=list(json.loads(Path('build/performance/earth-beam-source-sha256-v442.json').read_text()))
manifest=Path('build/performance/earth-beam-source-sha256-v444.json');manifest.write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
archive=Path('/tmp/earth-beam-v444.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p)
 t.add(manifest,arcname='earth-beam-source-sha256.json')
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-q','-i',str(key),'-o','BatchMode=yes',str(archive),'ubuntu@192.222.55.229:/tmp/earth-beam-v444.tar.gz'],check=True,timeout=55)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-earth-beam-v444');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_earth_beam_v444.py').write_text(program)
Path('build/performance/status_earth_beam_v444.py').write_text(Path('build/performance/status_earth_beam_v434.py').read_text().replace('earth-beam-v434','earth-beam-v444'))
print('Uploaded',hashlib.sha256(archive.read_bytes()).hexdigest())
