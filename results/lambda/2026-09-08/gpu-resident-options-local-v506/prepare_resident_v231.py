from pathlib import Path
import tarfile,os,subprocess
base=Path('build/performance')
benchmark=(base/'benchmark_retime_dp_v219.py').read_text()
benchmark=benchmark.replace('lambert.cuda_retime_dp=dispatch if native else lambda *args:NotImplemented','pass')
benchmark=benchmark.replace("with lambert.using_lambert_backend('cuda') as gpu:\n   start", "with lambert.using_lambert_backend('cuda') as gpu:\n   gpu.resident_retime_tables=native\n   start")
(base/'benchmark_resident_v230.py').write_text(benchmark)
launcher=(base/'launch_retime_dp_v220.py').read_text().replace('spacepdhcg-retime-dp-v220','spacepdhcg-resident-v231').replace('spacepdhcg-elements-v215/repo','spacepdhcg-retime-dp-v220/repo')
launcher=launcher.replace("'benchmark_retime_dp_v219.py'","'benchmark_resident_v230.py'")
launcher=launcher.replace(" run('cached-benchmark',[python,'benchmark_cached_dp_v219.py',str(root/'measurement'),source])\n",'')
launcher=launcher.replace("'tests/test_gtoc12_gpu_retime.py',", "'tests/test_gtoc12_gpu_retime.py','tests/test_gtoc12_resident_retime.py','tests/test_gtoc12_certified_objective.py',")
(base/'launch_resident_v231.py').write_text(launcher)
files=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_retime_c_api.h','cpp/cuda/src/gtoc12_retime.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','cpp/cuda/src/orbitweaver_gpu.cu','src/spacepdhcg/gtoc12/gpu_retime.py','src/spacepdhcg/gtoc12/retiming.py','src/spacepdhcg/gtoc12/lambert.py','src/spacepdhcg/gtoc12/gpu_lambert.py','tests/test_gtoc12_resident_retime.py','tests/test_gtoc12_certified_objective.py','results/lambda/2026-09-08/gpu-retime-dp-v220/h100/mission-v223/plan.json']
with tarfile.open(base/'resident-v231.tar.gz','w:gz') as t:
 for name in files:t.add(name,arcname=name)
 t.add(base/'benchmark_resident_v230.py',arcname='benchmark_resident_v230.py')
(base/'resident-v231-source-files.json').write_text(__import__('json').dumps(files))
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(Path('traj-key.pem').read_bytes())
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes',str(base/'resident-v231.tar.gz'),'ubuntu@192.222.55.229:/tmp/spacepdhcg-resident-v231.tar.gz'],check=True,timeout=55)
