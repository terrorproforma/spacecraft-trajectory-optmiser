from pathlib import Path

base = Path('build/performance/check_earth_beam_style_v445.py').read_text()
base = base.replace("root=Path('build/performance/earth-beam-style-v445')", "root=Path('build/performance/fused-tables-v447')")
base = base.replace("frozen=Path('/home/angus/build-spacepdhcg-earth-beam-v441/final')", "frozen=Path('/home/angus/build-spacepdhcg-fused-tables-v447/final');frozen.mkdir(parents=True,exist_ok=False)\nshutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',frozen/'libspacepdhcg_cuda.so')")
start=base.index('sources=');end=base.index('\nfor name in sources:',start)
names=['cpp/cuda/src/orbitweaver_gpu.cu','cpp/cuda/src/orbitweaver_beam.cuh','cpp/cuda/src/gtoc12_collect_dp.cu','cpp/cuda/include/spacepdhcg/cuda/orbitweaver_gpu_c_api.h','src/spacepdhcg/gtoc12/gpu_collect_tables.py','src/spacepdhcg/gtoc12/gpu_lambert.py','tests/test_gtoc12_gpu_fused_tables.py']
base=base[:start]+'sources='+repr(names)+base[end:]
base=base.replace("env.update(PYTHONPATH=", "env['SPACEPDHCG_TEST_GTOC12_FUSED_TABLES']='1'\nenv.update(PYTHONPATH=")
start=base.index('jobs=');end=base.index('\nr=dict(',start)
tests=['tests/test_gtoc12_gpu_fused_tables.py','tests/test_gtoc12_gpu_collect_tables.py','tests/test_gtoc12_gpu_resident_collect_tables.py','tests/test_gtoc12_gpu_harvest_window.py']
base=base[:start]+"jobs=[('pytest',[py,'-c',boot]+"+repr(tests)+"+['-q'])]\nfor tool in ['memcheck','synccheck','racecheck']:\n jobs.append((tool,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_fused_tables.py','-q']))"+base[end:]
base=base.replace('child.wait(timeout=300)','child.wait(timeout=900)')
Path('build/performance/check_fused_tables_v447.py').write_text(base)
