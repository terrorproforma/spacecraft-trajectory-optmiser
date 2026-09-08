from pathlib import Path
base=Path('build/performance/check_earth_beam_v431.py').read_text().replace("root=Path('build/performance/earth-beam-v431')", "root=Path('build/performance/earth-beam-v432')")
base=base.replace('frozen.mkdir(parents=True,exist_ok=False)','').replace("shutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)",'')
start=base.index('jobs=[');end=base.index('\nr=dict',start)
jobs="jobs=[('pytest',[py,'-c',boot,'tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_hops.py','tests/test_gtoc12_search2.py','tests/test_gtoc12_gpu_beam.py','-q'])]+[(name,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',py,'-c',boot,'tests/test_gtoc12_gpu_beam.py','-q']) for name in ['memcheck','synccheck','racecheck']]"
base=base[:start]+jobs+base[end:]
Path('build/performance/check_earth_beam_v432.py').write_text(base)
