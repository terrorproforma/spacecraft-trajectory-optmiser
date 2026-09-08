from pathlib import Path
base=Path('build/performance/check_earth_beam_v432.py').read_text().replace("root=Path('build/performance/earth-beam-v432')", "root=Path('build/performance/earth-beam-v435')")
base=base.replace("'cpp/cuda/src/orbitweaver_beam.cuh']", "'cpp/cuda/src/orbitweaver_beam.cuh','cpp/cuda/tests/orbitweaver_beam_test.cu']")
base=base.replace("binary='/home/angus/compact-options-probe-v423'", "binary='/home/angus/earth-beam-probe-v435'")
base=base.replace('cpp/cuda/tests/orbitweaver_options_test.cu\',\'-L', 'cpp/cuda/tests/orbitweaver_beam_test.cu\',\'-L')
start=base.index('jobs=[');end=base.index('\nr=dict',start)
base=base[:start]+"jobs=[('compile',compile_cmd),('probe',[binary])]+[(name,['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',name,'--error-exitcode','99',binary]) for name in ['memcheck','synccheck','racecheck']]"+base[end:]
Path('build/performance/check_earth_beam_v435.py').write_text(base)
