from pathlib import Path
base=Path('build/performance/check_compact_v424.py').read_text()
base=base.replace('compact-options-v424','earth-beam-v431').replace('build-spacepdhcg-compact-options-v423','build-spacepdhcg-earth-beam-v431')
base=base.replace("core=frozen/'libspacepdhcg_cuda.so'", "frozen.mkdir(parents=True,exist_ok=False)\ncore=frozen/'libspacepdhcg_cuda.so'\nshutil.copy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)")
base=base.replace("'tests/test_gtoc12_gpu_elements.py']", "'tests/test_gtoc12_gpu_elements.py','tests/test_gtoc12_gpu_beam.py','src/spacepdhcg/gtoc12/gpu_beam.py','cpp/cuda/src/orbitweaver_beam.cuh']")
base=base.replace("'tests/test_gtoc12_search2.py','-q'", "'tests/test_gtoc12_search2.py','tests/test_gtoc12_gpu_beam.py','-q'")
Path('build/performance/check_earth_beam_v431.py').write_text(base)
