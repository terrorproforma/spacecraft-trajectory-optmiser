from pathlib import Path
p=Path('build/performance')
worker=(p/'worker_layouts_v776.py').read_text().replace("boot,'-q','tests/test_gtoc12_gpu_joint_layouts.py']", "boot,'-q','tests/test_gtoc12_gpu_insertion_grid.py','tests/test_gtoc12_gpu_joint_layouts.py','tests/test_gtoc12_gpu_joint_insertions.py']")
worker=worker.replace("run('pytest',[py,'-c',boot,'-q','tests/test_gtoc12_gpu_insertion_grid.py','tests/test_gtoc12_gpu_joint_layouts.py','tests/test_gtoc12_gpu_joint_insertions.py'])", "run('pytest',[py,'-c',boot,'-q',*sorted(str(p.relative_to(repo)) for p in (repo/'tests').glob('test_gtoc12_gpu_joint*.py')),'tests/test_gtoc12_jointopt.py','tests/test_gtoc12_insertion_pivot.py','tests/test_gtoc12_gpu_insertion_grid.py'])")
(p/'worker_grid_v786.py').write_text(worker)
build=(p/'build_layouts_v776.py').read_text().replace('spacepdhcg-layouts-v776','spacepdhcg-grid-v786').replace('worker_layouts_v776','worker_grid_v786')
a=build.index('owned=');b=build.index('\nfor name in owned:',a)
owned=['cpp/cuda/src/gtoc12_joint_insertion.cuh','cpp/cuda/src/gtoc12_joint_layout.cuh','cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','src/spacepdhcg/gtoc12/gpu_joint_layouts.py','tests/test_gtoc12_gpu_insertion_grid.py']
build=build[:a]+'owned='+repr(owned)+build[b:]
(p/'build_grid_v786.py').write_text(build)
(p/'send_grid_v786.py').write_text((p/'send_layouts_v776.py').read_text().replace('spacepdhcg-layouts-v776','spacepdhcg-grid-v786').replace('layouts-source-v776','grid-source-v786').replace('launch_layouts_h100_v776','launch_grid_h100_v786'))
(p/'status_grid_v786.py').write_text((p/'status_layouts_v776.py').read_text().replace('spacepdhcg-layouts-v776','spacepdhcg-grid-v786'))
