from pathlib import Path
base=Path('build/performance/run_fused_tables_profile_v451.py').read_text().replace('fused-tables-profile-v451','solver-phase-v458').replace('fused_profile451','solver_phase458').replace('build-spacepdhcg-fused-tables-v447','build-spacepdhcg-solver-phase-v458')
base=base.replace("runpy.run_path('build/performance/profile_fused_tables_v451.py',run_name='__main__')", "runpy.run_module('spacepdhcg',run_name='__main__')")
base=base.replace("r=dict(source_sha256=", "from shutil import copy2\nPath(core).parent.mkdir(parents=True,exist_ok=False)\ncopy2('/home/angus/build-spacepdhcg-gtoc12-v141/cuda/libspacepdhcg_cuda.so',core)\ncopy2('cpp/cuda/src/gtoc12_scvx.cu',root/'gtoc12_scvx.cu')\nenv['SPACEPDHCG_TEST_GTOC12_PHASE_TRACE']='1'\nr=dict(source_sha256=")
base=base.replace("'build/performance/profile_fused_tables_v451.py'", "'cpp/cuda/src/gtoc12_scvx.cu'")
Path('build/performance/run_solver_phase_v458.py').write_text(base)
