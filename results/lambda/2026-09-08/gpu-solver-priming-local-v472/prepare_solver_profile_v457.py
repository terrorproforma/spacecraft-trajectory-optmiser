from pathlib import Path
base=Path('build/performance/run_fused_tables_profile_v451.py').read_text().replace('fused-tables-profile-v451','solver-profile-v457').replace('fused_profile451','solver_profile457').replace('build-spacepdhcg-fused-tables-v447','build-spacepdhcg-fused-tables-v454')
base=base.replace("runpy.run_path('build/performance/profile_fused_tables_v451.py',run_name='__main__')", "runpy.run_module('spacepdhcg',run_name='__main__')")
base=base.replace("r=dict(source_sha256=", "cmd=['/usr/local/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=graph','--force-overwrite=false','-o',str(root/'cuda-trace')]+cmd\nr=dict(source_sha256=")
Path('build/performance/run_solver_profile_v457.py').write_text(base)
