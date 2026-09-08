from pathlib import Path
import runpy
p=Path('build/performance');h=runpy.run_path(str(p/'prepare_scaled_followups.py'))
s=h['adapt']((p/'diagnose_scaled_pool_sanitizer_v546.py').read_text())
s=s.replace("   ('candidate','/home/ubuntu/spacepdhcg-scaled-pool-v545/core-build/cuda','/home/ubuntu/spacepdhcg-scaled-pool-v545/final')",'')
s=s.replace("'--tool','memcheck'", "'--tool','racecheck'")
h['launch'](554,s)
