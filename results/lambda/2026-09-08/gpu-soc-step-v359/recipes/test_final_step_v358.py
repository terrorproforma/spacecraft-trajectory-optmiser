from pathlib import Path
import subprocess
s=Path('build/performance/test_step_only_v352.py').read_text().replace('step-only-validation-v352','soc-step-validation-v358').replace('build-qoco-step-only-v352','build-qoco-soc-step-v358')
a=s.index(" for name,cmd in ");b=s.index("  start=time.perf_counter()",a)
s=s[:a]+" for name,cmd in [('pytest',['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','-m','pytest','tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','-q'])]:\n"+s[b:]
exec(compile(s,'final_step_validation','exec'))
r=Path('build/performance/replay_step_only_v352.py').read_text().replace('step-only','soc-step').replace('352','358').replace('step_only','soc_step')
Path('build/performance/replay_final_step_v358.py').write_text(r)
subprocess.run(['/home/angus/worktrees/spacepdhcg-literature-venv/bin/python','build/performance/replay_final_step_v358.py'],check=True)
