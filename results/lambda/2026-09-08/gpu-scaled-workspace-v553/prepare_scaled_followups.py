from pathlib import Path
p=Path('build/performance')
replacements={
 '/home/angus/build-spacepdhcg-scaled-pool-v539/final':'/home/ubuntu/spacepdhcg-scaled-pool-v545/core-build/cuda',
 '/home/angus/build-qoco-scaled-pool-v540/final':'/home/ubuntu/spacepdhcg-scaled-pool-v545/final',
 '/home/angus/build-spacepdhcg-retained-replay-v512/final':'/home/ubuntu/spacepdhcg-retained-replay-v514/core-build/cuda',
 '/home/angus/build-qoco-preserve-objective-v534/final':'/home/ubuntu/spacepdhcg-preserve-objective-v535/final',
 '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python',
 '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',
 '/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib':'/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib',
 '/usr/local/cuda-12.8':'/usr/local/cuda','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock'}
def adapt(s):
 for old,new in replacements.items():s=s.replace(old,new)
 return s
def launch(version,s,extra=''):
 root='/home/ubuntu/spacepdhcg-scaled-followup-v'+str(version)
 s=s.replace("root.mkdir(exist_ok=False)","root.mkdir(exist_ok=True)")
 s=s.replace("root=Path('build/performance/scaled-pool-v547')",'root=Path('+repr(root)+')')
 s=s.replace("root=Path('build/performance/scaled-pool-sanitizer-v546')",'root=Path('+repr(root)+')')
 code="from pathlib import Path\nimport subprocess,json\nroot=Path("+repr(root)+");root.mkdir(exist_ok=False)\n"
 code+="(root/'run.py').write_text("+repr(s)+")\n"+extra
 code+="with (root/'runner.log').open('x') as log:child=subprocess.Popen(['python3',str(root/'run.py')],cwd='/home/ubuntu/spacepdhcg-scaled-pool-v545/repo',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid,root=str(root))))\n"
 (p/f'launch_scaled_followup_v{version}.py').write_text(code)
s=adapt((p/'validate_scaled_pool_v547.py').read_text())
launch(549,s, "Path('/home/ubuntu/spacepdhcg-scaled-pool-v545/repo/build/performance/validate_scaled_pool_v547.py').write_text("+repr(s)+")\n")
s=adapt((p/'diagnose_scaled_pool_sanitizer_v546.py').read_text())
s=s.replace("'--tool','memcheck'", "'--tool',('synccheck' if label=='previous' else 'racecheck')")
launch(550,s)
