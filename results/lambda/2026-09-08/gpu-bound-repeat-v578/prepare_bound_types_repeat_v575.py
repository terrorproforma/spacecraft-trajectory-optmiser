from pathlib import Path
p=Path('build/performance')
s=(p/'validate_bound_types_v569.py').read_text().replace('bound-types-v569','bound-repeat-v575').replace('validate_bound_types_v569.py','validate_bound_repeat_v575.py')
a=s.index("  probe=");b=s.index('  replay=',a);s=s[:a]+s[b:]
s=s.replace("[('baseline','0'),('candidate','1')]","[('candidate','1'),('baseline','0')]")
replacements={'/home/angus/build-spacepdhcg-bound-types-v568/final':'/home/ubuntu/spacepdhcg-bound-types-v569/core-build/cuda','/home/angus/build-qoco-scaled-pool-v540/final':'/home/ubuntu/spacepdhcg-scaled-pool-v545/final','/home/angus/build-qoco-preserve-objective-v534/final':'/home/ubuntu/spacepdhcg-preserve-objective-v535/final','/home/angus/worktrees/spacepdhcg-literature-venv/bin/python':'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data':'/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data','/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib':'/home/ubuntu/spacepdhcg-recovery-v152/cudss/lib','/usr/local/cuda-12.8':'/usr/local/cuda','/home/angus/.spacepdhcg-gpu.lock':'/home/ubuntu/.spacepdhcg-gpu.lock'}
for a,b in replacements.items():s=s.replace(a,b)
code="from pathlib import Path\nimport subprocess,json\nrepo=Path('/home/ubuntu/spacepdhcg-bound-types-v569/repo')\n"
code+="path=repo/'build/performance/validate_bound_repeat_v575.py';assert not path.exists();path.write_text("+repr(s)+")\n"
code+="with Path('/home/ubuntu/bound-repeat-v575-runner.log').open('x') as log:child=subprocess.Popen(['python3',str(path)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=child.pid)))\n"
(p/'launch_bound_repeat_v575.py').write_text(code)
