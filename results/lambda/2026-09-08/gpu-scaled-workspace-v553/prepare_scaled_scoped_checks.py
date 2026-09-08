from pathlib import Path
import runpy
p=Path('build/performance')
s=(p/'diagnose_scaled_pool_sanitizer_v546.py').read_text()
s=s.replace('scaled-pool-sanitizer-v546','scaled-pool-scoped-sanitizer-v551')
a=s.index('  for label,core,qoco in [');b=s.index('   env.update',a)
s=s[:a]+'''  for label in ['memcheck','synccheck','racecheck']:
   core='/home/angus/build-spacepdhcg-scaled-pool-v539/final';qoco='/home/angus/build-qoco-scaled-pool-v540/final'
'''+s[b:]
s=s.replace("'--tool','memcheck'", "'--tool',label,'--kernel-name-exclude','kns=cudss'")
s=s.replace("rc=child.wait(timeout=300)","\n    try:rc=child.wait(timeout=300)\n    except subprocess.TimeoutExpired:child.kill();child.wait();rc=-999")
s=s.replace("report=dict(pid=os.getpid(),complete=False,cases=[])","report=dict(pid=os.getpid(),complete=False,scope='cuDSS kernels excluded from instrumentation; this is not a full solver sanitizer pass',cases=[])")
(p/'diagnose_scaled_scoped_v551.py').write_text(s)
helpers=runpy.run_path(str(p/'prepare_scaled_followups.py'))
s=helpers['adapt'](s).replace("root=Path('build/performance/scaled-pool-scoped-sanitizer-v551')", "root=Path('/home/ubuntu/spacepdhcg-scaled-followup-v552')")
helpers['launch'](552,s)
