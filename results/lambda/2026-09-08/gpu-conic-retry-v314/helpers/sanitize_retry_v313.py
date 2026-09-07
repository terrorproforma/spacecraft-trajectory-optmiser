from pathlib import Path
import subprocess,os,fcntl,json,time
root=Path('build/performance/conic-retry-v313');base=Path('/home/angus/build-spacepdhcg-conic-retry-v313/final')
env=dict(os.environ,LD_LIBRARY_PATH=str(base)+':/usr/local/cuda-12.8/lib64')
report=[]
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for tool in ['memcheck','initcheck','synccheck','racecheck']:
  start=time.monotonic();r=subprocess.run(['/usr/local/cuda-12.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(base/'gtoc12_scvx_test')],env=env,capture_output=True,text=True,timeout=120)
  (root/(tool+'.log')).write_text(r.stdout+r.stderr);report.append(dict(tool=tool,returncode=r.returncode,seconds=time.monotonic()-start));(root/'sanitizers.json').write_text(json.dumps(report,indent=2));print(tool,r.returncode,(r.stdout+r.stderr)[-300:]);r.check_returncode()
