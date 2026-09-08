import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
report=json.loads((root/'config.json').read_text())
report.update(pid=os.getpid(),complete=False)
def save():
 temp=root/'report.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(root/'report.json')
try:
 for relative,expected in report['source_sha256'].items():
  path=root/'source'/relative
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
 for key,name in [('SPACEPDHCG_GTOC12_CUDA_LIBRARY','core_sha256'),('SPACEPDHCG_QOCO_LIBRARY','qoco_sha256')]:
  report[name]=hashlib.sha256(Path(report['environment'][key]).read_bytes()).hexdigest()
 env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
 env.update(report['environment'])
 save()
 with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  start=time.perf_counter()
  with (root/'campaign.log').open('x') as log:
   child=subprocess.Popen(report['command'],cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
   report.update(child_pid=child.pid,stage='family_search');save();code=child.wait()
  report.update(complete=True,returncode=code,seconds=time.perf_counter()-start)
  path=root/'output/run_report.json'
  if path.exists():
   result=json.loads(path.read_text());best=result.get('best') or {}
   report.update(status=result.get('status'),best_independent=best.get('independent'),best_official=best.get('official'),columns=result.get('master',{}).get('columns'),best_fleet=best.get('fleet'))
except Exception as error:report.update(error=repr(error),complete=True)
save()
