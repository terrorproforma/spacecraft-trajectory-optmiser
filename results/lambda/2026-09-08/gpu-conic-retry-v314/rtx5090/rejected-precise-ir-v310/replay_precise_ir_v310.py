from pathlib import Path
import sys,subprocess,os,time,json,fcntl,hashlib
sys.path.insert(0,str(Path('build/performance/qp-ir-v309').resolve()))
from audit import problem,audit
root=Path('build/performance/precise-ir-v310');root.mkdir(exist_ok=False)
qp=Path('build/performance/qp-ir-v309/qp.txt').resolve();data=problem(qp)
binary=Path('build/performance/qp-regularization-v169/qoco_snapshot_replay').resolve()
lib=Path('/home/angus/build-qoco-precise-ir-v310/final/libqoco.so')
assert lib.exists()
env=dict(os.environ,LD_LIBRARY_PATH=str(lib.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
linked=subprocess.run(['ldd',str(binary)],env=env,capture_output=True,text=True,check=True).stdout
(root/'ldd.txt').write_text(linked);assert str(lib) in linked
report=dict(pid=os.getpid(),complete=False,qp_sha256=hashlib.sha256(qp.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),rows=[])
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,disabled in [('baseline',True),('precise',False),('precise_repeat',False),('baseline_repeat',True)]:
  flags=env.copy()
  if disabled:flags['SPACEPDHCG_TEST_QOCO_PRECISE_IR_DISABLE']='1'
  start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'16'],env=flags,capture_output=True,text=True,timeout=120)
  (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr);r.check_returncode()
  records=[json.loads(line[10:]) for line in r.stdout.splitlines() if line.startswith('QP_REPLAY ')]
  assert len(records)==16
  audits=[dict(iterations=x['iterations'],ir_iterations=x['ir_iterations'],status=x['status'],**audit(data,x)) for x in records]
  row=dict(name=name,seconds=time.perf_counter()-start,qualified=sum(x['qualified'] for x in audits),audits=audits)
  report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2));print(name,row['qualified'],'/16',flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
