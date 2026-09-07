from pathlib import Path
import os,json,subprocess,hashlib,time,fcntl
from audit import problem,audit
root=Path('/home/ubuntu/spacepdhcg-qp-sweep-v308')
qp=Path('/home/ubuntu/spacepdhcg-arc-snapshot-v307/snapshots/qp-153736-000000.txt')
binary=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/qoco_snapshot_replay')
assert b'QOCO_REPLAY_REGULARIZATION' in binary.read_bytes()
data=problem(qp)
report=dict(pid=os.getpid(),complete=False,qp_sha256=hashlib.sha256(qp.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),rows=[])
(root/'report.json').write_text(json.dumps(report))
with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for reg in ['1e-8','3e-8','1e-7','3e-7']:
  env=dict(os.environ,QOCO_REPLAY_REGULARIZATION=reg)
  start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'16'],env=env,capture_output=True,text=True,timeout=120)
  (root/(reg+'.log')).write_text(r.stdout);(root/(reg+'.err')).write_text(r.stderr);r.check_returncode()
  records=[json.loads(line[10:]) for line in r.stdout.splitlines() if line.startswith('QP_REPLAY ')]
  assert len(records)==16
  audits=[dict(iterations=x['iterations'],status=x['status'],**audit(data,x)) for x in records]
  row=dict(regularization=reg,seconds=time.perf_counter()-start,qualified=sum(x['qualified'] for x in audits),audits=audits)
  report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2));print(reg,row['qualified'],'/16',flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
