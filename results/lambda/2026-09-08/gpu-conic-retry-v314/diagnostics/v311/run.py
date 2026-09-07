from pathlib import Path
import os,json,subprocess,hashlib,time,fcntl
from audit import problem,audit
root=Path('/home/ubuntu/spacepdhcg-qp-scaling-v311')
original=Path('/home/ubuntu/spacepdhcg-arc-snapshot-v307/snapshots/qp-153736-000000.txt')
binary=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/qoco_snapshot_replay')
data=problem(original)
report=dict(pid=os.getpid(),complete=False,qp_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),rows=[])
(root/'report.json').write_text(json.dumps(report))
cases=[('ruiz'+str(k),k) for k in [0,1,2,4,8,12]]
with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,ruiz in cases:
  lines=original.read_text().splitlines();ints=lines[2].split();ints[1]=str(ruiz);lines[2]=' '.join(ints)
  qp=root/(name+'.txt');qp.write_text('\n'.join(lines)+'\n')
  start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'16'],capture_output=True,text=True,timeout=120)
  (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr);r.check_returncode()
  records=[json.loads(line[10:]) for line in r.stdout.splitlines() if line.startswith('QP_REPLAY ')]
  assert len(records)==16
  audits=[dict(iterations=x['iterations'],ir_iterations=x['ir_iterations'],status=x['status'],reported_primal=x['primal'],reported_dual=x['dual'],reported_gap=x['gap'],**audit(data,x)) for x in records]
  row=dict(name=name,ruiz=ruiz,seconds=time.perf_counter()-start,qualified=sum(x['qualified'] for x in audits),audits=audits)
  report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2));print(name,row['qualified'],'/16',flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
