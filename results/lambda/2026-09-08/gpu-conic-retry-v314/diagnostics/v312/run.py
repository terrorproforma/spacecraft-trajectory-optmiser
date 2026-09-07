from pathlib import Path
import os,json,subprocess,hashlib,time,fcntl
from audit import problem,audit
root=Path('/home/ubuntu/spacepdhcg-qp-objective-v312')
original=Path('/home/ubuntu/spacepdhcg-arc-snapshot-v307/snapshots/qp-153736-000000.txt')
binary=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/qoco_snapshot_replay')
data=problem(original)
report=dict(pid=os.getpid(),complete=False,qp_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),rows=[])
(root/'report.json').write_text(json.dumps(report))
cases=[('base',1.,1e-11),('cost01',.01,1e-13),('cost0001',.0001,1e-15),('cost100',100.,1e-11),('cost10',10.,1e-11)]
with open('/home/ubuntu/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,scale,tol in cases:
  lines=original.read_text().splitlines();assert data['shift']==0
  n,p,m,np_,na,ng,*_=map(int,lines[1].split())
  values=list(map(float,lines[11].split()[1:]));assert int(lines[11].split()[0])==len(values)
  for j in list(range(np_))+list(range(np_+na+ng,np_+na+ng+n)):values[j]*=scale
  lines[11]=str(len(values))+' '+' '.join(format(v,'.17g') for v in values)
  settings=lines[3].split();settings[5]=settings[6]=str(tol);lines[3]=' '.join(settings)
  qp=root/(name+'.txt');qp.write_text('\n'.join(lines)+'\n')
  start=time.perf_counter();r=subprocess.run([str(binary),str(qp),'16'],capture_output=True,text=True,timeout=120)
  (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr);r.check_returncode()
  records=[json.loads(line[10:]) for line in r.stdout.splitlines() if line.startswith('QP_REPLAY ')]
  assert len(records)==16
  for x in records:
   x['y']=[v/scale for v in x['y']];x['z']=[v/scale for v in x['z']]
  audits=[dict(iterations=x['iterations'],ir_iterations=x['ir_iterations'],status=x['status'],reported_primal=x['primal'],reported_dual=x['dual'],reported_gap=x['gap'],**audit(data,x)) for x in records]
  row=dict(name=name,objective_scale=scale,internal_tolerance=tol,seconds=time.perf_counter()-start,qualified=sum(x['qualified'] for x in audits),audits=audits)
  report['rows'].append(row);(root/'report.json').write_text(json.dumps(report,indent=2));print(name,row['qualified'],'/16',flush=True)
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
