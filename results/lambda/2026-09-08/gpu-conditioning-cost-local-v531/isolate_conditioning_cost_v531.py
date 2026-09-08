from pathlib import Path
import fcntl, hashlib, json, os, shutil, subprocess, time
import numpy as np
from analyse_qps_v169 import problem, audit

root=Path('build/performance/conditioning-cost-v531');root.mkdir(exist_ok=False)
prior=Path('build/performance/conditioning-qp-reg-v527')
binary=(prior/'qoco_snapshot_replay').resolve()
lib=Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1',LD_LIBRARY_PATH=str(lib.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
report=dict(pid=os.getpid(),complete=False,cases=[],binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
            scope='Externally applied equivalent objective and diagonal scalings, native Ruiz disabled. Native stopping tolerances tightened with objective scale; independent original-unit acceptance unchanged. Fresh process per repeat.')
def save():
 t=root/'report.tmp';t.write_text(json.dumps(report,indent=2));t.replace(root/'report.json')
save()
try:
 for name in ['analyse_qps_v169.py','isolate_conditioning_cost_v531.py']:shutil.copy2(Path('build/performance')/name,root/name)
 lines=(prior/'ruiz0.txt').read_text().splitlines();d=problem(prior/'ruiz0.txt')
 scale=next(json.loads(s[len('QP_SCALING '):]) for s in Path('build/performance/conditioning-matrices-v530/ruiz2-direct0.log').read_text().splitlines() if s.startswith('QP_SCALING '))
 (root/'scaling.json').write_text(json.dumps(scale))
 n,p,m,np_,na,ng,*_=map(int,lines[1].split())
 def vec(i,dtype=float):return np.array(lines[i].split()[1:],dtype=dtype)
 pp,pi,ap,ai,gp,gi=[vec(i,int) for i in range(4,10)]
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for diagonal,k in [(False,1.0),(True,1.0),(False,1e-4),(True,1e-4)]:
   name=f'diagonal{int(diagonal)}-cost{k}'
   D,E,F=[np.array(scale[key],float) if diagonal else np.ones(size) for key,size in [('D',n),('E',p),('F',m)]]
   values=d['values'].copy();offset=0
   for count,ptr,idx,row,column,factor in [(np_,pp,pi,D,D,k),(na,ap,ai,E,D,1),(ng,gp,gi,F,D,1)]:
    cols=np.repeat(np.arange(n),np.diff(ptr))
    values[offset:offset+count]*=factor*row[idx]*column[cols];offset+=count
   values[offset:offset+n]*=k*D;offset+=n
   values[offset:offset+p]*=E;offset+=p
   values[offset:]*=F
   changed=lines.copy();changed[11]=str(len(values))+' '+' '.join(format(v,'.17g') for v in values)
   numbers=changed[3].split()
   for idx in [5,6,7,8]:numbers[idx]=format(float(numbers[idx])*min(1,k),'.17g')
   changed[3]=' '.join(numbers)
   qp=root/(name+'.txt');qp.write_text('\n'.join(changed)+'\n')
   row=dict(diagonal=diagonal,cost_scale=k,iterations=[],audits=[],seconds=[]);report['cases'].append(row)
   for repeat in range(3):
    path=root/f'{name}-{repeat}.log';start=time.perf_counter()
    with path.open('x') as log:
     child=subprocess.Popen([str(binary),str(qp)],env=env,stdout=log,stderr=subprocess.STDOUT)
     report.update(stage=name,repeat=repeat,child_pid=child.pid);save();rc=child.wait(timeout=120)
    assert rc==0
    record=next(json.loads(s[10:]) for s in path.read_text().splitlines() if s.startswith('QP_REPLAY '))
    mapped=dict(record,x=(D*np.array(record['x'])).tolist(),y=(E*np.array(record['y'])/k).tolist(),z=(F*np.array(record['z'])/k).tolist(),s=(np.array(record['s'])/F).tolist())
    row['iterations'].append(record['iterations']);row['audits'].append(audit(d,mapped));row['seconds'].append(time.perf_counter()-start);save()
   print(name,sum(a['qualified'] for a in row['audits']),'/3',row['iterations'],flush=True)
 report['complete']=True
except Exception as error:report['error']=repr(error)
save()
