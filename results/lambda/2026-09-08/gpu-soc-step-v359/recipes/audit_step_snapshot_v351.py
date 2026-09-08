from pathlib import Path
from decimal import Decimal as D,getcontext
import numpy as np,json,hashlib
getcontext().prec=100
root=Path('/home/angus/build-qoco-step-snapshot-v351');out=Path('build/performance/step-snapshot-v351');out.mkdir(exist_ok=True)
def inner(a,b):return a[0]*b[0]-sum((x*y for x,y in zip(a[1:],b[1:])),D(0))
rows=[]
for path in sorted((root/'snapshots').glob('*.bin')):
 with path.open('rb') as f:
  l,nsoc,m=np.fromfile(f,np.int32,3);alpha,factor=np.fromfile(f,np.float64,2);q=np.fromfile(f,np.int32,nsoc);u=np.fromfile(f,np.float64,m);du=np.fromfile(f,np.float64,m);assert f.read()==b''
 row=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gpu_alpha=float(alpha),factor=float(factor),nonfinite_u=int(np.sum(~np.isfinite(u))),nonfinite_direction=int(np.sum(~np.isfinite(du))))
 if row['nonfinite_u'] or row['nonfinite_direction']:rows.append(row);continue
 lp=[(float(-u[i]/du[i]),i) for i in range(l) if du[i]<0];lpmin=min(lp,default=(1.,None))
 steps=[];idx=int(l)
 for c,n in enumerate(q):
  x=[D(float(v)) for v in u[idx:idx+n]];dx=[D(float(v)) for v in du[idx:idx+n]];idx+=n
  a=inner(dx,dx);b=2*inner(x,dx);cc=inner(x,x);disc=b*b-4*a*cc;roots=[D(1)]
  if dx[0]<0:roots.append(-x[0]/dx[0])
  if a==0 and b<0:roots.append(-cc/b)
  elif a!=0 and disc>=0:
   sq=disc.sqrt();roots.extend(r for r in [(-b-sq)/(2*a),(-b+sq)/(2*a)] if r>=0)
  steps.append(dict(cone=c,step=float(min(roots)),determinant=float(cc),relative_determinant=float(cc/(x[0]*x[0]))))
 winner=min(steps,key=lambda x:x['step']);bound=min(1.,lpmin[0],winner['step'])
 row.update(lp_bound=lpmin[0],lp_index=lpmin[1],soc_min=winner,reference_raw_bound=bound,reference_final_bound=0. if bound<1e-12 else factor*bound,soc_steps=steps)
 rows.append(row)
(out/'report.json').write_text(json.dumps(rows,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='soc_steps'} for r in rows],indent=2))
print('cone 230 independent step',rows[0]['soc_steps'][230]['step'])
