from pathlib import Path
from decimal import Decimal as D, getcontext
import numpy as np,json,hashlib
getcontext().prec=100
root=Path('/home/angus/build-qoco-division-trace-v370/snapshots')
rows=[];worst=None;valid=0;invalid=0;reasons={};first_invalid=None
for p in sorted(root.glob('division-*.bin')):
 with p.open('rb') as f:
  l,nsoc,m=map(int,np.fromfile(f,dtype='<i4',count=3));q=np.fromfile(f,dtype='<i4',count=nsoc)
  arrays=np.fromfile(f,dtype='<f8').reshape(3,m)
 offset=l;maxerr=0.;nonfinite=0
 for cone,size in enumerate(q):
  lam,v,out=arrays[:,offset:offset+size];offset+=size
  reason=None
  if not np.isfinite(lam).all():reason='nonfinite_lambda'
  elif not np.isfinite(v).all():reason='nonfinite_rhs'
  if reason:
   invalid+=1;reasons[reason]=reasons.get(reason,0)+1
   if first_invalid is None:first_invalid=dict(file=p.name,cone=cone,reason=reason,lambda_values=lam.tolist(),rhs=v.tolist())
   continue
  a=list(map(lambda x:D.from_float(float(x)),lam));b=list(map(lambda x:D.from_float(float(x)),v))
  determinant=a[0]*a[0]-sum(x*x for x in a[1:])
  if a[0]<=0 or determinant<=0:
   invalid+=1;reason='noninterior_lambda';reasons[reason]=reasons.get(reason,0)+1
   if first_invalid is None:first_invalid=dict(file=p.name,cone=cone,reason=reason,lambda_values=lam.tolist(),determinant=str(determinant))
   continue
  valid+=1
  d0=(a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:])))/determinant
  ref=[d0]+[(y-x*d0)/a[0] for x,y in zip(a[1:],b[1:])]
  if not np.isfinite(out).all():nonfinite+=1;continue
  err=float(max(abs(D.from_float(float(x))-y) for x,y in zip(out,ref))/max(D('1e-300'),max(map(abs,ref))))
  maxerr=max(maxerr,err)
  if worst is None or err>worst['relative_error']:
   worst=dict(file=p.name,cone=cone,relative_error=err,lambda_values=lam.tolist(),v=v.tolist(),output=out.tolist(),reference=list(map(str,ref)),determinant=str(determinant))
 rows.append(dict(file=p.name,max_relative_error=maxerr,nonfinite_outputs=nonfinite,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert rows
report=dict(valid_interior_pairs=valid,invalid_input_pairs=invalid,invalid_reasons=reasons,first_invalid=first_invalid,worst=worst,rows=rows)
out=Path('build/performance/division-trace-v370');out.mkdir(exist_ok=True)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
