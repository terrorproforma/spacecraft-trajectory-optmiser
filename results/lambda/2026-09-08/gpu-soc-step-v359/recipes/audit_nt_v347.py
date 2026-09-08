from pathlib import Path
from decimal import Decimal as D,getcontext
import numpy as np,json,sys
getcontext().prec=100
root=Path('build/performance/nt-normalization-v347')
def dot(a,b):return sum((x*y for x,y in zip(a,b)),D(0))
def determinant(a):return a[0]*a[0]-dot(a[1:],a[1:])
if sys.argv[1]=='generate':
 records=[json.loads(s[10:]) for s in Path('build/performance/nonfinite-replay-v336/guard_graph.log').read_text().splitlines() if s.startswith('QP_REPLAY ')]
 qplines=Path('build/performance/qp-ir-v309/qp.txt').read_text().splitlines();l=int(qplines[1].split()[6]);sizes=list(map(int,qplines[10].split()[1:]))
 pairs=[];labels=[];excluded=0
 def append(s,z,label):
  global excluded
  ss=list(map(lambda x:D(float(x)),s));zz=list(map(lambda x:D(float(x)),z))
  if determinant(ss)<=0 or determinant(zz)<=0 or ss[0]<=0 or zz[0]<=0:excluded+=1;return
  pairs.append((ss,zz));labels.append(label)
 for k,r in enumerate(records):
  idx=l
  for c,n in enumerate(sizes):append(r['s'][idx:idx+n],r['z'][idx:idx+n],f'QP{k}:{c}');idx+=n
 rng=np.random.default_rng(347)
 for n in [3,4,33,257]:
  for margin in [1e-14,1e-8,1.,100.]:
   for j in range(8):
    s=rng.normal(size=n);z=rng.normal(size=n)
    s[0]=np.linalg.norm(s[1:])*(1+margin);z[0]=np.linalg.norm(z[1:])*(1+margin)
    # Opposing spatial directions expose wbar cancellation/normalization.
    if j%2==0:z[1:]=-s[1:];z[0]=s[0]
    s*=10.**(j-4);z*=10.**(4-j)
    append(s,z,f'synthetic:{n}:{margin}:{j}')
 ns=[len(s) for s,z in pairs];refs=[]
 for s,z in pairs:
  ss=determinant(s).sqrt();zz=determinant(z).sqrt()
  gamma=((1+dot(s,z)/(ss*zz))/2).sqrt();eta=(ss/zz).sqrt()
  w=[(s[j]/ss+(z[j]/zz if j==0 else -z[j]/zz))/(2*gamma) for j in range(len(s))]
  mat=[eta*eta*(2*w[j]*w[k]+(-1 if j==0 else 1) if j==k else 2*w[j]*w[k]) for j in range(len(s)) for k in range(j+1)]
  refs.append(dict(nt=list(map(float,[eta]+w)),wtw=list(map(float,mat))))
 with (root/'inputs.bin').open('wb') as f:
  np.array([len(ns),sum(ns)],np.int32).tofile(f);np.array(ns,np.int32).tofile(f)
  np.array([float(x) for s,z in pairs for x in s]).tofile(f);np.array([float(x) for s,z in pairs for x in z]).tofile(f)
 (root/'reference.json').write_text(json.dumps(dict(labels=labels,sizes=ns,reference=refs,excluded_noninterior_pairs=excluded)))
 print('pairs',len(ns),'excluded',excluded)
else:
 ref=json.loads((root/'reference.json').read_text());ns=ref['sizes'];raw=np.fromfile(root/'output.bin',np.float64)
 nt_size=sum(n+1 for n in ns);tri=sum(n*(n+1)//2 for n in ns);assert len(raw)==2*(nt_size+tri)
 rows={}
 for variant in ['legacy','candidate']:
  base=0 if variant=='legacy' else nt_size+tri;ni=base;mi=base+nt_size
  errors=[];matrix_errors=[];identity=[];nonfinite=[]
  for i,(n,truth) in enumerate(zip(ns,ref['reference'])):
   nt=raw[ni:ni+n+1];m=raw[mi:mi+n*(n+1)//2];ni+=n+1;mi+=n*(n+1)//2
   nonfinite.append(not(np.isfinite(nt).all() and np.isfinite(m).all()))
   errors.append(float(np.max(np.abs(nt-np.array(truth['nt'])))/max(abs(x) for x in truth['nt'])))
   matrix_errors.append(float(np.max(np.abs(m-np.array(truth['wtw'])))/max(abs(x) for x in truth['wtw'])))
   # Identity checked in extended precision, using stored binary64 compact vector.
   w=nt[1:].astype(np.longdouble);identity.append(float(abs(w[0]*w[0]-np.dot(w[1:],w[1:])-1)))
  rows[variant]=dict(nonfinite_pairs=sum(nonfinite),max_relative_nt_error=float(np.nanmax(errors)),max_relative_wtw_error=float(np.nanmax(matrix_errors)),median_relative_wtw_error=float(np.nanmedian(matrix_errors)),max_lorentz_normalization_error=float(np.nanmax(identity)))
 summary=dict(cases=len(ns),excluded_noninterior_pairs=ref['excluded_noninterior_pairs'],rows=rows)
 (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 assert rows['candidate']['nonfinite_pairs']==0
 assert rows['candidate']['max_relative_nt_error']<1e-12
 assert rows['candidate']['max_relative_wtw_error']<1e-12
