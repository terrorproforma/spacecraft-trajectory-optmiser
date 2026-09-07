from pathlib import Path
import numpy as np,json,sys
from decimal import Decimal, getcontext
from types import SimpleNamespace
getcontext().prec=80
mp=SimpleNamespace(mpf=Decimal)
root=Path('/home/angus/build-qoco-linear-snapshot-v333')
L=np.longdouble
assert np.finfo(L).eps < np.finfo(float).eps

def read(path):
 with path.open('rb') as f:
  N,nnz,n,p,m,l,nsoc,ntnnz=np.fromfile(f,np.int32,8).tolist()
  reg=np.fromfile(f,np.float64,3).astype(L)
  rows=np.fromfile(f,np.int32,N+1);cols=np.fromfile(f,np.int32,nnz)
  val=np.fromfile(f,np.float64,nnz).astype(L)
  b,x,res=[np.fromfile(f,np.float64,N).astype(L) for _ in range(3)]
  nt=np.fromfile(f,np.float64,ntnnz).astype(L)
  ntidx,starts,q=[np.fromfile(f,np.int32,nsoc) for _ in range(3)]
  assert not f.read(1)
 ridx=np.repeat(np.arange(N),np.diff(rows));assert (cols>=ridx).all()
 def product(v):
  y=np.zeros(N,dtype=L);np.add.at(y,ridx,v*x[cols]);off=cols!=ridx
  np.add.at(y,cols[off],v[off]*x[ridx[off]]);return y
 def ntmul(u):
  y=np.zeros(m,dtype=L);y[:l]=nt[:l]*u[:l]
  for start,ni,size in zip(starts,ntidx,q):
   eta,w0=nt[ni:ni+2];w=nt[ni+2:ni+size+1];u0=u[start];uv=u[start+1:start+size]
   zeta=np.dot(w,uv);y[start]=eta*(w0*u0+zeta)
   y[start+1:start+size]=eta*(uv+(u0+zeta/(1+w0))*w)
  return y
 factor=product(val)
 core=val.copy();core[ridx>=n]=0
 true=product(core);true[:n]-=reg[0]*x[:n];true[n+p:]-=ntmul(ntmul(x[n+p:]))
 stored=factor.copy();stored[:n]-=reg[0]*x[:n];stored[n:n+p]+=reg[1]*x[n:n+p];stored[n+p:]+=reg[2]*x[n+p:]
 # Cross-check the worst finite row using independent 80-digit arithmetic.
 mp_error=None
 if np.isfinite(true).all():
  row=int(np.argmax(np.abs(b-true)))
  conv=lambda v:mp.mpf(float(v))
  mp_y=sum((conv(core[j])*conv(x[cols[j]]) for j in np.flatnonzero(ridx==row)),mp.mpf(0))
  mp_y+=sum((conv(core[j])*conv(x[ridx[j]]) for j in np.flatnonzero((cols==row)&(ridx!=row))),mp.mpf(0))
  if row<n:mp_y-=conv(reg[0])*conv(x[row])
  if row>=n+p:
   zrow=row-n-p
   if zrow<l:mp_y-=conv(nt[zrow])**2*conv(x[row])
   else:
    cone=next(i for i,(start,size) in enumerate(zip(starts,q)) if start<=zrow<start+size)
    start,ni,size=int(starts[cone]),int(ntidx[cone]),int(q[cone])
    eta,w0=map(conv,nt[ni:ni+2]);w=list(map(conv,nt[ni+2:ni+size+1]))
    def apply(u):
     dot=sum((a*v for a,v in zip(w,u[1:])),mp.mpf(0))
     return [eta*(w0*u[0]+dot)]+[eta*(v+(u[0]+dot/(1+w0))*a) for a,v in zip(w,u[1:])]
    mp_y-=apply(apply(list(map(conv,x[n+p+start:n+p+start+size]))))[zrow-start]
  mp_error=float(abs((conv(b[row])-mp_y)-mp.mpf(str(b[row]-true[row]))))
 norm=lambda a:float(np.max(np.abs(a)))
 return dict(decimal80_crosscheck_error=mp_error,file=path.name,rhs=norm(b),x=norm(x),factor_residual=norm(b-factor),true_residual=norm(b-true),gpu_residual=norm(res),gpu_residual_error=norm(res-(b-true)),stored_nt_error=norm(stored-true),worst_row=int(np.argmax(np.abs(b-true))))
rows=[read(p) for p in sorted((root/'snapshots').glob('*.bin'))]
(root/'linear-audit.json').write_text(json.dumps(dict(longdouble_eps=float(np.finfo(L).eps),rows=rows),indent=2))
print('worst residual arithmetic errors')
for r in sorted(rows,key=lambda r:r['gpu_residual_error'],reverse=True)[:6]:print(r)
print('last eight')
for r in rows[-8:]:print(r)
sys.path.insert(0,str(Path('build/performance/qp-ir-v309').resolve()))
from audit import problem,audit
record=[json.loads(s[10:]) for s in (root/'replay.log').read_text().splitlines() if s.startswith('QP_REPLAY ')][0]
print('original_QP_audit',audit(problem(Path('build/performance/qp-ir-v309/qp.txt')),record))
