from pathlib import Path
import json, numpy as np, scipy.sparse as sp, clarabel

def problem(path):
 lines=Path(path).read_text().splitlines()
 assert lines[0]=='SPACEPDHCG_QOCO_QP_V1'
 n,p,m,np_,na,ng,l,ns,shift=map(int,lines[1].split())
 def vector(i,dtype=float):
  parts=lines[i].split();assert int(parts[0])==len(parts)-1
  return np.array(parts[1:],dtype=dtype)
 values=vector(11);translated=vector(12);origin=vector(13)
 starts=np.cumsum([0,np_,na,ng,n,p,m])
 P,A,G=[sp.csc_matrix((values[starts[k]:starts[k+1]],vector(5+2*k,int),vector(4+2*k,int)),shape=(rows,n)) for k,rows in enumerate([n,p,m])]
 P=P+P.T-sp.diags(P.diagonal())
 c,b,h=[values[starts[k]:starts[k+1]] for k in range(3,6)]
 return dict(P=P,A=A,G=G,c=c,b=b,h=h,origin=origin,shift=shift,translated=translated,values=values,soc=vector(10,int),l=l,n=n,p=p,m=m,offset=float(lines[14]))

def audit(data,result):
 x=np.array(result['x']);x=x+data['origin'] if data['shift'] else x
 y,z,s=[np.array(result[k]) for k in ['y','z','s']]
 P,A,G,c,b,h=[data[k] for k in ['P','A','G','c','b','h']]
 # Long-double products provide an independent accumulation order and precision.
 x,y,z,s=[v.astype(np.longdouble) for v in [x,y,z,s]]
 px,ax,gx,aty,gtz=P@x,A@x,G@x,A.T@y,G.T@z
 inf=lambda v:float(np.max(np.abs(v),initial=0))
 pr=max(inf(ax-b),inf(gx+s-h));dr=inf(px+c+aty+gtz)
 pobj=.5*np.dot(x,px)+np.dot(c,x);dobj=-.5*np.dot(x,px)-np.dot(b,y)-np.dot(h,z)
 gap=float(abs(pobj-dobj))/max(1,float(abs(pobj)),float(abs(dobj)))
 pres=pr/(1+max(inf(ax),inf(b),inf(gx),inf(h),inf(s)))
 dres=dr/(1+max(inf(px),inf(c),inf(aty),inf(gtz)))
 violation=max(float(np.max(-s[:data['l']],initial=0)),float(np.max(-z[:data['l']],initial=0)))
 start=data['l']
 for size in data['soc']:
  for v in [s,z]: violation=max(violation,float(np.sqrt(np.dot(v[start+1:start+size],v[start+1:start+size]))-v[start]))
  start+=size
 return dict(primal=pres,dual=dres,gap=gap,primal_absolute=pr,dual_absolute=dr,cone_violation=violation,objective=float(pobj),qualified=result['status'] in [1,2] and max(pres,dres,gap)<=1e-9 and violation<=1e-8)

