"""Independent exact-map and FP64 iterate fixtures for opt-in mass/L1 PDHG.

Only standard-library CPU arithmetic is used. No solver/native library is loaded.
"""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = "fdf52ae31d259240ffebdcf79ed0965dce8b9298"
N = 15
MASS, GAMMA, NU, T = list(range(4)), [4, 5, 6], [7, 8, 9], [12, 13, 14]
ACTIVE = GAMMA + NU + [10, 11]
GAIN = [Q(1, 8), Q(1, 4), Q(1, 2)]
AFFINE = [Q(1, 16), Q(-1, 32), Q(1, 64)]
LAMBDA = [Q(1, 4), Q(4), Q(1, 8)]
THETA = Q(19, 20)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dot(a, b):
    return sum((x*y for x,y in zip(a,b,strict=True)), Q(0))


def mv(a, x):
    return [dot(row,x) for row in a]


def transpose(a, y):
    return [sum((row[j]*z for row,z in zip(a,y,strict=True)), Q(0)) for j in range(len(a[0]))]


def fixture():
    rows, upper, mass_rows = [], [], []
    def add(terms, rhs):
        row = [Q(0)]*N
        for j,value in terms.items(): row[j] = Q(value)
        rows.append(row); upper.append(Q(rhs)); return len(rows)-1
    add({0:1},1)
    for k in range(3): add({k:-1,k+1:1,GAMMA[k]:GAIN[k],NU[k]:-1},AFFINE[k])
    add({GAMMA[0]:1,10:1,NU[1]:-1},Q(1,4))
    add({11:1,GAMMA[2]:-1},Q(1,8))
    for node in range(4):
        for sign,rhs in [(1,2),(-1,0),(-1,Q(-3,4))]:
            mass_rows.append((add({node:sign},rhs),node,sign))
    for j in GAMMA: add({j:-1},0)
    pairs=[]
    for v,t in zip(NU,T,strict=True):
        pairs.append((v,t,add({v:1,t:-1},0),add({v:-1,t:-1},0)))
    f = [[Q(0)]*N for _ in range(3)]
    f[0][GAMMA[1]],f[0][10] = Q(2),Q(1)
    f[1][11],f[1][GAMMA[2]] = Q(1),Q(-1)
    f[2][GAMMA[0]] = Q(1,2)
    offset = [Q(1,8),Q(-1,4),Q(3,2)]
    c=[Q(0)]*N
    for j,v in zip(GAMMA,[Q(1,8),Q(-1,4),Q(3,8)],strict=True):c[j]=v
    for j,v in zip(NU,[Q(-3),Q(1,4),Q(4)],strict=True):c[j]=v
    c[10],c[11]=Q(1,2),Q(-1,8)
    for j,v in zip(T,LAMBDA,strict=True):c[j]=v
    return rows,upper,f,offset,c,mass_rows,pairs


def reduction(a,b,f,c,pairs):
    d=[Q(1)]
    s=[[Q(0)]*N]
    for k in range(3):
        d.append(d[-1]+AFFINE[k]); row=s[-1].copy()
        row[NU[k]]+=1;row[GAMMA[k]]-=GAIN[k];s.append(row)
    removed={0,1,2,3,*[r for _,_,p,n in pairs for r in (p,n)]}
    keep=[i for i in range(len(a)) if i not in removed]
    transformed=[];rhs=[]
    for i in keep:
        transformed.append([a[i][j]+sum((a[i][m]*s[m][j] for m in MASS),Q(0)) for j in ACTIVE])
        rhs.append(b[i]-sum((a[i][m]*d[m] for m in MASS),Q(0)))
    ff=[[row[j] for j in ACTIVE] for row in f]
    cc=[c[j]+sum((c[m]*s[m][j] for m in MASS),Q(0)) for j in ACTIVE]
    return d,s,keep,transformed,rhs,ff,cc


def downward(value):
    v=float(value)
    if Q(v)>value:v=math.nextafter(v,0)
    assert math.isfinite(v) and v>0 and Q(v)<=value
    return v


def metric(k,scalar_count):
    row=[sum(map(abs,r),Q(0)) for r in k]
    col=[sum((abs(r[j]) for r in k),Q(0)) for j in range(len(k[0]))]
    den=row.copy();den[scalar_count:]=[max(row[scalar_count:])]*3
    tau=[downward(THETA/(v or 1)) for v in col]
    sigma=[downward(THETA/(v or 1)) for v in den]
    alpha=max(Q(v)*r for v,r in zip(sigma,row,strict=True))
    beta=max(Q(v)*r for v,r in zip(tau,col,strict=True))
    assert alpha*beta<=THETA**2<1
    assert len(set(sigma[scalar_count:]))==1
    return tau,sigma,row,col,alpha*beta


def project_soc(v):
    n=math.hypot(v[0],v[1]);r=v[2]
    if n<=r:return v.copy()
    if n<=-r:return [0.,0.,0.]
    radius=.5*(n+r)
    return [radius*v[0]/n,radius*v[1]/n,radius]


def complete_pair(v,gradient,lam):
    if v>0:return lam,0.
    if v<0:return 0.,lam
    delta=max(-lam,min(lam,-gradient))
    return .5*(lam+delta),.5*(lam-delta)


def mass_duals(a,f,y,w):
    normal=[sum((a[i][m]*y[i] for i in range(4,len(a))),Q(0))+sum((f[i][m]*w[i] for i in range(3)),Q(0)) for m in MASS]
    return [-sum(normal,Q(0))]+[-sum(normal[j:],Q(0)) for j in range(1,4)]


def reconstruct(a,f,c,d,s,keep,pairs,xr,yr,wr):
    x=[0.]*N;y=[0.]*len(a)
    for j,value in zip(ACTIVE,xr,strict=True):x[j]=value
    for m in MASS:x[m]=float(d[m])+sum(float(s[m][j])*x[j] for j in ACTIVE)
    for i,value in zip(keep,yr,strict=True):y[i]=value
    y[:4]=list(map(float,mass_duals(a,f,y,wr)))
    gradient=[float(c[j])+sum(float(row[j])*v for row,v in zip(a,y,strict=True))+sum(float(row[j])*v for row,v in zip(f,wr,strict=True)) for j in range(N)]
    for (v,t,pos,neg),lam in zip(pairs,LAMBDA,strict=True):
        x[t]=abs(x[v]);y[pos],y[neg]=complete_pair(x[v],gradient[v],float(lam))
    return x,y,wr,gradient


def exact_gates(a,b,f,offset,c,x,y,w):
    # Reconstruct native slack but use exact FP64 values in rational products.
    x,y,w=list(map(Q,x)),list(map(Q,y)),list(map(Q,w))
    ax,fx=mv(a,x),mv(f,x)
    slack=[Q(float(b[i]-ax[i])) for i in range(6,len(a))]
    soc=[Q(float(v+o)) for v,o in zip(fx,offset,strict=True)]
    at,ft=transpose(a,y),transpose(f,w)
    grad=[cc+u+v for cc,u,v in zip(c,at,ft,strict=True)]
    eqpart=transpose(a[:6],y[:6]); gpart=[u-v for u,v in zip(at,eqpart,strict=True)]
    gpart=[u+v for u,v in zip(gpart,ft,strict=True)]
    equation=max([abs(ax[i]+slack[i-6]-b[i]) for i in range(6,len(a))]
                 +[abs(-v+s-o) for v,s,o in zip(fx,soc,offset,strict=True)])
    pinf=max(max(abs(ax[i]-b[i]) for i in range(6)),equation)
    pden=1+max(map(abs,ax[:6]+b[:6]+[-v for v in fx]+ax[6:]+b[6:]+offset+slack+soc))
    dinf=max(map(abs,grad));dden=1+max(map(abs,c+eqpart+gpart))
    pobj=dot(c,x);dobj=-dot(b,y)+dot(offset,w)
    den=max(Q(1),abs(pobj),abs(dobj))
    comp=[abs(s*z) for s,z in zip(slack,y[6:],strict=True)]+[abs(dot(soc,[-v for v in w]))]
    primal_cone=max(0.,float(-min(slack)),math.hypot(float(soc[0]),float(soc[1]))-float(soc[2]))
    dual_cone=max(0.,float(-min(y[6:])),math.hypot(float(w[0]),float(w[1]))+float(w[2]))
    gates={"normalized_primal":float(pinf/pden),"normalized_stationarity":float(dinf/dden),
           "normalized_gap":float(abs(pobj-dobj)/den),"normalized_max_block_complementarity":float(max(comp)/den),
           "primal_cone":primal_cone,"dual_cone":dual_cone,"primal_objective":float(pobj),"dual_objective":float(dobj),
           "maximum_mass_equation_absolute":float(max(abs(ax[i]-b[i]) for i in range(4))),"conic_equation_absolute":float(equation)}
    gates["passes"]=all(v<=1e-9 for k,v in gates.items() if k.startswith("normalized_")) and primal_cone<=1e-8 and dual_cone<=1e-8
    return gates


def main():
    a,b,f,offset,c,mass_rows,pairs=fixture()
    d,s,keep,ar,br,fr,cr=reduction(a,b,f,c,pairs)
    tau,sigma,rs,cs,certificate=metric(ar+fr,len(ar))
    # Exact affine map, signed adjoint, constant shift and dual objective checks.
    probes=0
    for seed in range(9):
        q=[Q(((j+3*seed)%11)-5,32) for j in range(N)]
        x=q.copy()
        for m in MASS:x[m]=d[m]+dot(s[m],q)
        for i in range(4):assert dot(a[i],x)==b[i]
        for ri,i in enumerate(keep):assert b[i]-dot(a[i],x)==br[ri]-dot(ar[ri],[q[j] for j in ACTIVE])
        normal=[Q(((m+seed)%7)-3,16) for m in MASS]
        pulled=[sum((s[m][j]*normal[m] for m in MASS),Q(0)) for j in range(N)]
        assert dot(normal,mv(s,q))==dot(pulled,q)
        ye=[-sum(normal,Q(0))]+[-sum(normal[m:],Q(0)) for m in range(1,4)]
        actual=transpose(a[:4],ye)
        assert all(actual[m]+normal[m]==0 for m in MASS)
        assert all(actual[j]==pulled[j] for j in ACTIVE)
        assert -dot(b[:4],ye)==dot(d,normal)
        probes+=1
    # Native dual-first update with diagonal Moreau/SOC prox and soft threshold.
    xr=[.25,-.125,.375,.5,-.5,.03125,.375,-.25]
    xbar=xr.copy();yr=[(.125 if i>=6 else -.25) for i in keep];wr=[.125,-.25,-.5]
    initial=reconstruct(a,f,c,d,s,keep,pairs,xr,yr,wr)
    initial={"x":initial[0],"scalar_dual":initial[1],"affine_dual":initial[2],
             "reduced_xbar":xbar.copy(),"original_gates":exact_gates(a,b,f,offset,c,*initial[:3])}
    records=[]
    for iteration in range(1,4):
        scalar=[sum(float(v)*x for v,x in zip(row,xbar,strict=True)) for row in ar]
        for i in range(len(ar)):
            value=yr[i]+sigma[i]*scalar[i]
            projection=float(br[i]) if keep[i]<6 else min(value/sigma[i],float(br[i]))
            yr[i]=value-sigma[i]*projection
        fwd=[sum(float(v)*x for v,x in zip(row,xbar,strict=True)) for row in fr]
        step=sigma[len(ar)]
        before=[wr[i]+step*fwd[i] for i in range(3)]
        projected=project_soc([before[i]/step+float(offset[i]) for i in range(3)])
        wr=[before[i]-step*(projected[i]-float(offset[i])) for i in range(3)]
        gradient=[float(cr[j])+sum(float(row[j])*v for row,v in zip(ar,yr,strict=True))+sum(float(row[j])*v for row,v in zip(fr,wr,strict=True)) for j in range(len(ACTIVE))]
        old=xr.copy()
        for j,original in enumerate(ACTIVE):
            argument=old[j]-tau[j]*gradient[j]
            lam=float(LAMBDA[NU.index(original)]) if original in NU else 0.
            threshold=tau[j]*lam
            xr[j]=(0. if abs(argument)<=threshold else math.copysign(abs(argument)-threshold,argument)) if lam else argument
        xbar=[2*v-u for v,u in zip(xr,old,strict=True)]
        x,y,w,full_gradient=reconstruct(a,f,c,d,s,keep,pairs,xr,yr,wr)
        assert all(math.isclose(full_gradient[j],value,rel_tol=1e-13,abs_tol=1e-13) for j,value in zip(ACTIVE,gradient,strict=True))
        records.append({"iteration":iteration,"x":x,"scalar_dual":y,"affine_dual":w,"reduced_xbar":xbar.copy(),
                        "reduced_gradient":gradient,"original_gates":exact_gates(a,b,f,offset,c,x,y,w)})
    # An original near-zero certificate must be accepted before canonical pairs.
    seed_c=[Q(0)]*N
    for t,lam in zip(T,LAMBDA,strict=True):seed_c[t]=lam
    seed_x=[0.]*N
    seed_x[10],seed_x[11]=.25,.125
    seed_x[NU[0]]=seed_x[T[0]]=1e-20
    for m in MASS:seed_x[m]=float(d[m])+sum(float(s[m][j])*seed_x[j] for j in ACTIVE)
    seed_y=[0.]*len(a)
    for (_,_,pos,neg),lam in zip(pairs,LAMBDA,strict=True):seed_y[pos]=seed_y[neg]=float(lam)/2
    original=exact_gates(a,b,f,offset,seed_c,seed_x,seed_y,[0.,0.,0.])
    assert original["passes"]
    canonical=reconstruct(a,f,seed_c,d,s,keep,pairs,[seed_x[j] for j in ACTIVE],[seed_y[i] for i in keep],[0.,0.,0.])
    changed=exact_gates(a,b,f,offset,seed_c,*canonical[:3])
    assert not changed["passes"] and changed["normalized_stationarity"]>.01
    source_names=["cpp/cuda/src/persistent_l1.cuh","cpp/cuda/src/persistent_l1_host.cuh",
                  "cpp/cuda/src/persistent_common_kkt.cuh","cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h"]
    source_hashes={name:sha(subprocess.check_output(["git","show",BASE+":"+name],cwd=ROOT)) for name in source_names}
    out={"scope":"Independent CPU fixture/oracle for pending implementation; no native loads/GPU or solution-derived step choice.",
         "base_commit":BASE,"base_source_sha256":source_hashes,"script_sha256":sha(Path(__file__).read_bytes()),
         "layout":{"variables":N,"equalities":6,"scalar_rows":len(a),"affine_SOC_rows":3,"SOC_radius_last":True,
                   "mass":MASS,"gamma":GAMMA,"virtual":NU,"epigraph":T,"active":ACTIVE,"retained_scalar_rows":keep},
         "map":{"initial_mass":1.,"gamma_coefficients":list(map(float,GAIN)),"affine_rhs":list(map(float,AFFINE)),
                "mass_constants":list(map(float,d)),"mass_scalar_rows":mass_rows,"L1_pairs":pairs},
         "original":{"A":a,"scalar_upper":b,"F":f,"affine_offset":offset,"c":c},
         "reduced":{"A":ar,"scalar_upper":br,"F":fr,"c":cr,"row_abs_sums":rs,"column_abs_sums":cs,
                    "tau":tau,"sigma":sigma,"norm_squared_bound_exact":str(certificate),"norm_squared_bound":float(certificate)},
         "exact_map_adjoint_dual_probes":probes,"initial_point":initial,"iterations":records,
         "qualified_near_zero_seed":{"x":seed_x,"scalar_dual":seed_y,"affine_dual":[0.,0.,0.],"c":seed_c,
                                     "original_gates":original,"canonicalized_gates":changed},
         "unsupported_scope":"Nonzero Q, mass costs, extra retained mass equality/SOC uses, altered causal coefficients or control overlap need rejection/proof; runtime lifecycle is reviewed separately."}
    def encode(value):
        if isinstance(value,Q):return float(value)
        raise TypeError(type(value).__name__)
    path=Path(__file__).with_name("fixtures.json")
    path.write_text(json.dumps(out,indent=2,default=encode,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"passed":True,"fixture_sha256":sha(path.read_bytes()),"exact_probes":probes,
                      "steps":len(records),"norm_squared_bound":float(certificate),"near_zero_seed_preservation_required":True},indent=2))


if __name__=="__main__":main()
