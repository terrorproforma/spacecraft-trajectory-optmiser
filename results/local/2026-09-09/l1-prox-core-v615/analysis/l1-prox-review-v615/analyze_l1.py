#!/usr/bin/env python3
"""Bounded CPU-only exact epigraph/soft-threshold/recovery analysis for v615.

No solver or CUDA calls. Original source/captures/points are never changed.
All audits use Decimal65 arithmetic from exact represented FP64 values.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import types

INPUTS = Path('build/performance/halpern-real-v612/inputs')
HASHES = {
    'conditioning.txt':'1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf',
    'difficult.txt':'14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080',
    'conditioning-initial.txt':'9caf303469dcd254d56b74f8bf3818c8044f51a3a20bc3c07835b76dd23c5632',
    'difficult-initial.txt':'b92be6c5a0051ab99c000c459ef466c69fc8b2fd669c38b3580aea000d926da6',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def detect(c, arows, grows, h, nonnegative, qvalues):
    if any(v != 0 for v in qvalues):
        raise ValueError('this diagnostic requires exactly zero Q')
    n = len(c)
    auses = [0]*n
    guses = [[] for _ in range(n)]
    for row in arows:
        for j,v in row:
            auses[j] += (v != 0)
    for i,row in enumerate(grows):
        for j,v in row:
            if v != 0:
                guses[j].append((i,v))
    found = []
    for t,lam in enumerate(c):
        if not math.isfinite(lam) or lam <= 0 or auses[t] or len(guses[t]) != 2:
            continue
        pair = []
        for i,coefficient in guses[t]:
            other = [(j,v) for j,v in grows[i] if j != t and v != 0]
            if i >= nonnegative or coefficient != -1 or h[i] != 0 or len(other) != 1:
                break
            pair.append((i,other[0][0],other[0][1]))
        if len(pair) == 2 and pair[0][1] == pair[1][1] and {v for _,_,v in pair} == {-1,1}:
            found.append(dict(t=t,v=pair[0][1],positive_row=next(i for i,_,v in pair if v==1),
                              negative_row=next(i for i,_,v in pair if v==-1),lambda_value=float(lam)))
    ts = [p['t'] for p in found]
    vs = [p['v'] for p in found]
    rows = [p[k] for p in found for k in ('positive_row','negative_row')]
    if len(set(ts)) != len(ts) or len(set(vs)) != len(vs) or set(ts)&set(vs) or len(set(rows)) != len(rows):
        raise ValueError('overlapping/nested roles unsupported by this bounded map')
    return found


def rows(snap,k):
    out = [[] for _ in range((snap.n,snap.p,snap.m)[k])]
    for j in range(snap.n):
        for at in range(snap.ptr[k][j],snap.ptr[k][j+1]):
            out[snap.idx[k][at]].append((j,snap.val[k][at]))
    return out


def soft(w,g,step,lam):
    if not all(math.isfinite(v) for v in (w,g,step,lam)) or step<=0 or lam<=0:
        raise ValueError('finite positive metric/cost required')
    threshold = step*lam
    centre = w-step*g
    if not math.isfinite(threshold) or threshold == 0 or not math.isfinite(centre):
        raise ValueError('overflow/underflow in positive threshold or centre')
    return centre-threshold if centre>threshold else centre+threshold if centre<-threshold else 0.0


def pair_dual(v,g_retained,lam):
    if not all(math.isfinite(a) for a in (v,g_retained,lam)) or lam<=0:
        raise ValueError('nonfinite dual completion')
    if v>0:
        return lam,0.0
    if v<0:
        return 0.0,lam
    delta = min(lam,max(-lam,-g_retained))
    # Avoid lambda+delta overflow. Form a dominant nonnegative multiplier,
    # then its complement; the original audit still sees actual FP64 values.
    if delta>=0:
        plus = 0.5*lam+0.5*delta
        minus = lam-plus
    else:
        minus = 0.5*lam-0.5*delta
        plus = lam-minus
    assert 0<=plus<=lam and 0<=minus<=lam
    return plus,minus


def fixtures():
    c=[0.0,10000.0]
    g=[[(0,1),(1,-1)],[(0,-1),(1,-1)]]
    assert len(detect(c,[],g,[0,0],2,[0]))==1
    tests={}
    for name,ar,gr,hr,l,q in (
        ('tiny_extra_equality_use',[[(1,5e-324)]],g,[0,0],2,[0]),
        ('tiny_extra_row_coefficient',[],[g[0]+[(2,5e-324)],g[1]],[0,0],2,[0]),
        ('nonzero_rhs',[],g,[0,5e-324],2,[0]),
        ('SOC_membership',[],g,[0,0],1,[0]),
        ('nonunit_target_coefficient',[],[[(0,math.nextafter(1,2)),(1,-1)],g[1]],[0,0],2,[0]),
        ('tiny_nonzero_Q',[],g,[0,0],2,[5e-324]),
    ):
        cc=c+[0] if name=='tiny_extra_row_coefficient' else c
        try:
            accepted=bool(detect(cc,ar,gr,hr,l,q))
        except ValueError:
            accepted=False
        assert not accepted
        tests[name]='rejected_or_no_pair'
    zeros=[g[0]+[(2,0.0)],g[1]]
    assert len(detect(c+[0], [[(1,0.0)]],zeros,[0,0],2,[0]))==1
    tests['structural_zeros']='accepted_without_treating_tiny_nonzeros_as_zero'
    duplicate=g+[[(0,1),(2,-1)],[(0,-1),(2,-1)]]
    try:
        detect([0,10000,10000],[],duplicate,[0]*4,4,[0])
    except ValueError:
        tests['duplicate_target']='rejected'
    else:
        raise AssertionError('duplicate target accepted')
    prox=[]
    for w,g0,d,lam in ((2.,1.,.25,3.),(-2.,-1.,.25,3.),(.1,.2,.5,1.)):
        v=soft(w,g0,d,lam)
        sub=(w-v)/d-g0
        assert abs(sub-lam*(1 if v>0 else -1))<1e-12 if v!=0 else abs(sub)<=lam
        prox.append(dict(w=w,smooth_gradient=g0,diagonal_step=d,lambda_value=lam,v=v,prox_witness=sub))
    lam=10000.;step=2.**-20;threshold=step*lam
    edge=[]
    for centre in (math.nextafter(threshold,0),threshold,math.nextafter(threshold,math.inf),
                   -math.nextafter(threshold,0),-threshold,-math.nextafter(threshold,math.inf)):
        v=soft(centre,0.,step,lam)
        zp,zm=pair_dual(v,0.,lam)
        assert (v==0)==(abs(centre)<=threshold)
        if v!=0:
            assert abs(v)<1e-15 and abs(zp-zm)==lam
        edge.append(dict(centre=centre,threshold=threshold,v=v,plus=zp,minus=zm))
    clipped=[]
    for gradient in (-20000.,-7000.,0.,7000.,20000.):
        zp,zm=pair_dual(0.,gradient,lam)
        assert zp+zm==lam and zp>=0 and zm>=0
        residual=gradient+zp-zm
        assert residual==math.copysign(max(abs(gradient)-lam,0),gradient)
        clipped.append(dict(retained_gradient=gradient,plus=zp,minus=zm,stationarity=residual))
    for name,args in [('threshold_overflow',(1.,0.,2.,float.fromhex('0x1.fffffffffffffp1023'))),
                      ('threshold_underflow',(1.,0.,5e-324,5e-324))]:
        try:
            soft(*args)
        except ValueError:
            tests[name]='rejected'
        else:
            raise AssertionError(name)
    B,O,D,eta,w,gradient,lam=.125,.25,2.,.5,2.,1.,3.
    native_step=eta*O/(B*D*D)
    native=soft(w,gradient,native_step,lam)
    scaled=soft(B*D*w,O*gradient/D,eta,O*lam/D)/(B*D)
    assert native==scaled==1.0
    assert B*O*lam*abs(w)==(O*lam/D)*abs(B*D*w)
    return dict(detection=tests,proximal_oracles=prox,one_ulp_threshold_oracles=edge,exact_zero_completion_oracles=clipped,
                scaled_coordinate_oracle=dict(B=B,O=O,D=D,eta=eta,native_step=native_step,
                                              lambda_scaled=O*lam/D,native_result=native,scaled_result_mapped_back=scaled))


def capture(name,mathmod):
    q=mathmod.Snapshot(INPUTS/(name+'.txt'))
    ar,gr=rows(q,1),rows(q,2)
    epigraphs=detect(q.c,ar,gr,q.h,q.l,q.val[0])
    point={}
    for line in (INPUTS/(name+'-initial.txt')).read_text().splitlines()[3:]:
        key,count,*values=line.split();assert int(count)==len(values);point[key]=list(map(float,values))
    point['termination_code']=1
    baseline=q.audit(point);assert baseline['qualified']
    reconstructed_baseline={k:list(v) if isinstance(v,list) else v for k,v in point.items()}
    gx=q.mul(2,[mathmod.dec(v) for v in point['x']])
    reconstructed_baseline['s']=[float(h-g) for h,g in zip(q.h,gx)]
    assert q.audit(reconstructed_baseline)['qualified']
    y,z=[[mathmod.dec(v) for v in point[k]] for k in ('y','z')]
    ay,gz=q.mul(1,y,True),q.mul(2,z,True)
    canonical={k:list(v) if isinstance(v,list) else v for k,v in point.items()}
    t_only={k:list(v) if isinstance(v,list) else v for k,v in point.items()}
    detail=[]
    for p in epigraphs:
        t,v,plus,minus=(p[k] for k in ('t','v','positive_row','negative_row'))
        lam=p['lambda_value'];xv=point['x'][v]
        gret=q.c[v]+ay[v]+gz[v]-(z[plus]-z[minus])
        zp,zm=pair_dual(xv,float(gret),lam)
        canonical['x'][t]=t_only['x'][t]=abs(xv)
        canonical['z'][plus]=zp;canonical['z'][minus]=zm
        detail.append(dict(**p,v_value=xv,t_value=point['x'][t],original_plus=point['z'][plus],original_minus=point['z'][minus],
                           retained_gradient=float(gret),canonical_plus=zp,canonical_minus=zm,
                           canonical_v_stationarity=float(gret+mathmod.dec(zp)-mathmod.dec(zm))))
    for changed in (canonical,t_only):
        gx=q.mul(2,[mathmod.dec(v) for v in changed['x']])
        changed['s']=[float(h-g) for h,g in zip(q.h,gx)]
    nz=[abs(p['v_value']) for p in detail if p['v_value']!=0]
    delta=sum((mathmod.dec(p['lambda_value'])*(abs(mathmod.dec(p['v_value']))-mathmod.dec(p['t_value'])) for p in detail),Decimal(0))
    reduced_c=[q.c[j] for j in range(q.n) if j not in {p['t'] for p in epigraphs}]
    reduced_objective=sum((q.c[j]*mathmod.dec(canonical['x'][j]) for j in range(q.n) if j not in {p['t'] for p in epigraphs}),Decimal(0))
    reduced_objective+=sum((mathmod.dec(p['lambda_value'])*abs(mathmod.dec(canonical['x'][p['v']])) for p in epigraphs),Decimal(0))
    original_canonical_objective=mathmod.dot(q.c,[mathmod.dec(v) for v in canonical['x']])
    assert abs(reduced_objective-original_canonical_objective)<Decimal('1e-55')
    all_rows=[p[k] for p in epigraphs for k in ('positive_row','negative_row')]
    excluded_rows=set(all_rows);excluded_cols={p['t'] for p in epigraphs}
    reduced_g_nnz=sum(v!=0 for i,row in enumerate(gr) if i not in excluded_rows for j,v in row if j not in excluded_cols)
    return dict(snapshot_sha256=HASHES[name+'.txt'],point_sha256=HASHES[name+'-initial.txt'],
                original_dimensions=dict(n=q.n,p=q.p,m=q.m,nonnegative=q.l),
                epigraph_count=len(detail),lambda_values=sorted({p['lambda_value'] for p in detail}),
                logically_reduced_dimensions=dict(n=q.n-len(detail),p=q.p,m=q.m-2*len(detail),nonnegative=q.l-2*len(detail)),
                reduced_numerical_G_nonzeros=reduced_g_nnz,
                near_zero=dict(nonzero_count=len(nz),exact_zero_count=len(detail)-len(nz),minimum_nonzero=min(nz),maximum_nonzero=max(nz),
                               nonzero_below_1e_15=sum(v<1e-15 for v in nz),interior_pair_count=sum(p['original_plus']>0 and p['original_minus']>0 for p in detail)),
                baseline_supplied_slack_audit=baseline,baseline_reconstructed_slack_audit=q.audit(reconstructed_baseline),
                after_replacing_t_only_audit=q.audit(t_only),
                after_premature_canonical_sign_reconstruction_audit=q.audit(canonical),
                objective_change_from_setting_t_abs_v=float(delta),
                original_canonical_and_reduced_objective_match=True,
                original_c_euclidean_norm=float(mathmod.dot(q.c,q.c).sqrt()),
                smooth_c_euclidean_norm=float(mathmod.dot(reduced_c,reduced_c).sqrt()),
                joint_unscaled_smooth_and_lambda_norm=float((mathmod.dot(reduced_c,reduced_c)+sum((mathmod.dec(p['lambda_value'])**2 for p in detail),Decimal(0))).sqrt()),
                detection_map=epigraphs,reference_examples=detail[:3],
                largest_canonical_stationarity=sorted(detail,key=lambda p:abs(p['canonical_v_stationarity']),reverse=True)[:3])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    for name,h in HASHES.items():assert sha((INPUTS/name).read_bytes())==h
    source=Path(__file__).with_name('decimal_audit_source.py')
    assert sha(source.read_bytes())=='b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4'
    mod=types.ModuleType('direct_source_decimal');mod.__file__=str(source)
    exec(compile(source.read_bytes(),str(source),'exec'),mod.__dict__)
    with localcontext() as ctx:
        ctx.prec=65
        out=dict(scope='CPU-only proposed exact reduction math/oracles; no implementation, timing or GPU claim',
                 decimal_precision=65,script_sha256=sha(Path(__file__).read_bytes()),decimal_source_sha256=sha(source.read_bytes()),
                 input_sha256=HASHES,oracles=fixtures(),captures={name:capture(name,mod) for name in ('conditioning','difficult')})
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
