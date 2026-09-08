"""Decimal65 original KKT plus exact coefficient/actual-step audit; CPU only."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import struct
import types

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'build/performance/mass-real-v622'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load_source(path,digest,name):
    assert sha(path)==digest
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module
dm=load_source(ROOT/'build/performance/l1-weight-real-review-v618/decimal_audit_source.py',
               'b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4','mass_decimal')
old=load_source(ROOT/'build/performance/l1-weight-real-review-v618/review_real.py',
                'd74aa8fdf9d7af5b58c540ee690b54643e1562d6c912bdd63a3cf11eb00753b7','mass_diagnosis')
def strict(text):
    return json.loads(text,parse_int=float,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def bits(values):return b''.join(struct.pack('<d',float(v)) for v in values)
def read_records(path):
    result={}
    for line in path.read_text().splitlines():
        if line.startswith('PERSISTENT_REPLAY'):
            prefix,value=line.split(' ',1);assert prefix not in result
            result[prefix]=strict(value)
    return result
def rows(q,index):
    result=[[] for _ in range((q.n,q.p,q.m)[index])]
    for j in range(q.n):
        for k in range(q.ptr[index][j],q.ptr[index][j+1]):
            v=F(q.val[index][k])
            if v:result[q.idx[index][k]].append((j,v))
    return result
def scan(values,reverse=False):
    a=list(values);distance=1
    while distance<len(a):
        a=[v+a[i+distance if reverse else i-distance]
           if 0<=(i+distance if reverse else i-distance)<len(a) else v for i,v in enumerate(a)]
        distance*=2
    return a

def metric_and_map(q,meta,final,seeded):
    nodes=[tuple(map(int,row)) for row in meta['mass_map']]
    pairs=meta['l1_map'];m=len(nodes)
    masses={node[0]:i for i,node in enumerate(nodes)}
    removed_equalities={node[1] for node in nodes}
    epigraphs={int(row[0]) for row in pairs}
    removed_g={int(row[k])-q.p for row in pairs for k in (2,3)}
    active=set(range(q.n))-set(masses)-epigraphs
    a,g=rows(q,1),rows(q,2)
    gamma=[F(0)];rhs=[float(q.b[nodes[0][1]])]
    assert nodes[0][2:]==(-1,-1) and a[nodes[0][1]]==[(nodes[0][0],F(1))] and rhs==[1.]
    for i,node in enumerate(nodes[1:],1):
        mass,eq,ga,nu=node;entries=dict(a[eq]);value=entries[ga]
        assert value>0 and entries=={mass:F(1),nodes[i-1][0]:F(-1),ga:value,nu:F(-1)}
        assert ga in active and nu in active
        gamma.append(value);rhs.append(float(q.b[eq]))
    mass_rows=[[] for _ in nodes]
    for r,entries in enumerate(g):
        touch=[(j,v) for j,v in entries if j in masses]
        if touch:
            assert len(entries)==len(touch)==1 and r<q.l and r not in removed_g
            j,v=touch[0];assert abs(v)==1
            mass_rows[masses[j]].append((r,v))
    assert all(len(r)==3 for r in mass_rows)
    native_g=list(range(q.p,q.p+q.l));soc_native=[]
    start=q.l
    for size in q.soc:
        first=q.p+start
        native_g += [first+size-1]+list(range(first,first+size-1))
        soc_native.append(list(range(first,first+size)));start+=size
    assert len(native_g)==q.m
    rsum=[F(0)]*(q.p+q.m);csum=[F(0)]*q.n
    for r,entries in enumerate(a):
        if r in removed_equalities:continue
        assert all(j not in masses for j,v in entries)
        for j,v in entries:
            if j in active:rsum[r]+=abs(v);csum[j]+=abs(v)
    mass_row_set={r for group in mass_rows for r,v in group}
    for r,entries in enumerate(g):
        if r in removed_g or r in mass_row_set:continue
        for j,v in entries:
            if j in active:rsum[native_g[r]]+=abs(v);csum[j]+=abs(v)
    prefix=F(0)
    for i,node in enumerate(nodes):
        if i:prefix+=1+gamma[i]
        for r,sign in mass_rows[i]:rsum[native_g[r]]=prefix
        if i:
            csum[node[2]]+=3*(m-i)*gamma[i];csum[node[3]]+=3*(m-i)
    steps=final['mass']['steps_original_layout']
    assert len(steps)==q.n+q.p+q.m and all(math.isfinite(x) and x>0 for x in steps)
    theta=F(float(.95))
    alpha=beta=F(0);max_relative_slack=0.
    for j in range(q.n):
        if j not in active:assert steps[j]==1.;continue
        denominator=csum[j] or F(1);exact=theta/denominator
        assert F(steps[j])<=exact
        max_relative_slack=max(max_relative_slack,float((exact-F(steps[j]))/exact))
        beta=max(beta,F(steps[j])*csum[j])
    removed_native=removed_equalities|{native_g[r] for r in removed_g}
    denominators=list(rsum)
    for block in soc_native:
        maximum=max(rsum[r] for r in block)
        for r in block:denominators[r]=maximum
        assert len({bits([steps[q.n+r]]) for r in block})==1
    for r in range(q.p+q.m):
        if r in removed_native:assert steps[q.n+r]==1.;continue
        exact=theta/(denominators[r] or F(1))
        assert F(steps[q.n+r])<=exact
        max_relative_slack=max(max_relative_slack,float((exact-F(steps[q.n+r]))/exact))
        alpha=max(alpha,F(steps[q.n+r])*rsum[r])
    assert alpha*beta<=theta*theta<1
    assert F(final['mass']['row_factor_upper'])>=alpha
    assert F(final['mass']['column_factor_upper'])>=beta
    assert F(final['mass']['norm_squared_upper'])>=alpha*beta
    assert max_relative_slack<1e-12
    x,y,z=final['x'],final['y'],final['z']
    constants=scan(rhs)
    linear=scan([0.]+[float(x[n[3]])-float(gamma[i])*float(x[n[2]]) for i,n in enumerate(nodes[1:],1)])
    predicted_m=[d+s for d,s in zip(constants,linear)]
    w=[sum((float(sign)*float(z[r]) for r,sign in group),0.) for group in mass_rows]
    predicted_y=[-v for v in scan(w,True)]
    primal_bits=bits(predicted_m)==bits([x[n[0]] for n in nodes])
    dual_bits=bits(predicted_y)==bits([y[n[1]] for n in nodes])
    if not seeded:assert primal_bits and dual_bits
    return {'nodes':m,'active_variables':len(active),'active_rows':q.p+q.m-len(removed_native),
            'actual_alpha':float(alpha),'actual_beta':float(beta),'exact_product_upper':float(alpha*beta),
            'conservative_report_bound':final['mass']['norm_squared_upper'],
            'max_relative_step_reduction_from_exact_formula':max_relative_slack,
            'SOC_steps_bitwise_tied':True,'all_direct_steps_inward':True,
            'cold_primal_prefix_bits_match':primal_bits,'cold_equality_dual_suffix_bits_match':dual_bits,
            'seed_mapping_not_required':seeded,
            'max_original_mass_dual_stationarity':float(max(abs(dm.dec(w[i])+dm.dec(y[nodes[i][1]])-
                (dm.dec(y[nodes[i+1][1]]) if i+1<m else Decimal(0))) for i in range(m)))}

def main():
    path=BASE/'run/report.json';report=strict(path.read_text())
    assert sha(path)=='121fc8226808a3c640faed8413a1181bf3cb15f33560bd3f54dd08db9400a28b'
    assert report['complete'] and report['executions_started']==6 and report['solve_API_calls']==8
    assert report['execution_blocks']==128 and report['bootstrap_iterations']<=2
    assert report['core_sha256']=='a89b8b30267399c033a88f3fd8ec7fa8ac1bceb1bd41e79b9841807af1060cb2'
    assert report['runner_sha256']==sha(BASE/'run.py')=='f4bfb29c683eaa1ac394e5b8e52e1f2f59bea73a77be20daae4c8df896579f37'
    assert report['manifest_sha256']==sha(BASE/'manifest.json')=='ca970d7acd9893ba71d7e58f85393cb2a26c003ca931e2071e2718e4292ad026'
    for name,digest in report['inputs_sha256'].items():assert sha(BASE/'inputs'/name)==digest
    findings=[]
    with localcontext() as context:
        context.prec=65
        for row in report['cases']:
            log=BASE/'run'/(row['name']+'.log');assert sha(log)==row['log_sha256']
            raw=read_records(log);meta=raw['PERSISTENT_REPLAY_META'];final=raw['PERSISTENT_REPLAY']
            q=dm.Snapshot(BASE/'inputs'/(row['capture']+'.txt'))
            assert q.q_all_zero and not q.shift and all(v==0 for v in q.origin)
            assert meta['input_sha256']==report['inputs_sha256'][row['capture']+'.txt']
            assert meta['library_sha256']==report['core_sha256'] and meta['source_tree_sha256']==report['source_tree_sha256']
            assert bits(final['x'])==bits(final['x_solver'])
            audit=q.audit(dict(final,termination_code=final['termination']))
            assert audit['passes']==row['audit']['passes_common_kkt_gate']
            assert audit['qualified']==final['qualified_original']==row['audit']['qualified']
            if final['gpu_common_kkt']['valid']:assert audit['passes']==final['gpu_common_kkt']['passes']
            for k in ('primal','dual','gap','primal_cone_violation','dual_cone_violation','objective','dual_objective'):
                assert math.isclose(audit[k],row['audit'][k],rel_tol=1e-9,abs_tol=1e-14),(row['name'],k)
            assert math.isclose(audit['block_complementarity_normalized'],row['audit']['complementarity_max_relative'],rel_tol=1e-9,abs_tol=1e-14)
            pairs=[{'t':int(p[0]),'v':int(p[1]),'positive_row':int(p[2])-q.p,'negative_row':int(p[3])-q.p,'lambda':p[4]} for p in meta['l1_map']]
            diagnosis=old.diagnose(q,final,pairs,dm)
            intervals=len(pairs)//7
            components=('position_x','position_y','position_z','velocity_x','velocity_y','velocity_z','mass')
            for item in diagnosis['equality_top']:
                index=item['index']
                if index<7*intervals:
                    item.update(block='dynamics',interval=index//7,component=components[index%7])
                elif index<7*intervals+7:
                    item.update(block='initial_state',component=components[index-7*intervals])
                elif index<7*intervals+13:
                    item.update(block='terminal_state',component=components[index-7*intervals-7])
                else:item.update(block='terminal_control')
            iu=7*(intervals+1)
            for item in diagnosis['stationarity_top']:
                index=item['index']
                if index<iu:item.update(block='state',node=index//7,component=components[index%7])
                elif index<iu+4*(intervals+1):
                    item.update(block='control',node=(index-iu)//4,component=('u_x','u_y','u_z','Gamma')[(index-iu)%4])
            seeded=row['seeded'];seed=None
            if seeded:
                initial=raw['PERSISTENT_REPLAY_INITIAL_POINT']
                lines=(BASE/'inputs'/(row['capture']+'-initial.txt')).read_text().splitlines()
                point={line.split()[0]:list(map(float,line.split()[2:])) for line in lines[3:]}
                assert all(bits(point[k])==bits(initial[k])==bits(final[k]) for k in ('x','y','z'))
                assert audit['qualified'] and final['iterations']==0
                seed={'xyz_bits_preserved':True,'bootstrap_updates':raw['PERSISTENT_REPLAY_BOOTSTRAP']['iterations']}
            else:
                assert final['iterations']<=100000
                assert diagnosis['completion']['exact_abs_t'] and diagnosis['completion']['strict_nonzero_endpoint']
            metric=metric_and_map(q,meta,final,seeded) if row['mode']=='mass' else None
            findings.append({'name':row['name'],'capture':row['capture'],'mode':row['mode'],'seeded':seeded,
                'log_sha256':sha(log),'iterations':final['iterations'],'termination':final['termination'],
                'solve_seconds':final['solve_seconds'],'scaling_seconds':final['scaling_seconds'],
                'wall_seconds':final['wall_seconds'],'outer_process_seconds':row['wall_seconds'],
                'deadline_requested':final['deadline_requested'],'audit':audit,'seed':seed,'diagnosis':diagnosis,'metric_and_map':metric})
    result={'scope':'Six saved original-coordinate vector sets: Decimal65 KKT, exact represented-coefficient actual-diagonal certificate, FP64 prefix/suffix export check. No reviewer GPU calls.',
            'report_sha256':sha(path),'core_sha256':report['core_sha256'],'source_tree_sha256':report['source_tree_sha256'],
            'row_column_labels_source':{'path':'cpp/cuda/src/gtoc12_conic.cu','sha256':'3e04dac39a852c44dbe4c7347f4f0157430c719d155cb74c004f9250296555ac','dynamics_lines':'153-168','boundary_and_cone_lines':'170-205'},
            'raw_inputs':report['inputs_sha256'],'all_original_gate_decisions_agree':True,
            'work':{'executions':6,'solve_APIs':8,'bootstrap_updates':report['bootstrap_iterations'],
                    'cold_updates':sum(c['iterations'] for c in findings if not c['seeded']),
                    'qualified_seeds':sum(c['audit']['qualified'] for c in findings if c['seeded']),
                    'qualified_cold':sum(c['audit']['qualified'] for c in findings if not c['seeded'])},'cases':findings}
    (OUT/'findings.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'report_sha256':sha(OUT/'findings.json'),'work':result['work']}))
if __name__=='__main__':main()
