"""CPU-only Decimal65 audit and reduced-scaling reconstruction for saved v615.

Run from the repository root. No imports of CUDA or numerical solver packages.
The projected known reference used for distance diagnostics is not asserted to
be an exact reduced nonsmooth certificate, nor used to select a solver weight.
"""
import argparse
from decimal import localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile
import types

BASE = Path('build/performance/l1-real-v615')
FROZEN = Path('build/performance/l1-v615c')
MAP = Path('build/performance/l1-prox-review-v615/findings.json')
SCALE = Path('build/performance/l1-scaling-v615b/findings.json')
BALANCE = Path('build/performance/l1-balance-v615/findings.json')
EXCLUDE = {'x', 'x_solver', 'y', 'z', 's'}
INPUTS = {
    'conditioning.txt': '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf',
    'difficult.txt': '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080',
    'conditioning-initial.txt': '9caf303469dcd254d56b74f8bf3818c8044f51a3a20bc3c07835b76dd23c5632',
    'difficult-initial.txt': 'b92be6c5a0051ab99c000c459ef466c69fc8b2fd669c38b3580aea000d926da6',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(values):
    return b''.join(struct.pack('<d', float(v)) for v in values)


def records(data):
    def invalid(value):
        raise ValueError(value)
    out = {}
    for line in data.decode().splitlines():
        if line.startswith('PERSISTENT_REPLAY'):
            key, text = line.split(' ', 1)
            assert key not in out
            out[key] = json.loads(text, parse_int=float, parse_constant=invalid)
    return out


def point(capture):
    lines = (BASE/'inputs'/f'{capture}-initial.txt').read_text().splitlines()
    assert len(lines) == 7 and lines[1] == 'snapshot_sha256 '+INPUTS[capture+'.txt']
    assert lines[2] == 'coordinates translated'
    result = {}
    for line in lines[3:]:
        key, count, *values = line.split()
        assert int(count) == len(values)
        result[key] = list(map(float, values))
    return result


def reduced_scaling(snap, pairs, expected, reference):
    """Independent standard-library ten-pass Ruiz and power20 computation."""
    ts = {p['t'] for p in pairs}
    removed = {p[k] for p in pairs for k in ('positive_row', 'negative_row')}
    columns = [j for j in range(snap.n) if j not in ts]
    grow = [r for r in range(snap.m) if r not in removed]
    cmap = {j:k for k,j in enumerate(columns)}
    rmap = {r:snap.p+k for k,r in enumerate(grow)}
    rows = snap.p + len(grow)
    entries = []
    for j in columns:
        for matrix in (1, 2):
            for k in range(snap.ptr[matrix][j], snap.ptr[matrix][j+1]):
                r, a = snap.idx[matrix][k], float(snap.val[matrix][k])
                if a == 0 or (matrix == 2 and r in removed):
                    continue
                entries.append((cmap[j], r if matrix == 1 else rmap[r], a))
    d, r = [1.0]*len(columns), [1.0]*rows
    for _ in range(10):
        cm, rm = [0.0]*len(d), [0.0]*len(r)
        for j,i,a in entries:
            value = abs(a)/(r[i]*d[j])
            cm[j] = max(cm[j],value)
            rm[i] = max(rm[i],value)
        at = snap.p + snap.l - len(removed)
        for size in snap.soc:
            maximum = max(rm[at:at+size])
            rm[at:at+size] = [maximum]*size
            at += size
        assert at == rows
        d = [v*(math.sqrt(a) if a > 1e-12 else 1.0) for v,a in zip(d,cm)]
        r = [v*(math.sqrt(a) if a > 1e-12 else 1.0) for v,a in zip(r,rm)]
    assert sha(bits(d)) == expected['D_sha256']
    assert sha(bits(r)) == expected['R_sha256']
    rhs = [float(v) for v in snap.b] + [float(snap.h[i]) for i in grow]
    b = 1/(1+math.sqrt(math.fsum((v/s)**2 for v,s in zip(rhs,r))))
    smooth_sq = math.fsum((float(snap.c[j])/d[k])**2 for k,j in enumerate(columns))
    penalty_sq = math.fsum((p['lambda_value']/d[cmap[p['v']]])**2 for p in pairs)
    o = 1/(1+math.sqrt(smooth_sq+penalty_sq))
    assert b == expected['B'] and o == expected['O']
    scaled = [(j,i,a/(r[i]*d[j])) for j,i,a in entries]
    v = [1/math.sqrt(len(d))]*len(d)
    for _ in range(20):
        row_terms = [[] for _ in r]
        for j,i,a in scaled:
            row_terms[i].append(a*v[j])
        product = [math.fsum(x) for x in row_terms]
        norm = math.sqrt(math.fsum(x*x for x in product))
        product = [x/norm for x in product]
        col_terms = [[] for _ in d]
        for j,i,a in scaled:
            col_terms[j].append(a*product[i])
        transpose = [math.fsum(x) for x in col_terms]
        estimate = math.sqrt(math.fsum(x*x for x in transpose))
        v = [x/estimate for x in transpose]
    eta = .9/max(1,estimate)
    assert math.isclose(eta, expected['eta'], rel_tol=1e-13)
    thresholds = [eta*o/(b*d[cmap[p['v']]]**2)*p['lambda_value'] for p in pairs]
    reference_y = reference['y'] + [reference['z'][i] for i in grow]
    x_norm = math.sqrt(math.fsum((b*s*reference['x'][j])**2 for j,s in zip(columns,d)))
    y_norm = math.sqrt(math.fsum((o*s*y)**2 for y,s in zip(reference_y,r)))
    return dict(columns=columns,grow=grow,D=d,R=r,B=b,O=o,eta=eta,
                minimum_threshold=min(thresholds),maximum_threshold=max(thresholds),
                reference_primal_norm=x_norm,reference_dual_norm=y_norm,
                projected_reference_weight_proxy=y_norm/x_norm,
                D_sha256=sha(bits(d)),R_sha256=sha(bits(r))), len(entries)


def decomposition(snap, final, pairs, scale, reference, dm):
    x,y,z,s = [[dm.dec(v) for v in final[k]] for k in ('x','y','z','s')]
    px,ax,gx = snap.mul(0,x),snap.mul(1,x),snap.mul(2,x)
    aty,gtz = snap.mul(1,y,True),snap.mul(2,z,True)
    rd = [q+c+a+g for q,c,a,g in zip(px,snap.c,aty,gtz)]
    eq = [a-b for a,b in zip(ax,snap.b)]
    ce = [h-g-a for h,g,a in zip(snap.h,gx,s)]
    f = dm.dot(x,px)/2 + dm.dot(snap.c,x)
    dual = -dm.dot(x,px)/2 - dm.dot(snap.b,y) - dm.dot(snap.h,z)
    denominator = max(dm.dec(1),abs(f),abs(dual))
    terms = [dm.dot(x,rd),-dm.dot(eq,y),dm.dot(s,z),dm.dot(ce,z)]
    error = f-dual-sum(terms,dm.dec(0))
    assert abs(error) < dm.dec(1e-45)*max(dm.dec(1),*(abs(v) for v in terms))
    ts,vs = {p['t'] for p in pairs},{p['v'] for p in pairs}
    removed = {p[k] for p in pairs for k in ('positive_row','negative_row')}
    stats = {'positive_v':0,'negative_v':0,'exact_zero_v':0,'noncanonical_t':0,
             'nonzero_not_endpoint_pairs':0}
    zero_rd,nonzero_rd,t_rd,pair_comp = [],[],[],[]
    for p in pairs:
        v,t,positive,negative,lam = p['v'],p['t'],p['positive_row'],p['negative_row'],dm.dec(p['lambda_value'])
        sign = 1 if x[v]>0 else -1 if x[v]<0 else 0
        stats['positive_v' if sign>0 else 'negative_v' if sign<0 else 'exact_zero_v'] += 1
        stats['noncanonical_t'] += int(x[t] != abs(x[v]))
        if sign:
            stats['nonzero_not_endpoint_pairs'] += int((z[positive],z[negative]) != ((lam,dm.dec(0)) if sign>0 else (dm.dec(0),lam)))
            nonzero_rd.append(rd[v])
        else:
            zero_rd.append(rd[v])
        t_rd.append(rd[t])
        pair_comp.extend((s[positive]*z[positive],s[negative]*z[negative]))
    stats.update(zero_v_max_stationarity=float(dm.inf(zero_rd)),nonzero_v_max_stationarity=float(dm.inf(nonzero_rd)),
                 epigraph_t_max_stationarity=float(dm.inf(t_rd)),pair_complementarity_max_absolute=float(dm.inf(pair_comp)))
    soc_blocks = []
    max_soc_comp = dm.dec(0)
    at = snap.l
    for index,size in enumerate(snap.soc):
        primal=max(dm.dec(0),dm.dot(s[at+1:at+size],s[at+1:at+size]).sqrt()-s[at])
        comp=dm.dot(s[at:at+size],z[at:at+size])
        max_soc_comp=max(max_soc_comp,abs(comp))
        soc_blocks.append(dict(index=index,row=at,length=size,primal_cone_violation=float(primal),complementarity=float(comp)))
        at+=size
    max_block=max(dm.inf([a*b for a,b in zip(s[:snap.l],z[:snap.l])]),max_soc_comp)
    groups = {'epigraph_t':ts,'absolute_target_v':vs,'other_retained':set(range(snap.n))-ts-vs}
    rows_with_v = {snap.idx[1][k] for j in vs for k in range(snap.ptr[1][j],snap.ptr[1][j+1]) if snap.val[1][k] != 0}
    columns,grow,d,r,b,o = (scale[k] for k in ('columns','grow','D','R','B','O'))
    final_y = final['y'] + [final['z'][i] for i in grow]
    reference_y = reference['y'] + [reference['z'][i] for i in grow]
    xerror=math.sqrt(math.fsum((b*s*(final['x'][j]-reference['x'][j]))**2 for j,s in zip(columns,d)))
    yerror=math.sqrt(math.fsum((o*s*(a-z))**2 for a,z,s in zip(final_y,reference_y,r)))
    return dict(signed_gap=float(f-dual),objective_denominator=float(denominator),
                terms=dict(zip(('x_dot_stationarity','minus_equality_error_dot_y','s_dot_z','slack_reconstruction_error_dot_z'),map(float,terms))),
                identity_error=float(error),equality_absolute=float(dm.inf(eq)),stationarity_absolute=float(dm.inf(rd)),
                equality_rows_with_virtual_control=len(rows_with_v),
                equality_virtual_control_rows_max=float(dm.inf([eq[i] for i in rows_with_v])),
                equality_other_rows_max=float(dm.inf([eq[i] for i in range(snap.p) if i not in rows_with_v])),
                group_max_stationarity={k:float(dm.inf([rd[j] for j in indices])) for k,indices in groups.items()},
                pair_completion=stats,max_block_complementarity_absolute=float(max_block),
                top_equality=[dict(row=i,residual=float(eq[i]),dual=float(y[i]),signed_gap_term=float(-eq[i]*y[i]),has_virtual_control=i in rows_with_v)
                              for i in sorted(range(snap.p),key=lambda i:abs(eq[i]),reverse=True)[:8]],
                top_stationarity=[dict(variable=j,group=next(k for k,v in groups.items() if j in v),x=float(x[j]),residual=float(rd[j]),c=float(snap.c[j]),Aeq_y=float(aty[j]),G_z=float(gtz[j]))
                                  for j in sorted(range(snap.n),key=lambda j:abs(rd[j]),reverse=True)[:8]],
                top_soc_primal=sorted(soc_blocks,key=lambda b:b['primal_cone_violation'],reverse=True)[:4],
                top_soc_complementarity=sorted(soc_blocks,key=lambda b:abs(b['complementarity']),reverse=True)[:4],
                distance_to_projected_reference=dict(scaled_primal_error=xerror,scaled_retained_dual_error=yerror,
                    primal_error_over_zero_start= xerror/scale['reference_primal_norm'],dual_error_over_zero_start=yerror/scale['reference_dual_norm'],
                    caveat='Projected approximate reference is not an exact reduced KKT certificate; distance is diagnostic only.'))


def audit(dm):
    report=json.loads((BASE/'run/report.json').read_bytes())
    manifest=json.loads((BASE/'manifest.json').read_bytes())
    assert sha((BASE/'run/report.json').read_bytes())=='baf58f2219dd1764a68b9be36171eab66aa54ed8ca1479bbac4e1311d5089e45'
    assert report['complete'] and report['executions_started']==len(report['cases'])==6
    assert report['solve_API_calls']==8 and report['bootstrap_iterations']==2 and report['execution_blocks']==128
    assert report['compute_processes_before']=='' and 'failure' not in report
    assert report['runner_sha256']==sha((BASE/'run.py').read_bytes())=='edc3592ffab5dc29f0589d9de0572cee01b2ac9b06341052c6ab6e7fe3b940cd'
    assert report['manifest_sha256']==sha((BASE/'manifest.json').read_bytes())==sha((FROZEN/'manifest.json').read_bytes())
    assert report['binary_sha256']==manifest['persistent_snapshot_replay_sha256']
    assert report['core_sha256']==manifest['library_sha256']=='83487574646fe67fce156c9a2055f448341c99f189ac2c66d856bfcf1b280047'
    assert manifest['frozen_commit']=='32efbed13eae47d2c8352771e884a2ad5ff5d07e'
    for name,expected in INPUTS.items():
        assert sha((BASE/'inputs'/name).read_bytes())==expected==report['inputs_sha256'][name]
    with tarfile.open(FROZEN/'source.tar.gz') as archive:
        for name in manifest['owned_paths']:
            assert sha(archive.extractfile(name).read())==manifest['source_sha256'][name]
        parts=['tests/persistent_snapshot.hpp','tests/persistent_snapshot_replay.cu','tests/cuda_test_support.hpp',
               'tests/persistent_l1_snapshot.hpp','include/spacepdhcg/cuda/persistent_pdhcg_c_api.h']
        compiled_identity=''.join(p+':'+sha(archive.extractfile('cpp/cuda/'+p).read())+'\n' for p in parts)
        assert sha(compiled_identity.encode())==manifest['compiled_snapshot_source_sha256']
    maps=json.loads(MAP.read_bytes())['captures']
    expected_scaling={c['capture']:c for c in json.loads(SCALE.read_bytes())['cases']}
    balance=json.loads(BALANCE.read_bytes())
    snaps={c:dm.Snapshot(BASE/'inputs'/f'{c}.txt') for c in maps}
    scales={}
    for c,snap in snaps.items():
        assert snap.q_all_zero and snap.shift==0 and snap.offset==0
        scale,nnz=reduced_scaling(snap,maps[c]['detection_map'],expected_scaling[c],point(c))
        assert nnz==expected_scaling[c]['reduced_numerical_entries']
        scales[c]=scale
        expected_weight=next(x for x in balance['cases'] if x['capture']==c)['policies'][0]['oracle_weight']
        assert math.isclose(scale['projected_reference_weight_proxy'],expected_weight,rel_tol=1e-13)
    rows=[]
    for case in report['cases']:
        name,capture,mode,seeded=(case[k] for k in ('name','capture','mode','seeded'))
        data=(BASE/'run'/f'{name}.log').read_bytes()
        assert sha(data)==case['log_sha256'] and case['returncode']==0
        rec=records(data);meta,final=rec['PERSISTENT_REPLAY_META'],rec['PERSISTENT_REPLAY']
        assert {k:v for k,v in final.items() if k not in EXCLUDE}==case['final']
        assert meta['input_sha256']==INPUTS[capture+'.txt'] and meta['library_sha256']==report['core_sha256']
        assert meta['source_commit']==manifest['frozen_commit'] and meta['source_sha256']==manifest['compiled_snapshot_source_sha256']
        assert meta['requested_execution_blocks']==final['execution_blocks']==128
        assert meta['halpern_mode']=='off' and meta['l1_prox']==(mode=='l1') and meta['repeat_mode']=='cold'
        assert not meta['fold_singleton_bounds'] and not meta['shifted']
        assert meta['stopping_policy']=='gpu_common_kkt_original_equations' and meta['audit_tolerance']==1e-9 and meta['cone_tolerance']==1e-8
        assert final['fresh_workspace'] and final['recovery_iterations']==final['recovery_seconds']==0
        assert not final['deadline_requested'] and final['within_requested_wall_deadline']
        assert bits(final['x'])==bits(final['x_solver'])
        snap=snaps[capture]
        exact=snap.audit(dict(final,termination_code=final['termination']))
        gpu=final['gpu_common_kkt']
        assert gpu['enabled'] and gpu['valid'] and gpu['finite']
        assert exact['passes']==final['kkt_qualified_original']==case['audit']['passes_common_kkt_gate']==gpu['passes']
        assert exact['qualified']==final['qualified_original']==case['audit']['qualified']==seeded
        seed=None
        if seeded:
            assert final['termination']==1 and final['iterations']==0 and case['solve_API_calls']==2
            pre,bootstrap,initial=(rec['PERSISTENT_REPLAY_'+k] for k in ('PRESTEP','BOOTSTRAP','INITIAL_POINT'))
            assert case['pre']==[pre] and case['bootstrap']==[bootstrap]
            assert bootstrap['iterations']==bootstrap['iteration_limit']==1 and bootstrap['recovery_iterations']==0
            assert pre['seeded_iterations']==pre['termination']==0 and pre['solve_epoch']==1
            assert pre['native_seed_verified_unchanged'] and not pre['inherited_report_counters_are_seed_work']
            assert initial['point_sha256']==meta['initial_point_sha256']==INPUTS[capture+'-initial.txt']
            encoded=point(capture)
            for key in ('x','y','z'):
                assert bits(encoded[key])==bits(initial[key])==bits(final[key])
            assert bits(encoded['s'])==bits(initial['s'])
            expected_dual=initial['y']+initial['z'][:snap.l]
            at=snap.l
            for size in snap.soc:
                z=initial['z'][at:at+size];expected_dual += [-v for v in z[1:]]+[-z[0]];at+=size
            assert bits(expected_dual)==bits(initial['dual_solver'])
            assert bits(initial['x_solver'])==bits(initial['x'])
            initial_exact=snap.audit(dict(initial,termination_code=1));assert initial_exact['qualified']
            seed=dict(encoded_initial_final_xyz_bits_equal=True,native_seed_order_bits_correct=True,
                      initial_decimal65=initial_exact,bootstrap_iterations=1)
        else:
            assert final['termination']==2 and final['iterations']==final['iteration_limit']==100000
            assert case['solve_API_calls']==1 and not case['bootstrap'] and not case['pre']
            assert not any('PERSISTENT_REPLAY_'+k in rec for k in ('PRESTEP','BOOTSTRAP','INITIAL_POINT'))
        l1=final.get('l1')
        scale=scales[capture]
        if mode=='l1':
            assert l1['enabled'] and l1['valid'] and l1['finite'] and l1['updates']==l1['completions']==final['iterations']
            assert l1['pairs']==len(maps[capture]['detection_map'])
            for actual,expected in [('bound_scale','B'),('objective_scale','O'),('eta','eta'),('minimum_threshold','minimum_threshold'),('maximum_threshold','maximum_threshold')]:
                assert math.isclose(l1[actual],scale[expected],rel_tol=1e-13),(name,actual,l1[actual],scale[expected])
        detail=decomposition(snap,final,maps[capture]['detection_map'],scale,point(capture),dm)
        if mode=='l1' and not seeded:
            assert detail['pair_completion']['noncanonical_t']==detail['pair_completion']['nonzero_not_endpoint_pairs']==0
            assert detail['pair_completion']['pair_complementarity_max_absolute']==0
        rows.append(dict(name=name,capture=capture,mode=mode,seeded=seeded,log_sha256=sha(data),
                         iterations=final['iterations'],termination=final['termination'],decimal65=exact,
                         residual_decomposition=detail,seed=seed,l1=l1,gpu_solve_seconds=final['solve_seconds'],
                         gpu_scaling_seconds=final['scaling_seconds'],process_wall_seconds=case['wall_seconds']))
    comparisons={}
    for capture in snaps:
        off=next(r for r in rows if r['capture']==capture and r['mode']=='off')
        on=next(r for r in rows if r['capture']==capture and r['mode']=='l1' and not r['seeded'])
        comparisons[capture]=dict(metric_ratios_l1_over_off={k:on['decimal65'][k]/off['decimal65'][k] for k in
            ('primal','dual','gap','block_complementarity_normalized','primal_cone_violation')},
            observed_fixed_100k_solve_time_ratio=on['gpu_solve_seconds']/off['gpu_solve_seconds'],qualified_cold_solutions=0,
            interpretation='Unqualified fixed-work endpoints; timing is not time to the same verified accuracy.')
    return dict(report_sha256=sha((BASE/'run/report.json').read_bytes()),manifest_sha256=report['manifest_sha256'],
                source_commit=manifest['frozen_commit'],core_sha256=report['core_sha256'],binary_sha256=report['binary_sha256'],
                runner_sha256=report['runner_sha256'],input_sha256=INPUTS,map_findings_sha256=sha(MAP.read_bytes()),
                prior_scaling_sha256=sha(SCALE.read_bytes()),prior_balance_sha256=sha(BALANCE.read_bytes()),
                gpu_identity=report['gpu_identity'].strip(),execution_blocks=128,
                executions=6,solve_api_calls=8,bootstrap_iterations=2,cold_updates=400000,seeded_updates=0,actual_total_updates=400002,
                qualified_seeds=2,qualified_cold=0,all_identities_and_verdicts_match=True,
                reconstructed_scaling={c:{k:v for k,v in s.items() if k not in ('columns','grow','D','R')} for c,s in scales.items()},
                rows=rows,comparisons=comparisons)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).with_name('decimal_audit_source.py')
    assert sha(source.read_bytes())=='b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4'
    dm=types.ModuleType('recorded_decimal_source');dm.__file__=str(source)
    exec(compile(source.read_bytes(),str(source),'exec'),dm.__dict__)
    with localcontext() as context:
        context.prec=65
        out=dict(scope='CPU-only exact-FP64 original-equation audit and independent FP64 reduced scaling; no GPU calls',
                 decimal_precision=65,script_sha256=sha(Path(__file__).read_bytes()),decimal_auditor_sha256=sha(source.read_bytes()),findings=audit(dm))
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output.read_bytes())}))


if __name__=='__main__':
    main()
