#!/usr/bin/env python3
"""CPU-only Decimal65 audit of the ten saved v612 real replay executions.

Run from repository root. No CUDA, solver calls, bytecode imports or changes
to original captures, raw logs, source code or acceptance gates.
"""
import argparse
from decimal import localcontext
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import types

BASE = Path('build/performance/halpern-real-v612')
FROZEN = Path('build/performance/halpern-v612d')
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
    result = {}
    for line in data.decode().splitlines():
        if line.startswith('PERSISTENT_REPLAY'):
            key, text = line.split(' ', 1)
            assert key not in result
            result[key] = json.loads(text)
    return result


def decomposition(snap, result, math):
    x, y, z, s = [[math.dec(v) for v in result[k]] for k in ('x', 'y', 'z', 's')]
    px, ax, gx = snap.mul(0, x), snap.mul(1, x), snap.mul(2, x)
    aty, gtz = snap.mul(1, y, True), snap.mul(2, z, True)
    rd = [q+c+a+g for q,c,a,g in zip(px,snap.c,aty,gtz)]
    eq = [a-b for a,b in zip(ax,snap.b)]
    ce = [h-g-a for h,g,a in zip(snap.h,gx,s)]
    f = math.dot(x,px)/2 + math.dot(snap.c,x)
    d = -math.dot(x,px)/2 - math.dot(snap.b,y) - math.dot(snap.h,z)
    scale = max(math.dec(1), abs(f), abs(d))
    terms = [math.dot(x,rd),-math.dot(eq,y),math.dot(s,z),math.dot(ce,z)]
    error = f-d-sum(terms,math.dec(0))
    assert abs(error) < math.dec(1e-45)*max(math.dec(1),*(abs(v) for v in terms))
    worst = sorted(range(snap.l), key=lambda i: abs(s[i]*z[i]), reverse=True)[:5]
    block = math.inf([a*b for a,b in zip(s[:snap.l],z[:snap.l])])
    at = snap.l
    for size in snap.soc:
        block = max(block,abs(math.dot(s[at:at+size],z[at:at+size])))
        at += size
    return dict(signed_gap=float(f-d),objective_normalization_scale=float(scale),
                max_block_complementarity_absolute=float(block),
                terms=dict(zip(('x_dot_stationarity','minus_equality_error_dot_y','s_dot_z','slack_reconstruction_error_dot_z'),map(float,terms))),
                identity_error=float(error),equality_absolute=float(math.inf(eq)),stationarity_absolute=float(math.inf(rd)),
                top_scalar_complementarity=[dict(row=i,slack=float(s[i]),z=float(z[i]),product=float(s[i]*z[i]),normalized=float(abs(s[i]*z[i])/scale)) for i in worst],
                top_stationarity=[dict(variable=j,residual=float(rd[j]),c=float(snap.c[j]),x=float(x[j]),Aeq_transpose_y=float(aty[j]),G_transpose_z=float(gtz[j]))
                                  for j in sorted(range(snap.n),key=lambda j:abs(rd[j]),reverse=True)[:5]])


def audit(math):
    report = json.loads((BASE/'run/report.json').read_bytes())
    manifest = json.loads((BASE/'manifest.json').read_bytes())
    assert report['complete'] and report['executions_started'] == len(report['cases']) == 10
    assert report['solve_API_calls'] == 14 and report['bootstrap_iterations'] == 4
    assert report['execution_blocks'] == 128 and report['compute_processes_before'] == ''
    assert report['runner_sha256'] == sha((BASE/'run.py').read_bytes())
    assert report['manifest_sha256'] == sha((BASE/'manifest.json').read_bytes()) == sha((FROZEN/'manifest.json').read_bytes())
    assert report['binary_sha256'] == manifest['persistent_snapshot_replay_sha256']
    assert report['core_sha256'] == manifest['library_sha256']
    assert manifest['frozen_commit'] == '6fbff324b152b75e6c27e3d91b4dc0835c65ab93'
    for name, expected in INPUTS.items():
        assert sha((BASE/'inputs'/name).read_bytes()) == expected == report['inputs_sha256'][name]
    with tarfile.open(FROZEN/'source.tar.gz') as archive:
        for path in manifest['owned_paths']:
            assert sha(archive.extractfile(path).read()) == manifest['source_sha256'][path]
    snaps = {k:math.Snapshot(BASE/'inputs'/f'{k}.txt') for k in ('conditioning','difficult')}
    assert all(s.q_all_zero and s.shift == 0 and s.offset == 0 for s in snaps.values())
    rows = []
    for case in report['cases']:
        name, capture, mode, seeded = (case[k] for k in ('name','capture','mode','seeded'))
        data = (BASE/'run'/f'{name}.log').read_bytes()
        assert sha(data) == case['log_sha256'] and case['returncode'] == 0
        rec = records(data)
        meta, final = rec['PERSISTENT_REPLAY_META'], rec['PERSISTENT_REPLAY']
        assert {k:v for k,v in final.items() if k not in EXCLUDE} == case['final']
        assert meta['input_sha256'] == INPUTS[f'{capture}.txt']
        assert meta['source_commit'] == manifest['frozen_commit']
        assert meta['source_sha256'] == manifest['compiled_snapshot_source_sha256']
        assert meta['library_sha256'] == report['core_sha256']
        assert meta['requested_execution_blocks'] == final['execution_blocks'] == 128
        assert meta['halpern_mode'] == mode and meta['repeat_mode'] == 'cold'
        assert not meta['fold_singleton_bounds'] and not meta['shifted']
        assert meta['stopping_policy'] == 'gpu_common_kkt_original_equations'
        assert meta['audit_tolerance'] == 1e-9 and meta['cone_tolerance'] == 1e-8
        assert final['requested_tolerance'] == 1e-9 and final['fresh_workspace']
        assert final['recovery_iterations'] == final['recovery_seconds'] == 0
        assert not final['deadline_requested'] and final['within_requested_wall_deadline']
        assert rec['PERSISTENT_REPLAY_SUMMARY']['repeats'] == 1
        assert bits(final['x']) == bits(final['x_solver'])
        snap = snaps[capture]
        exact = snap.audit(dict(final,termination_code=final['termination']))
        gpu = final['gpu_common_kkt']
        assert gpu['enabled'] and gpu['valid'] and gpu['finite']
        assert exact['passes'] == final['kkt_qualified_original'] == case['audit']['passes_common_kkt_gate'] == gpu['passes']
        assert exact['qualified'] == final['qualified_original'] == case['audit']['qualified'] == seeded
        seed = None
        if seeded:
            assert final['termination'] == 1 and final['iterations'] == 0
            pre, bootstrap, initial = (rec['PERSISTENT_REPLAY_'+k] for k in ('PRESTEP','BOOTSTRAP','INITIAL_POINT'))
            assert case['pre'] == [pre] and case['bootstrap'] == [bootstrap] and case['solve_API_calls'] == 2
            assert bootstrap['iterations'] == bootstrap['iteration_limit'] == 1 and bootstrap['recovery_iterations'] == 0
            assert pre['seeded_iterations'] == pre['termination'] == 0 and pre['solve_epoch'] == 1
            assert pre['native_seed_verified_unchanged'] and not pre['inherited_report_counters_are_seed_work']
            assert initial['point_sha256'] == meta['initial_point_sha256'] == INPUTS[f'{capture}-initial.txt']
            lines = (BASE/'inputs'/f'{capture}-initial.txt').read_text().splitlines()
            assert len(lines) == 7 and lines[1] == 'snapshot_sha256 '+INPUTS[f'{capture}.txt']
            assert lines[2] == 'coordinates translated'
            encoded = {}
            for line in lines[3:]:
                key,count,*values = line.split();assert int(count) == len(values);encoded[key]=list(map(float,values))
            for key in ('x','y','z'):
                assert bits(encoded[key]) == bits(initial[key]) == bits(final[key])
            expected_dual = initial['y'] + initial['z'][:snap.l]
            at = snap.l
            for size in snap.soc:
                z = initial['z'][at:at+size];expected_dual += [-v for v in z[1:]] + [-z[0]];at += size
            assert bits(expected_dual) == bits(initial['dual_solver'])
            assert bits(initial['x']) == bits(initial['x_solver'])
            initial_exact = snap.audit(dict(initial,termination_code=1))
            assert initial_exact['qualified']
            seed = dict(encoded_initial_and_final_xyz_bits_equal=True,native_seed_order_bits_correct=True,
                        x_entries=snap.n,y_entries=snap.p,z_entries=snap.m,bootstrap_iterations=1,
                        native_prestep_natural_residual=pre['native_natural_residual'],initial_decimal65=initial_exact)
        else:
            assert final['termination'] == 2 and final['iterations'] == final['iteration_limit'] == 100000
            assert case['solve_API_calls'] == 1 and case['bootstrap'] == case['pre'] == []
            assert not any('PERSISTENT_REPLAY_'+key in rec for key in ('PRESTEP','BOOTSTRAP','INITIAL_POINT'))
        h = final.get('halpern')
        if mode != 'off':
            assert h['valid'] and h['finite'] and h['updates'] == final['iterations']
            assert h['mode'] == (1 if mode == 'plain' else 2)
            assert h['weight_updates'] + h['weight_fallbacks'] == h['restarts']
            if mode == 'plain':
                assert h['restarts'] == h['weight_updates'] == 0 and h['primal_weight'] == 1
                assert h['inner_iterations'] == final['iterations']
            elif not seeded:
                assert h['last_restart_iteration'] % 200 == 0
                assert h['inner_iterations'] == final['iterations'] - h['last_restart_iteration']
                assert h['epoch_reference_iteration'] == h['last_restart_iteration'] + 1
                assert h['metric_evaluations'] == 500 + h['restarts']
        rows.append(dict(name=name,capture=capture,mode=mode,seeded=seeded,log_sha256=sha(data),
                         iterations=final['iterations'],termination=final['termination'],decimal65=exact,
                         gap_decomposition=decomposition(snap,final,math),seed=seed,halpern=h,
                         gpu_solve_seconds=final['solve_seconds'],gpu_scaling_seconds=final['scaling_seconds'],
                         process_wall_seconds=case['wall_seconds']))
    comparisons = {}
    for capture in snaps:
        cold = {r['mode']:r for r in rows if r['capture'] == capture and not r['seeded']}
        comparisons[capture] = {}
        for mode in ('plain','adaptive'):
            a,b = cold[mode],cold['off']
            comparisons[capture][mode] = dict(
                metric_ratios_to_off={k:a['decimal65'][k]/b['decimal65'][k] for k in
                    ('primal','dual','gap','block_complementarity_normalized','primal_cone_violation')},
                observed_100k_update_gpu_solve_time_ratio=a['gpu_solve_seconds']/b['gpu_solve_seconds'],
                qualified_solutions=0,interpretation='fixed-work unqualified endpoints, not time to equal accuracy')
    return dict(report_sha256=sha((BASE/'run/report.json').read_bytes()),manifest_sha256=report['manifest_sha256'],
                source_commit=manifest['frozen_commit'],core_sha256=report['core_sha256'],binary_sha256=report['binary_sha256'],
                runner_sha256=report['runner_sha256'],input_sha256=INPUTS,all_identities_and_verdicts_match=True,
                executions=10,solve_api_calls=14,bootstrap_iterations=4,seeded_updates=0,cold_updates=600000,
                actual_total_updates=600004,qualified_seeded_cases=4,qualified_cold_cases=0,
                rows=rows,comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name('decimal_audit_source.py')
    assert sha(source.read_bytes()) == 'b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4'
    math = types.ModuleType('source_compiled_decimal_auditor')
    math.__file__ = str(source)
    exec(compile(source.read_bytes(),str(source),'exec'),math.__dict__)
    with localcontext() as context:
        context.prec = 65
        out = dict(scope='CPU-only saved exact-FP64 original-equation audit; no solver/GPU calls',
                   decimal_precision=65,script_sha256=sha(Path(__file__).read_bytes()),
                   decimal_auditor_sha256=sha(source.read_bytes()),
                   auditor_loading='direct compile of recorded source bytes; no inherited Python bytecode',
                   findings=audit(math))
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
