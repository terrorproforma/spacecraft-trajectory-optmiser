"""Independent Decimal65 original-equation audit of the six saved v618 runs.

No solver/GPU calls, no numerical-package dependency, no weight tuning. Directly
compiles the pinned Decimal source, avoiding inherited Python bytecode.
"""
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile
import types

BASE = Path('build/performance/l1-weight-real-v618')
POLICY = Path('build/performance/l1-weight-review-v618/policy-findings.json')
MAP = Path('build/performance/l1-scaling-v615b/findings.json')
SOURCE_ARCHIVE = Path('build/performance/l1-weight-v618b/source.tar.gz')
REPORT_SHA = '6454ecc1d507fb8e781f49fe8c8e731e79d77cd567e1914c57c1a43ebe7c068b'
DECIMAL_SHA = 'b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4'
POLICY_SHA = 'aafb8c3207803d053524ec1447b8f253069c46b61d745b9e94a4be377627ed60'
MAP_SHA = '9bfd11597cf1a271ee9b0faa341c3c7d427a350fbe45f4bc153b991d7dc0282a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict(text):
    def reject(value):
        raise ValueError(value)
    return json.loads(text, parse_int=float, parse_constant=reject)


def bits(values):
    return b''.join(struct.pack('<d', float(x)) for x in values)


def records(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith('PERSISTENT_REPLAY'):
            prefix, value = line.split(' ', 1)
            assert prefix not in result
            result[prefix] = strict(value)
    return result


def top(values, count=5):
    return [dict(index=i, value=float(values[i]), absolute=float(abs(values[i])))
            for i in sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)[:count]]


def diagnose(q, final, pairs, dm):
    x, y, z, s = [[dm.dec(a) for a in final[k]] for k in ('x', 'y', 'z', 's')]
    px, ax, gx = q.mul(0, x), q.mul(1, x), q.mul(2, x)
    aty, gtz = q.mul(1, y, True), q.mul(2, z, True)
    eq = [a-b for a, b in zip(ax, q.b)]
    cone_eq = [a+b-h for a, b, h in zip(gx, s, q.h)]
    stationarity = [a+b+c+d for a, b, c, d in zip(px, q.c, aty, gtz)]
    epigraphs = {int(p['t']) for p in pairs}
    virtuals = {int(p['v']) for p in pairs}
    virtual_rows = {q.idx[1][k] for j in virtuals for k in range(q.ptr[1][j], q.ptr[1][j+1]) if q.val[1][k] != 0}
    # Pinned native assembly orders seven state components with mass last.
    intervals = len(pairs)//7
    assert len(pairs) == 7*intervals
    initial_mass_row = 7*intervals+6
    initial_entries = [(j, q.val[1][k]) for j in range(q.n) for k in range(q.ptr[1][j], q.ptr[1][j+1])
                       if q.idx[1][k] == initial_mass_row and q.val[1][k] != 0]
    assert initial_entries == [(6, Decimal(1))] and q.b[initial_mass_row] == 1
    completion = dict(pairs=len(pairs), exact_zero=0, positive=0, negative=0,
                      exact_abs_t=True, strict_nonzero_endpoint=True,
                      pair_complementarity_absolute=Decimal(0), pair_stationarity_absolute=Decimal(0),
                      pair_difference_max_absolute=Decimal(0), pair_difference_min_absolute=Decimal('Infinity'))
    for pair in pairs:
        t, v = int(pair['t']), int(pair['v'])
        rp, rm = int(pair['positive_row']), int(pair['negative_row'])
        lam = dm.dec(pair['lambda'])
        assert q.c[t] == lam and q.h[rp] == q.h[rm] == 0
        completion['exact_zero' if x[v] == 0 else 'positive' if x[v] > 0 else 'negative'] += 1
        completion['exact_abs_t'] &= x[t] == abs(x[v])
        if x[v] > 0:
            completion['strict_nonzero_endpoint'] &= z[rp] == lam and z[rm] == 0
        if x[v] < 0:
            completion['strict_nonzero_endpoint'] &= z[rp] == 0 and z[rm] == lam
        completion['pair_complementarity_absolute'] = max(completion['pair_complementarity_absolute'],
                                                         abs(s[rp]*z[rp]), abs(s[rm]*z[rm]))
        completion['pair_stationarity_absolute'] = max(completion['pair_stationarity_absolute'], abs(lam-z[rp]-z[rm]))
        completion['pair_difference_max_absolute'] = max(completion['pair_difference_max_absolute'], abs(z[rp]-z[rm]))
        completion['pair_difference_min_absolute'] = min(completion['pair_difference_min_absolute'], abs(z[rp]-z[rm]))
    scalar_pairs = {int(p[k]) for p in pairs for k in ('positive_row', 'negative_row')}
    scalar = [(i, s[i]*z[i], max(Decimal(0), -s[i])) for i in range(q.l) if i not in scalar_pairs]
    cones = []
    start = q.l
    for block, size in enumerate(q.soc):
        stop = start+size
        tail = dm.dot(s[start+1:stop], s[start+1:stop]).sqrt()
        cones.append(dict(block=block, first_original_row=start, size=size,
                          primal_cone_violation=float(max(Decimal(0), tail-s[start])),
                          signed_complementarity=float(dm.dot(s[start:stop], z[start:stop]))))
        start = stop
    primal_term, equality_term, cone_term, cone_eq_term = dm.dot(x, stationarity), -dm.dot(eq, y), dm.dot(s, z), -dm.dot(cone_eq, z)
    objective, dual_objective = dm.dot(x, px)/2+dm.dot(q.c, x), -dm.dot(x, px)/2-dm.dot(q.b, y)-dm.dot(q.h, z)
    signed_gap = objective-dual_objective
    defect = signed_gap-primal_term-equality_term-cone_term-cone_eq_term
    assert abs(defect) < Decimal('1e-50')
    return dict(original_objective=float(objective), original_dual_objective=float(dual_objective),
                objective_denominator=float(max(Decimal(1), abs(objective), abs(dual_objective))),
                signed_gap=float(signed_gap), gap_identity=dict(x_dot_stationarity=float(primal_term),
                    negative_equality_residual_dot_y=float(equality_term), s_dot_z=float(cone_term),
                    negative_conic_equation_residual_dot_z=float(cone_eq_term), decimal_identity_defect=str(defect)),
                equality_top=top(eq), stationarity_top=top(stationarity),
                mass_equalities=dict(initial_mass_row=initial_mass_row, initial_mass=float(x[6]),
                    initial_mass_residual=float(eq[initial_mass_row]),
                    maximum_dynamics_mass_residual=float(max(abs(eq[7*k+6]) for k in range(intervals))),
                    intervals=intervals),
                equality_with_virtual_variable_max=float(max((abs(eq[i]) for i in virtual_rows), default=Decimal(0))),
                equality_without_virtual_variable_max=float(max((abs(a) for i, a in enumerate(eq) if i not in virtual_rows), default=Decimal(0))),
                retained_stationarity_max=float(max(abs(a) for j, a in enumerate(stationarity) if j not in epigraphs)),
                retained_smooth_objective=float(sum((q.c[j]*x[j] for j in range(q.n) if j not in epigraphs), Decimal(0))),
                original_epigraph_penalty=float(sum((q.c[j]*x[j] for j in epigraphs), Decimal(0))),
                virtual_max_absolute=float(max(abs(x[j]) for j in virtuals)),
                equality_dual_max=float(dm.inf(y)),
                completion={k: float(v) if isinstance(v, Decimal) else v for k, v in completion.items()},
                SOC_by_primal_violation=sorted(cones, key=lambda c: c['primal_cone_violation'], reverse=True)[:3],
                SOC_by_complementarity=sorted(cones, key=lambda c: abs(c['signed_complementarity']), reverse=True)[:3],
                retained_scalar_complementarity_max=float(max((abs(c) for _, c, _ in scalar), default=Decimal(0))))


def main():
    report_path = BASE/'run/report.json'
    assert sha(report_path) == REPORT_SHA and sha(POLICY) == POLICY_SHA and sha(MAP) == MAP_SHA
    report = strict(report_path.read_text())
    manifest = strict((BASE/'manifest.json').read_text())
    assembly_source = 'cpp/cuda/src/gtoc12_conic.cu'
    with tarfile.open(SOURCE_ARCHIVE, 'r:gz') as tar:
        assembly_bytes = tar.extractfile(assembly_source).read()
    assert hashlib.sha256(assembly_bytes).hexdigest() == manifest['source_sha256'][assembly_source]
    assert report['complete'] and report['executions_started'] == 6 and report['solve_API_calls'] == 8 and report['bootstrap_iterations'] == 2
    assert report['runner_sha256'] == sha(BASE/'run.py') == 'ae8e3cf663cf563b810e2a75c8b0adc8eb736e00f308bc0cce28f0a4838984c2'
    assert report['manifest_sha256'] == sha(BASE/'manifest.json') == '6ad3dfc586c457afbeaa0da13417f1208484f908fb22977fd086ea41a4f3a4c2'
    assert report['core_sha256'] == '1002c69e2418ab8b4ba1cdb7376f2959b8af455ae7ea4815db751508009fceec'
    assert report['execution_blocks'] == 128
    for name, digest in report['inputs_sha256'].items():
        assert sha(BASE/'inputs'/name) == digest
    source = Path(__file__).with_name('decimal_audit_source.py')
    assert sha(source) == DECIMAL_SHA
    dm = types.ModuleType('decimal_v618_independent')
    exec(compile(source.read_bytes(), str(source), 'exec'), dm.__dict__)
    coefficient = {r['capture']: r for r in strict(MAP.read_text())['cases']}
    policy = {r['capture']: r for r in strict(POLICY.read_text())['cases']}
    findings = []
    with localcontext() as context:
        context.prec = 65
        for row in report['cases']:
            capture = row['capture']
            path = BASE/'run'/(row['name']+'.log')
            assert sha(path) == row['log_sha256']
            raw = records(path)
            meta, final = raw['PERSISTENT_REPLAY_META'], raw['PERSISTENT_REPLAY']
            q = dm.Snapshot(BASE/'inputs'/(capture+'.txt'))
            assert q.q_all_zero and not q.shift and all(v == 0 for v in q.origin)
            assert meta['input_sha256'] == coefficient[capture]['input_sha256'] == report['inputs_sha256'][capture+'.txt']
            assert meta['source_tree_sha256'] == report['source_tree_sha256'] and meta['library_sha256'] == report['core_sha256']
            assert bits(final['x']) == bits(final['x_solver'])
            for key, value in row['final'].items():
                assert final[key] == value, (row['name'], key)
            audit = q.audit(dict(final, termination_code=final['termination']))
            assert audit['passes'] == final['gpu_common_kkt']['passes'] == row['audit']['passes_common_kkt_gate']
            assert audit['qualified'] == final['qualified_original'] == row['audit']['qualified']
            for key in ('primal', 'dual', 'gap', 'primal_cone_violation', 'dual_cone_violation', 'objective', 'dual_objective'):
                assert math.isclose(audit[key], row['audit'][key], rel_tol=1e-9, abs_tol=1e-14), (row['name'], key)
            assert math.isclose(audit['block_complementarity_normalized'], row['audit']['complementarity_max_relative'], rel_tol=1e-9, abs_tol=1e-14)
            l, expected = final['l1'], policy[capture]
            assert l['finite'] and l['weight_valid'] and l['updates'] == final['iterations'] == l['completions']
            assert l['bound_scale'] == expected['B'] and l['objective_scale'] == expected['O']
            assert math.isclose(l['eta'], expected['eta'], rel_tol=1e-13)
            expected_omega = 1.0 if row['mode'] == 'unit' else expected['omega_cancel_global']
            assert l['omega'] == expected_omega and l['primal_base_step'] == l['eta']/expected_omega and l['dual_base_step'] == l['eta']*expected_omega
            ranges = expected['unit_ranges' if row['mode'] == 'unit' else 'cancel_global_ranges']
            assert math.isclose(l['minimum_threshold'], ranges['threshold'][0], rel_tol=1e-13)
            assert math.isclose(l['maximum_threshold'], ranges['threshold'][1], rel_tol=1e-13)
            assert not final['deadline_requested'] and final['recovery_iterations'] == final['recovery_seconds'] == 0
            seed_check = None
            if row['seeded']:
                assert audit['qualified'] and final['termination'] == 1 and final['iterations'] == 0
                initial = raw['PERSISTENT_REPLAY_INITIAL_POINT']
                point_lines = (BASE/'inputs'/(capture+'-initial.txt')).read_text().splitlines()
                point = {line.split()[0]: list(map(float, line.split()[2:])) for line in point_lines[3:]}
                assert point_lines[1] == 'snapshot_sha256 '+meta['input_sha256'] and point_lines[2] == 'coordinates translated'
                assert meta['initial_point_sha256'] == initial['point_sha256'] == sha(BASE/'inputs'/(capture+'-initial.txt'))
                for key in ('x', 'y', 'z'):
                    assert bits(point[key]) == bits(initial[key]) == bits(final[key])
                assert bits(point['s']) == bits(initial['s'])
                assert raw['PERSISTENT_REPLAY_PRESTEP']['native_seed_verified_unchanged']
                assert raw['PERSISTENT_REPLAY_BOOTSTRAP']['iterations'] == 1
                seed_check = dict(input_to_initial_to_final_xyz_bits_preserved=True, supplied_initial_slack_bits_preserved=True,
                                  final_slack_is_reconstructed=True, bootstrap_updates=1, actual_seed_updates=0)
            else:
                assert not audit['qualified'] and final['termination'] == 2 and final['iterations'] == 100000
            details = diagnose(q, final, coefficient[capture]['pairs'], dm)
            if not row['seeded']:
                assert details['completion']['exact_abs_t'] and details['completion']['strict_nonzero_endpoint']
                assert details['completion']['pair_complementarity_absolute'] == 0
            findings.append(dict(name=row['name'], capture=capture, mode=row['mode'], seeded=row['seeded'],
                                 log_sha256=sha(path), termination=final['termination'], iterations=final['iterations'],
                                 solve_seconds=final['solve_seconds'], scaling_seconds=final['scaling_seconds'],
                                 wall_seconds=final['wall_seconds'], outer_process_seconds=row['wall_seconds'],
                                 audit=audit, l1=l, seed_check=seed_check, diagnosis=details))
    result = dict(scope='Decimal65 from exact represented FP64 original coefficients and all six exported vector sets; CPU only',
                  script_sha256=sha(Path(__file__)), decimal_source_sha256=DECIMAL_SHA,
                  report_sha256=REPORT_SHA, policy_findings_sha256=POLICY_SHA, coefficient_map_sha256=sha(MAP),
                  raw_input_hashes=report['inputs_sha256'], source_tree_sha256=report['source_tree_sha256'], core_sha256=report['core_sha256'],
                  mass_row_source=dict(path=assembly_source, sha256=manifest['source_sha256'][assembly_source],
                                       dynamics_lines='153-168', initial_mass_line=174),
                  hardware=report['gpu_identity'].strip(), all_six_gate_decisions_agree=True,
                  work=dict(executions=6, solve_API_calls=8, bootstrap_updates=2, cold_updates=400000,
                            qualified_seed_cases=2, qualified_cold_cases=0), cases=findings)
    output = Path(__file__).with_name('findings.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), all_six_gate_decisions_agree=True)))


if __name__ == '__main__':
    main()
