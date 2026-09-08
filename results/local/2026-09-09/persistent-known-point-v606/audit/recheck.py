"""Read-only CPU recheck of the eight known-point GPU logs; writes only audit/."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import numpy as np

out = Path(__file__).resolve().parent
root = out.parent
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('recorded_auditor', root / 'source/audit_persistent_snapshot.py')
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
report = json.loads((root / 'run/report.json').read_text())
manifest = json.loads((root / 'run/adapter-manifest.json').read_text())
encoded = json.loads((root / 'encoded-points.json').read_text())
prepared = json.loads((root / 'preparation.json').read_text())
assert report['complete'] and len(report['cases']) == 8
assert digest(root / 'run/adapter-manifest.json') == report['manifest_sha256']
assert digest(root / 'run/tiny-report.json') == report['tiny_report_sha256']
assert digest(root / 'encoded-points.json') == report['point_manifest_sha256']
assert manifest['frozen_commit'] == report['source_pin']
assert manifest['executable_sha256'] == report['executable_sha256']
assert manifest['immutable_core_sha256'] == report['library_sha256']
assert report['library_sha256'] == 'd4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633'

def bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)

def same_bits(left, right):
    return np.array_equal(bits(left), bits(right))

def point_values(path, problem):
    lines = path.read_text().splitlines()
    assert len(lines) == 7 and lines[0] == 'SPACEPDHCG_QOCO_INITIAL_POINT_V1'
    assert lines[1].split() == ['snapshot_sha256', digest(root / 'inputs' / (path.name.split('-')[0] + '.txt'))]
    assert lines[2].split() == ['coordinates', 'translated']
    values = {}
    for line, key, length in zip(lines[3:], ('x', 'y', 'z', 's'),
                                 (problem['n'], problem['p'], problem['m'], problem['m']), strict=True):
        tokens = line.split()
        assert tokens[0] == key and int(tokens[1]) == length and len(tokens) == length + 2
        values[key] = np.asarray(tokens[2:], dtype=np.float64)
        assert np.isfinite(values[key]).all()
    return values

problems = {}
points = {}
input_evidence = []
for row in encoded['points']:
    label = row['label']
    snapshot = root / 'inputs' / (label + '.txt')
    point = root / 'inputs' / (label + '-initial.txt')
    qoco = root / 'inputs' / (label + '-qoco.log')
    neutral = root / 'inputs' / (label + '-point.json')
    before = next(x for x in prepared['points'] if x['label'] == label)
    assert digest(snapshot) == row['snapshot_sha256'] == before['snapshot_sha256']
    assert digest(point) == row['encoded_sha256']
    assert digest(qoco) == before['reference_log_sha256']
    assert digest(neutral) == before['point_sha256']
    problem = auditor.load_snapshot(snapshot)
    # Both actual captures are unshifted; shifted coordinate arithmetic is tested by the analytic bundle.
    assert problem['shift'] == 0 and problem['quadratic_numerical_nonzeros'] == 0
    problems[label] = problem
    points[label] = point_values(point, problem)
    initial_result = auditor.audit(problem, points[label], backend='persistent', coordinates='original')
    assert initial_result['passes_common_kkt_gate'] and not initial_result['qualified']
    input_evidence.append({'label': label, 'snapshot_sha256': digest(snapshot), 'point_sha256': digest(point),
                           'qoco_log_sha256': digest(qoco), 'neutral_point_sha256': digest(neutral),
                           'reference_repeat': before['reference_repeat'], 'initial_common_gate': initial_result,
                           'variables': problem['n'], 'equalities': problem['p'], 'conic_rows': problem['m'],
                           'numerical_hessian_nonzeros': problem['quadratic_numerical_nonzeros'],
                           'shifted': False})

cases = []
phase_groups = {}
work = {'executable_invocations': 0, 'native_solve_calls': 0, 'seeded_iterations': 0,
        'bootstrap_iterations': 0, 'recovery_iterations': 0,
        'requested_solve_wall_seconds_sum': 0.0, 'native_solve_event_seconds_sum': 0.0,
        'bootstrap_wall_seconds_sum': 0.0, 'executable_wall_seconds_sum': 0.0}
native_fields = {'primal': 'primal', 'dual': 'dual', 'gap': 'gap', 'objective': 'objective',
                 'dual_objective': 'dual_objective', 'block_complementarity_normalized': 'complementarity_max_relative'}
for row in report['cases']:
    path = root / 'run' / (row['name'] + '.log')
    assert digest(path) == row['log_sha256']
    records = {}
    for line in path.read_text().splitlines():
        prefix, payload = line.split(' ', 1)
        assert prefix not in records
        # Parsing integer-looking floating values as float preserves a JSON -0 sign bit.
        records[prefix] = json.loads(payload, parse_int=float, parse_float=float)
    assert set(records) == {'PERSISTENT_REPLAY_META', 'PERSISTENT_REPLAY_INITIAL_POINT',
                            'PERSISTENT_REPLAY_BOOTSTRAP', 'PERSISTENT_REPLAY_PRESTEP',
                            'PERSISTENT_REPLAY', 'PERSISTENT_REPLAY_SUMMARY'}
    meta = records['PERSISTENT_REPLAY_META']
    initial = records['PERSISTENT_REPLAY_INITIAL_POINT']
    bootstrap = records['PERSISTENT_REPLAY_BOOTSTRAP']
    pre = records['PERSISTENT_REPLAY_PRESTEP']
    final = records['PERSISTENT_REPLAY']
    p = problems[row['label']]
    point = points[row['label']]
    assert row['returncode'] == 0 and not row['timed_out']
    assert meta['input_sha256'] == row['input_sha256']
    assert meta['initial_point_sha256'] == row['point_sha256'] == initial['point_sha256']
    assert meta['source_commit'] == manifest['frozen_commit']
    assert meta['source_sha256'] == manifest['compiled_snapshot_source_sha256']
    assert meta['library_sha256'] == manifest['immutable_core_sha256']
    assert meta['fold_singleton_bounds'] == row['folded']
    assert meta['coordinate_system'] == initial['coordinate_system'] == 'original'
    assert meta['initial_point_coordinates'] == 'translated'
    assert meta['initial_point_supplied'] and initial['supplied_qualified'] and initial['mapped_reference_qualified']
    assert not initial['strict_reconstruction_is_import_gate']
    for key in ('x', 'y', 'z', 's'):
        assert same_bits(initial[key], point[key]), (row['name'], key)
    assert same_bits(initial['x_solver'], point['x'])
    # Independent expected native dual order: equalities, retained nonnegative rows, then each SOC tail/radius.
    keep = np.ones(p['l'], dtype=bool)
    singleton_count = 0
    if row['folded']:
        csr = p['G'].tocsr()
        for r in range(p['l']):
            values = csr.data[csr.indptr[r]:csr.indptr[r + 1]]
            nonzero = values[values != 0]
            if len(nonzero) == 1:
                assert abs(nonzero[0]) == 1  # Every actual singleton has an exact binary64 ratio.
                keep[r] = False
                singleton_count += 1
        assert singleton_count == meta['folded_rows']
    expected = [point['y'], point['z'][:p['l']][keep]]
    start = p['l']
    for dimension in p['soc']:
        end = start + dimension
        expected.extend([-point['z'][start + 1:end], -point['z'][start:start + 1]])
        start = end
    assert same_bits(np.concatenate(expected), initial['dual_solver'])
    assert pre['seeded_iterations'] == 0 and pre['termination'] == 0 and pre['state'] == 3
    assert pre['warm_start_mode'] == 2 and pre['warm_start_accepted']
    assert pre['solve_epoch'] == 1 and pre['native_seed_verified_unchanged']
    assert not pre['inherited_report_counters_are_seed_work'] and pre['inherited_report_iterations'] == 1
    assert bootstrap['iterations'] == 1 and bootstrap['recovery_iterations'] == 0
    assert bootstrap['termination'] == 2 and bootstrap['api_status'] == 0 and not bootstrap['deadline_requested']
    assert final['iterations'] == row['iterations_requested'] and final['recovery_iterations'] == 0
    assert final['termination'] == 2 and final['termination_name'] == 'iteration_limit'
    assert final['api_status'] == 0 and not final['deadline_requested'] and not final['solver_optimal']
    assert pre['native_natural_residual'] > report['requested_tolerance']
    phase_metrics = {key: value for key, value in pre.items() if key.startswith('native_') and key != 'native_pre_step_measured'}
    group = (row['label'], row['folded'])
    if group in phase_groups:
        assert phase_groups[group] == phase_metrics
    else:
        phase_groups[group] = phase_metrics
    fresh_initial = auditor.audit(p, initial, backend='persistent', coordinates='original')
    fresh_final = auditor.audit(p, final, backend='persistent', coordinates='original')
    assert fresh_initial['passes_common_kkt_gate'] and not fresh_initial['qualified']
    assert fresh_final['passes_common_kkt_gate'] == final['kkt_qualified_original']
    assert fresh_final['qualified'] == final['qualified_original'] == False
    for native, external in native_fields.items():
        assert math.isclose(final['audit'][native], fresh_final[external], rel_tol=1e-9, abs_tol=1e-12)
    for key, value in row['audits'][0].items():
        if key == 'repeat':
            assert value == final['repeat']
            continue
        observed = fresh_final[key]
        assert math.isclose(value, observed, rel_tol=1e-13, abs_tol=1e-20) if isinstance(value, float) else value == observed
    failures = [key for key in ('primal', 'dual', 'gap', 'complementarity_max_relative') if fresh_final[key] > 1e-9]
    if fresh_final['cone_violation'] > 1e-8:
        failures.append('absolute_cone_violation')
    delta = float(np.max(np.abs(np.asarray(final['x_solver']) - point['x'])))
    assert delta == row['maximum_primal_change_from_seed']
    cases.append({'name': row['name'], 'log_sha256': digest(path), 'input_sha256': row['input_sha256'],
                  'point_sha256': row['point_sha256'], 'supplied_and_reference_bits_match_point': True,
                  'native_dual_mapping_bits_match': True, 'internal_seed_verified_unchanged': True,
                  'prestep_natural_residual': pre['native_natural_residual'],
                  'strict_reconstructed_seed_qualified': initial['strict_reconstructed_qualified'],
                  'seeded_iterations': int(final['iterations']), 'bootstrap_iterations': int(bootstrap['iterations']),
                  'termination': 'ITERATION_LIMIT', 'fresh_initial_audit': fresh_initial,
                  'fresh_final_audit': fresh_final, 'common_gate_failures': failures,
                  'native_and_python_common_gate_agree': True, 'final_natural_residual': final['native_natural_residual'],
                  'maximum_primal_change_from_seed': delta, 'wall_seconds': final['wall_seconds'],
                  'native_solve_event_seconds': final['solve_seconds']})
    work['executable_invocations'] += 1
    work['native_solve_calls'] += 2
    work['seeded_iterations'] += int(final['iterations'])
    work['bootstrap_iterations'] += int(bootstrap['iterations'])
    work['recovery_iterations'] += int(final['recovery_iterations']) + int(bootstrap['recovery_iterations'])
    work['requested_solve_wall_seconds_sum'] += final['wall_seconds']
    work['native_solve_event_seconds_sum'] += final['solve_seconds']
    work['bootstrap_wall_seconds_sum'] += bootstrap['wall_seconds']
    work['executable_wall_seconds_sum'] += row['process_seconds']
result = {'complete': True, 'scope': 'independent saved-input/vector/phase/hash and original-conic-KKT evidence audit; no GPU launched',
          'source_pin': report['source_pin'], 'executable_sha256': report['executable_sha256'],
          'core_sha256': report['library_sha256'], 'report_sha256': digest(root / 'run/report.json'),
          'auditor_sha256': digest(root / 'source/audit_persistent_snapshot.py'),
          'input_evidence': input_evidence, 'cases': cases, 'work': work,
          'initial_reference_common_gate_passes': sum(c['fresh_initial_audit']['passes_common_kkt_gate'] for c in cases),
          'final_common_gate_passes': sum(c['fresh_final_audit']['passes_common_kkt_gate'] for c in cases),
          'native_optimal_terminations': 0, 'qualified_native_outputs': 0,
          'common_gate_note': 'A certificate can pass original KKT while native termination is ITERATION_LIMIT or UNSPECIFIED; qualified requires accepted termination as well.',
          'interpretation': 'Saved evidence confirms source qualification, exact retained-dual mapping, and unchanged internal pre-step vectors. Seven outputs lose the common gap gate after updates; one retains it but never meets the native absolute stopping test. This does not identify a specific numerical mechanism or prove global divergence.',
          'limitations': ['Two unshifted captured conic LPs; shifted mapping is covered only by the separate tiny analytic bundle.',
                          'Each of eight configurations was run once; no statistical repeatability or GPU speed claim.',
                          'No nonlinear trajectory certification, production backend integration, or fleet score change.']}
with (out / 'findings.json').open('x') as output:
    json.dump(result, output, indent=2, allow_nan=False)
print(json.dumps({key: result[key] for key in ('complete', 'initial_reference_common_gate_passes', 'final_common_gate_passes', 'native_optimal_terminations', 'qualified_native_outputs', 'work')}))
