"""CPU-only audit of the completed v618 tiny batch; no CUDA/test execution.

Raw tiny records contain metrics, not vectors. The frozen executable checks
downloaded vectors against its CPU oracle in-process. This script independently
reconstructs the two-step scalar/SOC fixture and its original objective gap.
"""
import hashlib
import json
import math
from pathlib import Path

BASE = Path('build/performance/l1-weight-tiny-v618')
REAL_RUNNER = Path('build/performance/run_l1_weight_real_v618.py')
REPORT_SHA = '0b76263d42ab50abe1d4d910fb1f5dac730fd5ffaa99b19360255dc1faf62feb'
REAL_RUNNER_SHA = 'ae8e3cf663cf563b810e2a75c8b0adc8eb736e00f308bc0cce28f0a4838984c2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict(text):
    def reject(value):
        raise ValueError(value)
    return json.loads(text, parse_constant=reject)


def simulate(cost, weight):
    eta = .9/math.sqrt(3)
    B = 1/(1+math.sqrt(2))
    O = 1/(1+math.sqrt(cost*cost+1+16))
    omega = O/B if weight == 0 else weight
    tau, sigma = eta/omega*O/B, eta*omega*B/O
    v = w = vbar = wbar = 0.0
    eq = a = b = c = 0.0
    for _ in range(2):
        eq += sigma*(vbar+wbar-1)
        sx, sy, sr = a/sigma+vbar, b/sigma+wbar, c/sigma+1
        norm = math.hypot(sx, sy)
        if norm <= -sr:
            px = py = pr = 0.0
        elif norm <= sr:
            px, py, pr = sx, sy, sr
        else:
            pr = (norm+sr)/2
            px, py = sx*pr/norm, sy*pr/norm
        a += sigma*(vbar-px)
        b += sigma*(wbar-py)
        c += sigma*(1-pr)
        oldv, oldw = v, w
        gradient = cost+eq+a
        argument = v-tau*gradient
        v = max(0, argument-4*tau) if argument >= 0 else min(0, argument+4*tau)
        w -= tau*(1+eq+b)
        t = abs(v)
        delta = 4 if v > 0 else -4 if v < 0 else max(-4, min(4, -gradient))
        zplus, zminus = (4+delta)/2, (4-delta)/2
        assert zplus >= 0 and zminus >= 0 and zplus+zminus == 4
        assert (t-v)*zplus == 0 and (t+v)*zminus == 0
        vbar, wbar = 2*v-oldv, 2*w-oldw
    objective, dual = cost*v+4*t+w, -eq+c
    return dict(eta=eta, B=B, O=O, omega=omega, threshold=4*tau,
                gap=abs(objective-dual)/max(1, abs(objective), abs(dual)))


def main():
    report_path = BASE/'report.json'
    assert sha(report_path) == REPORT_SHA and sha(REAL_RUNNER) == REAL_RUNNER_SHA
    report = strict(report_path.read_text())
    assert report['complete'] and report['status'] == 'passed'
    assert report['manifest_sha256'] == '6ad3dfc586c457afbeaa0da13417f1208484f908fb22977fd086ea41a4f3a4c2'
    assert report['core_sha256'] == '1002c69e2418ab8b4ba1cdb7376f2959b8af455ae7ea4815db751508009fceec'
    assert report['test_sha256'] == '2ed65a3ef07dc436e51d7ff6d5dd2e6a9d2aea6bf4401cdc6778f0deb067d8a6'
    assert report['runner_sha256'] == '874a46be33fc063caecec1f2bb3c652f02659e9981e9a7d84e281643aa76098c'
    all_cases, raw_hashes = {}, {}
    statuses = {1: 0, 2: 0, 3: 0, 4: 0}
    for item in report['cases']:
        path = BASE/(item['name']+'.log')
        raw_hashes[path.name] = sha(path)
        assert raw_hashes[path.name] == item['log_sha256'] and item['returncode'] == 0
        records = []
        for line in path.read_text().splitlines():
            prefix, value = line.split(' ', 1)
            records.append(dict(prefix=prefix, record=strict(value)))
        assert records == item['records']
        cases = [r['record'] for r in records if r['prefix'] == 'L1_TEST']
        all_cases[item['name']] = cases
        assert len(cases) == (9 if item['name'] == 'unit' else 13)
        assert sum(c['iterations'] for c in cases) == (10 if item['name'] == 'unit' else 12)
        for c in cases:
            statuses[c['termination']] += 1
            if 'weak_seed' in c['case']:
                assert c['iterations'] == 0 and c['termination'] == 1 and c['common_passes'] == 1 and c['completions'] == 0
            elif c['case'] == 'cancel_before_initial':
                assert c['iterations'] == 0 and c['termination'] == 3 and c['common_valid'] == 0
                if c['weight_mode'] == 2:
                    assert c['omega'] is c['primal_base_step'] is c['dual_base_step'] is None
            elif c['case'] == 'nonfinite_initial' or 'overflow' in c['case']:
                assert c['iterations'] == 0 and c['termination'] == 4 and c['finite'] == 0
                if 'overflow' in c['case']:
                    assert c['common_passes'] == 1  # Known feasible seed; invalid steps still reject.
            else:
                assert c['termination'] == 2 and c['common_passes'] == 0
    spec = {'positive_prox': (-8, 1), 'negative_prox': (8, 1), 'positive_zero_prox': (0, 1),
            'negative_zero_prox': (0, 1), 'weighted_quarter_prox': (-8, .25), 'weighted_four_prox': (8, 4),
            'weighted_positive_zero': (0, .25), 'weighted_negative_zero': (0, .25), 'cancel_global_prox': (-8, 0)}
    comparisons = []
    for cases in all_cases.values():
        for c in cases:
            if c['case'] not in spec:
                continue
            expected = simulate(*spec[c['case']])
            error = max(abs(c[k]-v) for k, v in expected.items())
            assert error < 3e-14
            comparisons.append(dict(case=c['case'], original_gap=expected['gap'], maximum_metric_absolute_error=error))
    weighted = {c['case']: c for c in all_cases['weighted']}
    switched = {k: v for k, v in weighted['weight_switch_unit'].items() if k not in ('case', 'weight_mode')}
    fresh = {k: v for k, v in weighted['fresh_unit_l1_control'].items() if k not in ('case', 'weight_mode')}
    assert switched == fresh
    unit = {c['case']: c for c in all_cases['unit']}
    assert {k: v for k, v in unit['disabled_original_scaling_refresh'].items() if k != 'case'} == {
        k: v for k, v in unit['fresh_original_control'].items() if k != 'case'}
    result = dict(scope='Saved tiny metadata plus independent scalar/SOC fixture calculation; no GPU execution',
                  script_sha256=sha(Path(__file__)), tiny_report_sha256=REPORT_SHA, raw_log_sha256=raw_hashes,
                  real_runner_sha256=REAL_RUNNER_SHA, actual_processes=2, solve_calls=22, actual_updates=22,
                  requested_update_cap=31, termination_counts=statuses,
                  scalar_soc_metric_reconstructions=comparisons, same_export_history_control_metrics=True,
                  caveat='No raw tiny vectors are emitted. In-process frozen test asserts downloaded vectors; this script independently replays fixture metrics, not absent raw vectors.',
                  real_runner_review=dict(disposition='GO for root-controlled finite run', executions=6,
                                          maximum_solve_calls=8, maximum_bootstrap_updates=2, blocks=128,
                                          cold_update_cap=100000, cold_deadline_seconds=30,
                                          preserves_original_gates=True, seed_bits_checked=True,
                                          retries=False, capacity_fallback=False))
    output = Path(__file__).with_name('tiny-findings.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
