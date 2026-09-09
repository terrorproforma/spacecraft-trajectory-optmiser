"""Read-only original fleet composition and saved checker audit; no propagation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def parts(path):
    sections, footprint, events = {}, {}, {}
    for raw in path.read_bytes().splitlines(keepends=True):
        row = raw.split()
        assert row
        ship, body = int(row[0]), int(row[1])
        sections.setdefault(ship, bytearray()).extend(raw)
        footprint.setdefault(ship, set())
        events.setdefault(ship, [])
        if body > 0:
            footprint[ship].add(body)
        if body != -1:
            events[ship].append((body, [float(v) for v in row[2:]]))
    assert set(sections) == set(range(1, 24))
    return sections, footprint, events


def near(a, b, tolerance=1e-8):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--bonus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'build/performance'
    kit = base / 'seeded-current-composition-v628'
    out = kit / 'output'
    route = base / 'seeded-candidate-boundary-merit-v627/campaign'
    current = root / 'results/lambda/2026-09-09/gpu-regeneration-v799/h100-best'
    plan = read(kit / 'plan.json')
    for name, expected in plan['source_sha256'].items():
        assert sha(root / name) == expected, name
    marker = read(kit / 'launch-marker.json')
    assert marker['plan_sha256'] == sha(kit / 'plan.json') and marker['run_sha256'] == sha(kit / 'run.py')
    launch, report = read(out / 'launch-report.json'), read(out / 'report.json')
    assert sha(out / 'report.json') == 'd3259631ca7fcfdef5e2aee24ce94bbaa0df6e98ad7c9528c2194b332f2a1806'
    assert sha(out / 'checker-binding.json') == '43e829e4baf76af30971a7681d5af909b816792e0d0037e5ea0477f054a14dce'
    assert report['complete'] and report['qualified'] and report['status'] == 'verified_current_fleet_improvement'
    assert launch['complete'] and launch['exit_code'] == launch['child_exit_after_cleanup'] == 0
    assert launch['hard_deadline_seconds'] == 120 and launch['TERM_grace_seconds'] == 10 and launch['KILL_reap_seconds'] == 30
    assert report['independent_calls'] == report['official_calls'] == 1
    assert all(report[k] == 0 for k in ('GPU_calls', 'native_solves', 'search_calls', 'extra_leg_certificates'))
    assert report['incumbent_promoted'] is False
    path = kit / 'inputs/Result.txt'
    result_hash = sha(path)
    assert result_hash == plan['composed_result_sha256'] == report['result_sha256'] == launch['result_sha256']
    assert result_hash == '63446ebf3ff1298911bccade7b789a8fb68a0f7db4171906c8e6783a7f174bab'
    old, old_fp, old_events = parts(current / 'Result.txt')
    added, added_fp, added_events = parts(route / 'output/fleet/Result.txt')
    actual, actual_fp, actual_events = parts(path)
    assert actual[8] == added[8] and actual_events[8] == added_events[8]
    assert all(actual[k] == old[k] for k in actual if k != 8)
    assert actual_fp == old_fp and actual_fp[8] == added_fp[8]
    assert len(set.union(*actual_fp.values())) == 200 and len(actual_fp[8]) == 8
    assert all(not actual_fp[8].intersection(v) for k, v in actual_fp.items() if k != 8)
    assert path.read_bytes() == b''.join(bytes(added[k] if k == 8 else old[k]) for k in range(1, 24))
    previous_review = base / 'boundary-candidate-review-v627'
    review_index = read(previous_review / 'index.json')
    assert sha(previous_review / 'index.json') == 'd8f29835244cb50147fb9853e4e04ea40282e9373b8faec76e74fdb7f4b03614'
    for name, pin in review_index['files'].items():
        assert (previous_review / name).stat().st_size == pin['bytes'] and sha(previous_review / name) == pin['sha256']
    assert read(previous_review / 'findings.json')['passed']
    binding = read(out / 'checker-binding.json')
    independent, official = read(out / 'independent-full.json'), read(out / 'official-full.json')
    assert independent['ok'] and not independent['violations'] and independent['ship_count'] == 23
    assert official['ok'] and official['return_code'] == 0 and 'Check successfully!' in official['stdout']
    assert sha(out / 'official/Result.txt') == result_hash
    assert sha(out / 'official/GTOC12_Asteroids_Data.txt') == report['catalogue_sha256'] == '99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675'
    for mode in ('independent', 'official'):
        summary = read(out / (mode + '.json'))
        assert summary == binding[mode] == report[mode]
        assert summary['result_sha256'] == binding['result_sha256'] == result_hash
        assert summary['ok'] and summary['ships'] == 23 and summary['mined_asteroids'] == 200
    assert independent['total_mass_kg'] == report['independent']['total_mass_kg']
    assert independent['weighted_score_fixed_bonus_kg'] == report['independent']['weighted_score_fixed_bonus_kg']
    assert independent['ship_limit'] >= 23
    lines = (out / 'official/ScoreData.txt').read_text().splitlines()
    assert int(lines[0]) == len(lines)-1 == 200
    mass = {int(row[0]): float(row[1]) for row in (line.split() for line in lines[1:])}
    assert len(mass) == 200 and {str(k): v for k, v in mass.items()} == official['score_data']
    assert set(mass) == set.union(*actual_fp.values())
    assert sha(args.bonus) == report['bonus_sha256'] == 'e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21'
    bonus_rows = [line.split() for line in args.bonus.read_text().splitlines() if line.strip()]
    assert len(bonus_rows) == 60000 and all(len(row) == 2 for row in bonus_rows)
    selected_bonus = {k: float(bonus_rows[k-1][0]) for k in mass}
    assert all(0 < value <= 1 for value in selected_bonus.values())
    score_raw = math.fsum(mass.values())
    score_weighted = math.fsum(value*selected_bonus[k] for k, value in mass.items())
    near(score_raw, independent['total_mass_kg'])
    near(score_weighted, independent['weighted_score_fixed_bonus_kg'])
    unloaded = {}
    for ship, ev in actual_events.items():
        assert len(ev) % 2 == 0 and ev[-2][0] == ev[-1][0] == -3
        assert ev[-2][1][0] == ev[-1][1][0]
        unloaded[ship] = ev[-2][1][7] - ev[-1][1][7]
    near(math.fsum(unloaded.values()), independent['total_mass_kg'])
    baseline = read(current / 'campaign-report.json')['independent']
    assert baseline == plan['baseline'] and baseline['ok']
    raw_delta = independent['total_mass_kg'] - baseline['total_mass_kg']
    weighted_delta = independent['weighted_score_fixed_bonus_kg'] - baseline['weighted_score_fixed_bonus_kg']
    assert raw_delta == report['raw_delta_kg'] > 0 and weighted_delta == report['weighted_delta_kg'] > 1e-8
    near(independent['total_mass_kg'], plan['expected_raw_kg'])
    near(independent['weighted_score_fixed_bonus_kg'], plan['expected_weighted_kg'])
    assert report['physics_tolerances_unchanged'] and report['reporting_bound_kg'] == 1e-8
    result = {'passed': True, 'scope': 'Saved bytes, reports and score arithmetic only. No checker rerun, propagation, GPU, optimizer or search.',
              'plan_sha256': sha(kit / 'plan.json'), 'report_sha256': sha(out / 'report.json'),
              'result_sha256': result_hash, 'checker_binding_sha256': sha(out / 'checker-binding.json'),
              'current_input_result_sha256': sha(current / 'Result.txt'), 'certified_replacement_result_sha256': sha(route / 'output/fleet/Result.txt'),
              'all_22_current_other_ship_sections_byte_identical': True, 'replacement_ship8_exact_certified_bytes': True,
              'ship8_bytes': len(actual[8]), 'ship8_sha256': hashlib.sha256(actual[8]).hexdigest(),
              'ships': 23, 'asteroids': 200, 'ship8_asteroids': sorted(actual_fp[8]), 'footprints_unchanged': True,
              'both_original_fullfleet_checkers_pass': True, 'raw_kg': independent['total_mass_kg'],
              'weighted_kg': independent['weighted_score_fixed_bonus_kg'], 'raw_gain_kg': raw_delta, 'weighted_gain_kg': weighted_delta,
              'official_scoredata_raw_sum': score_raw, 'official_scoredata_weighted_sum': score_weighted,
              'independently_summed_submitted_unloads_kg': math.fsum(unloaded.values()),
              'bonus_sha256': sha(args.bonus), 'used_bonus_coefficients': selected_bonus,
              'saved_calls': {'independent_fullfleet': 1, 'official_fullfleet': 1, 'GPU': 0, 'native': 0, 'search': 0},
              'seconds': {'independent': report['independent_seconds'], 'official': report['official']['wall_seconds'], 'worker': report['seconds']},
              'candidate_eligible_to_replace_pinned_v799_incumbent': True,
              'scope_limit': 'Eligibility is against the exact pinned v799 Result, not an uninspected later incumbent.',
              'output_sha256': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'passed': True, 'findings_sha256': sha(args.output), 'result_sha256': result_hash,
                      'weighted_gain_kg': weighted_delta, 'raw_gain_kg': raw_delta}))


if __name__ == '__main__':
    main()
