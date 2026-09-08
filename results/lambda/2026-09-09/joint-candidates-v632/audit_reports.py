"""Summarize downloaded reports without evaluating trajectories or using a GPU."""

from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
evidence = root/'evidence'


def read(relative):
    return json.loads((evidence/relative).read_text())


parent = read('campaign-v636/report.json')
names = ('baseline0', 'candidate0', 'candidate1', 'baseline1')
reports = {name: read(f'campaign-v636/{name}/report.json') for name in names}
rows = []
for name in names:
    report = reports[name]
    recorded = next(item for item in parent['runs'] if item['name'] == name)
    best = report['best']
    assert report['complete'] and best['official']['ok'] and best['independent']['ok']
    assert best['score_kg'] == recorded['best']['score_kg']
    assert report['native_sha256']['SPACEPDHCG_GTOC12_CUDA_LIBRARY'] == parent['core_sha256']
    assert report['native_sha256']['SPACEPDHCG_QOCO_LIBRARY'] == parent['qoco_sha256']
    rows.append({
        'name': name, 'device_selection': recorded['selection'],
        'seconds': report['seconds'], 'weighted_kg': best['score_kg'],
        'raw_kg': best['total_mass_kg'], 'official_pass': best['official']['ok'],
        'independent_pass': best['independent']['ok'],
        'mined_asteroids': best['independent']['mined_asteroids'],
        'ships': best['independent']['ships'],
        'native_solves': report['native_solves_completed'],
        'screening_telemetry': report['screening_telemetry'],
        'refinements': [{key: attempt.get(key) for key in (
            'ship', 'case', 'grid_days', 'certified', 'seconds', 'status', 'failures')}
            for attempt in report['refinements']],
    })
first = reports['baseline0']
scores = [row['weighted_kg'] for row in rows]
proxy_hashes = {}
for proxy in sorted((evidence/'campaign-v636/baseline0/proxies').glob('*.json')):
    proxy_hashes[proxy.name] = {
        name: hashlib.sha256((evidence/f'campaign-v636/{name}/proxies'/proxy.name).read_bytes()).hexdigest()
        for name in names}
summary = {
    'scope': 'Existing Lambda v632/v636 reports, retrieved read-only; no new tests, jobs or trajectory verification executed.',
    'core_sha256': parent['core_sha256'], 'qoco_sha256': parent['qoco_sha256'],
    'configuration': first['configuration'], 'scvx_settings': first['scvx_settings'],
    'joint_batch_requested': first['joint_batch_requested'],
    'physics_tolerances_changed': any(report['physics_tolerances_changed'] for report in reports.values()),
    'all_recorded_input_hashes_equal': all(report['input_sha256'] == first['input_sha256'] for report in reports.values()),
    'all_recorded_python_hashes_equal': all(report['source_sha256'] == first['source_sha256'] for report in reports.values()),
    'proxy_hashes': proxy_hashes,
    'all_proxy_artifacts_byte_identical': all(len(set(values.values())) == 1 for values in proxy_hashes.values()),
    'baseline_weighted_kg': first['baseline']['score_kg'],
    'baseline_raw_kg': first['baseline']['total_mass_kg'],
    'best_weighted_span_kg': max(scores)-min(scores),
    'candidate0_weighted_gain_kg': reports['candidate0']['best']['score_kg']-first['baseline']['score_kg'],
    'candidate0_raw_gain_kg': reports['candidate0']['best']['total_mass_kg']-first['baseline']['total_mass_kg'],
    'campaigns': rows,
    'timing_limit': 'Candidate1 failed its second refinement and therefore has different downstream verification work. The four durations do not establish a device-selection end-to-end speedup.',
}
(root/'audit-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({key: summary[key] for key in (
    'all_recorded_input_hashes_equal', 'all_recorded_python_hashes_equal',
    'all_proxy_artifacts_byte_identical', 'best_weighted_span_kg',
    'candidate0_weighted_gain_kg', 'candidate0_raw_gain_kg')}, indent=2))
