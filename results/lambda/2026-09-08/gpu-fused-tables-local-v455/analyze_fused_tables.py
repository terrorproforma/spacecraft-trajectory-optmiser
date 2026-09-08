from pathlib import Path
import json,statistics
for folder in ['fused-tables-v448','retrieved-fused-tables-v449']:
    root=Path('build/performance')/folder;r=json.loads((root/'report.json').read_text())
    assert r['complete'] and not r.get('error')
    reference=None;plans=None;rows=[]
    for c in r['campaigns']:
        f=json.loads((root/c['name']/'output/run_report.json').read_text());b=f['best']
        assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
        if reference is None:reference=f['screening'];plans=f['ships'][0]['search']['top_candidates']
        assert reference==f['screening']
        assert plans==f['ships'][0]['search']['top_candidates']
        rows.append(dict(name=c['name'],candidate=c['candidate'],seconds=f['wall_seconds_total'],score=b['independent']['weighted_score_fixed_bonus_kg']))
    before=statistics.median(x['seconds'] for x in rows if not x['candidate']);after=statistics.median(x['seconds'] for x in rows if x['candidate'])
    result=dict(rows=rows,baseline_median_seconds=before,candidate_median_seconds=after,less_time_percent=100*(1-after/before),scope='One ABBA batch, two samples per mode. All screening counters and complete initial candidate plans match exactly; both mission verifiers pass. Timing variation prevents attributing the full local campaign change to table construction.')
    (root/'analysis.json').write_text(json.dumps(result,indent=2));print(folder,result)
local=Path('build/performance/fused-tables-micro-v452/measurement.json')
remote=Path('build/performance/retrieved-fused-tables-v453/measurement.json')
assert json.loads(local.read_text())['table_sha256']==json.loads(remote.read_text())['table_sha256']
for path in [local,remote]:
    r=json.loads(path.read_text());rows=[x for x in r['rows'] if not x['name'].startswith('warm')]
    before=statistics.median(x['seconds'] for x in rows if not x['candidate']);after=statistics.median(x['seconds'] for x in rows if x['candidate'])
    result=dict(baseline_median_seconds=before,candidate_median_seconds=after,less_time_percent=100*(1-after/before),tables=r['tables'],cells=r['cells'],same_hashes_across_hardware=True)
    (path.parent/'analysis.json').write_text(json.dumps(result,indent=2));print(str(path),result)
