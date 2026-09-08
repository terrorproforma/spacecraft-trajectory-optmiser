from pathlib import Path
import json, statistics, sys
root=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text())
assert r['complete'] and not r.get('error')
cs=r['campaigns'];extra={'completed_compact_options','compact_option_download_bytes'}
logical=lambda c:{k:v for k,v in c['screening'].items() if k not in extra}
assert all(logical(c)==logical(cs[0]) for c in cs)
assert max(c['score'] for c in cs)-min(c['score'] for c in cs)<1e-6
for c in cs:
 raw=json.loads((root/c['name']/'output/run_report.json').read_text())
 assert raw['best']['accepted'] and raw['best']['official']['ok'] and raw['best']['independent']['ok']
b=statistics.median(c['seconds'] for c in cs if not c['candidate'])
a=statistics.median(c['seconds'] for c in cs if c['candidate'])
print(json.dumps(dict(campaigns=[{k:v for k,v in c.items() if k!='screening'} for c in cs],baseline_median=b,candidate_median=a,less_time_percent=100*(1-a/b),speedup=b/a,logical_counts_identical=True,scores_and_physics_preserved=True),indent=2))
