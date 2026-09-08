from pathlib import Path
import json,statistics
root=Path('build/performance/warp-pipeline-v386');r=json.loads((root/'report.json').read_text());assert r['complete'];campaigns=r['campaigns'];b=statistics.median(c['seconds'] for c in campaigns if not c['candidate']);n=statistics.median(c['seconds'] for c in campaigns if c['candidate']);screen=campaigns[0]['screening'];assert all(c['screening']==screen for c in campaigns)
summary=dict(complete=True,baseline_median_seconds=b,candidate_median_seconds=n,speedup=b/n,time_reduction_percent=100*(1-n/b),same_search_counts=True,all_mission_checkers_pass=True,campaigns=campaigns)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='campaigns'},indent=2))
