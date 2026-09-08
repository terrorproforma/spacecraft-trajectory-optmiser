from pathlib import Path
import json,statistics,sys
root=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
rows=[];plans=[]
extra={'completed_element_hops','completed_earth_beam_rows','earth_beam_download_bytes'}
logical=None
for c in r['campaigns']:
 f=json.loads((root/c['name']/'output/run_report.json').read_text());b=f['best'];assert b['accepted'] and b['official']['ok'] and b['independent']['ok']
 counts={k:v for k,v in f['screening'].items() if k not in extra}
 if logical is None:logical=counts
 assert counts==logical
 ship=f['ships'][0];plans.append(ship['search']['top_candidates'])
 rows.append(dict(name=c['name'],candidate=c['candidate'],seconds=f['wall_seconds_total'],score=c['score'],search_seconds=ship['search']['wall_seconds'],refinement_seconds=sum(x['refined']['wall_seconds'] for x in ship['refinements']),retiming_seconds=ship['retiming']['wall_seconds']))
def difference(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys();return max([difference(a[k],b[k]) for k in a]+[0])
 if isinstance(a,list):
  assert len(a)==len(b);return max([difference(x,y) for x,y in zip(a,b)]+[0])
 if isinstance(a,(int,float)) and not isinstance(a,bool):return abs(a-b)
 assert a==b;return 0
delta=max(difference(p,plans[0]) for p in plans)
a=statistics.median(c['seconds'] for c in rows if c['candidate']);b=statistics.median(c['seconds'] for c in rows if not c['candidate'])
summary=dict(rows=rows,baseline_median=b,candidate_median=a,less_time_percent=100*(1-a/b),max_initial_plan_numeric_delta=delta,scope='Complete CLI ABBA, two runs per mode. Initial plan comparison checks structure and nonnumeric fields exactly; maximum numeric difference is over all proxy costs, masses, scores and epochs, not a mission feasibility tolerance.')
(root/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
