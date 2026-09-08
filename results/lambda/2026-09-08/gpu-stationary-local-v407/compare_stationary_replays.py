from pathlib import Path
import json,sys,collections
root=Path(sys.argv[1])
a={v['index']:v for v in json.loads((root/'baseline/results.json').read_text())}
b={v['index']:v for v in json.loads((root/'candidate/results.json').read_text())}
assert set(a)==set(b)==set(range(225))
def summarize(rows):
 return dict(cases=len(rows),seconds=sum(v['seconds'] for v in rows.values()),statuses=dict(collections.Counter(v['status'] for v in rows.values())),outer_iterations=sum(v['iterations'] for v in rows.values()),converged_seconds=sum(v['seconds'] for v in rows.values() if v['status']=='converged'),unsuccessful_seconds=sum(v['seconds'] for v in rows.values() if v['status']!='converged'),uncertified=[i for i,v in rows.items() if v['status']=='converged' and (not v['certified'] or v['mass_margin_kg'] < -1e-6)])
sa,sb=summarize(a),summarize(b)
both=[i for i in a if a[i]['status']==b[i]['status']=='converged']
result=dict(baseline=sa,candidate=sb,speedup=sa['seconds']/sb['seconds'],less_time_percent=100*(1-sb['seconds']/sa['seconds']),lost_baseline=[i for i in a if a[i]['status']=='converged' and b[i]['status']!='converged'],gained=[i for i in a if a[i]['status']!='converged' and b[i]['status']=='converged'],lost_prior=[i for i in b if b[i]['prior_status']=='converged' and b[i]['status']!='converged'],max_final_mass_change_kg=max(abs(a[i]['certificate']['final_mass_kg']-b[i]['certificate']['final_mass_kg']) for i in both),stationary_failures=[i for i,v in b.items() if 'stationary penalized' in str(v['diagnostic'])],scope='One complete replay per mode, same frozen native binary, original boundaries/settings; independently propagated converged outputs. Timing includes GPU seed and solve, excluding independent certification. No global infeasibility or universal speedup claim.')
(root/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
