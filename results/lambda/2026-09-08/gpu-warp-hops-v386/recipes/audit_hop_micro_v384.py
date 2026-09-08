from pathlib import Path
import json,numpy as np
root=Path('build/performance/warp-hops-v384');mismatches=[];cases=0
for reference in sorted((root/'baseline0').glob('*.npz')):
 a=np.load(reference)
 for mode in ['candidate0','candidate1','baseline1']:
  b=np.load(root/mode/reference.name)
  for key in a.files:
   if not np.array_equal(a[key],b[key],equal_nan=True):
    good=np.isfinite(a[key])&np.isfinite(b[key]);mismatches.append(dict(case=reference.name,mode=mode,key=key,max_finite_difference=float(np.max(np.abs(a[key][good]-b[key][good]))) if np.any(good) else None))
  cases+=1
reports={m:json.loads((root/m/'report.json').read_text())['rows'] for m in ['baseline0','candidate0','candidate1','baseline1']};timings=[]
for i,row in enumerate(reports['baseline0']):
 baseline=np.median(reports['baseline0'][i]['seconds']+reports['baseline1'][i]['seconds']);candidate=np.median(reports['candidate0'][i]['seconds']+reports['candidate1'][i]['seconds']);timings.append(dict(count=row['count'],baseline_ms=1000*baseline,candidate_ms=1000*candidate,speedup=baseline/candidate))
audit=dict(compared_outputs=cases,all_fields_exact_equal=not mismatches,mismatches=mismatches,timings=timings)
(root/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
