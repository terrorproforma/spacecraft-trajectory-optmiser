from pathlib import Path
import numpy as np,json
root=Path('build/performance/nt-snapshot-v348');r=json.loads((root/'reference.json').read_text());raw=np.fromfile(root/'output.bin',np.float64)
nt=np.array(r['nt']);wtw=np.array(r['wtw']);assert len(raw)==2*(len(nt)+len(wtw))
rows={}
for variant,offset in [('legacy',0),('candidate',len(nt)+len(wtw))]:
 ns=[];ms=[];ni=0;mi=0
 for n in r['sizes']:
  ns.append(float(np.max(np.abs(raw[offset+ni:offset+ni+n+1]-nt[ni:ni+n+1]))/np.max(np.abs(nt[ni:ni+n+1]))))
  count=n*(n+1)//2
  ms.append(float(np.max(np.abs(raw[offset+len(nt)+mi:offset+len(nt)+mi+count]-wtw[mi:mi+count]))/np.max(np.abs(wtw[mi:mi+count]))))
  ni+=n+1;mi+=count
 rows[variant]=dict(nonfinite_pairs=int(np.sum(~np.isfinite(ns)|~np.isfinite(ms))),max_relative_nt_error=float(np.nanmax(ns)),max_relative_wtw_error=float(np.nanmax(ms)),median_relative_wtw_error=float(np.nanmedian(ms)))
summary=dict(cases=len(r['sizes']),rows=rows)
(root/'probe-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
assert rows['candidate']['nonfinite_pairs']==0 and rows['candidate']['max_relative_wtw_error']<1e-12
