from pathlib import Path
import json,statistics,hashlib
root=Path('results/lambda/2026-09-08/gpu-conic-retry-v314')
inputs={'rtx5090':root/'rtx5090/arc/report.json','h100':root/'diagnostics/v314/arc-report.json'}
summary={}
for label,path in inputs.items():
 data=json.loads(path.read_text());assert data['complete'] and len(data['rows'])==24
 counters=dict(qualified_arcs=0,accepted_steps=0,unchanged_retries=0,solver_rejected_steps=0,candidate_rejected_steps=0)
 for row in data['rows']:
  assert row['status']=='converged' and row['qualified'];counters['qualified_arcs']+=1
  history=row['history'];reports=row['conic_reports'];assert len(history)==len(reports)==row['iterations']
  for i,(h,r) in enumerate(zip(history,reports)):
   if h['accepted']:
    counters['accepted_steps']+=1
    assert r['qualified'] and max(r['primal_residual'],r['dual_residual'],r['relative_gap'])<=r['requested_tolerance']
   elif 'solver' in h:
    counters['candidate_rejected_steps' if r['qualified'] else 'solver_rejected_steps']+=1
   if h.get('retry_unchanged'):
    counters['unchanged_retries']+=1;assert not h['accepted'] and r['qoco_status']==2 and not r['qualified']
    if i+1<len(history):
     n=history[i+1];assert h['trust_state']==n['trust_state'] and h['trust_control']==n['trust_control']
  assert sum(bool(h['accepted']) for h in history)==row['accepted']
 counters.update(median_seconds=statistics.median(r['seconds'] for r in data['rows']),maximum_seconds=max(r['seconds'] for r in data['rows']),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 summary[label]=counters
(root/'history-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
