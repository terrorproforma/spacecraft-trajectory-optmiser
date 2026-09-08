from pathlib import Path
import json,sys,collections
root=Path(sys.argv[1]);log=root/('campaign.log' if (root/'campaign.log').exists() else 'default.log')
rows=[json.loads(s[len('SCVX_PHASE '):]) for s in log.read_text().splitlines() if s.startswith('SCVX_PHASE ')]
keys=['setup','priming','graph_build','graph_run','graph_close','download','cleanup']
result=dict(calls=len(rows),phase_seconds={k:sum(r[k] for r in rows) for k in keys},status_counts=dict(collections.Counter(r['status'] for r in rows)),rows=rows)
if (root/'calls.json').exists():
    calls=json.loads((root/'calls.json').read_text());assert len(calls)==len(rows)
    finite_reports=[[p for p in c['solver_reports'] if p['solve_seconds'] is not None] for c in calls]
    result['cold_reports']={k:sum(max((p[k] for p in ps),default=0.) for ps in finite_reports) for k in ['setup_seconds','update_seconds','solve_seconds','residual_seconds']}
    result['native_call_seconds']=sum(c['seconds'] for c in calls)
    result['cold_outer_attempts']=sum(len(ps) for ps in finite_reports)
    result['cold_inner_iterations']=sum(p['iterations'] for ps in finite_reports for p in ps)
    result['all_outer_attempts']=sum(c['iterations'] for c in calls)
    result['all_inner_iterations']=sum(p['iterations'] for c in calls for p in c['solver_reports'])
    result['dimensions']=dict(collections.Counter(c['nodes'] for c in calls))
(root/'phase-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
