from pathlib import Path
import json,statistics,sys
root=Path(sys.argv[1]);r=json.loads((root/'report.json').read_text())
assert r['complete'] and not r.get('error');rows=[];plans=None;counts=None
logical=['completed_branch_requests','completed_element_hops','completed_compact_options','completed_collection_options','completed_collection_queries','completed_batches','completed_collection_dp_passes','completed_retime_dp_calls','completed_retime_driver_calls','completed_earth_beam_rows']
for c in r['campaigns']:
 result=json.loads((root/c['name']/'output/run_report.json').read_text());best=result['best'];s=result['screening']
 assert best['accepted'] and best['official']['ok'] and best['independent']['ok']
 current=result['ships'][0]['search']['top_candidates'];actual={key:s[key] for key in logical}
 if plans is None:plans=current;counts=actual
 assert plans==current and counts==actual
 elapsed=c.get('process_seconds')
 if elapsed is None:elapsed=next(stage['seconds'] for stage in r['stages'] if stage['name']==c['name'])
 row=dict(name=c['name'],candidate=c['candidate'],process_seconds=elapsed,cli_seconds=result['wall_seconds_total'],score=best['independent']['weighted_score_fixed_bonus_kg'],option_download_bytes=s['compact_option_download_bytes'],option_upload_bytes=s['collection_option_upload_bytes'],host_table_read_bytes=s.get('resident_option_read_bytes',0),resident_builds=s.get('resident_option_builds',0),return_pruning_queries=s.get('completed_return_feasibility_queries',0),reported_selection_download_bytes=s.get('resident_option_selection_download_bytes',0))
 if c['candidate']:
  assert row['resident_builds']==2267 and row['option_upload_bytes']==row['host_table_read_bytes']==0 and row['option_download_bytes']==9068
 rows.append(row)
before=statistics.median(x['process_seconds'] for x in rows if not x['candidate']);after=statistics.median(x['process_seconds'] for x in rows if x['candidate'])
for row in rows:
 log=(root/(row['name']+'.log')).read_text()
 phases=[json.loads(s.split('SCVX_PHASE ',1)[1]) for s in log.splitlines() if s.startswith('SCVX_PHASE ')]
 assert len(phases)==47
 row['phase_seconds']={k:sum(p[k] for p in phases) for k in ['setup','priming','graph_build','graph_run','graph_close','download','cleanup']}
 row['retained_replays']=log.count('RETAINED_REPLAY submitted=0 updated=1')
 assert row['retained_replays']==(30 if row['candidate'] else 0)
 assert log.count('RETAINED_REPLAY ')==row['retained_replays']
 row['solver_calls']=len(json.loads((root/row['name']/'calls.json').read_text()))
 assert row['solver_calls']==47
out=dict(scope='ABBA two observations per mode, same native binary. Process time includes teardown. Exact initial plans and listed logical counts; both final checkers pass. Traces add diagnostic overhead to both modes.',baseline_process_median=before,candidate_process_median=after,less_process_time_percent=100*(1-after/before),logical_counts=counts,rows=rows,telemetry_note='Adapter transfer fields include cumulative counters from retained audit/conversion objects. Do not treat them as a per-leg byte ledger.')
(root/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
