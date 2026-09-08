"""Audit saved native tiny results and bind them to the reviewed source/oracle."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
D=ROOT/'build/performance/mass-tiny-v622'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'report.json').read_text())
assert sha(D/'report.json')=='471537ac89b61cf1623d03ec4d18a61f5f2a6fb83343a54387668cd25b234618'
b=json.loads((ROOT/'build/performance/mass-core-v622a/manifest.json').read_text())
assert r['manifest_sha256']==sha(ROOT/'build/performance/mass-core-v622a/manifest.json')
assert r['complete'] and r['status']=='passed' and r['gpu_executions_started']==1
assert r['core_sha256']==b['library']['sha256'] and r['test_sha256']==b['test']['sha256']
assert r['source_tree_sha256']==b['source_tree_sha256']
assert sha(D/'tiny.log')==r['cases'][0]['log_sha256']
rows=[{'prefix':s.split(' ',1)[0],'record':json.loads(s.split(' ',1)[1])}
      for s in (D/'tiny.log').read_text().splitlines() if s.startswith('MASS_')]
assert rows==r['cases'][0]['records']
cases=[item['record'] for item in rows if item['prefix']=='MASS_TEST']
expected=[('three_interval_oracle',2,3),('disabled_to_L1',2,1),('fresh_L1_control',2,1),
          ('original_seed_zero_step',1,0),('cancel_before_initial',3,0),('nonfinite_seed',4,0),('metric_overflow',4,0)]
assert [(c['case'],c['termination'],c['iterations']) for c in cases]==expected
assert r['actual_solve_api_calls']==7 and r['actual_optimization_iterations']==sum(c['iterations'] for c in cases)==5
oracle=json.loads((ROOT/'build/performance/mass-native-oracle-v622/fixtures.json').read_text())
gap=oracle['iterations'][-1]['original_gates']['normalized_gap']
assert abs(cases[0]['gap']-gap)<1e-15
assert cases[0]['mass_valid'] and cases[0]['finite'] and cases[0]['norm_squared_upper']<1
assert cases[1]['gap']==cases[2]['gap']
assert cases[3]['common_passes'] and cases[3]['completions']==0 and cases[3]['gap']==2.5e-21
assert not cases[4]['mass_valid'] and not cases[4]['common_valid']
assert not cases[5]['finite'] and not cases[6]['finite']
pre=ROOT/'build/performance/mass-real-v622-preflight/run/report.json'
p=json.loads(pre.read_text())
assert p['status']=='prepared_only_zero_GPU_calls' and p['executions_started']==0
assert p['tiny_report_sha256']==sha(D/'report.json') and p['core_sha256']==r['core_sha256']
assert p['runner_sha256']==sha(ROOT/'build/performance/run_mass_real_v622.py')=='f4bfb29c683eaa1ac394e5b8e52e1f2f59bea73a77be20daae4c8df896579f37'
out={'scope':'Saved outputs and frozen native assertion/oracle review; no reviewer GPU calls.',
     'tiny_report_sha256':sha(D/'report.json'),'tiny_log_sha256':sha(D/'tiny.log'),
     'core_sha256':r['core_sha256'],'test_sha256':r['test_sha256'],'cases':cases,
     'oracle_final_gap':gap,'actual_final_gap':cases[0]['gap'],
     'vector_evidence_limit':'Native test asserts full downloaded vectors and actual diagonal values against the independent fixture; this tiny log exports metrics only, not those vectors.',
     'solve_APIs':7,'actual_updates':5,'passed':True,
     'real_preflight_sha256':sha(pre),'reviewed_real_runner_sha256':p['runner_sha256'],
     'real_launch_decision':'GO for exactly6 processes/8 solve APIs including <=2 bootstrap updates; fixed128 blocks,2 original-seed guards and4 cold100k/30s cases. No retry or fallback.',
     'reviewer_GPU_calls':0}
(OUT/'tiny-findings.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'report_sha256':sha(OUT/'tiny-findings.json')}))
