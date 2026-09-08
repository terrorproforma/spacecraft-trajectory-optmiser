from pathlib import Path
for family,versions in [('solver-phase',['459','462']),('early-graph',['466','469','471'])]:
 for version in versions:
  tag=family+'-v'+version
  for kind in ['archive','retrieve']:
   base=Path('build/performance/'+kind+'_compact_v426.py').read_text().replace('compact-options-v426',tag)
   if kind=='archive' and version in ['459','466']:
    base=base.replace("assert r['complete'] and not r.get('error')", "assert not r['complete'] and r.get('error')")
   Path('build/performance/'+kind+'_'+family.replace('-','_')+'_v'+version+'.py').write_text(base)
