from pathlib import Path
p=Path('build/performance')
for tag in ['workspace-pool-v478','workspace-pool-replay-v480','workspace-pool-v484','workspace-pool-replay-v486','workspace-pool-v487','workspace-pool-v491']:
 for kind in ['archive','retrieve']:
  source=(p/(kind+'_early_graph_v471.py')).read_text().replace('early-graph-v471',tag)
  if kind=='archive':
   source=source.replace("and p.name!='files-sha256.json'", "and p.name not in ['files-sha256.json','workspace-pool-probe']")
   analysis=p/('analyze_workspace_pool.py' if '-replay-' not in tag else 'analyze_early_graph_replay.py')
   if tag in ['workspace-pool-v478','workspace-pool-v484']:analysis=p/'analyze_workspace_pool_baselines.py'
   if tag!='workspace-pool-v491':source="import sys\nsys.argv=['analysis','/home/ubuntu/spacepdhcg-"+tag+"']\nexec("+repr(analysis.read_text())+")\n"+source
  (p/(kind+'_'+tag.replace('-','_')+'.py')).write_text(source)
