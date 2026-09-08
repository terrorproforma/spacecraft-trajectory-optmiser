from pathlib import Path
p=Path('build/performance')
for tag in ['resident-options-v497','resident-options-v500']:
 for kind in ['archive','retrieve']:
  source=(p/(kind+'_early_graph_v471.py')).read_text().replace('early-graph-v471',tag)
  if kind=='archive':
   source=source.replace("and p.name!='files-sha256.json'", "and p.name not in ['files-sha256.json','resident-options-probe']")
   if tag.endswith('497'):source="import sys\nsys.argv=['analysis','/home/ubuntu/spacepdhcg-"+tag+"']\nexec("+repr((p/'analyze_resident_options.py').read_text())+")\n"+source
  (p/(kind+'_'+tag.replace('-','_')+'.py')).write_text(source)
