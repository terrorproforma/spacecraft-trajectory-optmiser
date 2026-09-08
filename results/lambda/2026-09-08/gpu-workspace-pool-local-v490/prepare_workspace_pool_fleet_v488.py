from pathlib import Path
import json
p=Path('build/performance')
source=(p/'run_early_graph_fleet_v472.py').read_text().replace('early-graph-fleet-v472','workspace-pool-fleet-v488').replace('early_graph_fleet472','workspace_pool_fleet488').replace('build-spacepdhcg-early-graph-v465','build-spacepdhcg-workspace-pool-v482').replace('SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH','SPACEPDHCG_TEST_GTOC12_QOCO_POOL')
a=source.index('for p in [')+len('for p in ');b=source.index(']},pid',a)+1
old=source[a:b]
import ast
names=list(dict.fromkeys(ast.literal_eval(old)+list(json.loads((p/'workspace-pool-v482/report.json').read_text())['source_sha256'])))
source=source[:a]+repr(names)+source[b:]
(p/'run_workspace_pool_fleet_v488.py').write_text(source)
