from pathlib import Path
import json
p=Path.home()/'spacepdhcg-collect-geometry-bench-v861'
r=json.loads((p/'report.json').read_text())
print({k:r.get(k) for k in ['pid','child_pid','stage','complete','success','error','medians']})
print('child_alive',Path('/proc/'+str(r.get('child_pid'))).exists())
print([(x['name'],[(y['ship'],round(y['seconds'],6),y['max_numeric_difference']) for y in x['routes']]) for x in r['runs']])
