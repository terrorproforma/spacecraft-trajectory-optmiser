from pathlib import Path
import json

root = Path.home() / 'spacepdhcg-collect-geometry-v860'
r = json.loads((root / 'report.json').read_text())
print(json.dumps({k: r.get(k) for k in ('pid', 'child_pid', 'stage', 'complete', 'success', 'error')}, indent=2))
for key in ('pid', 'child_pid'):
    path = Path('/proc') / str(r.get(key)) / 'cmdline'
    print(key, path.read_bytes().replace(b'\0', b' ').decode() if path.exists() else 'terminal/missing')
print((root / (r['stage'] + '.log')).read_text()[-5000:])
