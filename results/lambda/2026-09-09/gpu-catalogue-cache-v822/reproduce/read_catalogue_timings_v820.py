from pathlib import Path
import json
r=json.loads((Path.home()/'spacepdhcg-catalogue-bench-v820/report.json').read_text())
for run in r['runs']:
    print(json.dumps(dict(name=run['name'],routes=[{k:row[k] for k in ('ship','seconds','candidates')}|{'hashes':row['telemetry'].get('completion_catalogue_hashes'),'reuses':row['telemetry'].get('completion_catalogue_hash_reuses'),'completion_batches':row['telemetry'].get('completion_batches'),'packing_seconds':row['telemetry'].get('completion_pack_seconds')} for row in run['routes']])))
