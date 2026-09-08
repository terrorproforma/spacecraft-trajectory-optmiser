from pathlib import Path
import json
root=Path('/home/angus/spacepdhcg-joint-geometry-v651/campaign-v656')
report=json.loads((root/'report.json').read_text());reference=None
keys=('asteroids','orphaned','deploy_epochs','collect_epochs','foreign_deploy_epochs','earth_return_epoch','collected_mass_kg')
for run in report['runs']:
 proxies={p.name:{k:json.loads(p.read_text())[k] for k in keys} for p in (root/run['name']/'proxies').glob('*.json')}
 if reference is None:reference=proxies
 assert proxies==reference and len(proxies)==4
 print(run['name'],'all four proxy event sequences and payloads match exactly')
