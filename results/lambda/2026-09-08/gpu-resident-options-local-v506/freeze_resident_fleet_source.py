from pathlib import Path
import hashlib,json,shutil
root=Path('build/performance/resident-options-fleet-v499');r=json.loads((root/'report.json').read_text())
assert r['complete'] and r['returncode']==0
for name,sha in r['source_sha256'].items():
 source=Path(name);assert hashlib.sha256(source.read_bytes()).hexdigest()==sha,name
 target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
print('Fleet source snapshot verified')
