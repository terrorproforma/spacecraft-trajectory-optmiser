from pathlib import Path
import hashlib,json,shutil
p=Path('build/performance');root=p/'preserve-campaign-v537'
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
assert '107 passed' in (root/'pytest.log').read_text()
for name,sha in r['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
names=list(r['source_sha256'])+[n for n in r['pytest']['command'] if n.startswith('tests/')]
names+=['build/performance/run_preserve_campaign_v537.py']
names += [str(q) for q in Path('src/spacepdhcg/gtoc12').glob('*.py')]
for name in names:
 dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
print('Frozen',len(set(names)),'campaign source files')
