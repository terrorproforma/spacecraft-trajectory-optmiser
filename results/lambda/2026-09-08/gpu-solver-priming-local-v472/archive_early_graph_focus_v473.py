from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/ubuntu/spacepdhcg-early-graph-focus-v473')
r=json.loads((root/'report.json').read_text());assert r['complete'] and not r.get('error')
assert len(r['stages'])==4 and all(s['returncode']==0 for s in r['stages'])
repo=Path('/home/ubuntu/spacepdhcg-early-graph-v471/repo')
for name in ['build/performance/replay_early_graph.py','build/performance/grid-cache-fleet-v403/scvx-calls.json','cpp/cuda/src/native_qoco_adapter.cpp','cpp/cuda/src/gtoc12_scvx.cu']:
 target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/name).read_bytes())
rows={}
for name in ['baseline0','candidate0','candidate1','baseline1']:
 data=json.loads((root/name/'results.json').read_text());assert [r['index'] for r in data]==[44,201,98]
 rows[name]=[{k:v for k,v in r.items() if k in ['index','seconds','iterations','status','diagnostic','priming']} for r in data]
(root/'analysis.json').write_text(json.dumps(rows,indent=2))
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='files-sha256.json'}
(root/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
archive=root.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for name in list(manifest)+['files-sha256.json']:t.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(manifest))))
