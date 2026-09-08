from pathlib import Path
import json,hashlib,tarfile,statistics
v=Path('/home/ubuntu/spacepdhcg-nonfinite-ir-v337');r=Path('/home/ubuntu/spacepdhcg-nonfinite-replay-v338');c=Path('/home/ubuntu/spacepdhcg-nonfinite-confirm-v341')
a=json.loads((v/'report.json').read_text());b=json.loads((r/'report.json').read_text());d=json.loads((c/'report.json').read_text());assert a['complete'] and b['complete'] and d['complete']
rows=b['campaigns']+d['rows'];med={str(mode):statistics.median(x['seconds'] for x in rows if x['candidate']==mode) for mode in [False,True]}
summary=dict(campaigns=rows,median_seconds=med,time_reduction_percent=100*(1-med['True']/med['False']),baseline_qp_qualified=sum(x['qualified'] for x in b['rows'] if x['name'].startswith('baseline')),candidate_qp_qualified=sum(x['qualified'] for x in b['rows'] if x['name'].startswith('guard')),qp_repeats_per_mode=32)
(r/'summary.json').write_text(json.dumps(summary,indent=2))
files=[]
for p in v.iterdir():
 if p.is_file() and p.suffix in ['.log','.json','.py']:files.append((p,'validation/'+p.name))
for p in (v/'overlay').rglob('*'):
 if p.is_file():files.append((p,'validation/overlay/'+str(p.relative_to(v/'overlay'))))
for name in ['algebra/cuda/qoco_ir_runtime.cuh','algebra/cuda/cudss_backend.cu']:files.append((v/'qoco'/name,'validation/patched-qoco/'+name))
for base,label in [(r,'replay'),(c,'confirmation')]:
 for p in base.rglob('*'):
  if p.is_file():files.append((p,label+'/'+str(p.relative_to(base))))
manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for p,name in files}
mp=Path('/tmp/nonfinite-ir-v337-sha256.json');mp.write_text(json.dumps(manifest,indent=2))
archive=Path('/tmp/nonfinite-ir-v337-results.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for p,name in files:t.add(p,arcname=name)
 t.add(mp,arcname='remote-sha256.json')
print(json.dumps(dict(summary=summary,files=len(files),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
