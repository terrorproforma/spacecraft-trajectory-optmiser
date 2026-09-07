from pathlib import Path
import json,tarfile,hashlib
base=Path('/home/ubuntu/spacepdhcg-conic-retry-v314')
out=base/'diagnostic-evidence';out.mkdir(exist_ok=False)
roots={306:'spacepdhcg-resident-capture-v306',307:'spacepdhcg-arc-snapshot-v307',308:'spacepdhcg-qp-sweep-v308',309:'spacepdhcg-qp-ir-v309',311:'spacepdhcg-qp-scaling-v311',312:'spacepdhcg-qp-objective-v312',314:'spacepdhcg-conic-retry-v314'}
raw_hashes={}
for version,name in roots.items():
 root=Path('/home/ubuntu')/name;target=out/('v'+str(version));target.mkdir()
 for p in root.iterdir():
  if not p.is_file() or p.suffix not in ['.json','.py','.log','.err']:continue
  dest=target/p.name;data=p.read_bytes()
  # Full-vector replay logs can be very large. Retain their SHA and one complete
  # first/last replay for an independently recomputable compact evidence bundle.
  if p.suffix=='.log' and b'QP_REPLAY ' in data:
   raw_hashes[str(p)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   records=[json.loads(x[10:]) for x in data.decode().splitlines() if x.startswith('QP_REPLAY ')]
   (target/(p.stem+'-samples.json')).write_text(json.dumps(dict(total_replays=len(records),first=records[0],last=records[-1]),indent=2))
  else:dest.write_bytes(data)
 if version==307:
  for p in (root/'snapshots').glob('*.txt'):
   (target/p.name).write_bytes(p.read_bytes())
 if version==314:
  (target/'arc-report.json').write_bytes((root/'arc/report.json').read_bytes())
  for relative in json.loads((root/'repo/retry-source-sha256.json').read_text()):
   dest=target/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/'repo'/relative).read_bytes())
(out/'remote-full-log-sha256.json').write_text(json.dumps(raw_hashes,indent=2))
(out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2))
archive=base/'diagnostic-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname='diagnostics')
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
