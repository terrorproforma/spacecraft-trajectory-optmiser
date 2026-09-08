from pathlib import Path
import hashlib,json
p=Path('cpp/cuda/src/orbitweaver_gpu.cu');s=p.read_text();assert s.count('if(count<=1024)hop_warp_kernel')==1;s=s.replace('if(count<=1024)hop_warp_kernel','if(count<=16384)hop_warp_kernel');p.write_text(s)
root=Path('build/performance/warp-hops-v385');root.mkdir(exist_ok=False);(root/'orbitweaver.candidate.cu').write_bytes(p.read_bytes());(root/'source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),change='Extend cooperative scan to batches through 16384'),indent=2))
for old,new in [('run_hop_micro_v384.py','run_hop_micro_v385.py'),('audit_hop_micro_v384.py','audit_hop_micro_v385.py')]:
 s=Path('build/performance',old).read_text().replace('warp-hops-v384','warp-hops-v385');Path('build/performance',new).write_text(s)
