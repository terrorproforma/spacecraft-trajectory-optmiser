from pathlib import Path
import re,hashlib,json
p=Path('cpp/cuda/src/orbitweaver_gpu.cu');s=p.read_text();root=Path('build/performance/warp-hops-v384');root.mkdir(exist_ok=False)
(root/'orbitweaver.before.cu').write_bytes(p.read_bytes())
patterns=[('hop_kernel<<<static_cast<unsigned>((count+63)/64),64,0,\n        reinterpret_cast<cudaStream_t>(stream.native_handle)>>>(requests,count,samples,results,nullptr);','launch_hops(requests,count,samples,results,nullptr,reinterpret_cast<cudaStream_t>(stream.native_handle));'),('hop_kernel<<<static_cast<unsigned>((count+63)/64),64,0,w->stream>>>(\n            w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid);','launch_hops(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);'),('hop_kernel<<<static_cast<unsigned>((count+63)/64),64,0,w->stream>>>(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid);','launch_hops(w->hops,count,w->config.scan_samples_per_band,w->hop_results,w->scan_grid,w->stream);')]
counts=[]
for old,new in patterns:
 counts.append(s.count(old));assert old in s;s=s.replace(old,new)
assert sum(counts)==4,counts
anchor='__device__ bool element_state(';assert s.count(anchor)==1
s=s.replace(anchor,Path('build/performance/warp_hops_v384.cuh').read_text()+'\n'+anchor)
p.write_text(s);(root/'orbitweaver.candidate.cu').write_bytes(p.read_bytes());(root/'source.json').write_text(json.dumps(dict(replaced_launches=counts,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
