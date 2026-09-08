from pathlib import Path
import sys,time,json
import numpy as np
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert,HOP_REQUEST,HOP_RESULT
from spacepdhcg.gtoc12 import constants as C
root=Path(sys.argv[1]);root.mkdir(exist_ok=False);rng=np.random.default_rng(384)
count=16384;requests=np.zeros(count,dtype=HOP_REQUEST);q=requests['lambert']
q['id']=np.arange(count);q['mu']=C.MU_SUN_KM3_S2;q['tof']=rng.uniform(30,1500,count)*C.DAY_S;q['tolerance']=1e-8;q['iterations']=256
for field in ['r1','r2']:
 vectors=rng.normal(size=(count,3));q[field]=vectors/np.linalg.norm(vectors,axis=1)[:,None]*rng.uniform(.8,3.5,count)[:,None]*C.AU_KM
requests['v1_body']=rng.normal(size=(count,3))*20;requests['v2_body']=rng.normal(size=(count,3))*20
requests['departure_allowance'][::3]=6;requests['arrival_allowance'][::7]=6
q['tof'][2:4]=[0,-1];q['r2'][4]=q['r1'][4];requests['v1_body'][5,0]=np.nan;q['r1'][6,1]=np.inf
rows=[];output=np.zeros(count,HOP_RESULT)
with GpuLambert(count) as gpu:
 for scan in [16,31,32,127,256,8192]:
  gpu._prepare(scan)
  for n in [1,31,32,127,257,451,480,533,1024,1025,4096,16384]:
   gpu._check(gpu.evaluate_hops(gpu.handle,requests.ctypes.data,n,output.ctypes.data,n))
   np.savez_compressed(root/f'scan-{scan}-n-{n}.npz',**{key:output[key][:n] for key in HOP_RESULT.names})
   if scan==256:
    times=[]
    for repeat in range(12):
     start=time.perf_counter();gpu._check(gpu.evaluate_hops(gpu.handle,requests.ctypes.data,n,output.ctypes.data,n));elapsed=time.perf_counter()-start
     if repeat>=2:times.append(elapsed)
    rows.append(dict(count=n,seconds=times,median_seconds=float(np.median(times))))
(root/'report.json').write_text(json.dumps(dict(rows=rows,accuracy_cases=72),indent=2))
