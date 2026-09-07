from pathlib import Path
import os,subprocess,json,time
root=Path('/home/ubuntu/spacepdhcg-finish-profile-v268');root.mkdir(exist_ok=False)
repo=Path('/home/ubuntu/spacepdhcg-driver-retime-v256/repo')
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/ubuntu/spacepdhcg-driver-retime-v256/core-build/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data')
report=dict(complete=False,steps=[])
for mode in ['old','candidate']:
    env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']='/home/ubuntu/'+('spacepdhcg-driver-retime-v256' if mode=='old' else 'spacepdhcg-finish-barriers-v266')+'/core-build/cuda/libspacepdhcg_cuda.so'
    cmd=['/usr/local/bin/nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(root/mode),'/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/tmp/profile_driver_v258.py',str(root/(mode+'.json')),'/home/ubuntu/spacepdhcg-native-campaign-v209/output/ship_01/refinements.json',str(repo/'input-return.json'),'device']
    start=time.time()
    with (root/(mode+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
    report['steps'].append(dict(command=cmd,returncode=r.returncode,seconds=time.time()-start));(root/'report.json').write_text(json.dumps(report,indent=2));r.check_returncode()
    with (root/(mode+'-stats.log')).open('x') as log:r=subprocess.run(['/usr/local/bin/nsys','stats','--report','cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum','--format','csv','--output','.',str(root/(mode+'.nsys-rep'))],stdout=log,stderr=subprocess.STDOUT,timeout=30)
    r.check_returncode()
report['complete']=True;(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
