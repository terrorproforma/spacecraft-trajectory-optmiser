"""Compile/run only the host arithmetic/conversion checks; never load CUDA."""
from pathlib import Path
import hashlib,json,os,subprocess,time
repo=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
out=repo/'build/performance/l1-weight-cpu-v618b'
out.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
exe=out/'conversion-test'
commands=[['/usr/bin/c++','-std=c++20','-O2','-Wall','-Wextra','-Wpedantic','-Werror',
    '-I'+str(repo/'cpp/cuda/include'),'-I'+str(repo/'cpp/include'),
    str(repo/'cpp/cuda/tests/persistent_snapshot_conversion_test.cpp'),'-o',str(exe)],
    [str(exe)]]
result={'complete':False,'cuda_visible_devices':'','stages':[]}
for i,cmd in enumerate(commands):
    start=time.perf_counter()
    with (out/f'{i}.log').open('x') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=60)
    result['stages'].append({'command':cmd,'returncode':p.returncode,'seconds':time.perf_counter()-start})
    if p.returncode:break
result['complete']=len(result['stages'])==2 and all(x['returncode']==0 for x in result['stages'])
result['source_sha256']={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [repo/'cpp/cuda/include/spacepdhcg/cuda/l1_epigraph_arithmetic.hpp',
              repo/'cpp/cuda/tests/persistent_snapshot_conversion_test.cpp']}
(out/'report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
raise SystemExit(0 if result['complete'] else 1)
