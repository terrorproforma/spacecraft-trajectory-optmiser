"""One approved bounded tiny batch, under the shared GPU lock; never overwrites."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

root=Path('/home/angus/spacepdhcg-common-kkt-v609d')
manifest=json.loads((root/'manifest.json').read_text())
assert manifest['complete']
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
binary=root/'build/cuda-tests/persistent_common_kkt_test'
core=root/'build/cuda/libspacepdhcg_cuda.so'
assert digest(binary)==manifest['persistent_common_kkt_test_sha256']
assert digest(core)==manifest['library_sha256']
out=root/'tiny';out.mkdir(exist_ok=False)
shutil.copy2(__file__,out/'run_tiny.py')
report={'complete':False,'pid':os.getpid(),'manifest_sha256':digest(root/'manifest.json'),
        'runner_sha256':digest(out/'run_tiny.py'),'binary_sha256':digest(binary),'core_sha256':digest(core),
        'cases':[],'gpu_invocations':0,'scope':'two execution strategies; seven bounded solve calls each; no real captures'}
def save():(out/'report.json').write_text(json.dumps(report,indent=2))
save()
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_','QOCO_','PDHCG_','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES'))}
try:
    with Path('/home/angus/.spacepdhcg-gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report['lock_acquired']=True;save()
        for blocks in (0,2):
            name='legacy' if blocks==0 else 'cooperative'
            command=[str(binary),'--execution-blocks',str(blocks)]
            start=time.perf_counter();report['gpu_invocations']+=1;save()
            with (out/(name+'.log')).open('x') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
            records=[]
            for line in (out/(name+'.log')).read_text().splitlines():
                if line.startswith('COMMON_KKT_'):
                    prefix,payload=line.split(' ',1);records.append({'prefix':prefix,'record':json.loads(payload)})
            case={'name':name,'command':command,'returncode':result.returncode,'wall_seconds':time.perf_counter()-start,
                  'log_sha256':digest(out/(name+'.log')),'records':records}
            report['cases'].append(case);save()
            assert result.returncode==0,(name,(out/(name+'.log')).read_text()[-5000:])
            summary=[item['record'] for item in records if item['prefix']=='COMMON_KKT_SUMMARY']
            tests=[item['record'] for item in records if item['prefix']=='COMMON_KKT_TEST']
            assert len(summary)==1 and summary[0]['complete'] and summary[0]['solve_calls']==7 and len(tests)==7
        report['complete']=True;save()
except BaseException as error:
    report['error']=str(error);save();raise
finally:
    report['lock_scope_ended']=True;save()
    destination=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/common-kkt-v609d/tiny')
    shutil.copytree(out,destination)
print(json.dumps({'complete':report['complete'],'report':str(out/'report.json'),'report_sha256':digest(out/'report.json')}))
