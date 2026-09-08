from pathlib import Path
import fcntl,json,os,subprocess,time
root=Path('build/performance/bound-types-native-v573');root.mkdir(exist_ok=False)
core=Path('/home/angus/build-spacepdhcg-bound-types-v568/final/libspacepdhcg_cuda.so')
qoco=Path('/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so')
native=str(Path('build/performance/bound-types-v568/native-conversion-test').resolve())
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_TEST_QOCO_DEVICE_BOUND_TYPES='1',SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION='1',LD_LIBRARY_PATH=str(core.parent)+':'+str(qoco.parent)+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
report=dict(pid=os.getpid(),complete=False,stages=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
try:
 with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for ruiz in (0,2,5):
   for origin in ('plain','origin'):
    name=f'native-{ruiz}-{origin}';cmd=[native,str(ruiz),'device-validation',origin,'device-init']
    with (root/(name+'.log')).open('x') as log:
     start=time.perf_counter();child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);report.update(stage=name,child_pid=child.pid);save();code=child.wait(timeout=90)
    report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-start,command=cmd));save();assert code==0,(name,code)
 report['complete']=True
except Exception as error:report['error']=repr(error)
save()
