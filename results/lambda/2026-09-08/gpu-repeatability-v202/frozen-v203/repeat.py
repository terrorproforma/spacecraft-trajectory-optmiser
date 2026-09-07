from pathlib import Path
import os,sys,json,time,fcntl,hashlib,ast
root=Path('/home/ubuntu/spacepdhcg-frozen-v203');integrated=Path('/home/ubuntu/spacepdhcg-collection-full-v198');repo=integrated/'repo'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
for key in list(os.environ):
 if key.startswith('SPACEPDHCG_TEST_'):del os.environ[key]
core=integrated/'core-build/cuda/libspacepdhcg_cuda.so';qoco=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so')
os.environ.update(SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
# cuDSS is dynamically opened by the adapter using its configured absolute paths.
import subprocess
binary=Path('/home/ubuntu/spacepdhcg-diagnose-v174/final/qoco_snapshot_replay')
qp=next(Path('/home/ubuntu/spacepdhcg-capture-v202/qp-0').glob('*000000.txt'))
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
report=dict(pid=os.getpid(),start=time.time(),rows=[],qp_sha256=hashlib.sha256(qp.read_bytes()).hexdigest())
for name,flags in [('default',{}),('factor_uncaptured',{'SPACEPDHCG_TEST_QOCO_FACTOR_GRAPH_DISABLE':'1'}),('ipm_graph',{'SPACEPDHCG_TEST_QOCO_IPM_GRAPH':'1'}),('initcheck',{})]:
 env=dict(os.environ,**flags)
 command=[str(binary),str(qp),'8' if name!='initcheck' else '2']
 if name=='initcheck':command=['/usr/local/cuda/bin/compute-sanitizer','--tool','initcheck','--error-exitcode','88',*command]
 started=time.monotonic();r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
 (root/(name+'.log')).write_text(r.stdout);(root/(name+'.err')).write_text(r.stderr)
 records=[json.loads(line[10:]) for line in r.stdout.splitlines() if line.startswith('QP_REPLAY ')]
 row=dict(name=name,returncode=r.returncode,seconds=time.monotonic()-started,records=[{k:v for k,v in x.items() if k not in ['x','y','z','s']} for x in records]);report['rows'].append(row)
 (root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
report.update(complete=True,end=time.time());(root/'report.json').write_text(json.dumps(report,indent=2))
