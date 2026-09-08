from pathlib import Path
import ctypes as ct,fcntl,hashlib,json,os,subprocess,sys,traceback
home=Path.home();root=home/'spacepdhcg-fleet-profile-v746';root.mkdir(exist_ok=False)
base=home/'spacepdhcg-fleet-v742/repo';remote=home.name=='ubuntu'
cuda='/usr/local/cuda' if remote else '/usr/local/cuda-12.8'
source=Path(__file__).with_suffix('.cu');(root/'profile.cu').write_bytes(source.read_bytes())
report=dict(pid=os.getpid(),complete=False,success=False,rows=[])
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
save()
try:
    cmd=[cuda+'/bin/nvcc','-O3','-std=c++17','--shared','-Xcompiler','-fPIC','-arch=sm_'+('90' if remote else '120'),'-I'+str(base/'cpp/cuda/include'),'-I'+str(base/'cpp/cuda/src'),str(root/'profile.cu'),'-o',str(root/'libprofile.so')]
    with (root/'build.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    report['build_command']=cmd;report['library_sha256']=hashlib.sha256((root/'libprofile.so').read_bytes()).hexdigest()
    sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(base/'src'))
    os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(root/'libprofile.so')
    import numpy as np
    from spacepdhcg.gtoc12.cooperative import FleetColumn
    from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace,REPORT
    pool=home/'spacepdhcg-fleet-pool-v716/pool.json';data=json.loads(pool.read_text())
    columns=[FleetColumn(r['identifier'],r['ship_id'],r['label'],*({int(k):v for k,v in r[key].items()} for key in ('deploys','collects','foreign','mass')),True) for r in data['rows']]
    weights={int(k):v for k,v in data['weights'].items()};warm_ids=set(data['warm']);incumbent=tuple(c for c in columns if c.identifier in warm_ids)
    report['pool_sha256']=hashlib.sha256(pool.read_bytes()).hexdigest();save()
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with CudaFleetWorkspace(columns,weights=weights) as workspace:
            profile=workspace._library.profile_fleet_stages;profile.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_uint64,ct.c_void_p,ct.c_void_p,ct.c_void_p];profile.restype=ct.c_int
            warm=np.asarray([c.identifier in warm_ids for c in workspace._columns],dtype=np.uint8)
            for cap in (0,200000):
                for rounds in (0,16):
                    expected=workspace.solve(incumbent=incumbent,node_cap=cap,exchange_rounds=rounds)
                    for repeat in range(3):
                        timings=np.zeros(4+2*rounds);selected=np.empty(len(warm),dtype=np.uint8);stats=np.zeros(1,dtype=REPORT)
                        status=profile(workspace._handle,warm.ctypes.data,rounds,cap,timings.ctypes.data,selected.ctypes.data,stats.ctypes.data);assert status==0,status
                        ids=sorted(c.identifier for i,c in enumerate(workspace._columns) if selected[i]);assert ids==[c.identifier for c in expected.selected]
                        assert stats[0]['objective']==expected.objective and stats[0]['nodes']==expected.nodes
                        report['rows'].append(dict(cap=cap,rounds=rounds,repeat=repeat,seed_ms=timings[0],start_ms=timings[1],evaluate_ms=timings[2:2+2*rounds:2].tolist(),accept_ms=timings[3:2+2*rounds:2].tolist(),search_ms=timings[-2],finish_ms=timings[-1],sum_ms=float(sum(timings))));save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
