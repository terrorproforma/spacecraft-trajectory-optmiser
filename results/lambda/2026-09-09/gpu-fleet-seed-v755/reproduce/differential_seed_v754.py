from pathlib import Path
import dataclasses,fcntl,hashlib,importlib.util,json,os,sys,traceback
home=Path.home();root=home/'spacepdhcg-fleet-differential-v754';root.mkdir(exist_ok=False)
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-fleet-v750/repo/src'))
import numpy as np
from spacepdhcg.gtoc12.cooperative import FleetColumn
from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
spec=importlib.util.spec_from_file_location('spacepdhcg.gtoc12.old_seed',home/'spacepdhcg-fleet-v742/repo/src/spacepdhcg/gtoc12/gpu_fleet.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
report=dict(pid=os.getpid(),complete=False,success=False,python=sys.version,numpy=np.__version__,cases=[])
report['libraries']={v:hashlib.sha256((home/f'spacepdhcg-fleet-{v}/final/libspacepdhcg_cuda.so').read_bytes()).hexdigest() for v in ('v742','v750')}
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
def col(i,ds,cs,mass,foreign=None):
    return FleetColumn(i,i,str(i),dict.fromkeys(ds,65000.),dict.fromkeys(cs,66000.),foreign or {},dict.fromkeys(cs,mass/max(1,len(cs))),True)
save()
try:
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for seed in range(128):
            rng=np.random.default_rng(seed);columns=[]
            for i in range(int(rng.integers(2,45))):
                ids=rng.choice(60,size=int(rng.integers(1,5)),replace=False).tolist()
                c=col(i,ids,ids,float(rng.choice([0,20,200,500,1000,1400])))
                if rng.random()<.08:c=dataclasses.replace(c,certified=False)
                columns.append(c)
            columns.extend([col(100,[100],[101],700,{101:65000.}),col(101,[101],[100],800,{100:65000.}),col(102,[102],[103],900,{103:65000.}),col(103,[103],[104],700,{999:65000.})])
            a,b=col(110,[110],[110],700),col(111,[111],[111],800)
            columns.extend([a,b,FleetColumn.from_bundle(112,'pair',[a,b])])
            weights={i:float(rng.choice([-.5,0,.3,1,2])) for i in range(120)}
            warm=tuple(columns[i] for i in rng.choice(len(columns),size=min(12,len(columns)),replace=False))
            options=dict(max_ships=[0,1,2,5,10,100][seed%6],node_cap=[0,1,17,1024][seed%4],prefix_bits=3,exchange_rounds=[0,1,16][seed%3])
            outputs=[]
            for version,fn in [('v742',old.solve_fleet_cuda),('v750',solve_fleet_cuda)]:
                os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(home/f'spacepdhcg-fleet-{version}/final/libspacepdhcg_cuda.so')
                r=fn(columns,weights=weights,incumbent=warm,**options)
                outputs.append(dict(ids=[c.identifier for c in r.selected],objective=r.objective,bound=r.upper_bound,greedy=r.greedy_objective,nodes=r.nodes,exhaustive=r.exhaustive,proposals=r.exchange_proposals,moves=r.exchange_moves,rounds=r.exchange_rounds))
            assert outputs[0]==outputs[1],(seed,outputs)
            report['cases'].append(dict(seed=seed,columns=len(columns),options=options,result=outputs[0]));save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
