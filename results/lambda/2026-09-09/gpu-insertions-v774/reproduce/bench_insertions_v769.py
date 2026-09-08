from pathlib import Path
import dataclasses,fcntl,hashlib,json,os,statistics,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent;source=home/'spacepdhcg-insertions-v768'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(source/'repo/src'))
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.jointopt import JointItinerary,JointSettings,route_from_summary
from spacepdhcg.gtoc12.bundles import ClusterPricingSettings,cluster_search_settings,cluster_retime_settings
from spacepdhcg.gtoc12.retiming import Retimer,visits_of
from spacepdhcg.gtoc12.clusters import ClusterBands
from spacepdhcg.gtoc12.returnsweep import neighbourhood
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.gpu_joint_insertions import insertions
report=dict(pid=os.getpid(),complete=False,success=False,trials=[])
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save()
try:
    os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_BATCH']='1';os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY']='1'
    os.environ.pop('SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_INSERTIONS',None)
    report['core_sha256']=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest()
    catalogue,bonus=load_catalogue(),load_bonus_table();weights={int(a):float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    old=home/'spacepdhcg-fleet-routes-v766';pool=json.loads((old/'input/pool.json').read_text());campaign=json.loads((old/'input/campaign.json').read_text());selected=campaign['selected']
    occupied={r['identifier']:set(map(int,r['deploys']))|set(map(int,r['collects'])) for r in pool['rows'] if r['identifier'] in selected}
    pricing=ClusterPricingSettings(collect_dp_inflation_fit=str(old/'input/fit.json'))
    search=cluster_search_settings(pricing,60);retime=cluster_retime_settings(pricing,last=True)
    fixtures=[]
    for identifier in (1786,2297):
        path=old/'routes'/str(identifier)/'route_summary.json';route=route_from_summary(json.loads(path.read_text()))
        joint=JointItinerary(catalogue,Retimer(catalogue,search,retime,weights),weights=weights)
        assert joint.learn(route)==len(route.legs)
        visits,arr,dep=visits_of(route.plan);arr=np.asarray(arr);dep=np.asarray(dep)
        bands=dataclasses.replace(getattr(search,'cluster_bands',None) or ClusterBands.collect_window(),radius=JointSettings().insert_radius)
        candidates=neighbourhood(catalogue,route.plan,SimpleNamespace(cluster_bands=bands),count=60).tolist()
        banned=set(route.plan.asteroids)|set().union(*(v for k,v in occupied.items() if k!=identifier))
        candidates=[int(a) for a in candidates if int(a) not in banned]
        fixtures.append((identifier,joint,visits,arr,dep,candidates))
    report['fixtures']=[dict(identifier=f[0],candidates=f[-1]) for f in fixtures];save()
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_lambert_backend('cuda') as gpu:
            for repeat in range(6):
                for identifier,template,visits,arr,dep,candidates in fixtures:
                    outputs={}
                    modes=['scalar','native'] if repeat%2==0 else ['native','scalar']
                    for mode in modes:
                        joint=deepcopy(template);before=dict(gpu.telemetry)
                        started=time.perf_counter()
                        result=joint.insertions(visits,arr,dep,candidates) if mode=='scalar' else insertions(joint,visits,arr,dep,candidates)
                        seconds=time.perf_counter()-started
                        serial=[dict(asteroid=r[4],arr=r[1].tolist(),dep=r[2].tolist(),evaluation=dataclasses.asdict(r[3])) for r in result]
                        signature=hashlib.sha256(json.dumps(serial,sort_keys=True).encode()).hexdigest()
                        outputs[mode]=(joint.evaluations,len(result),signature)
                        report['trials'].append(dict(repeat=repeat,identifier=identifier,mode=mode,seconds=seconds,evaluations=joint.evaluations,feasible=len(result),signature=signature,telemetry={k:v-before.get(k,0) for k,v in gpu.telemetry.items() if v!=before.get(k,0)}));save()
                    assert outputs['scalar']==outputs['native'],outputs
    report['medians']={str(identifier):{mode:statistics.median(t['seconds'] for t in report['trials'] if t['identifier']==identifier and t['mode']==mode and t['repeat']>0) for mode in ('scalar','native')} for identifier,*_ in fixtures}
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
