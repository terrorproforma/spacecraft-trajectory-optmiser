from pathlib import Path
import collections,dataclasses,fcntl,hashlib,json,os,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-grid-v786/repo/src'))
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
from spacepdhcg.gtoc12.gpu_joint_layouts import PreparedInsertions
from spacepdhcg.gtoc12.gpu_joint import FAILURES,_decode_evaluation
report=dict(pid=os.getpid(),complete=False,success=False,trials=[])
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save()
try:
    os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_BATCH']='1'
    os.environ['SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY']='1'
    catalogue,bonus=load_catalogue(),load_bonus_table();weights={int(a):float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    old=home/'spacepdhcg-fleet-routes-v779';inputs=home/'spacepdhcg-fleet-routes-v766/input'
    pool=json.loads((inputs/'pool.json').read_text());campaign=json.loads((inputs/'campaign.json').read_text());selected=campaign['selected']
    occupied={r['identifier']:set(map(int,r['deploys']))|set(map(int,r['collects'])) for r in pool['rows'] if r['identifier'] in selected}
    pricing=ClusterPricingSettings(collect_dp_inflation_fit=str(inputs/'fit.json'))
    search=cluster_search_settings(pricing,60);retime=cluster_retime_settings(pricing,last=True)
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_lambert_backend('cuda') as gpu:
            report['core_sha256']=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest()
            for identifier in (1786,2297):
                route=route_from_summary(json.loads((old/'routes'/str(identifier)/'route_summary.json').read_text()))
                template=JointItinerary(catalogue,Retimer(catalogue,search,retime,weights),weights=weights);template.learn(route)
                visits,arr,dep=visits_of(route.plan);arr=np.asarray(arr);dep=np.asarray(dep)
                report.setdefault('routes',{})[identifier]=dict(visits=[dict(body=v.body,deploy=v.deploy,collect=v.collect,dwell=float(dep[j]-arr[j])) for j,v in enumerate(visits)],weighted_kg=sum(weights[a]*m for a,m in route.collected_mass.items()))
                banned=set(route.plan.asteroids)|set().union(*(v for k,v in occupied.items() if k!=identifier))
                for radius,count,points in ((2.5,60,1),(2.5,60,3),(4.,600,3),(8.,3000,3),(8.,3000,5)):
                    bands=dataclasses.replace(getattr(search,'cluster_bands',None) or ClusterBands.collect_window(),radius=radius)
                    begin=time.perf_counter();ids=neighbourhood(catalogue,route.plan,SimpleNamespace(cluster_bands=bands),count=count).tolist()
                    candidates=[int(a) for a in ids if int(a) not in banned]
                    joint=deepcopy(template);before=dict(gpu.telemetry)
                    prepared=PreparedInsertions(joint,visits,arr,dep,candidates,layouts_per_batch=max(1,4096//points**2),split_points=points)
                    failures=collections.Counter();best=[]
                    for first in range(0,prepared.total,prepared.batch_size):
                        size=min(prepared.batch_size,prepared.total-first)
                        values,enabled,aa,dd,mm,ii,pp,cc,_,_=prepared.run(first,size)
                        failures.update(int(v) for v in values['failure'][enabled.astype(bool)])
                        for row in np.flatnonzero(enabled & (values['failure']==0)):
                            evvisits,asteroid=prepared.layout(first+row//prepared.rows_per_layout)
                            ev=_decode_evaluation(joint,evvisits,aa[row],dd[row],values[row],mm[row],ii[row],pp[row],cc[row])
                            best.append(dict(asteroid=asteroid,objective=ev.objective,weighted_kg=ev.weighted_kg,arrivals=aa[row].tolist(),departures=dd[row].tolist(),plan=dataclasses.asdict(ev.plan)))
                    best.sort(key=lambda x:(-x['objective'],x['asteroid']))
                    record=dict(identifier=identifier,split_points=points,radius=radius,requested_neighbours=count,neighbourhood=len(ids),candidates=len(candidates),layouts=prepared.total,evaluations=joint.evaluations,seconds=time.perf_counter()-begin,failures={FAILURES[k] if k<len(FAILURES) else str(k):v for k,v in failures.items()},feasible=len(best),top=best[:10],telemetry={k:v-before.get(k,0) for k,v in gpu.telemetry.items() if v!=before.get(k,0)})
                    report['trials'].append(record);save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
