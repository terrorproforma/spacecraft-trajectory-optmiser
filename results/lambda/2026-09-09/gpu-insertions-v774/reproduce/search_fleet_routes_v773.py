from pathlib import Path
import argparse,dataclasses,fcntl,hashlib,json,os,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent;repo=home/'spacepdhcg-insertions-v772/repo'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.low_thrust import ScvxSettings
from spacepdhcg.gtoc12.jointopt import JointSettings,optimise_ship,route_from_summary
from spacepdhcg.gtoc12.bundles import ClusterPricingSettings,cluster_search_settings,cluster_retime_settings
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.pipeline import write_route_artifacts
from spacepdhcg.gtoc12.solution import Solution,ShipTrajectory
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset
from spacepdhcg.gtoc12 import gpu_scvx
report=dict(pid=os.getpid(),complete=False,success=False,ships=[],native_solves=0,core_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_QOCO_LIBRARY']).read_bytes()).hexdigest())
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2,default=float));tmp.replace(root/'report.json')
save();started=time.perf_counter();native=gpu_scvx.solve_native
def measured(*args,**kwargs):
    begin=time.perf_counter();r=None;record=dict(column=report.get('column'),number=report['native_solves']+1)
    try:
        r=native(*args,**kwargs);record.update(status=r.status,iterations=r.iterations,accepted=r.accepted_iterations);return r
    except BaseException as e:record['error']=repr(e);raise
    finally:
        record['seconds']=time.perf_counter()-begin;report['native_solves']+=1
        with (root/'native-solves.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        save()
gpu_scvx.solve_native=measured
try:
    for name,digest in json.loads((root/'input-hashes.json').read_text()).items():assert hashlib.sha256((root/'input'/name).read_bytes()).hexdigest()==digest,name
    catalogue,bonus=load_catalogue(),load_bonus_table();report['bonus_sha256']=bonus.source_sha256
    weights={int(i):float(bonus.coefficient[int(i)-1]) for i in catalogue.ids}
    original=json.loads((root/'input/campaign.json').read_text());pool=json.loads((root/'input/pool.json').read_text());selected=original['selected']
    assert hashlib.sha256((root/'input/incumbent.txt').read_bytes()).hexdigest()=='fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f'
    scvx=ScvxSettings(max_iterations=40,node_days=2.,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',qoco_ruiz_iterations=0,outer_loop_backend='cuda',seed_backend='cuda',certification_backend='cuda')
    settings=JointSettings(time_budget_seconds=180.,max_certifications=4,min_gain_kg=.1,insert=True,insert_neighbours=60,insert_trials=2,earth_leg=True,earth_leg_certifications=2)
    pricing=ClusterPricingSettings(collect_dp_inflation_fit=str(root/'input/fit.json'))
    search=cluster_search_settings(pricing,60);retime=cluster_retime_settings(pricing,last=True)
    report.update(scvx=dataclasses.asdict(scvx),joint=dataclasses.asdict(settings),baseline_score_kg=12842.970672270894);save()
    replacements={};occupied={i:set(map(int,r['deploys']))|set(map(int,r['collects'])) for i in selected for r in pool['rows'] if r['identifier']==i}
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_gpu_execution(argparse.Namespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)),using_lambert_backend('cuda') as gpu:
            for entry in original['refinements']:
                identifier=entry['identifier'];report.update(stage='joint_search',column=identifier);save()
                route=route_from_summary(entry['summary']);excluded=set().union(*(v for k,v in occupied.items() if k!=identifier))
                result=optimise_ship(route,catalogue,Retimer(catalogue,search,retime,weights),weights=weights,scvx=scvx,settings=settings,search_settings=search,excluded=excluded)
                record=dict(identifier=identifier,**result.summary());report['ships'].append(record)
                if result.route is not None:
                    out=root/'routes'/str(identifier);out.mkdir(parents=True);write_route_artifacts(result.route,catalogue,out)
                    replacements[identifier]=Solution.read(out/'Result.txt').ships[0]
                    occupied[identifier]=set(result.route.plan.deploy_epochs)|set(result.route.plan.collect_epochs)
                report['gpu_telemetry']=dict(gpu.telemetry);save()
    report.update(stage='assemble',replaced=sorted(replacements));save()
    incumbent=Solution.read(root/'input/incumbent.txt');ships=[]
    assert len(incumbent.ships)==len(selected)
    for i,(identifier,ship) in enumerate(zip(selected,incumbent.ships,strict=True),1):
        value=replacements.get(identifier,ship);ships.append(ShipTrajectory(i,value.items))
    fleet=root/'fleet';fleet.mkdir();path=fleet/'Result.txt';Solution(ships).write(path)
    report.update(stage='independent_full_fleet');save();histories={};begin=time.perf_counter()
    check=Gtoc12Verifier(catalogue,bonus=bonus,history=histories).verify_file(path)
    report.update(independent=check.summary(),independent_seconds=time.perf_counter()-begin,stage='official_full_fleet');save()
    official=run_official_verifier(path,keep_directory=fleet/'official');report['official']=official.summary()
    (fleet/'official.stdout').write_text(official.stdout);(fleet/'official.stderr').write_text(official.stderr)
    report.update(qualified=check.ok and official.ok,score_kg=check.weighted_score_fixed_bonus_kg,total_mass_kg=check.total_mass_kg,solution_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['improved']=report['qualified'] and report['score_kg']>report['baseline_score_kg']+1e-6
    if report['improved']:report['viewer_manifest']=write_viewer_dataset(fleet/'viewer',Solution.read(path),histories,catalogue,run_id='gpu_insertion_search_v773',commit='e80b9cc4',verification=check.summary(),solution_path=path)
    report['success']=report['qualified']
except BaseException:report['error']=traceback.format_exc()
report.update(complete=True,seconds=time.perf_counter()-started);save()
