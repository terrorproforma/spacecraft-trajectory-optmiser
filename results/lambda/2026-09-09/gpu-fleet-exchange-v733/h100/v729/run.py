from pathlib import Path
import argparse,dataclasses,fcntl,hashlib,json,os,sys,time,traceback
home=Path.home();root=home/'spacepdhcg-fleet-refine-v729';repo=home/'spacepdhcg-fleet-v727/repo'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.low_thrust import ScvxSettings
from spacepdhcg.gtoc12.pipeline import plan_from_route_summary,refine_route,write_route_artifacts
from spacepdhcg.gtoc12.solution import Solution,ShipTrajectory
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset
from spacepdhcg.gtoc12 import gpu_scvx
report=dict(pid=os.getpid(),complete=False,success=False,refinements=[],native_solves=0,
            core_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_QOCO_LIBRARY']).read_bytes()).hexdigest())
def save():
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2,default=float));tmp.replace(root/'report.json')
save();start=time.perf_counter();native=gpu_scvx.solve_native
def measured(*args,**kwargs):
    begin=time.perf_counter();record=dict(column=report.get('column'),number=report['native_solves']+1)
    try:
        r=native(*args,**kwargs);record.update(status=r.status,iterations=r.iterations,accepted=r.accepted_iterations);return r
    except BaseException as e:record['error']=repr(e);raise
    finally:
        record['seconds']=time.perf_counter()-begin;report['native_solves']+=1
        with (root/'native-solves.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        save()
gpu_scvx.solve_native=measured
try:
    catalogue,bonus=load_catalogue(),load_bonus_table();report['bonus_sha256']=bonus.source_sha256
    settings=ScvxSettings(max_iterations=40,node_days=2.0,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',qoco_ruiz_iterations=0,outer_loop_backend='cuda',seed_backend='cuda',certification_backend='cuda')
    report['settings']=dataclasses.asdict(settings);save();solutions={}
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_gpu_execution(argparse.Namespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)):
            for identifier in (1786,2297):
                report.update(stage='native_refinement',column=identifier);save();began=time.perf_counter()
                plan=plan_from_route_summary(json.loads((root/'input'/f'{identifier}.json').read_text()))
                route=refine_route(plan,catalogue,scvx=settings)
                out=root/'routes'/str(identifier);out.mkdir(parents=True)
                (out/'summary.json').write_text(json.dumps(route.summary(),indent=2,default=float))
                report['refinements'].append(dict(identifier=identifier,seconds=time.perf_counter()-began,certified=route.certified,summary=route.summary()));save()
                if route.certified:
                    write_route_artifacts(route,catalogue,out);solutions[identifier]=Solution.read(out/'Result.txt').ships[0]
    if len(solutions)!=2:raise RuntimeError('Replacement route failed CUDA refinement; candidate not promoted')
    report.update(stage='assemble');save()
    pool=json.loads((root/'input/pool.json').read_text());selected=json.loads((root/'input/selected.json').read_text())
    incumbent=Solution.read(root/'input/incumbent.txt');ships=[]
    for identifier in selected:
        row=next(r for r in pool['rows'] if r['identifier']==identifier)
        ship=solutions.get(identifier) if identifier in solutions else next(s for s in incumbent.ships if s.ship_id==row['ship_id'])
        if identifier not in solutions and identifier not in pool['warm']:raise ValueError('Unexpected unverified source')
        ships.append(ShipTrajectory(len(ships)+1,ship.items))
    fleet=root/'fleet';fleet.mkdir();path=fleet/'Result.txt';Solution(ships).write(path)
    report.update(stage='independent_full_fleet');save();histories={};t=time.perf_counter()
    independent=Gtoc12Verifier(catalogue,bonus=bonus,history=histories).verify_file(path)
    report['independent']=independent.summary();report['independent_seconds']=time.perf_counter()-t;save()
    report.update(stage='official_full_fleet');save();official=run_official_verifier(path,keep_directory=fleet/'official')
    report['official']=official.summary();(fleet/'official.stdout').write_text(official.stdout);(fleet/'official.stderr').write_text(official.stderr)
    report['score_kg']=independent.weighted_score_fixed_bonus_kg;report['total_mass_kg']=independent.total_mass_kg
    report['qualified']=independent.ok and official.ok;report['improved']=report['qualified'] and report['score_kg']>12810.135953048562+1e-6
    report['solution_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();save()
    if report['qualified']:
        report.update(stage='viewer_export');save()
        report['viewer_manifest']=write_viewer_dataset(fleet/'viewer',Solution.read(path),histories,catalogue,run_id='cuda_fleet_exchange_v729',commit='experimental',verification=independent.summary(),solution_path=path)
    report['success']=report['improved']
except BaseException:report['error']=traceback.format_exc()
report.update(complete=True,seconds=time.perf_counter()-start);save();print(json.dumps({k:v for k,v in report.items() if k not in ('refinements','viewer_manifest')},default=float))
