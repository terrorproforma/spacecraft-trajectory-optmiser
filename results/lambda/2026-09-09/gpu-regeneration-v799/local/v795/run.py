from pathlib import Path
import argparse,dataclasses,fcntl,hashlib,json,math,os,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent;prior=home/'spacepdhcg-raw-regeneration-v794'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-grid-v788/repo/src'))
import numpy as np
from spacepdhcg.gtoc12 import pipeline as p
from spacepdhcg.gtoc12.low_thrust import ScvxSettings,ZohTrajectorySeed
from spacepdhcg.gtoc12.solution import Solution,ShipTrajectory,BurnArc
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
report=dict(pid=os.getpid(),complete=False,success=False,attempts=[],native_solves=0)
def save():
    t=root/'report.tmp';t.write_text(json.dumps(report,indent=2,default=float));t.replace(root/'report.json')
save();solve=p.solve_leg;started=time.perf_counter()
try:
    while not json.loads((prior/'report.json').read_text())['complete']:time.sleep(2)
    started=time.perf_counter();state=json.loads((prior/'report.json').read_text());assert state['success']
    fleet=Solution.read(prior/'fleet.txt');catalogue,bonus=load_catalogue(),load_bonus_table()
    digest=hashlib.sha256((prior/'fleet.txt').read_bytes()).hexdigest();weights={int(a):float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    scvx=ScvxSettings(max_iterations=40,node_days=2.,discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',qoco_ruiz_iterations=0,outer_loop_backend='cuda',seed_backend='cuda',certification_backend='cuda')
    report.update(source_solution_sha256=digest,core_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),qoco_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_QOCO_LIBRARY']).read_bytes()).hexdigest(),scvx=dataclasses.asdict(scvx));save()
    jobs=[]
    for row in state['routes']:
        ship_id=row['ship'];ship=fleet.ships[ship_id-1]
        raw=sum(max(0.,e.after.mass-e.before.mass) for e in ship.asteroid_visits())
        plans=[RoutePlan.from_summary(s) for s in json.loads((prior/f"ship-{ship_id:02d}/plans.json").read_text())]
        seen=set();accepted=[]
        for rank,plan in enumerate(plans):
            score=sum(weights[a]*m for a,m in plan.collected_mass.items())
            if sum(plan.collected_mass.values())<=raw+.1 or score<row['incumbent_weighted_kg']-30:continue
            signature=tuple((l.from_id,l.to_id,l.departure_epoch,l.arrival_epoch) for l in plan.legs)
            if signature in seen:continue
            accepted.append((ship_id,rank,plan,score,row['incumbent_weighted_kg']));seen.add(signature)
            if len(accepted)>=2:break
        jobs.extend(accepted)
    jobs.sort(key=lambda j:-(j[3]-j[4]))
    report['jobs']=[dict(ship=j,rank=k,surrogate_kg=s,incumbent_kg=i) for j,k,_,s,i in jobs];save()
    from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
    from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace
    columns=[];artifacts={};warm=[]
    for ship in fleet.ships:
        dep={};col={};mass={}
        for event in ship.asteroid_visits():
            delta=event.after.mass-event.before.mass
            if delta<0:
                assert abs(delta+40.)<1e-7 and event.event_id not in dep
                dep[event.event_id]=event.epoch
            elif delta>0:
                assert event.event_id not in col
                col[event.event_id]=event.epoch;mass[event.event_id]=delta
        assert set(col)==set(dep)
        c=FleetColumn(ship.ship_id,ship.ship_id,'verified incumbent',dep,col,{},mass,True)
        columns.append(c);warm.append(c);artifacts[c.identifier]=ship
    assert fleet_feasible(warm)==''
    replacements={}
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_gpu_execution(argparse.Namespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)):
            for ship_id,rank,plan,score,incumbent in jobs:
                ship=fleet.ships[ship_id-1];first=ship.asteroid_visits()[0]
                def measured(boundary,settings):
                    begin=time.perf_counter();seed=None;record=dict(ship=ship_id,rank=rank,t0=boundary.departure_epoch,tf=boundary.arrival_epoch,number=report['native_solves']+1)
                    if boundary.departure_epoch==ship.launch.epoch and boundary.arrival_epoch==first.epoch:
                        whole=math.floor(boundary.duration_days/settings.node_days+1e-9);days=np.arange(whole+1)*settings.node_days
                        if boundary.duration_days-days[-1]>1e-9:days=np.append(days,boundary.duration_days)
                        epochs=boundary.departure_epoch+days;u=np.zeros((len(epochs),3));used=np.zeros(len(epochs)-1,bool)
                        for arc in ship.burns:
                            if arc.start>=first.epoch or arc.end<=ship.launch.epoch:continue
                            assert arc.start>=ship.launch.epoch and arc.end<=first.epoch
                            vals=np.asarray([s.thrust for s in arc.interior]);assert len(vals)>0 and np.all(vals==vals[0])
                            a=np.flatnonzero(epochs==arc.start);b=np.flatnonzero(epochs==arc.end)
                            assert len(a)==len(b)==1 and b[0]>a[0] and not used[a[0]:b[0]].any()
                            u[a[0]:b[0]]=vals[0];used[a[0]:b[0]]=True
                        initial=np.r_[boundary.departure_position,ship.launch.after.velocity,boundary.initial_mass]
                        seed=ZohTrajectorySeed(epochs,initial,u,digest)
                        record.update(seed='archived_zoh',position_serialization_difference_km=float(np.linalg.norm(boundary.departure_position-ship.launch.after.position)),seed_nodes=len(epochs))
                    try:
                        result=solve(boundary,settings,seed=seed) if seed is not None else solve(boundary,settings)
                        record.update(status=result.status,iterations=result.iterations,seed_backend=result.seed_backend);return result
                    except BaseException as e:record['error']=repr(e);raise
                    finally:
                        record['seconds']=time.perf_counter()-begin;report['native_solves']+=1
                        with (root/'solves.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        save()
                p.solve_leg=measured
                report.update(stage='refine',ship=ship_id,rank=rank);save();before=time.perf_counter()
                route=p.refine_route(plan,catalogue,scvx=scvx)
                out=root/f'ship-{ship_id:02d}-rank-{rank:03d}';out.mkdir()
                summary=route.summary();(out/'route_summary.json').write_text(json.dumps(summary,indent=2,default=float))
                weighted=sum(weights[a]*m for a,m in route.collected_mass.items())
                entry=dict(ship=ship_id,rank=rank,certified=route.certified,surrogate_kg=score,actual_weighted_kg=weighted,incumbent_kg=incumbent,seconds=time.perf_counter()-before,failures=route.failures)
                if route.certified:
                    p.write_route_artifacts(route,catalogue,out)
                    c=FleetColumn.from_plan(1000+len(columns),ship_id,f'ship-{ship_id}-rank-{rank}',route.plan,route.collected_mass,certified=route.certified)
                    columns.append(c);artifacts[c.identifier]=Solution.read(out/'Result.txt').ships[0]
                report['attempts'].append(entry);save()
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with CudaFleetWorkspace(columns,weights=weights) as master:
            selection=master.solve(incumbent=warm,max_ships=23,node_cap=2_000_000,exchange_rounds=32)
    report['master']=selection.summary();report['selected_ids']=[c.identifier for c in selection.selected]
    report['replacements']=[c.slot for c in selection.selected if c.identifier>=1000]
    out=root/'fleet';out.mkdir();path=out/'Result.txt'
    Solution([ShipTrajectory(i,artifacts[c.identifier].items) for i,c in enumerate(sorted(selection.selected,key=lambda c:c.slot),1)]).write(path)
    check=Gtoc12Verifier(catalogue,bonus=bonus).verify_file(path);official=run_official_verifier(path,keep_directory=out/'official')
    report.update(independent=check.summary(),official=official.summary(),score_kg=check.weighted_score_fixed_bonus_kg,qualified=check.ok and official.ok,solution_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'official.stdout').write_text(official.stdout);(out/'official.stderr').write_text(official.stderr)
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
finally:p.solve_leg=solve
report.update(complete=True,seconds=time.perf_counter()-started);save()
