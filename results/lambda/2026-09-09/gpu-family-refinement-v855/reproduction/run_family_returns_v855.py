"""Three fixed prescriptions, CUDA SCvx, exact-boundary reuse, original fleet gates."""
from pathlib import Path
import argparse, copy, dataclasses, fcntl, hashlib, json, os, sys, time, traceback
ROOT=Path(__file__).resolve().parent
HOME=Path.home()
RUNTIME=HOME/'spacepdhcg-return-cache-v844'
PRIOR=HOME/'spacepdhcg-family-departures-v850'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(RUNTIME/'repo/src'))
import numpy as np
from spacepdhcg.gtoc12 import pipeline as p
from spacepdhcg.gtoc12.fixed_refinement import refine_fixed
from spacepdhcg.gtoc12.refinement_admission import FixedCargoRequest
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.low_thrust import ScvxSettings
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.solution import Solution,ShipTrajectory
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier

def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):
    value=json.loads(json.dumps(value,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)),parse_constant=lambda x:x)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))
report=dict(pid=os.getpid(),complete=False,success=False,attempts=[],native_solves=0,cache_hits=0,fleet_checks=0,
            fixed_requests=3,incumbent_sha256=sha((PRIOR/'fleet.txt').read_bytes()),protocol='Use the +30-day deployment schedule from v854 rank 1 and its 16 certified flights. Test original Earth return +60, +120, +180 days; cold CUDA initialization for each changed return. Cargo fixed. Exact saved boundary/settings prefix reuse, fresh flight certificates. Stop at first gain passing both full-fleet checkers; no changed physics tolerance.')
def save():
    write(ROOT/'report.tmp',report);(ROOT/'report.tmp').replace(ROOT/'report.json')
save()
started=time.perf_counter();original_solve=p.solve_leg
try:
    assert report['incumbent_sha256']=='1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da'
    prior=json.loads((PRIOR/'report.json').read_text());assert prior['complete'] and prior['success']
    catalogue,bonus=load_catalogue(),load_bonus_table()
    fleet=Solution.read(PRIOR/'fleet.txt')
    settings=ScvxSettings(max_iterations=40,node_days=2.,discretisation_backend='cuda',assembly_backend='cuda',
        convex_solver_backend='qoco',qoco_ruiz_iterations=0,outer_loop_backend='cuda',seed_backend='cuda',
        certification_backend='cuda',ephemeris_backend='cuda',time_limit_s=120.)
    report.update(settings=dataclasses.asdict(settings),core_sha256=sha(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()),
        qoco_sha256=sha(Path(os.environ['SPACEPDHCG_QOCO_LIBRARY']).read_bytes()),source_manifest_sha256=sha((RUNTIME/'source-manifest.json').read_bytes()))
    assert report['core_sha256']==prior['core_sha256']
    old_refine=HOME/'spacepdhcg-family-retiming-v854'
    old_report=json.loads((old_refine/'report.json').read_text())
    assert old_report['complete'] and old_report['success']
    assert old_report['attempts'][1]['certified_legs']==16 and old_report['attempts'][1]['legs']==17
    assert dataclasses.asdict(settings)==old_report['settings']
    original=json.loads((old_refine/'requests.json').read_text())[1]['plan']
    unit=[];jobs=[]
    for rank,delay in enumerate((60.,120.,180.)):
        summary=copy.deepcopy(original)
        summary['legs'][-1]['tf']+=delay
        summary['earth_return_epoch']+=delay
        assert summary['legs'][-1]['to']==0 and summary['earth_return_epoch']<=p.C.MISSION_END_MJD
        unit.append(summary)
        cargo={int(k):v for k,v in summary['collected_mass_kg'].items()}
        request=FixedCargoRequest.from_summary(summary,cargo)
        jobs.append((rank,RoutePlan.from_summary(summary),request.sha256))
    assert len({j[2] for j in jobs})==3
    report['return_policy']=dict(arrivals_mjd=[u['earth_return_epoch'] for u in unit],
        prefix_source='v854 rank 1, 16 certified flights',cargo_unchanged=True,proxy_costs_recomputed=False)
    write(ROOT/'requests.json',[dict(rank=r,request_sha256=d,plan=unit[r]) for r,_,d in jobs])
    cache={}
    prefix_dirs=[HOME/'spacepdhcg-family-refinement-v851'/f'solve-{i:03d}' for i in (0,1)]
    # v854 rank 0 performs solves 0..14; rank 1 performs 15..29.
    # Rank 1 flights 2..15 therefore map to native solves 15..28.
    prefix_dirs.extend(old_refine/f'solve-{i:03d}' for i in range(15,29))
    for number,base in enumerate(prefix_dirs):
        boundary=json.loads((base/'boundary.json').read_text())
        for k in ('departure_position','departure_velocity','arrival_position','arrival_velocity'):boundary[k]=np.asarray(boundary[k])
        boundary=p.LegBoundary(**boundary)
        fields=json.loads((base/'solution.json').read_text())
        assert fields['status'] in ('converged','iteration_limit') and fields['max_defect']<=settings.defect_tolerance
        record=json.loads((old_refine/f'rank-001/leg-{number:02d}.json').read_text())
        assert record['certified'] and record['departure']==boundary.departure_epoch and record['arrival']==boundary.arrival_epoch
        assert record['initial_mass_kg']==boundary.initial_mass
        z=np.load(base/'raw.npz')
        fields.update(boundary=boundary,node_epochs_mjd=z['epochs'],thrust_n=z['thrust'],states_scaled=z['states'],
            departure_vinf_km_s=z['departure_vinf'],arrival_vinf_km_s=z['arrival_vinf'])
        sol=p.LegSolution(**fields)
        encoded=json.dumps(dataclasses.asdict(boundary),sort_keys=True,default=lambda v:v.tolist(),allow_nan=False).encode()
        key=sha(encoded+json.dumps(dataclasses.asdict(settings),sort_keys=True).encode())
        cache[key]=(str(base),sol)
    assert len(cache)==16
    with (HOME/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_gpu_execution(argparse.Namespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)):
            for rank,plan,digest in jobs:
                report.update(stage='refine',rank=rank);save()
                out=ROOT/f'rank-{rank:03d}';out.mkdir()
                def measured(boundary,scvx,**kwargs):
                    assert not kwargs,'Cold GPU initializer is the fixed policy'
                    encoded=json.dumps(dataclasses.asdict(boundary),sort_keys=True,default=lambda v:v.tolist(),allow_nan=False).encode()
                    key=sha(encoded+json.dumps(dataclasses.asdict(scvx),sort_keys=True).encode())
                    if key in cache:
                        report['cache_hits']+=1
                        with (out/'cache.jsonl').open('a') as f:f.write(json.dumps(dict(boundary_sha256=key,source_solve=cache[key][0]))+'\n')
                        save();return copy.deepcopy(cache[key][1])
                    serial=report['native_solves'];report['native_solves']+=1
                    prefix=ROOT/f'solve-{serial:03d}';prefix.mkdir()
                    write(prefix/'boundary.json',dataclasses.asdict(boundary));save()
                    sol=original_solve(boundary,scvx)
                    np.savez_compressed(prefix/'raw.npz',epochs=sol.node_epochs_mjd,thrust=sol.thrust_n,states=sol.states_scaled,
                                        departure_vinf=sol.departure_vinf_km_s,arrival_vinf=sol.arrival_vinf_km_s)
                    write(prefix/'solution.json',{k:v for k,v in dataclasses.asdict(sol).items() if k not in ('boundary','node_epochs_mjd','thrust_n','states_scaled','departure_vinf_km_s','arrival_vinf_km_s')})
                    cache[key]=(serial,copy.deepcopy(sol));return sol
                p.solve_leg=measured
                def on_leg(index,item,record):
                    write(out/f'leg-{index:02d}.json',record)
                    if item.solution is not None:
                        np.savez_compressed(out/f'leg-{index:02d}.npz',epochs=item.solution.node_epochs_mjd,
                            thrust=item.solution.thrust_n,states=item.solution.states_scaled)
                    report['last_leg']=dict(rank=rank,index=index,certified=item.certified);save()
                route=refine_fixed(plan,catalogue,dict(plan.collected_mass),settings=settings,on_leg=on_leg,deadline=started+600.)
                write(out/'route.json',route.summary())
                entry=dict(rank=rank,request_sha256=digest,certified=route.certified,legs=len(route.legs),
                           certified_legs=sum(l.certified for l in route.legs),seconds=route.wall_seconds,failures=route.failures)
                if route.certified:
                    p.write_route_artifacts(route,catalogue,out)
                    replacement=Solution.read(out/'Result.txt').ships[0]
                    path=out/'fleet.txt'
                    Solution([ShipTrajectory(s.ship_id,replacement.items if s.ship_id==18 else s.items) for s in fleet.ships]).write(path)
                    check=Gtoc12Verifier(catalogue,bonus=bonus).verify_file(path)
                    official=run_official_verifier(path,keep_directory=out/'official')
                    report['fleet_checks']+=2
                    (out/'official.stdout').write_text(official.stdout);(out/'official.stderr').write_text(official.stderr)
                    entry.update(independent=check.summary(),official=official.summary(),result_sha256=sha(path.read_bytes()),
                        verified_gain=check.ok and official.ok and check.weighted_score_fixed_bonus_kg>prior['fleet_weighted_kg']+.1)
                report['attempts'].append(entry);save()
                if entry.get('verified_gain'):break
    report['success']=True
except BaseException:
    report['error']=traceback.format_exc()
finally:
    p.solve_leg=original_solve
report.update(complete=True,seconds=time.perf_counter()-started);save()
