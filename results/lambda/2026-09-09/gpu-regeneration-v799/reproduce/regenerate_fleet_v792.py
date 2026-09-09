from pathlib import Path
import cProfile,dataclasses,fcntl,gc,hashlib,json,math,os,pstats,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-grid-v788/repo/src'))
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.solution import Solution
from spacepdhcg.gtoc12.clusters import ClusterBands,ComovingClusters
from spacepdhcg.gtoc12.bundles import ClusterPricingSettings,cluster_search_settings
from spacepdhcg.gtoc12.search import RouteSearch,EarthLeg
from spacepdhcg.gtoc12.screening import screen_earth_to_asteroids
from spacepdhcg.gtoc12.lambert import using_lambert_backend
report=dict(pid=os.getpid(),complete=False,success=False,routes=[])
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2,default=float));p.replace(root/'report.json')
save()
try:
    for name,digest in json.loads((root/'input-hashes.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    fleet=Solution.read(root/'fleet.txt');catalogue,bonus=load_catalogue(),load_bonus_table()
    weights={int(a):float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    occupied={s.ship_id:{e.event_id for e in s.asteroid_visits()} for s in fleet.ships}
    cargo={s.ship_id:sum(weights[e.event_id]*max(0.,e.after.mass-e.before.mass) for e in s.asteroid_visits()) for s in fleet.ships}
    order=sorted(fleet.ships,key=lambda s:(cargo[s.ship_id],s.ship_id))
    families=ComovingClusters(catalogue,catalogue.ids,ClusterBands.collect_window(radius=4.))
    pricing=ClusterPricingSettings(beam_width=16,neighbours=48,max_deploys=10,collect_dp=True,collect_dp_inflation_fit=str(root/'fit.json'),harvest_substitution=False)
    report.update(core_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((home/'spacepdhcg-grid-v788/source-manifest.json').read_bytes()).hexdigest(),bonus_sha256=bonus.source_sha256,order=[s.ship_id for s in order],incumbent_weighted=cargo);save()
    for key in ('COMPLETION_BATCH','COMPLETION_NATIVE_MODEL'):os.environ['SPACEPDHCG_TEST_GTOC12_'+key]='1'
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for ship in order:
            begin=time.perf_counter();report.update(stage='route_search',ship=ship.ship_id);save()
            first=ship.asteroid_visits()[0];launch=ship.launch.epoch;tof=first.epoch-launch
            excluded=set().union(*(v for k,v in occupied.items() if k!=ship.ship_id))
            ids=np.asarray(sorted((set(map(int,families.neighbours(first.event_id)[:192]))|occupied[ship.ship_id])-excluded),np.int64)
            settings=dataclasses.replace(cluster_search_settings(pricing,len(ids)),time_budget_seconds=30.,first_level_window_days=0.)
            with using_lambert_backend('cuda') as gpu:
                screen=screen_earth_to_asteroids(catalogue,np.asarray([first.event_id]),np.asarray([launch]),np.asarray([tof]))
                dv=float(screen['total_delta_v'][0,0,0]);assert math.isfinite(dv)
                seed=EarthLeg(first.event_id,launch,tof,dv,ship.launch.after.mass-first.before.mass,True)
                search=RouteSearch(catalogue,ids,settings,weights=weights,first_level=[seed])
                profile=cProfile.Profile();profile.enable();result=search.run();profile.disable()
                prefix=root/f'ship-{ship.ship_id:02d}';prefix.mkdir();profile.dump_stats(prefix/'profile.pstats')
                with (prefix/'profile.txt').open('w') as out:pstats.Stats(profile,stream=out).sort_stats('cumulative').print_stats(45)
                plans=[p for p in result.candidates if p.feasible];plans.sort(key=lambda p:(-sum(weights[a]*m for a,m in p.collected_mass.items()),p.asteroids))
                record=dict(ship=ship.ship_id,target=first.event_id,pool=len(ids),incumbent_weighted_kg=cargo[ship.ship_id],seed=dataclasses.asdict(seed),settings=dataclasses.asdict(settings),seconds=time.perf_counter()-begin,search_seconds=result.wall_seconds,expansions=result.expansions,depth=result.depth_reached,lambert_evaluations=result.lambert_evaluations,candidates=len(plans),failures=result.failures,best_weighted_kg=max((sum(weights[a]*m for a,m in p.collected_mass.items()) for p in plans),default=None),gpu_telemetry=dict(gpu.telemetry))
                (prefix/'plans.json').write_text(json.dumps([p.summary() for p in plans],indent=2,default=float))
                (prefix/'report.json').write_text(json.dumps(record,indent=2,default=float))
                report['routes'].append({k:v for k,v in record.items() if k not in ('settings','failures','gpu_telemetry')});save()
            del search,result,plans,profile;gc.collect()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report['complete']=True;save()
