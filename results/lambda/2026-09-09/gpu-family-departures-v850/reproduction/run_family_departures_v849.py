"""Finite new-departure campaign; forecasts never count as verified score."""
from pathlib import Path
import dataclasses, fcntl, gc, hashlib, json, math, os, sys, time, traceback

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
RUNTIME = HOME / 'spacepdhcg-return-cache-v844'
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
sys.path.insert(0, str(RUNTIME / 'repo/src'))
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue, load_bonus_table
from spacepdhcg.gtoc12.solution import Solution
from spacepdhcg.gtoc12.clusters import ClusterBands, ComovingClusters
from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, cluster_search_settings
from spacepdhcg.gtoc12.search import RouteSearch, EarthLeg
from spacepdhcg.gtoc12.screening import screen_earth_to_asteroids, propellant_for_delta_v, thrust_authority_km_s
from spacepdhcg.gtoc12.lambert import using_lambert_backend

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    # Search settings legitimately contain infinity for disabled limits.
    value = json.loads(json.dumps(value, default=float), parse_constant=lambda x:x)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

report = dict(pid=os.getpid(), complete=False, success=False, routes=[], trajectory_solves=0,
              certificates=0, fleet_checks=0, objective='new Earth departure targets in four different families')

def save():
    write(ROOT / 'report.tmp', report)
    (ROOT / 'report.tmp').replace(ROOT / 'report.json')

save()
started = time.perf_counter()
try:
    assert sha(ROOT / 'fleet.txt') == '1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da'
    validation = json.loads((RUNTIME / 'report.json').read_text())
    assert validation['complete'] and validation['success']
    fleet = Solution.read(ROOT / 'fleet.txt')
    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    occupied = {s.ship_id: {e.event_id for e in s.asteroid_visits()} for s in fleet.ships}
    raw = {s.ship_id: sum(max(0., e.after.mass-e.before.mass) for e in s.asteroid_visits()) for s in fleet.ships}
    weighted = {s.ship_id: sum(weights[e.event_id]*max(0., e.after.mass-e.before.mass) for e in s.asteroid_visits()) for s in fleet.ships}
    n = len(fleet.ships)
    required = n * math.log(n / 2.) / .004
    margin = sum(raw.values()) - required
    assert n == 24 and 5.6045 < margin < 5.6047
    report.update(fleet_sha256=sha(ROOT/'fleet.txt'), core_sha256=sha(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY'])),
                  source_manifest_sha256=sha(RUNTIME/'source-manifest.json'), bonus_sha256=bonus.source_sha256,
                  fleet_raw_kg=sum(raw.values()), fleet_weighted_kg=sum(weighted.values()), raw_margin_kg=margin,
                  ships=[2,5,12,18], arms=['unit','bonus'], max_refinements=0,
                  protocol='8 searches; each uses up to 3 NEW targets; 64-wide beam, depth 10, 60 seconds per search. No inherited incumbent seed. Exclude every retained ship asteroid before screening. Same epochs as replaced ship; new target must pass existing inflated authority filter. No tuning after seeing results.')
    write(ROOT/'incumbent.json', dict(raw=raw, weighted=weighted, occupied={k: sorted(v) for k,v in occupied.items()}))
    save()
    families = ComovingClusters(catalogue, catalogue.ids, ClusterBands.collect_window(radius=4.))
    pricing = ClusterPricingSettings(beam_width=64, neighbours=96, max_deploys=10, collect_dp=True,
              collect_dp_inflation_fit=str(ROOT/'fit.json'), harvest_substitution=False)
    for key in ('COMPLETION_BATCH','COMPLETION_NATIVE_MODEL'):
        os.environ['SPACEPDHCG_TEST_GTOC12_'+key] = '1'
    with (HOME/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for ship_id in report['ships']:
            ship = fleet.ships[ship_id-1]
            assert ship.ship_id == ship_id
            first = ship.asteroid_visits()[0]
            launch = ship.launch.epoch
            tof = first.epoch-launch
            excluded = set().union(*(v for k,v in occupied.items() if k != ship_id))
            ids = np.asarray(sorted((set(map(int,families.neighbours(first.event_id)[:192])) | occupied[ship_id])-excluded), np.int64)
            settings = dataclasses.replace(cluster_search_settings(pricing,len(ids)), time_budget_seconds=60.,
                       first_level_window_days=0., max_per_first=24)
            targets = ids[ids != first.event_id]
            screen_path = ROOT/f'ship-{ship_id:02d}-screen.json'
            if screen_path.exists():
                seeds = [EarthLeg(**x) for x in json.loads(screen_path.read_text())['selected']]
            else:
              with using_lambert_backend('cuda') as gpu:
                grid = screen_earth_to_asteroids(catalogue, targets, np.asarray([launch]), np.asarray([tof]))
                dv = grid['total_delta_v'][:,0,0]
                ok = grid['feasible'][:,0,0] & np.isfinite(dv) & (dv*settings.earth_out_inflation <= settings.earth_out_authority_ratio*thrust_authority_km_s(settings.initial_mass, tof, 1.))
                costs = propellant_for_delta_v(settings.initial_mass, dv*settings.earth_out_inflation)
                order = sorted(np.flatnonzero(ok), key=lambda i:(float(costs[i]),int(targets[i])))[:3]
                seeds = [EarthLeg(int(targets[i]),launch,tof,float(dv[i]),float(costs[i]),False) for i in order]
                write(screen_path, dict(targets=targets.tolist(), launch=launch,tof=tof,
                      feasible=ok.tolist(), delta_v_km_s=[float(x) if math.isfinite(x) else None for x in dv],
                      selected=[dataclasses.asdict(x) for x in seeds], gpu_telemetry=dict(gpu.telemetry)))
            for arm in report['arms']:
                out = ROOT/f'ship-{ship_id:02d}-{arm}'
                if (out/'candidates.json').exists():
                    # Preserve the completed search preceding the reporting-only abort.
                    candidates = json.loads((out/'candidates.json').read_text())
                    record = dict(ship=ship_id,arm=arm,seeds=[dataclasses.asdict(x) for x in seeds],
                        candidates=len(candidates), search_seconds=None, lambert_evaluations=None,
                        recovered_from='v849; completed plans retained, unsaved search telemetry unavailable',
                        best_weighted_kg=candidates[0]['weighted_kg'] if candidates else None,
                        refinement_candidates=[c for c in candidates if c['qualifies_for_refinement']])
                    write(out/'report.json',record)
                    report['routes'].append(record)
                    save()
                    continue
                out.mkdir()
                report.update(stage='search', ship=ship_id, arm=arm)
                save()
                if not seeds:
                    report['routes'].append(dict(ship=ship_id,arm=arm,skipped='No new Earth target passes unchanged proxy authority filter'))
                    save()
                    continue
                with using_lambert_backend('cuda') as gpu:
                    search = RouteSearch(catalogue,ids,settings,weights=weights if arm=='bonus' else {int(a):1. for a in catalogue.ids},first_level=seeds)
                    result = search.run()
                    plans = [p for p in result.candidates if p.feasible]
                    plans.sort(key=lambda p:(-sum(weights[a]*m for a,m in p.collected_mass.items()),p.asteroids))
                    write(out/'plans.json',[p.summary() for p in plans])
                    candidates = []
                    for rank, plan in enumerate(plans):
                        r = sum(plan.collected_mass.values())
                        w = sum(weights[a]*m for a,m in plan.collected_mass.items())
                        conflicts = sorted(set(plan.asteroids)&excluded)
                        candidates.append(dict(rank=rank,raw_kg=r,weighted_kg=w,raw_delta_kg=r-raw[ship_id],
                            weighted_delta_kg=w-weighted[ship_id],fleet_raw_margin_kg=margin+r-raw[ship_id],conflicts=conflicts,
                            qualifies_for_refinement=not conflicts and margin+r-raw[ship_id]>=1e-6 and w-weighted[ship_id]>.1))
                    write(out/'candidates.json',candidates)
                    record = dict(ship=ship_id,arm=arm,seeds=[dataclasses.asdict(x) for x in seeds],pool=len(ids),
                        settings=dataclasses.asdict(settings),candidates=len(plans),search_seconds=result.wall_seconds,
                        expansions=result.expansions,depth=result.depth_reached,lambert_evaluations=result.lambert_evaluations,
                        gpu_telemetry=dict(gpu.telemetry),failures=result.failures,
                        best_weighted_kg=candidates[0]['weighted_kg'] if candidates else None,
                        refinement_candidates=[c for c in candidates if c['qualifies_for_refinement']])
                    write(out/'report.json',record)
                    report['routes'].append({k:v for k,v in record.items() if k not in ('settings','gpu_telemetry','failures')})
                    save()
                del search,result,plans
                gc.collect()
    report['success'] = True
except BaseException:
    report['error'] = traceback.format_exc()
report.update(complete=True,seconds=time.perf_counter()-started)
save()
