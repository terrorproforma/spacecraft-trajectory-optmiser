from pathlib import Path
import fcntl,hashlib,json,os,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-grid-v788/repo/src'))
from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.solution import Solution,ShipTrajectory
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset
report=dict(pid=os.getpid(),complete=False,success=False,stage='waiting_for_refinement')
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save();started=time.perf_counter()
try:
    prior=home/'spacepdhcg-raw-refine-v795'
    while not json.loads((prior/'report.json').read_text())['complete']:time.sleep(2)
    assert json.loads((prior/'report.json').read_text())['success']
    started=time.perf_counter();report['stage']='pack_certified';save()
    catalogue,bonus=load_catalogue(),load_bonus_table();weights={int(a):float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    original=home/'spacepdhcg-regeneration-v792/fleet.txt';fleet=Solution.read(original);columns=[];artifacts={};warm=[]
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
        assert set(dep)==set(col)
        c=FleetColumn(ship.ship_id,ship.ship_id,'verified incumbent',dep,col,{},mass,True)
        warm.append(c);columns.append(c);artifacts[c.identifier]=ship
    assert fleet_feasible(warm)==''
    baseline=sum(c.value(weights) for c in warm)
    report['input_routes']=[]
    for dirname in ('spacepdhcg-regeneration-refine-v793','spacepdhcg-raw-refine-v795'):
        prior=home/dirname;state=json.loads((prior/'report.json').read_text());assert state['complete'] and state['success']
        for attempt in state['attempts']:
            if not attempt['certified']:continue
            route=prior/f"ship-{attempt['ship']:02d}-rank-{attempt['rank']:03d}"
            summary=json.loads((route/'route_summary.json').read_text());assert summary['certified'] and summary['master_certified']
            assert all(l['certified'] and l['certification_backend']=='cuda' for l in summary['legs'])
            mass={int(a):m for a,m in summary['collected_mass_kg'].items()}
            c=FleetColumn.from_plan(1000+len(columns),attempt['ship'],dirname+'/'+route.name,RoutePlan.from_summary(summary['plan']),mass,certified=True)
            columns.append(c);artifacts[c.identifier]=Solution.read(route/'Result.txt').ships[0]
            report['input_routes'].append(dict(identifier=c.identifier,slot=c.slot,label=c.label,weighted_kg=c.value(weights),raw_kg=c.collected_kg,solution_sha256=hashlib.sha256((route/'Result.txt').read_bytes()).hexdigest()))
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with CudaFleetWorkspace(columns,weights=weights) as master:
            report['budget_sweep']=[]
            for cap in (2_000_000,20_000_000,200_000_000,2_000_000_000):
                result=master.solve(incumbent=warm,max_ships=23,node_cap=cap,exchange_rounds=32)
                report['budget_sweep'].append(dict(cap=cap,objective=result.objective,seconds=result.native_seconds,nodes=result.nodes,exhaustive=result.exhaustive,selected=[c.identifier for c in result.selected]))
                warm=list(result.selected);save()
                if result.exhaustive:break
    report.update(master=result.summary(),replacements=[c.slot for c in result.selected if c.identifier>=1000],baseline_score_kg=baseline,core_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest())
    out=root/'fleet';out.mkdir();path=out/'Result.txt'
    Solution([ShipTrajectory(i,artifacts[c.identifier].items) for i,c in enumerate(sorted(result.selected,key=lambda c:c.slot),1)]).write(path)
    report['stage']='independent_full_fleet';save();history={};before=time.perf_counter()
    check=Gtoc12Verifier(catalogue,bonus=bonus,history=history).verify_file(path)
    report.update(independent=check.summary(),independent_seconds=time.perf_counter()-before,stage='official_full_fleet');save()
    official=run_official_verifier(path,keep_directory=out/'official')
    (out/'official.stdout').write_text(official.stdout);(out/'official.stderr').write_text(official.stderr)
    report.update(official=official.summary(),qualified=check.ok and official.ok,score_kg=check.weighted_score_fixed_bonus_kg,total_mass_kg=check.total_mass_kg,solution_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['improved']=report['qualified'] and report['score_kg']>report['baseline_score_kg']+1e-6
    if report['improved']:
        report['viewer_manifest']=write_viewer_dataset(out/'viewer',Solution.read(path),history,catalogue,run_id='gpu_regeneration_v798',commit='ccf5de37',verification=check.summary(),solution_path=path)
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report.update(complete=True,seconds=time.perf_counter()-started);save()
