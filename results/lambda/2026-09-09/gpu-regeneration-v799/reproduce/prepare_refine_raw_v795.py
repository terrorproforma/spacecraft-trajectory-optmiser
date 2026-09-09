from pathlib import Path
import os,subprocess
p=Path('build/performance');root=Path.home()/'spacepdhcg-raw-refine-v795';root.mkdir()
script=(p/'refine_regeneration_v793.py').read_text().replace('spacepdhcg-regeneration-v792','spacepdhcg-raw-regeneration-v794')
script=script.replace("state=json.loads((prior/'report.json').read_text());assert state['complete'] and state['success']", "while not json.loads((prior/'report.json').read_text())['complete']:time.sleep(2)\n    started=time.perf_counter();state=json.loads((prior/'report.json').read_text());assert state['success']")
a=script.index('    jobs=[]');b=script.index("    report['jobs']",a)
script=script[:a]+'''    jobs=[]
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
''' +script[b:]
script=script.replace('    replacements={}', '''    from spacepdhcg.gtoc12.cooperative import FleetColumn,fleet_feasible
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
    replacements={}''')
old="                    if weighted>incumbent+.01 and weighted>replacements.get(ship_id,(None,-math.inf))[1]:replacements[ship_id]=(Solution.read(out/'Result.txt').ships[0],weighted)"
new="""                    c=FleetColumn.from_plan(1000+len(columns),ship_id,f'ship-{ship_id}-rank-{rank}',route.plan,route.collected_mass,certified=route.certified)
                    columns.append(c);artifacts[c.identifier]=Solution.read(out/'Result.txt').ships[0]"""
assert old in script;script=script.replace(old,new)
a=script.index("    report['replacements']");b=script.index("    report['success']=True",a)
script=script[:a]+'''    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
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
''' +script[b:]
(root/'run.py').write_text(script)
launch=(Path.home()/'spacepdhcg-regeneration-refine-v793/launch.py').read_text().replace('spacepdhcg-regeneration-refine-v793','spacepdhcg-raw-refine-v795');(root/'launch.py').write_text(launch)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-raw-refine-v795';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
