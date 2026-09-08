from pathlib import Path
import json,shutil,subprocess
home=Path.home();old=home/'spacepdhcg-fleet-refine-v729';root=home/'spacepdhcg-fleet-refine-v732';root.mkdir()
shutil.copytree(old/'input',root/'input');shutil.copyfile(old/'input-hashes.json',root/'input-hashes.json')
s=(old/'run.py').read_text().replace('fleet-refine-v729','fleet-refine-v732').replace('cuda_fleet_exchange_v729','cuda_fleet_exchange_v732')
needle="    settings=ScvxSettings("
insert="""    from spacepdhcg.gtoc12.cooperative import FleetColumn
    from spacepdhcg.gtoc12.gpu_fleet import solve_fleet_cuda
    pool=json.loads((root/'input/pool.json').read_text());columns=[]
    for row in pool['rows']:
        columns.append(FleetColumn(row['identifier'],row['ship_id'],row['label'],*({int(k):v for k,v in row[key].items()} for key in ('deploys','collects','foreign','mass')),True))
    warm=tuple(c for c in columns if c.identifier in pool['warm'])
    weights={int(a):float(bonus.coefficient[int(a)-1]) for a in catalogue.ids}
    report['stage']='gpu_fleet_search';save()
    with (home/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);began=time.perf_counter()
        master=solve_fleet_cuda(columns,weights=weights,incumbent=warm,node_cap=0)
    report['master_seconds']=time.perf_counter()-began
    report['master']={k:v for k,v in master.summary().items() if k not in ('selected','rejected')}
    selection=[c.identifier for c in master.selected]
    assert selection==json.loads((root/'input/selected.json').read_text())
    report['selected']=selection;save()
"""
assert needle in s;s=s.replace(needle,insert+needle)
(root/'run.py').write_text(s)
launch=(old/'launch.py').read_text().replace('fleet-refine-v729','fleet-refine-v732').replace('spacepdhcg-joint-search-v702','spacepdhcg-fleet-v727')
(root/'launch.py').write_text(launch)
exec(launch)
