from pathlib import Path
import os,subprocess
root=Path.home()/'spacepdhcg-regeneration-master-v798';root.mkdir()
script=Path('build/performance/master_regeneration_v796.py').read_text().replace("run_id='gpu_regeneration_v796'","run_id='gpu_regeneration_v798'")
old='            result=master.solve(incumbent=warm,max_ships=23,node_cap=2_000_000,exchange_rounds=32)'
new='''            report['budget_sweep']=[]
            for cap in (2_000_000,20_000_000,200_000_000,2_000_000_000):
                result=master.solve(incumbent=warm,max_ships=23,node_cap=cap,exchange_rounds=32)
                report['budget_sweep'].append(dict(cap=cap,objective=result.objective,seconds=result.native_seconds,nodes=result.nodes,exhaustive=result.exhaustive,selected=[c.identifier for c in result.selected]))
                warm=list(result.selected);save()
                if result.exhaustive:break'''
assert old in script;script=script.replace(old,new)
# The starting baseline remains the original fleet, rather than the evolving warm start.
script=script.replace("    assert fleet_feasible(warm)==''", "    assert fleet_feasible(warm)==''\n    baseline=sum(c.value(weights) for c in warm)")
script=script.replace('baseline_score_kg=sum(c.value(weights) for c in warm)','baseline_score_kg=baseline')
(root/'run.py').write_text(script)
launch=(Path.home()/'spacepdhcg-regeneration-master-v796/launch.py').read_text().replace('spacepdhcg-regeneration-master-v796','spacepdhcg-regeneration-master-v798');(root/'launch.py').write_text(launch)
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-regeneration-master-v798';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
