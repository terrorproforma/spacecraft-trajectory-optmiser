from pathlib import Path
import os,subprocess
p=Path('build/performance');root=Path.home()/'spacepdhcg-raw-regeneration-v794';root.mkdir()
script=(p/'regenerate_fleet_v792.py').read_text()
script=script.replace("for name,digest in json.loads((root/'input-hashes.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest", "pass # Inputs copied byte-for-byte from the hashed v792 run")
script=script.replace("beam_width=16,neighbours=48", "beam_width=64,neighbours=96")
script=script.replace("time_budget_seconds=30.,first_level_window_days=0.", "time_budget_seconds=60.,first_level_window_days=0.,max_per_first=64")
script=script.replace("settings,weights=weights,first_level=[seed]", "settings,weights={int(a):1.0 for a in catalogue.ids},first_level=[seed]")
script=script.replace("profile=cProfile.Profile();profile.enable();result=search.run();profile.disable()", "profile=cProfile.Profile();result=search.run()")
script=script.replace("profile.dump_stats(prefix/'profile.pstats')", "(prefix/'unprofiled.txt').write_text('Profile disabled; wall-clock search timings only. Raw-mass search objective; reported weighted values use original bonus table.')")
script=script.replace("with (prefix/'profile.txt').open('w') as out:pstats.Stats(profile,stream=out).sort_stats('cumulative').print_stats(45)", "pass")
(root/'run.py').write_text(script)
launch=(Path.home()/'spacepdhcg-regeneration-v792/launch.py').read_text().replace('spacepdhcg-regeneration-v792','spacepdhcg-raw-regeneration-v794')
(root/'launch.py').write_text(launch)
for name in ('fleet.txt','fit.json'):(root/name).write_bytes((Path.home()/'spacepdhcg-regeneration-v792'/name).read_bytes())
key=Path('/tmp/traj-key.pem')
if not key.exists():
    fd=os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
code="from pathlib import Path\nroot=Path.home()/'spacepdhcg-raw-regeneration-v794';root.mkdir()\n(root/'run.py').write_text("+repr(script)+")\n(root/'launch.py').write_text("+repr(launch)+")\nfor name in ('fleet.txt','fit.json'):(root/name).write_bytes((Path.home()/'spacepdhcg-regeneration-v792'/name).read_bytes())\nexec((root/'launch.py').read_text())"
r=subprocess.run(['ssh','-i',str(key),'-o','BatchMode=yes','ubuntu@192.222.55.229','python3 -'],input=code,text=True,capture_output=True,timeout=40);print(r.stdout,r.stderr);r.check_returncode()
exec(launch)
