from pathlib import Path
script=Path(__file__).with_name('launch_family_refinement_v851.py').read_text()
script=script.replace('run_family_refinement_v851.py','run_family_penalty_v852.py')
script=script.replace("root=home/'spacepdhcg-family-refinement-v851'", "root=home/'spacepdhcg-family-penalty-v852'")
script=script.replace("'family-refinement-launch-v851.json'", "'family-penalty-launch-v852.json'")
exec(compile(script,str(Path(__file__).with_name('launch_family_refinement_v851.py')),'exec'))
