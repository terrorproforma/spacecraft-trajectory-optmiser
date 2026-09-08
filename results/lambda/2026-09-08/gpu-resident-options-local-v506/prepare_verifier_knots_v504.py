from pathlib import Path
p=Path('build/performance')
run=(p/'check_resident_options_v495.py').read_text().replace("root=Path('build/performance/resident-options-v495')", "root=Path('build/performance/verifier-knots-v504')")
a=run.index('jobs=');b=run.index('\nr=dict',a)
run=run[:a]+"jobs=[('diagnose',[py,'build/performance/diagnose_verifier_knots_v504.py',str(root/'cases')])]"+run[b:]
(p/'run_verifier_knots_v504.py').write_text(run)
