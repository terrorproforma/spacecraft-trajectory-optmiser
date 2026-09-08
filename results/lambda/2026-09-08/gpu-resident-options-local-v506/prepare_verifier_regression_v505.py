from pathlib import Path
import json,numpy as np
p=Path('build/performance');root=p/'verifier-knots-v504/cases';rows=json.loads((root/'analysis.json').read_text())
i=max(range(len(rows)),key=lambda i:abs(rows[i]['old_mass']-rows[i]['quadrature32_mass']))
data=np.load(root/(str(i)+'.npz'));dest=Path('tests/fixtures/gtoc12_lagrange_replay.json');dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(dict(source='Captured synthetic Lagrange-hold GPU trajectory, verifier-knots-v504 case '+str(i),**{key:data[key].tolist() for key in data.files}),indent=2))
run=(p/'check_resident_options_v495.py').read_text().replace("root=Path('build/performance/resident-options-v495')", "root=Path('build/performance/verifier-regression-v505')")
a=run.index('jobs=');b=run.index('\nr=dict',a)
run=run[:a]+"jobs=[('pytest',[py,'-c',boot,'tests/test_gtoc12_verifier_knots.py','-s','-q'])]"+run[b:]
run=run.replace('sources=[',"sources=['src/spacepdhcg/gtoc12/verifier.py','tests/test_gtoc12_verifier_knots.py','tests/fixtures/gtoc12_lagrange_replay.json',")
(p/'check_verifier_regression_v505.py').write_text(run)
