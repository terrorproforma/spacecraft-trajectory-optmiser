"""CPU diagnostic only: distance balance after removal of epigraph dual pairs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

import numpy as np
import scipy.sparse as sp

root = Path(__file__).resolve().parents[2]
out = root / 'build/performance/l1-balance-v615'
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
array_sha = lambda a: hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()
prior = root / 'build/performance/l1-scaling-v615b/findings.json'
findings = json.loads(prior.read_text())
source = root / 'scripts/gpu/audit_persistent_snapshot.py'
spec = importlib.util.spec_from_file_location('balance_auditor', source)
audit = importlib.util.module_from_spec(spec)
exec(compile(source.read_bytes(), str(source), 'exec'), audit.__dict__)
(out / 'run.py').write_bytes(Path(__file__).read_bytes())
results = []
for old in findings['cases']:
    name = old['capture']
    base = root / 'build/performance/known-point-replay-v606/inputs'
    q = audit.load_snapshot(base / (name + '.txt'))
    pairs = old['pairs']; ts = {r['t'] for r in pairs}
    removed = {r[k] for r in pairs for k in ('positive_row', 'negative_row')}
    columns = np.asarray([j for j in range(q['n']) if j not in ts])
    grow = np.asarray([i for i in range(q['G'].shape[0]) if i not in removed])
    K = sp.vstack((q['A'], q['G'][grow]), format='csc')[:, columns].astype(np.float64)
    K.eliminate_zeros()
    col = np.repeat(np.arange(len(columns)), np.diff(K.indptr)); row = K.indices
    D = np.ones(K.shape[1]); R = np.ones(K.shape[0])
    for _ in range(10):
        values = np.abs(K.data) / (R[row] * D[col])
        cm = np.zeros(len(D)); rm = np.zeros(len(R))
        np.maximum.at(cm, col, values); np.maximum.at(rm, row, values)
        offset = q['p'] + q['l'] - len(removed)
        for size in q['soc']:
            size = int(size); rm[offset:offset + size] = rm[offset:offset + size].max(); offset += size
        D *= np.where(cm > 1e-12, np.sqrt(cm), 1)
        R *= np.where(rm > 1e-12, np.sqrt(rm), 1)
    assert array_sha(D) == old['D_sha256'] and array_sha(R) == old['R_sha256']
    point = base / (name + '-initial.txt')
    vectors = {}
    for line in point.read_text().splitlines()[3:]:
        fields = line.split(); vectors[fields[0]] = np.asarray(fields[2:], dtype=np.float64)
        assert int(fields[1]) == len(vectors[fields[0]])
    x = vectors['x'][columns]
    y = np.concatenate((vectors['y'], vectors['z'][grow]))
    primal_norm = float(np.linalg.norm(old['B'] * D * x))
    unscaled_dual_norm = float(np.linalg.norm(R * y))
    smooth = np.asarray(q['c'], dtype=np.float64)[columns]
    O_smooth = 1 / (1 + math.sqrt(math.fsum(float(v)**2 for v in smooth / D)))
    rows = []
    for mode, O in [('joint_smooth_and_l1_norm', old['O']), ('smooth_norm_only_diagnostic', O_smooth)]:
        dual_norm = O * unscaled_dual_norm
        weight = dual_norm / primal_norm
        rows.append({'mode': mode, 'O': O, 'scaled_primal_distance': primal_norm,
                     'scaled_dual_distance': dual_norm, 'oracle_weight': weight,
                     'unit_over_oracle_distance_bound_coefficient': (primal_norm**2 + dual_norm**2) / (2 * primal_norm * dual_norm),
                     'same_eta_primal_step_multiplier_versus_joint': O / old['O'],
                     'L1_lambda_remains_in_objective_and_prox': True})
    results.append({'capture': name, 'point_sha256': sha(point), 'policies': rows})
report = {'scope': 'CPU diagnostic using known reference distances; no solver calls and no deployable oracle tuning',
          'source_sha256': sha(Path(__file__)), 'scaling_findings_sha256': sha(prior),
          'meaning': 'O is a positive coordinate normalization; both policies must still scale every smooth and L1 objective term consistently',
          'caveat': 'The projected reference is not certified for the reduced nonsmooth KKT system; near-zero v require original initial acceptance',
          'cases': results}
(out / 'findings.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(results, indent=2))
