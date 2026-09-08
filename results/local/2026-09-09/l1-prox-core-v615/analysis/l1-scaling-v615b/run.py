"""Independent CPU spectral check of the proposed exact L1 reduced operator."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

root = Path(__file__).resolve().parents[2]
out = root / 'build/performance/l1-scaling-v615b'
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
array_sha = lambda a: hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()
source = root / 'scripts/gpu/audit_persistent_snapshot.py'
assert sha(source) == '0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
spec = importlib.util.spec_from_file_location('l1_spectral_auditor', source)
audit = importlib.util.module_from_spec(spec)
exec(compile(source.read_bytes(), str(source), 'exec'), audit.__dict__)
(out / 'run.py').write_bytes(Path(__file__).read_bytes())
(out / 'auditor.py').write_bytes(source.read_bytes())
cases = []
for name, expected_count in [('conditioning', 1470), ('difficult', 1631)]:
    start = time.perf_counter()
    path = root / 'build/performance/known-point-replay-v606/inputs' / (name + '.txt')
    q = audit.load_snapshot(path)
    assert q['quadratic_numerical_nonzeros'] == 0 and not q['shift']
    A = q['A'].tocsc(copy=True); A.eliminate_zeros()
    G = q['G'].tocsc(copy=True); G.eliminate_zeros()
    rows = G.tocsr()
    pairs = []
    for t, cost in enumerate(q['c']):
        if not (cost > 0 and math.isfinite(cost)) or A.indptr[t] != A.indptr[t + 1]:
            continue
        incidence = list(zip(G.indices[G.indptr[t]:G.indptr[t + 1]], G.data[G.indptr[t]:G.indptr[t + 1]]))
        if len(incidence) != 2 or any(i >= q['l'] or a != -1 or q['h'][i] != 0 for i, a in incidence):
            continue
        targets = []
        for i, _ in incidence:
            terms = list(zip(rows.indices[rows.indptr[i]:rows.indptr[i + 1]], rows.data[rows.indptr[i]:rows.indptr[i + 1]]))
            if len(terms) != 2:
                break
            targets.extend((j, float(a), int(i)) for j, a in terms if j != t)
        if len(targets) != 2 or targets[0][0] != targets[1][0] or {x[1] for x in targets} != {-1, 1}:
            continue
        pairs.append({'t': t, 'v': int(targets[0][0]), 'lambda': float(cost),
                      'positive_row': next(x[2] for x in targets if x[1] == 1),
                      'negative_row': next(x[2] for x in targets if x[1] == -1)})
    assert len(pairs) == expected_count
    ts = {r['t'] for r in pairs}; vs = {r['v'] for r in pairs}
    removed_rows = {r[k] for r in pairs for k in ('positive_row', 'negative_row')}
    assert len(vs) == len(pairs) and len(removed_rows) == 2 * len(pairs) and ts.isdisjoint(vs)
    keep_columns = np.asarray([j for j in range(q['n']) if j not in ts])
    keep_g = np.asarray([i for i in range(G.shape[0]) if i not in removed_rows])
    column_map = {int(j): k for k, j in enumerate(keep_columns)}
    K = sp.vstack((A, G[keep_g]), format='csc')[:, keep_columns].astype(np.float64)
    K.eliminate_zeros()
    n = K.shape[1]; m = K.shape[0]
    columns = np.repeat(np.arange(n), np.diff(K.indptr)); indices = K.indices
    D = np.ones(n); R = np.ones(m)
    l = q['l'] - len(removed_rows)
    for _ in range(10):
        values = np.abs(K.data) / (R[indices] * D[columns])
        colmax = np.zeros(n); rowmax = np.zeros(m)
        np.maximum.at(colmax, columns, values); np.maximum.at(rowmax, indices, values)
        offset = q['p'] + l
        for size in q['soc']:
            size = int(size)
            rowmax[offset:offset + size] = rowmax[offset:offset + size].max()
            offset += size
        assert offset == m
        D *= np.where(colmax > 1e-12, np.sqrt(colmax), 1)
        R *= np.where(rowmax > 1e-12, np.sqrt(rowmax), 1)
    rhs = np.concatenate((q['b'], np.asarray(q['h'])[keep_g])).astype(np.float64)
    smooth = np.asarray(q['c'], dtype=np.float64)[keep_columns]
    penalties = np.asarray([r['lambda'] / D[column_map[r['v']]] for r in pairs])
    B = 1 / (math.sqrt(math.fsum(float(x)**2 for x in rhs / R)) + 1)
    O = 1 / (math.sqrt(math.fsum(float(x)**2 for x in smooth / D) + math.fsum(float(x)**2 for x in penalties)) + 1)
    scaled = K.copy(); scaled.data /= R[indices] * D[columns]
    v = np.ones(n) / math.sqrt(n); history = []
    for _ in range(20):
        product = scaled @ v; norm = np.linalg.norm(product)
        assert norm > 1e-12
        transpose = scaled.T @ (product / norm)
        estimate = np.linalg.norm(transpose); v = transpose / estimate
        history.append(float(estimate))
    eta = .9 / max(1, float(estimate))
    gram = scaled.T @ scaled
    values, vectors = spla.eigsh(gram, k=1, which='LA', tol=1e-12, maxiter=2000,
                                v0=np.random.default_rng(615).normal(size=n))
    eigenvalue = float(values[0]); vector = vectors[:, 0]
    absK = abs(scaled)
    upper = math.sqrt(float(np.max(np.asarray(absK.sum(axis=0)))) * float(np.max(np.asarray(absK.sum(axis=1)))))
    tau = eta * O / (B * D**2)
    thresholds = [float(tau[column_map[pair['v']]] * pair['lambda']) for pair in pairs]
    cases.append({'capture': name, 'input_sha256': sha(path), 'original_variables': q['n'],
                  'original_rows': A.shape[0] + G.shape[0], 'original_numerical_entries': int(A.nnz + G.nnz),
                  'reduced_variables': n, 'reduced_rows': m, 'reduced_numerical_entries': int(K.nnz),
                  'pairs': pairs, 'D_sha256': array_sha(D), 'R_sha256': array_sha(R),
                  'D_range': [float(D.min()), float(D.max())], 'R_range': [float(R.min()), float(R.max())],
                  'B': B, 'O': O, 'eta': eta, 'power20_history': history,
                  'numeric_largest_singular_value': math.sqrt(eigenvalue),
                  'numeric_eigenpair_residual_l2': float(np.linalg.norm(gram @ vector - eigenvalue * vector)),
                  'eta_squared_numeric_norm_squared': eta**2 * eigenvalue,
                  'safe_one_infinity_upper_bound': upper,
                  'eta_squared_safe_upper_squared': (eta * upper)**2,
                  'primal_step_range': [float(tau.min()), float(tau.max())],
                  'soft_threshold_range': [min(thresholds), max(thresholds)],
                  'cpu_seconds': time.perf_counter() - start})
report = {'scope': 'CPU-only proposed reduced scaling; no solver or CUDA calls',
          'script_sha256': sha(Path(__file__)), 'auditor_sha256': sha(source),
          'norm_caveat': 'Numerical eigenpair evidence for two captures, not a formal general spectral bound',
          'O_definition': '1/(1+sqrt(sum_active((c_j/D_j)^2)+sum_pairs((lambda/D_v)^2)))',
          'cases': cases}
(out / 'findings.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps([{k: r[k] for k in ('capture', 'reduced_variables', 'reduced_rows', 'reduced_numerical_entries',
                                    'eta_squared_numeric_norm_squared', 'B', 'O', 'eta', 'soft_threshold_range')}
                  for r in cases], indent=2))
