"""CPU-only exact mass-row/scan audit; reads coefficients, not solution vectors."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import tarfile
import types

INPUTS = Path('build/performance/known-point-replay-v606/inputs')
FROZEN = Path('build/performance/l1-weight-v618b')
CAPTURES = {
    'conditioning': (210, '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf'),
    'difficult': (233, '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frac(value):
    return F(float(value))


def rows(q, matrix):
    result = [[] for _ in range((q.n, q.p, q.m)[matrix])]
    for j in range(q.n):
        for at in range(q.ptr[matrix][j], q.ptr[matrix][j+1]):
            if q.val[matrix][at] != 0:
                result[q.idx[matrix][at]].append((j, frac(q.val[matrix][at])))
    return result


def capture(name, intervals, expected, dm):
    path = INPUTS/(name+'.txt')
    assert sha(path) == expected
    q = dm.Snapshot(path)
    assert not q.shift and q.offset == 0 and q.q_all_zero
    mass = [7*k+6 for k in range(intervals+1)]
    mass_index = {j: k for k, j in enumerate(mass)}
    erows = [7*intervals+6]+[7*k+6 for k in range(intervals)]
    eset = set(erows)
    iu, inu = 7*(intervals+1), 11*(intervals+1)
    a, g = rows(q, 1), rows(q, 2)
    assert a[erows[0]] == [(6, F(1))] and q.b[erows[0]] == 1
    assert all(q.c[j] == 0 for j in mass)
    retained_mass_uses = [(i, j) for i, row in enumerate(a) if i not in eset for j, v in row if j in mass_index]
    assert retained_mass_uses == []  # Captured zero-thrust linearizations, not a general assumption.
    gamma, nu, coefficient, affine = [], [], [], []
    for k in range(intervals):
        gm, vk = iu+4*k+3, inu+7*k+6
        row = dict(a[7*k+6])
        assert set(row) == {mass[k], mass[k+1], gm, vk}
        assert row[mass[k]] == -1 and row[mass[k+1]] == 1 and row[vk] == -1
        assert row[gm] > 0
        gamma.append(gm); nu.append(vk); coefficient.append(row[gm]); affine.append(frac(q.b[7*k+6]))
    mass_grows = []
    for i, row in enumerate(g):
        uses = [(j, v) for j, v in row if j in mass_index]
        if uses:
            assert i < q.l and len(row) == 1 and abs(uses[0][1]) == 1
            mass_grows.append((i, mass_index[uses[0][0]], uses[0][1]))
    assert len(mass_grows) == 3*(intervals+1)
    assert all(sum(node == k for _, node, _ in mass_grows) == 3 for k in range(intervals+1))
    # Deterministic dyadic probes, independent of any known or computed solution.
    retained = [F((j % 13)-6, 2**20) for j in range(q.n)]
    d, linear = [F(1)], [F(0)]
    for k in range(intervals):
        d.append(d[-1]+affine[k])
        linear.append(linear[-1]+retained[nu[k]]-coefficient[k]*retained[gamma[k]])
    x = retained.copy()
    for k, j in enumerate(mass):
        x[j] = d[k]+linear[k]
    for i in erows:
        assert sum((v*x[j] for j, v in a[i]), F(0)) == frac(q.b[i])
    for i, k, sign in mass_grows:
        original_slack = frac(q.h[i])-sign*x[mass[k]]
        reduced_slack = frac(q.h[i])-sign*d[k]-sign*linear[k]
        assert original_slack == reduced_slack
    # S^T uses a suffix sum. Check exact adjoint and triangular dual recovery.
    gm = [F((k % 9)-4, 32) for k in range(intervals+1)]
    pulled = [F(0)]*q.n
    suffix = F(0)
    ye = [F(0)]*(intervals+1)
    for k in range(intervals-1, -1, -1):
        suffix += gm[k+1]
        pulled[nu[k]] += suffix
        pulled[gamma[k]] -= coefficient[k]*suffix
        ye[k+1] = -suffix
    ye[0] = -suffix-gm[0]
    assert sum((a*b for a, b in zip(gm, linear)), F(0)) == sum((a*b for a, b in zip(pulled, retained)), F(0))
    original_normal = [F(0)]*q.n
    for i, dual in zip(erows, ye):
        for j, value in a[i]:
            original_normal[j] += value*dual
    assert all(original_normal[j]+gm[k] == 0 for k, j in enumerate(mass))
    assert all(original_normal[j] == pulled[j] for j in range(q.n) if j not in mass_index)
    assert -sum((frac(q.b[i])*y for i, y in zip(erows, ye)), F(0)) == sum((a*b for a, b in zip(d, gm)), F(0))
    # Floating inclusive scan is numerically approximate, never a new physics gate.
    fp = [1.0]
    for k in range(intervals):
        fp.append(fp[-1]+float(affine[k])+float(retained[nu[k]])-float(coefficient[k])*float(retained[gamma[k]]))
    fp_residual = max(abs(F(fp[k+1])-F(fp[k])+coefficient[k]*retained[gamma[k]]-retained[nu[k]]-affine[k])
                      for k in range(intervals))
    numeric_before = sum(map(len, a))+sum(map(len, g))
    mass_e_nnz = sum(len(a[i]) for i in erows)
    expanded_mass_g = sum(2*k for _, k, _ in mass_grows)
    numeric_after = numeric_before-mass_e_nnz-len(mass_grows)+expanded_mass_g
    # A unit vector of equal mass-virtual values gives an exact norm lower bound.
    lower_squared = F(3, intervals)*sum(k*k for k in range(1, intervals+1))
    return dict(capture=name, input_sha256=expected, intervals=intervals,
                original_variables=q.n, original_equalities=q.p, original_cone_rows=q.m,
                eliminated_mass_variables=len(mass), eliminated_equalities=len(erows),
                all_virtual_variables_retained=7*intervals,
                reduced_variables=q.n-len(mass), reduced_equalities=q.p-len(erows),
                combined_L1_reduced_variables=q.n-len(mass)-7*intervals,
                current_mass_coefficient=-1, next_mass_coefficient=1, virtual_mass_coefficient=-1,
                gamma_coefficient_range=[float(min(coefficient)), float(max(coefficient))],
                affine_rhs_range=[float(min(affine)), float(max(affine))],
                initial_mass=1, hessian_exactly_zero=True, original_mass_costs_exactly_zero=True,
                retained_equality_mass_entries=0, scalar_mass_bound_rows=len(mass_grows), SOC_mass_entries=0,
                scan_constant_last=float(d[-1]),
                exact_fraction_checks=dict(eliminated_equalities=True, original_reduced_mass_slacks=True,
                    forward_reverse_adjoint=True, eliminated_dual_stationarity=True, retained_dual_pullback=True,
                    dual_objective_offset_sign=True),
                synthetic_FP64_mass_residual_max=float(fp_residual),
                numerical_operator_entries_original=numeric_before,
                numerical_operator_entries_explicit_mass_reduction=numeric_after,
                numerical_operator_entries_combined_L1_before=numeric_before-4*(7*intervals),
                numerical_operator_entries_combined_L1_after=numeric_after-4*(7*intervals),
                explicit_cumulative_mass_entries=expanded_mass_g,
                cumulative_mass_operator_norm_lower_bound=math.sqrt(float(lower_squared)),
                cumulative_mass_operator_norm_lower_bound_squared_exact=str(lower_squared),
                note='Three scalar mass rows per node. Their retained mass-virtual columns have unit entries, so infinity-norm Ruiz alone leaves this cumulative subblock unscaled.')


def main():
    reader = Path(__file__).with_name('snapshot_reader.py')
    assert sha(reader) == 'b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4'
    dm = types.ModuleType('mass_snapshot_reader')
    exec(compile(reader.read_bytes(), str(reader), 'exec'), dm.__dict__)
    manifest = json.loads((FROZEN/'manifest.json').read_text())
    assert sha(FROZEN/'manifest.json') == '6ad3dfc586c457afbeaa0da13417f1208484f908fb22977fd086ea41a4f3a4c2'
    sources = {}
    with tarfile.open(FROZEN/'source.tar.gz', 'r:gz') as tar:
        for name in ('cpp/cuda/src/gtoc12_conic.cu', 'cpp/cuda/src/gtoc12_discretisation.cu'):
            data = tar.extractfile(name).read()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == manifest['source_sha256'][name]
            sources[name] = digest
    result = dict(scope='Bounded coefficient-only CPU mathematical review; no point files, GPU, compiler or solver calls',
                  script_sha256=sha(Path(__file__)), reader_sha256=sha(reader), source_hashes=sources,
                  frozen_source_tree=manifest['source_tree_sha256'],
                  cases=[capture(name, k, digest, dm) for name, (k, digest) in CAPTURES.items()],
                  disposition='Exact affine reduction is possible for these captures; do not reuse the old operator scaling/step estimate or materialize the cumulative fill without review.')
    output = Path(__file__).with_name('findings.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
