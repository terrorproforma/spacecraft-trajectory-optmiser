"""CPU-only coefficient-derived L1 balance review. No point files or solver calls.

Run from the repository root with Python's standard library. This deliberately
does not read known solutions, initial-point files, balance proxies or GPU logs.
The single experimental policy is omega=O/B after reduced ten-pass Ruiz.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct

INPUTS = Path('build/performance/known-point-replay-v606/inputs')
HASHES = {
    'conditioning': '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf',
    'difficult': '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(values):
    return sha(b''.join(struct.pack('<d', x) for x in values))


def soft(x, t):
    return x-t if x > t else x+t if x < -t else x*0


def exact_oracles():
    count = 0
    eta, B, O, D, R, lam = map(F, ('1/2', '1/4', '1/8', '2', '4', '8'))
    for omega in map(F, ('1/4', '1', '4', '1/2')):
        tau, sigma = eta/omega*O/(B*D*D), eta*omega*B/(O*R*R)
        assert tau*sigma == eta*eta/(D*D*R*R)
        if omega == O/B:
            assert tau == eta/(D*D) and sigma == eta/(R*R)
        for x in map(F, ('-1', '0', '1')):
            for g in map(F, ('-16', '0', '16')):
                native = soft(x-tau*g, tau*lam)
                scaled = soft(B*D*x-eta/omega*O*g/D, eta/omega*O*lam/D)
                assert B*D*native == scaled
                delta = lam if native > 0 else -lam if native < 0 else max(-lam, min(lam, -g))
                zp, zm = (lam+delta)/2, (lam-delta)/2
                assert zp >= 0 and zm >= 0 and zp+zm == lam
                assert (abs(native)-native)*zp == 0
                assert (abs(native)+native)*zm == 0
                count += 1
    return {'fraction_exact_prox_cases': count,
            'product_and_cancel_global_identities': True,
            'pair_duals_are_native_without_weight_factor': True}


def capture(name):
    path = INPUTS/(name+'.txt')
    data = path.read_bytes()
    assert sha(data) == HASHES[name]
    lines = data.decode().splitlines()
    assert len(lines) == 15 and lines[0] == 'SPACEPDHCG_QOCO_QP_V1'
    n, ne, ng, nq, na, nz, nn, ns, shifted = map(int, lines[1].split())
    assert shifted == 0 and float(lines[14]) == 0
    ptr = [[int(x) for x in lines[k].split()[1:]] for k in (4, 6, 8)]
    idx = [[int(x) for x in lines[k].split()[1:]] for k in (5, 7, 9)]
    soc = [int(x) for x in lines[10].split()[1:]]
    assert len(soc) == ns and nn+sum(soc) == ng
    vals = [float(x) for x in lines[11].split()[1:]]
    assert len(vals) == nq+na+nz+n+ne+ng and all(math.isfinite(x) for x in vals)
    assert all(x == 0 for x in vals[:nq])
    matrix = [vals[:nq], vals[nq:nq+na], vals[nq+na:nq+na+nz]]
    c = vals[nq+na+nz:nq+na+nz+n]
    b = vals[nq+na+nz+n:nq+na+nz+n+ne]
    h = vals[nq+na+nz+n+ne:]
    rows = [[[] for _ in range(m)] for m in (n, ne, ng)]
    for k, m in enumerate((n, ne, ng)):
        assert len(ptr[k]) == n+1 and ptr[k][0] == 0 and ptr[k][-1] == len(matrix[k]) == len(idx[k])
        for j in range(n):
            inds = idx[k][ptr[k][j]:ptr[k][j+1]]
            assert inds == sorted(set(inds)) and all(0 <= i < m for i in inds)
            for at in range(ptr[k][j], ptr[k][j+1]):
                if matrix[k][at] != 0:
                    rows[k][idx[k][at]].append((j, matrix[k][at]))
    auses, guses = [0]*n, [[] for _ in range(n)]
    for row in rows[1]:
        for j, _ in row:
            auses[j] += 1
    for i, row in enumerate(rows[2]):
        for j, value in row:
            guses[j].append((i, value))
    pairs = []
    for t, lam in enumerate(c):
        if lam <= 0 or auses[t] or len(guses[t]) != 2:
            continue
        pair = []
        for i, coefficient in guses[t]:
            other = [(j, value) for j, value in rows[2][i] if j != t]
            if i >= nn or coefficient != -1 or h[i] != 0 or len(other) != 1:
                break
            pair.append((i, *other[0]))
        if len(pair) == 2 and pair[0][1] == pair[1][1] and {p[2] for p in pair} == {-1, 1}:
            pairs.append((t, pair[0][1], lam, pair[0][0], pair[1][0]))
    ts, vs = [p[0] for p in pairs], [p[1] for p in pairs]
    removed = {i for p in pairs for i in p[3:]}
    assert len(set(ts)) == len(ts) == len(set(vs)) and not set(ts)&set(vs) and len(removed) == 2*len(pairs)
    columns = [j for j in range(n) if j not in set(ts)]
    grows = [i for i in range(ng) if i not in removed]
    cmap = {j: k for k, j in enumerate(columns)}
    rmap = {i: ne+k for k, i in enumerate(grows)}
    m = ne+len(grows)
    entries = []
    for k in (1, 2):
        for i, row in enumerate(rows[k]):
            if k == 2 and i in removed:
                continue
            for j, a in row:
                assert j in cmap
                entries.append((cmap[j], i if k == 1 else rmap[i], a))
    d, r = [1.0]*len(columns), [1.0]*m
    for _ in range(10):
        cm, rm = [0.0]*len(d), [0.0]*m
        for j, i, a in entries:
            value = abs(a)/(r[i]*d[j])
            cm[j], rm[i] = max(cm[j], value), max(rm[i], value)
        start = ne+nn-len(removed)
        for size in soc:
            maximum = max(rm[start:start+size])
            rm[start:start+size] = [maximum]*size
            start += size
        assert start == m
        d = [v*(math.sqrt(a) if a > 1e-12 else 1) for v, a in zip(d, cm)]
        r = [v*(math.sqrt(a) if a > 1e-12 else 1) for v, a in zip(r, rm)]
    rhs = b+[h[i] for i in grows]
    bound_norm = math.sqrt(math.fsum((v/s)**2 for v, s in zip(rhs, r)))
    objective_norm = math.sqrt(math.fsum((c[j]/d[k])**2 for k, j in enumerate(columns))
                               + math.fsum((p[2]/d[cmap[p[1]]])**2 for p in pairs))
    B, O = 1/(1+bound_norm), 1/(1+objective_norm)
    scaled = [(j, i, a/(d[j]*r[i])) for j, i, a in entries]
    v = [1/math.sqrt(len(d))]*len(d)
    for _ in range(20):
        rowterms = [[] for _ in r]
        for j, i, a in scaled:
            rowterms[i].append(a*v[j])
        product = [math.fsum(x) for x in rowterms]
        norm = math.sqrt(math.fsum(x*x for x in product))
        product = [x/norm for x in product]
        colterms = [[] for _ in d]
        for j, i, a in scaled:
            colterms[j].append(a*product[i])
        transpose = [math.fsum(x) for x in colterms]
        estimate = math.sqrt(math.fsum(x*x for x in transpose))
        v = [x/estimate for x in transpose]
    eta, omega = .9/max(1, estimate), O/B
    assert all(math.isfinite(x) and x > 0 for x in (B, O, eta, omega, 1/omega))
    sx, sy = [O/(x*x*B) for x in d], [B/(x*x*O) for x in r]
    def steps(weight):
        tau, sigma = [(eta/weight)*x for x in sx], [(eta*weight)*x for x in sy]
        thresholds = [tau[cmap[p[1]]]*p[2] for p in pairs]
        assert all(math.isfinite(x) and x > 0 for x in tau+sigma+thresholds)
        return tau, sigma, thresholds
    unit = steps(1.0)
    selected = steps(omega)
    product_error = max(abs((selected[0][j]*selected[1][i])/(unit[0][j]*unit[1][i])-1) for j, i, _ in entries)
    cancellation_error = max([abs(x/(eta/(d[j]*d[j]))-1) for j, x in enumerate(selected[0])]
                            + [abs(x/(eta/(r[i]*r[i]))-1) for i, x in enumerate(selected[1])])
    assert max(product_error, cancellation_error) < 2e-15
    start = ne+nn-len(removed)
    for size in soc:
        assert len(set(selected[1][start:start+size])) == 1
        start += size
    return dict(capture=name, input_sha256=sha(data), pairs=len(pairs), variables=len(d), rows=m,
                numerical_entries=len(entries), D_sha256=fingerprint(d), R_sha256=fingerprint(r),
                B=B, O=O, eta=eta, omega_cancel_global=omega,
                represented_product_max_relative_error=product_error,
                represented_cancellation_max_relative_error=cancellation_error,
                soc_metric_uniform=True,
                unit_ranges={k: [min(v), max(v)] for k, v in zip(('tau', 'sigma', 'threshold'), unit)},
                cancel_global_ranges={k: [min(v), max(v)] for k, v in zip(('tau', 'sigma', 'threshold'), selected)},
                scaled_joint_objective_to_rhs_norm=(O*objective_norm)/(B*bound_norm))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('policy-findings.json'))
    args = parser.parse_args()
    result = dict(scope='CPU only; coefficient inputs only; no initial points, known solutions, pilot, solver, or GPU calls',
                  script_sha256=sha(Path(__file__).read_bytes()),
                  policy='once after reduced scaling, omega=O/B; freeze within each solve',
                  limitations=['This is an interpretable ablation, not a predicted optimal weight or improvement.',
                               'A fixed positive reciprocal weight preserves the same exact-arithmetic spectral condition; the power20 eta remains heuristic.',
                               'The current common original-coordinate qualification gates remain mandatory.',
                               'No guarantee for adaptive changes is inferred from the fixed-weight argument.'],
                  exact_oracles=exact_oracles(), cases=[capture(name) for name in HASHES])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output.read_bytes()),
                          omega={c['capture']: c['omega_cancel_global'] for c in result['cases']})))


if __name__ == '__main__':
    main()
