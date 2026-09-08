"""Coefficient-only diagonal metric for the exact mass/L1 reduced operator.

No point files, native libraries, GPU work or optimization solves are used.
Fraction arithmetic proves absolute-sum construction and the stored FP64-step
certificate; bounded power iteration is labelled numerical evidence only.
"""
from __future__ import annotations
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import types

ROOT = Path(__file__).resolve().parents[3]
THETA = F(19, 20)
CAPTURES = {
    "conditioning": (210, "1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf"),
    "difficult": (233, "14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(b"".join(struct.pack("<d", x) for x in values)).hexdigest()


def rows(q, matrix):
    result = [[] for _ in range((q.n, q.p, q.m)[matrix])]
    for j in range(q.n):
        for at in range(q.ptr[matrix][j], q.ptr[matrix][j+1]):
            value = F(q.val[matrix][at])
            if value:
                result[q.idx[matrix][at]].append((j, value))
    return result


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "p10": values[(len(values)-1)//10],
            "median": values[len(values)//2], "p90": values[9*(len(values)-1)//10], "max": values[-1]}


def downward(value):
    result = float(value)
    if F(result) > value:
        result = math.nextafter(result, 0.0)
    assert math.isfinite(result) and result > 0 and F(result) <= value
    return result


def one_case(name, intervals, expected_hash, reader, old_scaling):
    path = ROOT / "build/performance/known-point-replay-v606/inputs" / (name + ".txt")
    assert sha(path) == expected_hash
    q = reader.Snapshot(path)
    assert q.q_all_zero and not q.shift and q.offset == 0
    a, g = rows(q, 1), rows(q, 2)
    mass = [7*k+6 for k in range(intervals+1)]
    mass_index = {j: k for k, j in enumerate(mass)}
    eliminated = {7*intervals+6, *(7*k+6 for k in range(intervals))}
    assert a[7*intervals+6] == [(6, F(1))] and q.b[7*intervals+6] == 1
    iu, inu = 7*(intervals+1), 11*(intervals+1)
    gamma, virtual, coefficients = [], [], []
    for k in range(intervals):
        gg, vv = iu+4*k+3, inu+7*k+6
        row = dict(a[7*k+6])
        assert set(row) == {mass[k], mass[k+1], gg, vv}
        assert row[mass[k]] == -1 and row[mass[k+1]] == 1 and row[vv] == -1 and row[gg] > 0
        gamma.append(gg); virtual.append(vv); coefficients.append(row[gg])
    assert all(q.c[j] == 0 for j in mass)
    assert all(j not in mass_index for i, row in enumerate(a) if i not in eliminated for j, _ in row)

    pairs = []
    for t in range(q.n):
        if q.c[t] <= 0 or any(q.val[1][k] for k in range(q.ptr[1][t], q.ptr[1][t+1])):
            continue
        incidence = [(q.idx[2][k], F(q.val[2][k])) for k in range(q.ptr[2][t], q.ptr[2][t+1]) if q.val[2][k]]
        if len(incidence) != 2 or any(i >= q.l or value != -1 or q.h[i] != 0 for i, value in incidence):
            continue
        targets = [(j, value) for i, _ in incidence for j, value in g[i] if j != t]
        if any(len(g[i]) != 2 for i, _ in incidence) or len(targets) != 2:
            continue
        if targets[0][0] != targets[1][0] or {x[1] for x in targets} != {F(-1), F(1)}:
            continue
        pairs.append((t, targets[0][0], F(q.c[t]), tuple(i for i, _ in incidence)))
    assert len(pairs) == 7*intervals
    removed_t = {x[0] for x in pairs}
    removed_g = {i for p in pairs for i in p[3]}
    assert len(removed_t) == len(pairs) and len(removed_g) == 2*len(pairs)
    assert removed_t.isdisjoint(mass)
    columns = [j for j in range(q.n) if j not in removed_t and j not in mass_index]
    active_rows = [i for i in range(q.p+q.m) if i not in eliminated and not (i >= q.p and i-q.p in removed_g)]
    base, mass_rows = [], []
    row_sum, col_sum = [F(0)]*(q.p+q.m), [F(0)]*q.n
    for i in active_rows:
        row = a[i] if i < q.p else g[i-q.p]
        uses = [(j, value) for j, value in row if j in mass_index]
        if uses:
            assert i < q.p+q.l and len(row) == 1 and abs(uses[0][1]) == 1
            mass_rows.append((i, mass_index[uses[0][0]], uses[0][1]))
        else:
            for j, value in row:
                assert j in columns
                base.append((i, j, value))
                row_sum[i] += abs(value)
                col_sum[j] += abs(value)
    assert len(mass_rows) == 3*(intervals+1)
    assert all(sum(node == k for _, node, _ in mass_rows) == 3 for k in range(intervals+1))

    prefix = [F(0)]
    for coefficient in coefficients:
        prefix.append(prefix[-1] + 1 + abs(coefficient))
    for i, node, sign in mass_rows:
        row_sum[i] = abs(sign)*prefix[node]
    for k in range(intervals):
        suffix_count = F(3*(intervals-k))
        col_sum[virtual[k]] += suffix_count
        col_sum[gamma[k]] += abs(coefficients[k])*suffix_count

    # Compare scan formulas with every explicitly expanded numerical coefficient.
    explicit_row, explicit_col = [F(0)]*(q.p+q.m), [F(0)]*q.n
    for i, j, value in base:
        explicit_row[i] += abs(value); explicit_col[j] += abs(value)
    expanded_entries = len(base)
    for i, node, sign in mass_rows:
        for k in range(node):
            for j, value in ((virtual[k], sign), (gamma[k], -sign*coefficients[k])):
                explicit_row[i] += abs(value); explicit_col[j] += abs(value)
                expanded_entries += 1
    assert explicit_row == row_sum and explicit_col == col_sum
    denominator = row_sum.copy()
    offset, tied_blocks, lowered_rows = q.p+q.l, [], 0
    for size in q.soc:
        greatest = max(row_sum[offset:offset+size])
        lowered_rows += sum(value < greatest for value in row_sum[offset:offset+size])
        denominator[offset:offset+size] = [greatest]*size
        tied_blocks.append((offset, size))
        offset += size
    assert offset == q.p+q.m
    tau, sigma = [0.0]*q.n, [0.0]*(q.p+q.m)
    for j in columns:
        tau[j] = downward(THETA / (col_sum[j] or 1))
    for i in active_rows:
        sigma[i] = downward(THETA / (denominator[i] or 1))
    row_factor = max(F(sigma[i])*row_sum[i] for i in active_rows)
    col_factor = max(F(tau[j])*col_sum[j] for j in columns)
    assert row_factor <= THETA and col_factor <= THETA
    certificate = row_factor*col_factor
    assert certificate <= THETA**2 < 1
    assert all(len(set(sigma[i:i+size])) == 1 for i, size in tied_blocks)

    # Matrix-free numerical norm probe. Constants/offsets never enter K or K^T.
    numeric = [(i, j, float(value)) for i, j, value in base]
    gg = list(map(float, coefficients))
    def forward(x):
        out = [0.0]*(q.p+q.m)
        for i, j, value in numeric:
            out[i] += value*x[j]
        scan = [0.0]
        for k in range(intervals):
            scan.append(scan[-1]+x[virtual[k]]-gg[k]*x[gamma[k]])
        for i, node, sign in mass_rows:
            out[i] = float(sign)*scan[node]
        return out
    def transpose(y):
        out = [0.0]*q.n
        for i, j, value in numeric:
            out[j] += value*y[i]
        node_values = [0.0]*(intervals+1)
        for i, node, sign in mass_rows:
            node_values[node] += float(sign)*y[i]
        suffix = 0.0
        for k in range(intervals-1, -1, -1):
            suffix += node_values[k+1]
            out[virtual[k]] += suffix
            out[gamma[k]] -= gg[k]*suffix
        return out
    st, ss = list(map(math.sqrt, tau)), list(map(math.sqrt, sigma))
    v = [1/math.sqrt(len(columns)) if j in columns else 0.0 for j in range(q.n)]
    for _ in range(80):
        w = forward([s*x for s, x in zip(st, v)])
        z = transpose([sigma_i*x for sigma_i, x in zip(sigma, w)])
        z = [s*x for s, x in zip(st, z)]
        norm = math.sqrt(math.fsum(x*x for x in z))
        assert norm > 0 and math.isfinite(norm)
        v = [x/norm for x in z]
    w = forward([s*x for s, x in zip(st, v)])
    rayleigh = math.fsum(s*x*x for s, x in zip(sigma, w))
    z = [s*x for s, x in zip(st, transpose([s*x for s, x in zip(sigma, w)]))]
    residual = math.sqrt(math.fsum((x-rayleigh*y)**2 for x, y in zip(z, v)))
    assert 0 <= rayleigh <= float(certificate)+1e-12

    # Recompute the old reduced-operator Ruiz arrays using coefficients only.
    old_rows = [i for i in range(q.p+q.m) if not (i >= q.p and i-q.p in removed_g)]
    old_cols = [j for j in range(q.n) if j not in removed_t]
    old_terms = [(i, j, float(value)) for i in old_rows for j, value in (a[i] if i < q.p else g[i-q.p])]
    D, R = [1.0]*q.n, [1.0]*(q.p+q.m)
    for _ in range(10):
        cm, rm = [0.0]*q.n, [0.0]*(q.p+q.m)
        for i, j, value in old_terms:
            value = abs(value)/(R[i]*D[j])
            cm[j] = max(cm[j], value); rm[i] = max(rm[i], value)
        for i, size in tied_blocks:
            rm[i:i+size] = [max(rm[i:i+size])]*size
        D = [d*(math.sqrt(v) if v > 1e-12 else 1.0) for d, v in zip(D, cm)]
        R = [r*(math.sqrt(v) if v > 1e-12 else 1.0) for r, v in zip(R, rm)]
    assert all(D[j] == 1 for j in virtual) and all(R[i] == 1 for i, _, _ in mass_rows)
    assert array_sha([D[j] for j in old_cols]) == old_scaling["D_sha256"]
    assert array_sha([R[i] for i in old_rows]) == old_scaling["R_sha256"]
    lower_squared = F((intervals+1)*(2*intervals+1), 2)
    reused_lower = F(old_scaling["eta"])**2 * lower_squared
    old_tau = [old_scaling["eta"]*old_scaling["O"]/(old_scaling["B"]*D[j]**2) for j in virtual]
    old_sigma = [old_scaling["eta"]*old_scaling["B"]/(old_scaling["O"]*R[i]**2) for i, node, _ in mass_rows if node]
    return {
        "capture": name, "input_sha256": expected_hash, "intervals": intervals,
        "retained_variables": len(columns), "retained_rows": len(active_rows),
        "base_numeric_entries": len(base), "logical_numeric_entries": expanded_entries,
        "mass_rows": len(mass_rows), "SOC_blocks_tied": len(tied_blocks), "SOC_rows_given_smaller_step": lowered_rows,
        "exact_expansion_matches_scan_absolute_sums": True,
        "stored_FP64_step_norm_squared_certificate": float(certificate),
        "certificate_exact_rational": str(certificate),
        "theta_exact": str(THETA), "zero_operator_rows_retained": sum(row_sum[i] == 0 for i in active_rows),
        "zero_operator_columns_retained": sum(col_sum[j] == 0 for j in columns),
        "tau": distribution([tau[j] for j in columns]),
        "tau_mass_virtual": distribution([tau[j] for j in virtual]),
        "tau_gamma_in_mass_dynamics": distribution([tau[j] for j in gamma]),
        "sigma": distribution([sigma[i] for i in active_rows]),
        "sigma_nonconstant_mass_rows": distribution([sigma[i] for i, node, _ in mass_rows if node]),
        "L1_thresholds": distribution([tau[v]*float(lam) for _, v, lam, _ in pairs]),
        "numerical_power_iterations": 80,
        "numerical_scaled_Gram_Rayleigh_quotient": rayleigh,
        "numerical_scaled_Gram_eigenpair_residual_l2": residual,
        "numerical_norm_note": "Rayleigh quotient/residual are a coefficient-only numerical probe, not the certificate or an optimization solve.",
        "old_Ruiz_D_R_hashes_reproduced": True,
        "invalid_reused_old_metric_norm_squared_lower_bound": float(reused_lower),
        "old_unit_weight_tau_mass_virtual": distribution(old_tau),
        "old_unit_weight_sigma_mass_rows": distribution(old_sigma),
        "tau_sha256": array_sha([tau[j] for j in columns]),
        "sigma_sha256": array_sha([sigma[i] for i in active_rows]),
    }


def main():
    mass_dir = ROOT / "build/performance/mass-structure-review-v621"
    reader_path = mass_dir / "snapshot_reader.py"
    assert sha(reader_path) == "b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4"
    reader = types.ModuleType("coefficient_reader")
    exec(compile(reader_path.read_bytes(), str(reader_path), "exec"), reader.__dict__)
    prior = mass_dir / "findings.json"
    assert sha(prior) == "60efbc1f49c3f485f654bd08822b8504f0d488497d7efe0b69b81df3c75f1104"
    scaling_path = ROOT / "build/performance/l1-scaling-v615b/findings.json"
    scaling = {case["capture"]: case for case in json.loads(scaling_path.read_text())["cases"]}
    result = {
        "scope": "Planned coefficient-only metric hypothesis; no core implementation, GPU, point data, parameter sweep or optimization solve.",
        "script_sha256": sha(Path(__file__)), "reader_sha256": sha(reader_path),
        "prior_mass_review_sha256": sha(prior), "old_scaling_report_sha256": sha(scaling_path),
        "prior_mass_review_source_binding": json.loads(prior.read_text())["source_hashes"],
        "cases": [one_case(name, k, digest, reader, scaling[name]) for name, (k, digest) in CAPTURES.items()],
    }
    out = Path(__file__).with_name("findings.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "sha256": sha(out), "cases": [{k: c[k] for k in ("capture", "stored_FP64_step_norm_squared_certificate", "numerical_scaled_Gram_Rayleigh_quotient", "invalid_reused_old_metric_norm_squared_lower_bound")} for c in result["cases"]]}, indent=2))


if __name__ == "__main__":
    main()
