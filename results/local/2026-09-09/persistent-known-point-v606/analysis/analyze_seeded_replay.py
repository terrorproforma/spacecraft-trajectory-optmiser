#!/usr/bin/env python3
"""Read immutable v606 evidence; diagnose native stopping and original gap drift.

Standard library only; no CUDA/solver execution or acceptance-policy changes.
Run from repository root:
  python -B build/performance/known-point-replay-v606/analysis/analyze_seeded_replay.py --output build/performance/known-point-replay-v606/analysis/findings.json
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def dec(value):
    return Decimal.from_float(float(value))


def bits(values):
    return b"".join(struct.pack("<d", value) for value in values)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Decimal(0))


def peak(values):
    index = max(range(len(values)), key=lambda i: abs(values[i])) if values else None
    return {"index": index, "absolute": float(abs(values[index])) if index is not None else 0.0,
            "signed": float(values[index]) if index is not None else 0.0}


class Problem:
    def __init__(self, path):
        lines = path.read_text().splitlines()
        require(len(lines) == 15 and lines[0] == "SPACEPDHCG_QOCO_QP_V1", "snapshot format")
        self.n, self.p, self.m, nq, na, ng, self.l, ns, shift = map(int, lines[1].split())
        require(shift == 0, "this analysis covers unshifted captures only")
        v = [dec(x) for x in lines[11].split()[1:]]
        require(len(v) == nq + na + ng + self.n + self.p + self.m
                and all(x == 0 for x in v[:nq]), "zero-P analysis precondition")
        self.ptr = [[int(x) for x in lines[i].split()[1:]] for i in (6, 8)]
        self.idx = [[int(x) for x in lines[i].split()[1:]] for i in (7, 9)]
        pp, pi = [[int(x) for x in lines[i].split()[1:]] for i in (4, 5)]
        self.P_pattern = {(pi[k], j) for j in range(self.n) for k in range(pp[j], pp[j + 1])}
        self.data = [v[nq:nq + na], v[nq + na:nq + na + ng]]
        at = nq + na + ng
        self.c, self.b, self.h = v[at:at + self.n], v[at + self.n:at + self.n + self.p], v[-self.m:]
        self.soc = [int(x) for x in lines[10].split()[1:]]
        require(len(self.soc) == ns and self.l + sum(self.soc) == self.m, "cone count")
        rows = [[] for _ in range(self.l)]
        for j in range(self.n):
            for k in range(self.ptr[1][j], self.ptr[1][j + 1]):
                row, value = self.idx[1][k], self.data[1][k]
                if row < self.l and value != 0:
                    rows[row].append((j, value))
        self.folded, self.lower, self.upper = set(), [Decimal('-Infinity')] * self.n, [Decimal('Infinity')] * self.n
        for row, entries in enumerate(rows):
            if len(entries) == 1:
                j, value = entries[0]
                ratio = Fraction(self.h[row]) / Fraction(value)
                bound = float(ratio)
                if math.isfinite(bound) and Fraction(bound) == ratio:
                    self.folded.add(row)
                    if value > 0:
                        self.upper[j] = min(self.upper[j], dec(bound))
                    else:
                        self.lower[j] = max(self.lower[j], dec(bound))

    def multiply(self, matrix, x):
        result = [Decimal(0)] * (self.p if matrix == 0 else self.m)
        for j in range(self.n):
            for k in range(self.ptr[matrix][j], self.ptr[matrix][j + 1]):
                result[self.idx[matrix][k]] += self.data[matrix][k] * x[j]
        return result

    def transpose(self, matrix, y):
        return [sum((self.data[matrix][k] * y[self.idx[matrix][k]]
                     for k in range(self.ptr[matrix][j], self.ptr[matrix][j + 1])), Decimal(0))
                for j in range(self.n)]


def soc_project(values):
    radius, vector = values[0], values[1:]
    length = dot(vector, vector).sqrt()
    if length <= radius:
        return values
    if length <= -radius:
        return [Decimal(0)] * len(values)
    projected = (radius + length) / 2
    return [projected] + [projected * v / length for v in vector]


def equivalent_objective_assessment(root, label):
    """Structural/normalization calculations only; no augmented solver is built."""
    path = root / "inputs" / (label + ".txt")
    q = Problem(path)
    entries, equality_rows = [], [[] for _ in range(q.p)]
    for matrix in (0, 1):
        for j in range(q.n):
            for k in range(q.ptr[matrix][j], q.ptr[matrix][j + 1]):
                i, value = q.idx[matrix][k], float(q.data[matrix][k])
                entries.append((i if matrix == 0 else q.p + i, j, abs(value)))
                if matrix == 0:
                    equality_rows[i].append((j, value))
    D, R = [1.0] * q.n, [1.0] * (q.p + q.m)
    for _ in range(10):
        columns, rows = [0.0] * q.n, [0.0] * (q.p + q.m)
        for i, j, value in entries:
            value /= D[j] * R[i]
            columns[j], rows[i] = max(columns[j], value), max(rows[i], value)
        start = q.p + q.l
        for size in q.soc:
            rows[start:start + size] = [max(rows[start:start + size])] * size
            start += size
        D = [d * (math.sqrt(v) if v > 1e-12 else 1.0) for d, v in zip(D, columns)]
        R = [r * (math.sqrt(v) if v > 1e-12 else 1.0) for r, v in zip(R, rows)]
    total = 0.0
    for value, scale in zip(q.c, D):
        total += (float(value) / scale) ** 2
    C = math.sqrt(total)
    stored, nonzero = set(), set()
    for row in equality_rows:
        for slot, (i, a) in enumerate(row):
            for j, b in row[slot:]:
                stored.add((i, j))
                if a != 0 and b != 0:
                    nonzero.add((i, j))
    def counts(pattern):
        diagonal = sum(i == j for i, j in pattern)
        return {"upper": len(pattern), "full": 2 * len(pattern) - diagonal}
    return {
        "label": label, "input_sha256": sha(path), "ruiz_passes": 10, "C_after_Ruiz": C,
        "objective_scaling": [{"exponent": exponent, "alpha": 2.0 ** exponent,
                               "effective_original_primal_ratio": (2.0 ** exponent) * (C + 1) / ((2.0 ** exponent) * C + 1),
                               "effective_original_dual_ratio": ((2.0 ** exponent) * C + 1) / ((2.0 ** exponent) * (C + 1))}
                              for exponent in (-13, -19)],
        "equality_augmentation": {"n": q.n, "p": q.p, "minimum_nullity_of_AtA": q.n - q.p,
                                  "A_stored_entries": len(q.data[0]),
                                  "A_nonzero_entries": sum(v != 0 for v in q.data[0]),
                                  "maximum_A_row_width": max(map(len, equality_rows)),
                                  "original_P": counts(q.P_pattern), "AtA_stored_pattern": counts(stored),
                                  "AtA_nonzero_support_pattern": counts(nonzero),
                                  "augmented_union_with_original_P": counts(stored | q.P_pattern),
                                  "augmented_nonzero_support_union": counts(nonzero | q.P_pattern)},
    }


def analyze(root, case):
    label, name, folded = case["label"], case["name"], case["folded"]
    source = root / "inputs" / (label + ".txt")
    point = root / "inputs" / (label + "-initial.txt")
    log = root / "run" / (name + ".log")
    require(sha(source) == case["input_sha256"] and sha(point) == case["point_sha256"]
            and sha(log) == case["log_sha256"], "input/point/log identity mismatch")
    records = {}
    for line in log.read_text().splitlines():
        if line.startswith("PERSISTENT_REPLAY"):
            prefix, payload = line.split(" ", 1)
            # parse_int=float preserves a serialized '-0' iterate component.
            records.setdefault(prefix, []).append(json.loads(payload, parse_int=float))
    meta, initial, pre, final = [records[key][0] for key in (
        "PERSISTENT_REPLAY_META", "PERSISTENT_REPLAY_INITIAL_POINT",
        "PERSISTENT_REPLAY_PRESTEP", "PERSISTENT_REPLAY")]
    require(meta["library_sha256"] == "d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633"
            and meta["source_commit"] == "659f5abd7b41d16dbeac980a6e10d6ca92666314", "runtime identity")
    require(pre["seeded_iterations"] == 0 and pre["termination"] == 0
            and pre["native_seed_verified_unchanged"] and pre["solve_epoch"] > 0, "pre-step lifecycle")
    q = Problem(source)
    point_lines = point.read_text().splitlines()
    pv = [[float(x) for x in point_lines[i].split()[2:]] for i in (3, 4, 5, 6)]
    for key, vector in zip(("x", "y", "z", "s"), pv):
        require(bits(initial[key]) == bits(vector), "encoded/reference bit mismatch")
    native_dual = pv[1] + [v for i, v in enumerate(pv[2][:q.l]) if not folded or i not in q.folded]
    start = q.l
    for size in q.soc:
        native_dual.extend(-pv[2][i] for i in range(start + 1, start + size))
        native_dual.append(-pv[2][start])
        start += size
    require(bits(native_dual) == bits(initial["dual_solver"])
            and bits(pv[0]) == bits(initial["x_solver"]), "native seed conversion mismatch")
    x0, y0, z0 = [[dec(x) for x in initial[k]] for k in ("x", "y", "z")]
    x1, y1, z1 = [[dec(x) for x in final[k]] for k in ("x", "y", "z")]
    ax0, gx0 = q.multiply(0, x0), q.multiply(1, x0)
    slack0 = [h - gx for h, gx in zip(q.h, gx0)]
    scalar = [abs(a - b) for a, b in zip(ax0, q.b)]
    scalar_rows = list(range(q.p))
    candidates = []
    for i in range(q.l):
        natural = abs(gx0[i] - min(gx0[i] + z0[i], q.h[i]))
        candidates.append({"G_row": i, "natural": float(natural), "slack": float(slack0[i]),
                           "dual": float(z0[i]), "slack_times_dual": float(slack0[i] * z0[i]),
                           "folded": i in q.folded})
        if not folded or i not in q.folded:
            scalar.append(natural)
            scalar_rows.append(q.p + i)
    scalar_peak = peak(scalar)
    scalar_peak["original_A_then_G_row"] = scalar_rows[scalar_peak["index"]]
    soc_values = []
    start = q.l
    for size in q.soc:
        projected = soc_project([slack0[i] - z0[i] for i in range(start, start + size)])
        soc_values.extend(slack0[start + i] - projected[i] for i in range(size))
        start += size
    zmapped = [Decimal(0) if folded and i in q.folded else v for i, v in enumerate(z0)]
    native_gradient = [c + a + g for c, a, g in zip(q.c, q.transpose(0, y0), q.transpose(1, zmapped))]
    stationarity = [x0[j] - min(q.upper[j], max(q.lower[j], x0[j] - native_gradient[j]))
                    if folded else native_gradient[j] for j in range(q.n)]
    ax1, gx1 = q.multiply(0, x1), q.multiply(1, x1)
    residual = [c + a + g for c, a, g in zip(q.c, q.transpose(0, y1), q.transpose(1, z1))]
    dx, dy, dz = [[b - a for a, b in zip(v0, v1)] for v0, v1 in ((x0, x1), (y0, y1), (z0, z1))]
    delta_parts = [dot(q.c, dx), dot(q.b, dy), dot(q.h[:q.l], dz[:q.l]), dot(q.h[q.l:], dz[q.l:])]
    gap_parts = [dot(x1, residual), dot(y1, [b - a for b, a in zip(q.b, ax1)]),
                 dot(z1, [h - g for h, g in zip(q.h, gx1)])]
    f0, d0 = dot(q.c, x0), -dot(q.b, y0) - dot(q.h, z0)
    f1, d1 = dot(q.c, x1), -dot(q.b, y1) - dot(q.h, z1)
    require(abs(sum(gap_parts) - (f1 - d1)) < Decimal('1e-40')
            and abs(sum(delta_parts) - ((f1 - d1) - (f0 - d0))) < Decimal('1e-40'), "gap decomposition")
    audit = case["audits"][0]
    require(bool(audit["passes_common_kkt_gate"]) == bool(final["kkt_qualified_original"])
            and bool(audit["qualified"]) == bool(final["qualified_original"]), "audit gate mismatch")
    failing = [key for key in ("primal", "dual", "gap", "complementarity_max_relative") if audit[key] > 1e-9]
    if audit["cone_violation"] > 1e-8:
        failing.append("cone_violation")
    return {
        "name": name, "input_sha256": case["input_sha256"], "point_sha256": case["point_sha256"],
        "log_sha256": case["log_sha256"], "encoded_and_native_seed_bits_verified": True,
        "reported_pre_step": {k: v for k, v in pre.items() if k.startswith("native_") or k in ("seeded_iterations", "termination", "solve_epoch")},
        "cpu_seed_natural_components": {"scalar": scalar_peak, "SOC": peak(soc_values), "stationarity": peak(stationarity)},
        "largest_original_nonnegative_natural_rows": sorted(candidates, key=lambda row: row["natural"], reverse=True)[:3],
        "source_reference_qualified": initial["mapped_reference_qualified"],
        "strict_reconstructed_seed_qualified": initial["strict_reconstructed_qualified"],
        "iterations": int(final["iterations"]), "termination": int(final["termination"]),
        "external_kkt_pass": audit["passes_common_kkt_gate"], "native_optimal": final["solver_optimal"],
        "external_failing_components": failing, "final_external_audit": audit,
        "primal_change": peak(dx), "equality_dual_change": peak(dy), "conic_dual_change": peak(dz),
        "source_primal_objective": float(f0), "source_dual_objective": float(d0),
        "final_primal_objective": float(f1), "final_dual_objective": float(d1),
        "signed_gap_change_decomposition": dict(zip(("c_dot_delta_x", "b_dot_delta_y", "h_dot_delta_z_nonnegative", "h_dot_delta_z_SOC"), map(float, delta_parts))),
        "final_signed_gap_decomposition": dict(zip(("x_dot_stationarity", "y_dot_b_minus_Ax", "z_dot_h_minus_Gx"), map(float, gap_parts))),
        "folded_dual_caveat": "Folded delta_z includes the export reconstruction, since those original multipliers are absent from native state." if folded else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/performance/known-point-replay-v606"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report_path = args.root / "run/report.json"
    require(sha(report_path) == "8c51d621edb648f8fdb355972c4c01add7365fa7c43c9004bf1790cc0f6c452c", "report identity")
    report = json.loads(report_path.read_text())
    require(report["complete"] and len(report["cases"]) == 8, "incomplete campaign")
    require(sha(args.root / "run/adapter-manifest.json") == report["manifest_sha256"]
            and sha(args.root / "run/tiny-report.json") == report["tiny_report_sha256"], "manifest binding")
    with localcontext() as context:
        context.prec = 65
        manifest = json.loads((args.root / "run/adapter-manifest.json").read_text())
        findings = {"scope": "saved-evidence CPU diagnosis; no solver/GPU runs or gate changes",
                    "script_sha256": sha(Path(__file__)), "report_sha256": sha(report_path),
                    "source_pin": report["source_pin"], "core_sha256": report["library_sha256"],
                    "executable_sha256": report["executable_sha256"],
                    "auditor_sha256": sha(args.root / "source/audit_persistent_snapshot.py"),
                    "core_sources_sha256": {name: manifest["source_sha256"][name] for name in
                        ("cpp/cuda/src/persistent_pdhcg.cu", "cpp/cuda/src/cooperative_pdhg.cuh")},
                    "source_locations": {"scalar_natural": "cpp/cuda/src/persistent_pdhcg.cu:992",
                                         "natural_residual_without_complementarity": "cpp/cuda/src/persistent_pdhcg.cu:1102",
                                         "scalar_dual_update": "cpp/cuda/src/persistent_pdhcg.cu:1226",
                                         "explicit_primal_update": "cpp/cuda/src/persistent_pdhcg.cu:1274",
                                         "first_check_after_update": "cpp/cuda/src/persistent_pdhcg.cu:1295",
                                         "absolute_natural_stop": "cpp/cuda/src/persistent_pdhcg.cu:1305",
                                         "cooperative_update": "cpp/cuda/src/cooperative_pdhg.cuh:898",
                                         "cooperative_stop": "cpp/cuda/src/cooperative_pdhg.cuh:929"},
                    "arithmetic": "exact FP64 inputs extended to 65-digit Decimal products/projections; native GPU reductions may round differently",
                    "cases": [analyze(args.root, case) for case in report["cases"]],
                    "equivalent_objective_assessments": [equivalent_objective_assessment(args.root, label)
                                                         for label in ("conditioning", "difficult")]}
    encoded = json.dumps(findings, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
