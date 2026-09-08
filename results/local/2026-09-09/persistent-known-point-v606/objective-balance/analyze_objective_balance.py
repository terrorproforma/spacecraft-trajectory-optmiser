#!/usr/bin/env python3
"""CPU-only scaling assessment for two immutable zero-P, generic v605 captures.

Run from the repository root with Python 3.10+ (standard library only):
  python -B build/performance/objective-balance-v606/analyze_objective_balance.py \
    --output build/performance/objective-balance-v606/findings.json

No solver, CUDA, network, source mutation, or acceptance-gate change is involved.
The ten Ruiz max-equilibration passes reproduce the reviewed source policy.
Norms use sequential FP64 sums; GPU parallel reduction rounding can differ.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile


CAPTURES = {
    "conditioning": "1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf",
    "difficult": "14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080",
}
CORE_SHA = "d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633"
SOURCE_SHA = {
    "cpp/cuda/src/persistent_pdhcg.cu": "af2ffcc7d101bb1c9cf5069274c03c52855653d17476675047573539729c5246",
    "cpp/cuda/src/cooperative_pdhg.cuh": "3ca44177992eaadd6acc3f4be7c9360949c27b445dede4db15247dd63cfead98",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(values):
    total = 0.0
    for value in values:
        total += value * value
    return math.sqrt(total)


def array_sha(values):
    return hashlib.sha256(b"".join(struct.pack("<d", v) for v in values)).hexdigest()


def identity(root):
    base = root / "build/performance"
    manifest_path = base / "persistent-replay-v603/manifest.json"
    archive = base / "persistent-replay-v603/source.tar.gz"
    adapter_path = base / "persistent-bound-ablation-v605/adapter-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    adapter = json.loads(adapter_path.read_text())
    require(manifest["complete"] and adapter["complete"], "incomplete build evidence")
    require(manifest["build/cuda/libspacepdhcg_cuda.so"]["sha256"] == CORE_SHA
            == adapter["immutable_core_sha256"], "core identity mismatch")
    with tarfile.open(archive, "r:gz") as source:
        for name, expected in SOURCE_SHA.items():
            actual = hashlib.sha256(source.extractfile(name).read()).hexdigest()
            require(actual == expected == manifest["source_sha256"][name]
                    == adapter["source_sha256"][name], "frozen source identity mismatch")
    return {
        "core_sha256": CORE_SHA,
        "core_bytes": manifest["build/cuda/libspacepdhcg_cuda.so"]["bytes"],
        "core_path_recorded_in_existing_evidence": adapter["immutable_core_path"],
        "core_identity_scope": "cross-checked existing build manifests and run metadata; binary not loaded or rehashed here",
        "core_source_commit": manifest["frozen_commit"],
        "core_manifest_sha256": sha(manifest_path),
        "core_source_archive_sha256": sha(archive),
        "reviewed_source_sha256": SOURCE_SHA,
        "adapter_manifest_sha256": sha(adapter_path),
        "adapter_source_commit": adapter["frozen_commit"],
        "adapter_executable_sha256": adapter["executable_sha256"],
    }


def analyze(root, label, expected):
    base = root / "build/performance/persistent-bound-ablation-v605"
    path = base / "inputs" / (label + ".txt")
    log_path = base / (label + "-generic.log")
    require(sha(path) == expected, "capture identity mismatch")
    meta = [json.loads(line.split(" ", 1)[1]) for line in log_path.read_text().splitlines()
            if line.startswith("PERSISTENT_REPLAY_META ")]
    require(len(meta) == 1 and meta[0]["input_sha256"] == expected
            and meta[0]["library_sha256"] == CORE_SHA
            and meta[0]["fold_singleton_bounds"] is False, "generic replay binding mismatch")
    lines = path.read_text().splitlines()
    require(len(lines) == 15 and lines[0] == "SPACEPDHCG_QOCO_QP_V1", "unsupported capture")
    n, p, m, nq, na, ng, nonnegative, nsoc, shifted = map(int, lines[1].split())

    def vector(index, convert):
        words = lines[index].split()
        require(int(words[0]) == len(words) - 1, "vector count mismatch")
        return [convert(value) for value in words[1:]]

    values = vector(11, float)
    require(len(values) == nq + na + ng + n + p + m and all(map(math.isfinite, values)), "bad values")
    require(shifted == 0 and not vector(12, float) and not vector(13, float), "only unshifted captures supported")
    require(all(value == 0 for value in values[:nq]), "zero-P specialization does not apply")
    offset = float(lines[14])
    require(offset == 0, "unexpected unshifted objective offset")
    soc = vector(10, int)
    require(len(soc) == nsoc and all(size >= 3 for size in soc)
            and nonnegative + sum(soc) == m, "invalid cone inventory")
    ap, ai, gp, gi = vector(6, int), vector(7, int), vector(8, int), vector(9, int)
    av, gv = values[nq:nq + na], values[nq + na:nq + na + ng]
    c = values[nq + na + ng:nq + na + ng + n]
    b, h = values[nq + na + ng + n:nq + na + ng + n + p], values[-m:]
    # Generic scalar rows [A; G_nonnegative], followed by -G_SOC with radius last.
    permutation = list(range(nonnegative))
    start = nonnegative
    for size in soc:
        permutation.extend(range(start + 1, start + size))
        permutation.append(start)
        start += size
    inverse = {old: new for new, old in enumerate(permutation)}
    entries = []
    for offsets, indices, data, rows in ((ap, ai, av, p), (gp, gi, gv, m)):
        require(len(offsets) == n + 1 and offsets[0] == 0
                and offsets[-1] == len(data) == len(indices), "CSC count mismatch")
        for column in range(n):
            require(offsets[column] <= offsets[column + 1], "CSC offsets unsorted")
            prior = -1
            for k in range(offsets[column], offsets[column + 1]):
                row = indices[k]
                require(prior < row < rows, "CSC row order/range mismatch")
                prior = row
                native_row = row if offsets is ap else p + inverse[row]
                entries.append((native_row, column, abs(data[k])))
    D, R = [1.0] * n, [1.0] * (p + m)
    passes = []
    for iteration in range(10):
        columns, rows = [0.0] * n, [0.0] * (p + m)
        for row, column, value in entries:
            scaled = value / (R[row] * D[column])
            columns[column] = max(columns[column], scaled)
            rows[row] = max(rows[row], scaled)
        start = p + nonnegative
        for size in soc:
            maximum = max(rows[start:start + size])
            rows[start:start + size] = [maximum] * size
            start += size
        D = [d * (math.sqrt(v) if v > 1e-12 else 1.0) for d, v in zip(D, columns)]
        R = [r * (math.sqrt(v) if v > 1e-12 else 1.0) for r, v in zip(R, rows)]
        passes.append({"pass": iteration + 1, "D_min": min(D), "D_max": max(D),
                       "R_min": min(R), "R_max": max(R), "C": norm(v / d for v, d in zip(c, D))})
    C = norm(value / scale for value, scale in zip(c, D))
    rhs = b + [h[i] for i in permutation]
    B = 1.0 / (norm(value / scale for value, scale in zip(rhs, R)) + 1.0)
    scenarios = []
    for exponent in (-13, -19):
        alpha = math.ldexp(1.0, exponent)
        for value in values[:nq] + c + [offset]:
            scaled = math.ldexp(value, exponent)
            require(math.isfinite(scaled) and Fraction(scaled) == Fraction(value) * Fraction(alpha),
                    "objective scaling loses an exact binary64 coefficient")
        rho = alpha * (C + 1.0) / (alpha * C + 1.0)
        scenarios.append({"exponent": exponent, "alpha": alpha, "alpha_C": alpha * C,
                          "objective_coefficients_exact_in_binary64": True,
                          "effective_original_primal_step_ratio": rho,
                          "effective_original_dual_step_ratio": 1.0 / rho,
                          "primal_step_change_percent": 100.0 * (rho - 1.0),
                          "dual_step_change_percent": 100.0 * (1.0 / rho - 1.0)})
    return {"capture": label, "input_relative_path": str(path.relative_to(root)).replace("\\", "/"),
            "input_sha256": expected, "generic_log_sha256": sha(log_path),
            "n": n, "p": p, "m": m, "nonnegative_rows": nonnegative, "soc_count": nsoc,
            "quadratic_structural_entries": nq, "quadratic_numerical_nonzeros": 0,
            "ruiz_passes": passes, "D_sha256_le_float64": array_sha(D), "R_sha256_le_float64": array_sha(R),
            "c_max_abs": max(map(abs, c)), "c_norm": norm(c), "C": C,
            "bound_scale_B": B, "objective_scale_O": 1.0 / (C + 1.0), "scenarios": scenarios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "SPACEPDHCG_OBJECTIVE_BALANCE_CPU_V1",
        "scope": "frozen-source scaling analysis, generic zero-P captures; no solver execution or speedup claim",
        "script_sha256": sha(Path(__file__)),
        "identities": identity(args.repo_root),
        "arithmetic": "FP64 ten Ruiz passes and sequential norm sums; GPU reduction rounding may differ",
        "derivation": {
            "ruiz": "D/R start at one; ten max-norm square-root passes on A/F; share affine SOC row maximum; factor=1 when maximum<=1e-12; P/c not used",
            "normalizers": "C=||c/D||2; O=1/(C+1); B=1/(||rhs/R||2+1)",
            "diagonal_steps": "primal=O/(D^2 B), dual=B/(R^2 O)",
            "zero_P": "P=0 implies q_norm=0; operator norm, B, D, R and both base steps unchanged by objective scaling",
            "ratios": "rho=alpha*(C+1)/(alpha*C+1); original-coordinate primal ratio=rho, dual ratio=1/rho",
            "general_P_caveat": "For nonzero P, q_norm scales by rho and primal ratio gains max(1,q_norm+Knorm)/max(1,rho*q_norm+Knorm); not evaluated here",
        },
        "mapping_and_guards": [
            "Scale all P/c/objective-offset coefficients by positive exact power-of-two alpha; preserve constraints, topology, origin, primal and slack.",
            "Scale initial y/z by alpha; recover original y/z by 1/alpha before auditing original unscaled equations.",
            "Reject overflow and every inexact subnormal conversion, including nonzero-to-nonzero rounding; retain signed zeros.",
            "Absolute native stopping tolerance is not transformation invariant; preserve original external gates and compare fixed budgets or equal qualified accuracy.",
        ],
        "cases": [analyze(args.repo_root, label, expected) for label, expected in CAPTURES.items()],
        "finding": "alpha=2^-13 mostly cancels existing objective normalization: only about 2% effective balance change. alpha=2^-19 changes balance materially but improvement is untested. Do not repeat 2^-13 expecting an 8192-fold step-balance change.",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
