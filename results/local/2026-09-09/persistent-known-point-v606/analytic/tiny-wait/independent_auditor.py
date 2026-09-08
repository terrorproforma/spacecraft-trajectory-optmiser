#!/usr/bin/env python3
"""Audit replay vectors against original QOCO snapshot equations on the CPU.

No persistent cone conversion is reused. Sparse products and scalar reductions
use long double. This checks a conic KKT gate, not nonlinear trajectory physics
or positive semidefiniteness of an arbitrary supplied Hessian.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_snapshot(path: Path) -> dict:
    lines = path.read_text().splitlines()
    if len(lines) != 15 or lines[0] != "SPACEPDHCG_QOCO_QP_V1":
        raise ValueError("expected a 15-line QOCO V1 snapshot")
    n, p, m, np_, na, ng, nonnegative, ns, shift = map(int, lines[1].split())
    if n < 1 or min(p, m, np_, na, ng, nonnegative, ns) < 0 or shift not in (0, 1):
        raise ValueError("invalid snapshot dimensions")

    def vector(line: int, dtype=np.longdouble):
        parts = lines[line].split()
        if not parts or int(parts[0]) != len(parts) - 1:
            raise ValueError(f"invalid vector at line {line + 1}")
        # The native reader consumes FP64. Extend those exact binary values,
        # rather than independently rounding the decimal text to long double.
        input_dtype = np.float64 if dtype == np.longdouble else dtype
        result = np.asarray(parts[1:], dtype=input_dtype).astype(dtype, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite snapshot input")
        return result

    values, translated, origin = [vector(i) for i in (11, 12, 13)]
    counts = (np_, na, ng, n, p, m)
    starts = np.cumsum((0, *counts))
    if (
        len(values) != starts[-1]
        or len(translated) != (len(values) if shift else 0)
        or len(origin) != (n if shift else 0)
    ):
        raise ValueError("inconsistent numerical vector lengths")
    matrices = []
    for k, rows in enumerate((n, p, m)):
        indptr, indices = vector(4 + 2 * k, np.int64), vector(5 + 2 * k, np.int64)
        if (
            len(indptr) != n + 1
            or len(indices) != counts[k]
            or indptr[0] != 0
            or indptr[-1] != counts[k]
            or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0)
            or np.any(indices >= rows)
        ):
            raise ValueError("invalid CSC topology")
        for column in range(n):
            column_rows = indices[indptr[column] : indptr[column + 1]]
            if np.any(np.diff(column_rows) <= 0):
                raise ValueError("duplicate or unsorted CSC rows")
            if k == 0 and np.any(column_rows > column):
                raise ValueError("snapshot P must use its upper triangle")
        matrices.append(
            sp.csc_matrix((values[starts[k] : starts[k + 1]], indices, indptr), shape=(rows, n))
        )
    upper, A, G = matrices
    P = upper + upper.T - sp.diags(upper.diagonal())
    soc = vector(10, np.int64)
    if len(soc) != ns or np.any(soc < 2) or nonnegative + int(soc.sum()) != m:
        raise ValueError("invalid cone inventory")
    c, b, h = [values[starts[k] : starts[k + 1]] for k in (3, 4, 5)]
    offset = np.longdouble(float(lines[14]))
    if not np.isfinite(offset):
        raise ValueError("invalid offset")
    return dict(
        P=P,
        A=A,
        G=G,
        c=c,
        b=b,
        h=h,
        n=n,
        p=p,
        m=m,
        l=nonnegative,
        soc=soc,
        shift=shift,
        origin=origin,
        translated=translated,
        offset=offset,
        quadratic_structural_nonzeros=np_,
        quadratic_numerical_nonzeros=int(np.count_nonzero(upper.data)),
    )


def audit(
    problem: dict,
    record: dict,
    *,
    backend: str,
    coordinates: str,
    residual_tolerance: float = 1e-9,
    cone_tolerance: float = 1e-8,
) -> dict:
    if backend not in ("persistent", "qoco") or coordinates not in ("original", "translated"):
        raise ValueError("explicit backend and coordinates required")
    if not all(np.isfinite(t) and t > 0 for t in (residual_tolerance, cone_tolerance)):
        raise ValueError("tolerances must be positive and finite")
    vectors = []
    for name, length in (
        ("x", problem["n"]),
        ("y", problem["p"]),
        ("z", problem["m"]),
        ("s", problem["m"]),
    ):
        value = np.asarray(record[name], dtype=np.float64)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid replay vector {name}")
        vectors.append(value)
    x, y, z, s = vectors
    if coordinates == "translated" and problem["shift"]:
        x = x + problem["origin"].astype(np.float64)
    x, y, z, s = [value.astype(np.longdouble) for value in (x, y, z, s)]
    P, A, G, c, b, h = [problem[name] for name in ("P", "A", "G", "c", "b", "h")]
    px, ax, gx, aty, gtz = P @ x, A @ x, G @ x, A.T @ y, G.T @ z

    def inf(v):
        return np.max(np.abs(v), initial=np.longdouble(0))

    equality_absolute = inf(ax - b)
    conic_equation_absolute = inf(gx + s - h)
    primal_absolute = max(equality_absolute, conic_equation_absolute)
    dual_absolute = inf(px + c + aty + gtz)
    primal = primal_absolute / (1 + max(inf(ax), inf(b), inf(gx), inf(h), inf(s)))
    dual = dual_absolute / (1 + max(inf(px), inf(c), inf(aty), inf(gtz)))
    objective = np.dot(x, px) / 2 + np.dot(c, x)
    dual_objective = -np.dot(x, px) / 2 - np.dot(b, y) - np.dot(h, z)
    objective_scale = max(1, abs(objective), abs(dual_objective))
    gap = abs(objective - dual_objective) / objective_scale
    nonnegative = problem["l"]
    primal_cone = max(np.longdouble(0), np.max(-s[:nonnegative], initial=np.longdouble(0)))
    dual_cone = max(np.longdouble(0), np.max(-z[:nonnegative], initial=np.longdouble(0)))
    block_complementarity = inf(s[:nonnegative] * z[:nonnegative])
    start = nonnegative
    for size in problem["soc"]:
        end = start + int(size)
        primal_cone = max(
            primal_cone, np.sqrt(np.dot(s[start + 1 : end], s[start + 1 : end])) - s[start]
        )
        dual_cone = max(
            dual_cone, np.sqrt(np.dot(z[start + 1 : end], z[start + 1 : end])) - z[start]
        )
        block_complementarity = max(block_complementarity, abs(np.dot(s[start:end], z[start:end])))
        start = end
    metrics = dict(
        primal=primal,
        dual=dual,
        gap=gap,
        primal_absolute=primal_absolute,
        equality_absolute=equality_absolute,
        conic_equation_absolute=conic_equation_absolute,
        dual_absolute=dual_absolute,
        primal_cone_violation=primal_cone,
        dual_cone_violation=dual_cone,
        cone_violation=max(primal_cone, dual_cone),
        objective=objective,
        dual_objective=dual_objective,
        complementarity_absolute=abs(np.dot(s, z)),
        complementarity_max_absolute=block_complementarity,
        complementarity_max_relative=block_complementarity / objective_scale,
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("nonfinite KKT audit")
    passes = bool(
        max(primal, dual, gap) <= residual_tolerance
        and block_complementarity / objective_scale <= residual_tolerance
        and max(primal_cone, dual_cone) <= cone_tolerance
    )
    stopped = (
        record.get("termination") == 1
        if backend == "persistent"
        else record.get("status") in (1, 2)
    )
    return {
        **{key: float(value) for key, value in metrics.items()},
        "passes_common_kkt_gate": passes,
        "solver_terminated_accepted": stopped,
        "qualified": passes and stopped,
        "backend": backend,
        "coordinates": coordinates,
        "residual_tolerance": residual_tolerance,
        "cone_tolerance": cone_tolerance,
    }


def audit_log(problem, path, snapshot_sha256, *, backend, coordinates, record_prefix):
    lines = path.read_text().splitlines()
    metadata = [
        json.loads(line[len("PERSISTENT_REPLAY_META ") :])
        for line in lines
        if line.startswith("PERSISTENT_REPLAY_META ")
    ]
    if backend == "persistent":
        if len(metadata) != 1 or metadata[0].get("input_sha256") != snapshot_sha256:
            raise ValueError("persistent replay metadata does not identify this exact snapshot")
        if metadata[0].get("coordinate_system") != coordinates:
            raise ValueError("replay coordinate declaration does not match requested audit")
    results = []
    prefix = record_prefix.rstrip() + " "
    for line in lines:
        if line.startswith(prefix):
            record = json.loads(line[len(prefix) :])
            try:
                result = audit(problem, record, backend=backend, coordinates=coordinates)
            except (ValueError, KeyError, TypeError) as error:
                result = {"qualified": False, "error": str(error)}
            results.append({"repeat": record.get("repeat"), **result})
    if not results:
        raise ValueError("no matching replay records")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("persistent", "qoco"))
    parser.add_argument("--coordinates", required=True, choices=("original", "translated"))
    parser.add_argument("--record-prefix", required=True)
    args = parser.parse_args()
    problem = load_snapshot(args.snapshot)
    snapshot_sha256 = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    results = audit_log(
        problem,
        args.log,
        snapshot_sha256,
        backend=args.backend,
        coordinates=args.coordinates,
        record_prefix=args.record_prefix,
    )
    report = {
        "scope": "original-equation conic KKT audit; no nonlinear or PSD certification",
        "snapshot_sha256": snapshot_sha256,
        "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "numeric_input": "FP64 parsing and origin reconstruction, then long-double products",
        "gate": "relative primal/dual/gap and max block complementarity <=1e-9; cones <=1e-8",
        "gate_note": "Blockwise check prevents cancellation; production tolerances unchanged",
        "quadratic_structural_nonzeros": problem["quadratic_structural_nonzeros"],
        "quadratic_numerical_nonzeros": problem["quadratic_numerical_nonzeros"],
        "qualified_count": sum(result["qualified"] for result in results),
        "results": results,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({key: report[key] for key in ("snapshot_sha256", "qualified_count", "results")})
    )
    return 0 if all(result["qualified"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
