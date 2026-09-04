"""Cross-solver correctness suite for deterministic trajectory-banded fixtures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from typing import Any

from spacepdhcg.backends import PDHCGOneShot, PersistentClarabel, PersistentOSQP
from spacepdhcg.benchmarks.trajectory_banded import (
    BandedControlConstraint,
    TrajectoryBandedConfig,
    TrajectoryBandedFixture,
)
from spacepdhcg.cqp import CQPSolution


def _solution_record(
    fixture: TrajectoryBandedFixture,
    solution: CQPSolution,
) -> dict[str, Any]:
    diagnostics = fixture.diagnostics(solution.primal)
    return {
        "status": solution.status,
        "objective": solution.objective,
        "known_objective": fixture.known_objective,
        "objective_gap_abs": diagnostics.objective_gap_abs,
        "solution_error_inf": diagnostics.solution_error_inf,
        "scalar_violation_inf": diagnostics.scalar_violation_inf,
        "dynamics_defect_inf": diagnostics.dynamics_defect_inf,
        "control_violation_inf": diagnostics.control_violation_inf,
        "reported_primal_residual": solution.primal_residual,
        "reported_dual_residual": solution.dual_residual,
        "iterations": solution.iterations,
        "solve_seconds": solution.solve_seconds,
    }


def _assert_solution(
    name: str,
    fixture: TrajectoryBandedFixture,
    solution: CQPSolution,
    tolerance: float,
) -> None:
    if not solution.solved:
        raise RuntimeError(f"{name} failed with status {solution.status!r}")
    diagnostics = fixture.diagnostics(solution.primal)
    acceptance = max(1.0e-6, 100.0 * tolerance)
    if (
        max(
            diagnostics.solution_error_inf,
            diagnostics.scalar_violation_inf,
            diagnostics.dynamics_defect_inf,
            diagnostics.control_violation_inf,
        )
        > acceptance
    ):
        raise RuntimeError(f"{name} failed independent checks: {diagnostics}")


def run_suite(
    *,
    seeds: Iterable[int] = (17, 29, 41),
    intervals: int = 8,
    tolerance: float = 1.0e-8,
    include_pdhcg: bool = False,
) -> dict[str, Any]:
    """Run exact-optimum QP/SOCP fixtures and return a machine-readable record."""

    cases: list[dict[str, Any]] = []
    pdhcg_backend_version: str | None = None
    for seed in seeds:
        for control_constraint in (
            BandedControlConstraint.BOX,
            BandedControlConstraint.SECOND_ORDER_CONE,
        ):
            fixture = TrajectoryBandedFixture(
                TrajectoryBandedConfig(
                    intervals=intervals,
                    seed=int(seed),
                    control_constraint=control_constraint,
                )
            )
            solvers: dict[str, Any] = {
                "clarabel": PersistentClarabel(
                    fixture.canonical,
                    tolerance=tolerance,
                )
            }
            if control_constraint is BandedControlConstraint.BOX:
                solvers["osqp"] = PersistentOSQP(fixture.canonical)
            if include_pdhcg:
                pdhcg_backend = PDHCGOneShot(
                    fixture.canonical,
                    params={"LogLevel": 0},
                )
                pdhcg_backend_version = pdhcg_backend.upstream_version
                solvers["pdhcg_one_shot"] = pdhcg_backend

            records: dict[str, Any] = {}
            for name, backend in solvers.items():
                if isinstance(backend, PersistentClarabel):
                    solution = backend.solve()
                else:
                    solution = backend.solve(tolerance=tolerance)
                _assert_solution(name, fixture, solution, tolerance)
                record = _solution_record(fixture, solution)
                record["setup_seconds"] = float(getattr(backend, "setup_seconds", 0.0))
                record["total_seconds"] = float(
                    getattr(backend, "last_total_seconds", solution.solve_seconds)
                )
                records[name] = record

            cases.append(
                {
                    "seed": int(seed),
                    "control_constraint": str(control_constraint),
                    "intervals": intervals,
                    "variables": fixture.structure.n_variables,
                    "scalar_constraints": fixture.structure.n_constraints,
                    "affine_cone_rows": fixture.structure.n_affine_constraints,
                    "solvers": records,
                }
            )

    return {
        "suite": "trajectory-banded exact-optimum correctness",
        "tolerance": tolerance,
        "include_pdhcg": include_pdhcg,
        "pdhcg_version": pdhcg_backend_version,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--intervals", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--pdhcg",
        action="store_true",
        help="also run the optional CUDA-enabled one-shot PDHCG backend",
    )
    arguments = parser.parse_args()
    result = run_suite(
        seeds=arguments.seeds,
        intervals=arguments.intervals,
        tolerance=arguments.tolerance,
        include_pdhcg=arguments.pdhcg,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
