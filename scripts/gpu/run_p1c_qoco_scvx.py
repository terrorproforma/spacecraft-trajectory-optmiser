#!/usr/bin/env python3
"""Run the displaced P1-C nonlinear lifecycle with the pure QOCO-GPU backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spacepdhcg.backends import QOCOGPU
from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.scvx import (
    PoweredDescentOuterConfig,
    PoweredDescentSCvxSolver,
    TrustRegionConfig,
)
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def run(library: Path, intervals: int, dispersion: float, maximum_attempts: int) -> dict:
    config = PoweredDescentSCvxConfig(
        intervals=intervals,
        step_seconds=0.25,
        virtual_l1_weight=10.0,
        virtual_quadratic_weight=1.0e-3,
        virtual_epigraph_regularisation=1.0e-3,
    )
    model = PoweredDescent3DOFModel()
    subproblem = PoweredDescent3DOFSubproblem(model, config)
    nominal_initial = np.asarray([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 2_000.0])
    controls = np.asarray(
        [
            [0.0, 0.0, 7_422.0 - 0.5 * interval, 7_422.0 - 0.5 * interval]
            for interval in range(intervals)
        ]
    )
    nominal_states = model.rollout(nominal_initial, controls, config.step_seconds)
    initial = nominal_initial.copy()
    initial[0] += 10.0 * dispersion
    initial[2] += 100.0 * dispersion
    initial[3] -= 5.0 * dispersion
    reference_states = model.rollout(initial, controls, config.step_seconds)

    workspaces: list[QOCOGPU] = []

    def builder(problem, **settings):
        backend = QOCOGPU(problem, library_path=library, **settings)
        workspaces.append(backend)
        return backend

    solver = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(
            max_iterations=maximum_attempts,
            minimum_iterations=1,
            convergence_tolerance=1.0e-6,
            step_tolerance=2.0e-2,
            acceptance_threshold=0.05,
            restoration_reduction=0.9,
            feasibility_penalty=100.0,
            virtual_penalty=100.0,
            max_resolves_per_iteration=1,
        ),
        trust_config=TrustRegionConfig(
            initial_radius=1.0,
            minimum_radius=1.0e-4,
            maximum_radius=8.0,
            shrink_factor=0.5,
            growth_factor=1.8,
            rejection_threshold=0.05,
            strong_agreement=0.75,
            boundary_fraction=0.8,
        ),
        backend_builder=builder,
    )
    result = solver.solve(
        initial,
        nominal_states[-1, :3],
        nominal_states[-1, 3:6],
        reference_states=reference_states,
        reference_controls=controls,
    )

    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "status": result.status,
        "accepted": result.accepted_iterations,
        "terminal": result.residual.terminal,
        "workspace_creations": len(workspaces),
        "numeric_updates": sum(backend.update_count for backend in workspaces),
        "solves": sum(backend.solve_count for backend in workspaces),
        "iterations": [
            {
                "outer": record.iteration,
                "trust_before": record.trust_radius_before,
                "trust_after": record.trust_radius_after,
                "accepted": record.accepted,
                "step_fraction": record.step_fraction,
                "predicted": finite(record.predicted_reduction),
                "actual": finite(record.actual_reduction),
                "ratio": finite(record.agreement),
                "primal": record.primal_residual,
                "dual": record.dual_residual,
            }
            for record in result.iterations
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--dispersion", type=float, default=0.01)
    parser.add_argument("--maximum-attempts", type=int, default=15)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(
        run(
            arguments.library,
            arguments.intervals,
            arguments.dispersion,
            arguments.maximum_attempts,
        ),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
