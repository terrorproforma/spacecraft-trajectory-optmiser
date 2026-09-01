#!/usr/bin/env python3
"""Replay exact dumped P1-C trust retries through QOCO and nonlinear RK4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from diagnose_g3_pd3 import _rk4_step, load_dump

from spacepdhcg.backends import QOCOGPU
from spacepdhcg.cqp import CanonicalCQP
from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.scvx import PoweredDescentOuterConfig, PoweredDescentSCvxSolver
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def _rollout(
    model: PoweredDescent3DOFModel,
    initial: np.ndarray,
    controls: np.ndarray,
    step_seconds: float,
) -> np.ndarray:
    states = np.empty((controls.shape[0] + 1, initial.size), dtype=np.float64)
    states[0] = initial
    for interval, control in enumerate(controls):
        states[interval + 1] = _rk4_step(model, states[interval], control, step_seconds)
    return states


def run(dump: Path, library: Path, intervals: int) -> dict:
    canonical = load_dump(dump)
    config = PoweredDescentSCvxConfig(
        intervals=intervals,
        step_seconds=0.25,
        virtual_l1_weight=10.0,
        virtual_quadratic_weight=1.0e-3,
        virtual_epigraph_regularisation=1.0e-3,
    )
    model = PoweredDescent3DOFModel()
    subproblem = PoweredDescent3DOFSubproblem(model, config)
    if canonical.structure.n_variables != subproblem.layout.n_variables:
        raise ValueError("dump does not match the P1-C decision layout")
    nominal_initial = np.asarray([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 2_000.0])
    controls = np.asarray(
        [
            [0.0, 0.0, 7_422.0 - 0.5 * interval, 7_422.0 - 0.5 * interval]
            for interval in range(intervals)
        ]
    )
    target_states = _rollout(model, nominal_initial, controls, config.step_seconds)
    initial = nominal_initial.copy()
    initial[[0, 2, 3]] += np.asarray([0.1, 1.0, -0.05])
    reference_states = _rollout(model, initial, controls, config.step_seconds)
    merit_owner = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(
            feasibility_penalty=100.0,
            virtual_penalty=100.0,
        ),
    )
    current_merit = merit_owner._actual_merit(
        reference_states,
        controls,
        target_states[-1, :3],
        target_states[-1, 3:6],
    )
    stage_start = 4 * intervals + 3 * (intervals + 1)
    terminal_start = stage_start + 12 * intervals
    attempts: list[dict] = []
    backend: QOCOGPU | None = None
    try:
        radius = 1.0
        while radius >= 1.0e-4:
            values = canonical.values.copy()
            for interval in range(intervals):
                values.affine_offset[stage_start + 12 * interval + 11] = radius
            values.affine_offset[terminal_start + 7] = radius
            problem = CanonicalCQP(canonical.structure, values)
            if backend is None:
                backend = QOCOGPU(
                    problem,
                    library_path=library,
                    tolerance=1.0e-8,
                    iteration_limit=2_000,
                )
            else:
                backend.update(values)
            accepted = False
            for resolve in range(2):
                solution = backend.solve(tolerance=1.0e-8, iteration_limit=2_000)
                states, candidate_controls, virtual, _ = subproblem.decode(solution.primal)
                rollout = _rollout(model, initial, candidate_controls, config.step_seconds)
                model_merit = merit_owner._model_merit(
                    states,
                    candidate_controls,
                    virtual,
                    target_states[-1, :3],
                    target_states[-1, 3:6],
                )
                actual_merit = merit_owner._actual_merit(
                    rollout,
                    candidate_controls,
                    target_states[-1, :3],
                    target_states[-1, 3:6],
                )
                predicted = current_merit - model_merit
                actual = current_merit - actual_merit
                ratio = actual / predicted if predicted > 1.0e-12 else -np.inf
                step_fraction = merit_owner._step_fraction(
                    states,
                    candidate_controls,
                    reference_states,
                    controls,
                    radius,
                )
                forcing_satisfied = bool(
                    max(solution.primal_residual, solution.dual_residual) <= 1.0e-6
                )
                accepted = bool(
                    forcing_satisfied
                    and step_fraction <= 1.0 + 1.0e-7
                    and actual > 1.0e-10
                    and ratio >= 0.05
                )
                attempts.append(
                    {
                        "trust_radius": radius,
                        "resolve": resolve,
                        "accepted": accepted,
                        "forcing_satisfied": forcing_satisfied,
                        "step_fraction": step_fraction,
                        "maximum_trust_distance": max(
                            0.0,
                            radius * (step_fraction - 1.0),
                        ),
                        "predicted": predicted,
                        "actual": actual,
                        "ratio": ratio,
                        "primal": solution.primal_residual,
                        "dual": solution.dual_residual,
                    }
                )
                if accepted or forcing_satisfied:
                    break
            if accepted:
                break
            radius *= 0.5
    finally:
        if backend is not None:
            backend.close()
    for attempt in attempts:
        for key in ("predicted", "actual", "ratio"):
            if not np.isfinite(attempt[key]):
                attempt[key] = None
    return {"current_merit": current_merit, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(
        run(arguments.dump, arguments.library, arguments.intervals),
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
