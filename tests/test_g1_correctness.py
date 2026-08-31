from __future__ import annotations

import numpy as np
import pytest

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.benchmarks.g1_correctness import (
    _objective,
    _project_soc,
    evaluate_pdhcg_quality,
)
from spacepdhcg.cqp import CQPSolution
from spacepdhcg.models import (
    CWRendezvousConfig,
    CWRendezvousProblem,
    ThrustConstraint,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([0.25, 0.0, 1.0], [0.25, 0.0, 1.0]),
        ([1.0, 0.0, -2.0], [0.0, 0.0, 0.0]),
        ([2.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
    ],
)
def test_native_soc_projection(value: list[float], expected: list[float]) -> None:
    np.testing.assert_allclose(_project_soc(np.asarray(value)), expected)


@pytest.mark.parametrize(
    "constraint",
    [ThrustConstraint.BOX, ThrustConstraint.SECOND_ORDER_CONE],
)
def test_independent_quality_accepts_high_accuracy_reference_in_pdhcg_dual_convention(
    constraint: ThrustConstraint,
) -> None:
    initial = np.array([10.0, -5.0, 2.0, 0.01, -0.02, 0.005])
    target = np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0])
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=20,
            max_component_acceleration=2.0e-3,
            thrust_constraint=constraint,
        )
    )
    canonical = problem.canonical(initial, target)
    reference = PersistentClarabel(canonical, tolerance=1.0e-10).solve()

    # Clarabel's scalar output is a normal multiplier, while its transformed
    # affine-cone output and PDHCG's Pi are dual-cone values.
    scalar = -reference.dual[: problem.structure.n_constraints]
    affine = reference.dual[problem.structure.n_constraints :]
    pdhcg_convention = CQPSolution(
        status=reference.status,
        primal=reference.primal,
        dual=np.concatenate((scalar, affine)),
        objective=reference.objective,
        primal_residual=reference.primal_residual,
        dual_residual=reference.dual_residual,
        iterations=reference.iterations,
        solve_seconds=reference.solve_seconds,
    )
    objective = _objective(canonical, reference.primal)
    quality = evaluate_pdhcg_quality(
        problem,
        canonical,
        pdhcg_convention,
        initial_state=initial,
        target_state=target,
        reference_objective=objective,
    )

    assert quality.objective_gap_relative < 1.0e-12
    assert quality.reported_objective_error_relative < 1.0e-9
    assert quality.scalar_primal_violation_inf < 1.0e-9
    assert quality.cone_primal_violation_inf < 1.0e-9
    assert quality.relative_natural_residual_inf < 1.0e-8
    assert quality.maximum_trajectory_violation < 1.0e-9
