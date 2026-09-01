"""Optional ABI integration against the pinned builtin (CPU) QOCO library."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from spacepdhcg.backends import QOCOGPU, PersistentClarabel
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem, ThrustConstraint

QOCO_LIBRARY = os.environ.get("SPACEPDHCG_QOCO_LIBRARY")
pytestmark = pytest.mark.skipif(
    not QOCO_LIBRARY,
    reason="SPACEPDHCG_QOCO_LIBRARY does not name a pinned CPU/GPU QOCO build",
)


@pytest.mark.parametrize(
    "thrust_constraint",
    [ThrustConstraint.BOX, ThrustConstraint.SECOND_ORDER_CONE],
    ids=["trajectory-qp", "trajectory-socp"],
)
def test_pinned_qoco_c_abi_matches_cpu_clarabel_reference(
    thrust_constraint: ThrustConstraint,
) -> None:
    library = Path(QOCO_LIBRARY or "")
    problem_model = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=3,
            thrust_constraint=thrust_constraint,
        )
    )
    problem = problem_model.canonical(
        np.array([10.0, -2.0, 1.0, 0.0, 0.0, 0.0]),
        np.zeros(6),
    )
    reference = PersistentClarabel(problem, tolerance=1.0e-8).solve()

    with QOCOGPU(problem, library_path=library) as backend:
        solution = backend.solve(tolerance=1.0e-8)
        report = backend.last_report

    assert reference.solved
    assert solution.solved, solution.status
    assert report is not None
    assert report.failure_class is None
    assert solution.primal_residual < 2.0e-6
    assert solution.dual_residual < 2.0e-6
    np.testing.assert_allclose(solution.primal, reference.primal, atol=3.0e-5, rtol=3.0e-5)
    assert solution.objective == pytest.approx(reference.objective, abs=3.0e-6, rel=3.0e-6)


def test_pinned_qoco_c_abi_updates_and_accepts_unscaled_primal_start() -> None:
    library = Path(QOCO_LIBRARY or "")
    problem_model = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=3,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    first_problem = problem_model.canonical(np.zeros(6), np.zeros(6))
    second_values = problem_model.values(
        np.array([4.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        np.zeros(6),
    )

    with QOCOGPU(first_problem, library_path=library) as backend:
        first = backend.solve()
        backend.update(second_values)
        backend.warm_start(first.primal, first.dual)
        second = backend.solve()
        report = backend.last_report

    assert second.solved
    assert report is not None
    assert report.warm_start.primal_accepted
    assert report.warm_start.dual_discarded
    assert report.update_seconds > 0.0
    assert second.primal_residual < 2.0e-6
    assert second.dual_residual < 2.0e-6
