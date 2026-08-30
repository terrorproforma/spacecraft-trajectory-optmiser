import numpy as np
import pytest

from spacepdhcg.backends import PersistentClarabel, PersistentOSQP
from spacepdhcg.benchmarks.trajectory_banded import (
    BandedControlConstraint,
    TrajectoryBandedConfig,
    TrajectoryBandedFixture,
)


@pytest.mark.parametrize(
    "control_constraint",
    [BandedControlConstraint.BOX, BandedControlConstraint.SECOND_ORDER_CONE],
)
def test_generated_known_solution_is_strictly_feasible_and_optimal(control_constraint) -> None:
    fixture = TrajectoryBandedFixture(
        TrajectoryBandedConfig(
            intervals=5,
            seed=31,
            control_constraint=control_constraint,
        )
    )
    diagnostics = fixture.diagnostics(fixture.known_solution)

    assert diagnostics.solution_error_inf == 0.0
    assert diagnostics.scalar_violation_inf < 1.0e-14
    assert diagnostics.dynamics_defect_inf < 1.0e-14
    assert diagnostics.control_violation_inf == 0.0
    assert diagnostics.objective_gap_abs == 0.0


def test_box_fixture_agrees_between_osqp_and_clarabel() -> None:
    fixture = TrajectoryBandedFixture(
        TrajectoryBandedConfig(
            intervals=6,
            seed=41,
            weight_log10_span=0.25,
            control_constraint=BandedControlConstraint.BOX,
        )
    )
    osqp_solution = PersistentOSQP(fixture.canonical).solve(tolerance=1.0e-9)
    clarabel_solution = PersistentClarabel(
        fixture.canonical,
        tolerance=1.0e-9,
    ).solve()

    assert osqp_solution.solved, osqp_solution.status
    assert clarabel_solution.solved, clarabel_solution.status
    osqp_diagnostics = fixture.diagnostics(osqp_solution.primal)
    clarabel_diagnostics = fixture.diagnostics(clarabel_solution.primal)

    assert osqp_diagnostics.solution_error_inf < 1.0e-6
    assert clarabel_diagnostics.solution_error_inf < 1.0e-6
    assert osqp_diagnostics.scalar_violation_inf < 1.0e-7
    assert clarabel_diagnostics.scalar_violation_inf < 1.0e-7
    assert osqp_diagnostics.objective_gap_abs < 1.0e-8
    assert clarabel_diagnostics.objective_gap_abs < 1.0e-8
    np.testing.assert_allclose(
        osqp_solution.primal,
        clarabel_solution.primal,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_soc_fixture_matches_exact_optimum_with_clarabel() -> None:
    fixture = TrajectoryBandedFixture(
        TrajectoryBandedConfig(
            intervals=6,
            seed=53,
            weight_log10_span=0.25,
            control_constraint=BandedControlConstraint.SECOND_ORDER_CONE,
        )
    )
    solution = PersistentClarabel(fixture.canonical, tolerance=1.0e-9).solve()
    diagnostics = fixture.diagnostics(solution.primal)

    assert solution.solved, solution.status
    assert diagnostics.solution_error_inf < 2.0e-6
    assert diagnostics.scalar_violation_inf < 1.0e-7
    assert diagnostics.dynamics_defect_inf < 1.0e-7
    assert diagnostics.control_violation_inf < 1.0e-8
    assert diagnostics.objective_gap_abs < 1.0e-8
