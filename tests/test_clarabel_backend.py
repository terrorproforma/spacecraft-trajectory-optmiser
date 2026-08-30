import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem, ThrustConstraint


def test_persistent_clarabel_solves_and_updates_soc_rendezvous() -> None:
    config = CWRendezvousConfig(
        intervals=20,
        step_seconds=30.0,
        max_component_acceleration=0.5,
        thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
    )
    problem = CWRendezvousProblem(config)
    initial = np.array([100.0, -30.0, 10.0, 0.01, -0.02, 0.0])
    target = np.zeros(6)

    backend = PersistentClarabel(problem.canonical(initial, target), tolerance=1.0e-8)
    first = backend.solve()
    assert first.solved, first.status
    first_diagnostics = problem.diagnostics(first.primal, initial, target)
    assert first_diagnostics.feasible(1.0e-5)
    assert first_diagnostics.maximum_acceleration_norm <= 0.5 + 1.0e-6

    second_initial = np.array([50.0, 20.0, -5.0, 0.0, 0.0, 0.0])
    second_target = np.array([5.0, -2.0, 1.0, 0.0, 0.0, 0.0])
    backend.update(problem.values(second_initial, second_target))
    second = backend.solve()
    second_diagnostics = problem.diagnostics(second.primal, second_initial, second_target)

    assert second.solved, second.status
    assert second_diagnostics.feasible(1.0e-5)
    assert second_diagnostics.maximum_acceleration_norm <= 0.5 + 1.0e-6
    assert backend.update_count == 1
    assert backend.structure is problem.structure
    assert second.dual.shape == (problem.structure.n_duals,)
