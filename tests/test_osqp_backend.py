import numpy as np

from spacepdhcg.backends import PersistentOSQP
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem


def test_persistent_workspace_solves_and_updates_rendezvous() -> None:
    config = CWRendezvousConfig(
        intervals=20,
        step_seconds=30.0,
        max_component_acceleration=0.5,
    )
    problem = CWRendezvousProblem(config)
    initial = np.array([100.0, -30.0, 10.0, 0.01, -0.02, 0.0])
    target = np.zeros(6)

    backend = PersistentOSQP(problem.canonical(initial, target))
    first = backend.solve(tolerance=1.0e-8)
    assert first.solved, first.status
    assert problem.diagnostics(first.primal, initial, target).feasible(1.0e-5)

    second_initial = np.array([50.0, 20.0, -5.0, 0.0, 0.0, 0.0])
    second_target = np.array([5.0, -2.0, 1.0, 0.0, 0.0, 0.0])
    backend.update(problem.values(second_initial, second_target))
    backend.warm_start(first.primal, first.dual)
    second = backend.solve(tolerance=1.0e-8)

    assert second.solved, second.status
    assert problem.diagnostics(second.primal, second_initial, second_target).feasible(1.0e-5)
    assert backend.update_count == 1
    assert backend.warm_start_count == 1
    assert backend.structure is problem.structure
