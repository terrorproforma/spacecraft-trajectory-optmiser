from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from spacepdhcg.backends.pdhcg_oneshot import PDHCGOneShot
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem, ThrustConstraint


class FakeConeSpec:
    def __init__(self, types, starts, v_dims, power_alphas) -> None:
        self.types = list(types)
        self.starts = np.asarray(starts).copy()
        self.v_dims = np.asarray(v_dims).copy()
        self.power_alphas = np.asarray(power_alphas).copy()


class FakeModel:
    instances: ClassVar[list[FakeModel]] = []

    def __init__(self, **arguments) -> None:
        self.arguments = arguments
        self.params: dict[str, object] = {}
        self.primal_start = None
        self.dual_start = None
        self.Status = None
        self.X = None
        self.Pi = None
        self.ObjVal = None
        self.RelPrimalResidual = None
        self.RelDualResidual = None
        self.IterCount = None
        self.Runtime = None
        self.__class__.instances.append(self)

    def setParams(self, **parameters) -> None:
        self.params.update(parameters)

    def setWarmStart(self, primal=None, dual=None) -> None:
        self.primal_start = None if primal is None else np.asarray(primal).copy()
        self.dual_start = None if dual is None else np.asarray(dual).copy()

    def optimize(self) -> None:
        variables = self.arguments["objective_vector"].size
        scalar_rows = self.arguments["constraint_matrix"].shape[0]
        affine = self.arguments["affine_cone_matrix"]
        affine_rows = 0 if affine is None else affine.shape[0]
        self.Status = "OPTIMAL"
        self.X = np.arange(variables, dtype=np.float64)
        self.Pi = np.arange(scalar_rows + affine_rows, dtype=np.float64)
        self.ObjVal = 12.5
        self.RelPrimalResidual = 2.0e-7
        self.RelDualResidual = 3.0e-7
        self.IterCount = 37
        self.Runtime = 0.125


def fake_pdhcg_module() -> SimpleNamespace:
    FakeModel.instances.clear()
    return SimpleNamespace(
        Model=FakeModel,
        ConeSpec=FakeConeSpec,
        __version__="test-upstream",
    )


def test_maps_native_soc_problem_and_solver_settings() -> None:
    intervals = 3
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=intervals,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    initial = np.array([100.0, -20.0, 4.0, 0.01, 0.0, 0.0])
    target = np.zeros(6)
    backend = PDHCGOneShot(
        problem.canonical(initial, target),
        params={"LogLevel": 0, "Presolve": False},
        pdhcg_module=fake_pdhcg_module(),
    )

    solution = backend.solve(tolerance=1.0e-5, iteration_limit=1234)
    model = FakeModel.instances[-1]
    arguments = model.arguments

    assert backend.is_persistent is False
    assert backend.upstream_version == "test-upstream"
    assert arguments["objective_matrix"].shape == (
        problem.layout.n_variables,
        problem.layout.n_variables,
    )
    assert arguments["constraint_matrix"].shape == (
        problem.layout.n_constraints,
        problem.layout.n_variables,
    )
    assert arguments["affine_cone_matrix"].shape == (
        problem.layout.n_affine_constraints,
        problem.layout.n_variables,
    )
    cone_spec = arguments["affine_cones"]
    assert cone_spec.types == ["soc"] * intervals
    np.testing.assert_array_equal(cone_spec.starts, np.arange(intervals) * 4)
    np.testing.assert_array_equal(cone_spec.v_dims, np.full(intervals, 2))
    np.testing.assert_allclose(cone_spec.power_alphas, 0.0)
    assert arguments["variable_cones"] is None

    assert model.params == {
        "LogLevel": 0,
        "Presolve": False,
        "OptimalityTol": 1.0e-5,
        "FeasibilityTol": 1.0e-5,
        "IterationLimit": 1234,
    }
    assert solution.solved
    assert solution.status == "Solved (OPTIMAL)"
    assert solution.primal.shape == (problem.structure.n_variables,)
    assert solution.dual.shape == (problem.structure.n_duals,)
    assert solution.objective == 12.5
    assert solution.primal_residual == 2.0e-7
    assert solution.dual_residual == 3.0e-7
    assert solution.iterations == 37
    assert solution.solve_seconds == 0.125
    assert backend.solve_count == 1
    assert backend.last_model is model


def test_update_and_primal_dual_warm_start_reach_upstream_model() -> None:
    problem = CWRendezvousProblem(CWRendezvousConfig(intervals=4))
    first_initial = np.array([80.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    first_target = np.zeros(6)
    second_initial = np.array([40.0, -15.0, 2.0, 0.0, 0.0, 0.0])
    second_target = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    backend = PDHCGOneShot(
        problem.canonical(first_initial, first_target),
        pdhcg_module=fake_pdhcg_module(),
    )
    second_values = problem.values(second_initial, second_target)
    backend.update(second_values)

    primal = np.linspace(0.0, 1.0, problem.structure.n_variables)
    dual = np.linspace(0.0, 1.0, problem.structure.n_duals)
    backend.warm_start(primal, dual)
    backend.solve()
    model = FakeModel.instances[-1]

    np.testing.assert_allclose(model.arguments["objective_vector"], second_values.linear)
    np.testing.assert_allclose(model.arguments["constraint_lower_bound"], second_values.lower)
    np.testing.assert_allclose(model.arguments["constraint_upper_bound"], second_values.upper)
    np.testing.assert_allclose(model.primal_start, primal)
    np.testing.assert_allclose(model.dual_start, dual)
    assert backend.update_count == 1
    assert backend.warm_start_count == 1


def test_warm_start_shapes_use_native_dual_dimension() -> None:
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=2,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    backend = PDHCGOneShot(
        problem.canonical(np.ones(6), np.zeros(6)),
        pdhcg_module=fake_pdhcg_module(),
    )

    with pytest.raises(ValueError, match="primal"):
        backend.warm_start(primal=np.zeros(problem.structure.n_variables - 1))
    with pytest.raises(ValueError, match="dual"):
        backend.warm_start(dual=np.zeros(problem.structure.n_duals - 1))
    with pytest.raises(ValueError, match="at least one"):
        backend.warm_start()
