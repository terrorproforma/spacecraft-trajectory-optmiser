from __future__ import annotations

from dataclasses import dataclass

import clarabel
import numpy as np
import pytest
import scipy.sparse as sp

from spacepdhcg.backends import (
    QOCOGPU,
    PDHCGQOCOHybrid,
    PersistentClarabel,
    QOCOAdapterError,
    QOCORawSolution,
    QOCOSettings,
    QOCOSetupError,
    QOCOUnsupportedError,
    UnsupportedQOCOClass,
    canonical_primal_residual,
    convert_to_qoco,
    independent_residuals,
)
from spacepdhcg.cqp import (
    CanonicalCQP,
    ConeBlock,
    ConeKind,
    CQPSolution,
    CQPStructure,
    CQPValues,
    CSCStructure,
)
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem, ThrustConstraint


def _mixed_problem(*, cone_kind: ConeKind = ConeKind.SECOND_ORDER) -> CanonicalCQP:
    quadratic = sp.csc_matrix((3, 3), dtype=np.float64)
    scalar = sp.eye(3, format="csc")
    affine = sp.eye(3, format="csc")
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(quadratic),
        constraint=CSCStructure.from_matrix(scalar),
        affine_cone=CSCStructure.from_matrix(affine),
        affine_cones=(ConeBlock(cone_kind, 0, 1),),
    )
    return CanonicalCQP(
        structure,
        CQPValues(
            quadratic=structure.quadratic.values_from(quadratic),
            constraint=structure.constraint.values_from(scalar),
            linear=np.zeros(3),
            lower=np.array([1.0, -2.0, -1.0]),
            upper=np.array([1.0, 3.0, np.inf]),
            affine_cone=structure.affine_cone.values_from(affine),
            affine_offset=np.array([0.0, 0.0, 2.0]),
            variable_lower=np.full(3, -np.inf),
            variable_upper=np.array([4.0, np.inf, np.inf]),
        ),
    )


def test_exact_bound_soc_signs_and_sparse_ordering() -> None:
    formulation = convert_to_qoco(_mixed_problem())

    assert formulation.equality_dimension == 1
    np.testing.assert_allclose(formulation.a.toarray(), [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(formulation.b, [1.0])
    assert formulation.nonnegative_dimension == 4
    np.testing.assert_allclose(
        formulation.g.toarray()[:4],
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
    )
    np.testing.assert_allclose(formulation.h[:4], [3.0, 2.0, 1.0, 4.0])
    np.testing.assert_allclose(
        formulation.g.toarray()[4:],
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )
    np.testing.assert_allclose(formulation.h[4:], [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(formulation.soc_dimensions, [3])
    assert np.all(np.diff(formulation.g.indptr) >= 0)


def test_rotated_soc_uses_orthonormal_exact_map() -> None:
    formulation = convert_to_qoco(_mixed_problem(cone_kind=ConeKind.ROTATED_SECOND_ORDER))
    transform = -formulation.g.toarray()[4:]

    np.testing.assert_allclose(transform.T @ transform, np.eye(3), atol=1.0e-15)
    native = np.array([1.0, 2.0, 1.0])
    mapped = transform @ native
    assert mapped[0] >= np.linalg.norm(mapped[1:])


def test_independent_residuals_do_not_trust_native_values() -> None:
    problem = _mixed_problem()
    formulation = convert_to_qoco(problem)
    raw = QOCORawSolution(
        x=np.array([1.0, 0.0, 0.0]),
        y=np.zeros(formulation.equality_dimension),
        z=np.zeros(formulation.m),
        objective=999.0,
        primal_residual=999.0,
        dual_residual=999.0,
        gap=999.0,
        iterations=1,
        setup_seconds=0.1,
        solve_seconds=0.2,
        analysis_seconds=0.05,
        status=1,
    )

    residuals = independent_residuals(problem, formulation, raw)
    assert residuals.primal == pytest.approx(0.0)
    assert residuals.dual == pytest.approx(0.0)
    assert canonical_primal_residual(problem, raw.x) == pytest.approx(0.0)
    assert canonical_primal_residual(problem, np.array([0.0, 0.0, 3.0])) == pytest.approx(1.0)


@dataclass
class _FakeHandle:
    formulation: object


class FakeQOCOAPI:
    def __init__(self, *, status: int = 1, setup_error: int | None = None) -> None:
        self.status = status
        self.setup_error = setup_error
        self.setup_calls = 0
        self.update_calls = 0
        self.starts: list[np.ndarray | None] = []
        self.cleanup_calls = 0

    def setup(self, formulation, settings):
        del settings
        self.setup_calls += 1
        if self.setup_error is not None:
            raise QOCOSetupError(self.setup_error)
        return _FakeHandle(formulation)

    def update(self, handle, formulation, settings) -> None:
        del settings
        handle.formulation = formulation
        self.update_calls += 1

    def set_primal_start(self, handle, primal) -> None:
        del handle
        self.starts.append(None if primal is None else primal.copy())

    def solve(self, handle, formulation) -> QOCORawSolution:
        del handle
        return QOCORawSolution(
            x=np.zeros(formulation.n),
            y=np.zeros(formulation.equality_dimension),
            z=np.zeros(formulation.m),
            objective=0.0,
            primal_residual=123.0,
            dual_residual=456.0,
            gap=789.0,
            iterations=7,
            setup_seconds=0.01,
            solve_seconds=0.02,
            analysis_seconds=0.003,
            status=self.status,
        )

    def cleanup(self, handle) -> None:
        del handle
        self.cleanup_calls += 1


class DualQOCOAPI(FakeQOCOAPI):
    def solve(self, handle, formulation) -> QOCORawSolution:
        del handle
        return QOCORawSolution(
            x=np.array([1.0, 0.0, 0.0]),
            y=np.array([2.0]),
            z=np.arange(3.0, 3.0 + formulation.m),
            objective=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
            gap=0.0,
            iterations=1,
            setup_seconds=0.01,
            solve_seconds=0.02,
            analysis_seconds=0.003,
            status=1,
        )


def _zero_cw_problem() -> CanonicalCQP:
    model = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=2,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    return model.canonical(np.zeros(6), np.zeros(6))


def test_persistent_lifecycle_warm_start_reporting_and_handback() -> None:
    problem = _zero_cw_problem()
    api = FakeQOCOAPI()
    backend = QOCOGPU(
        problem,
        qoco_api=api,
        warm_start_primal_tolerance=1.0e-12,
    )
    backend.warm_start(
        np.zeros(problem.structure.n_variables),
        np.zeros(problem.structure.n_duals),
    )
    solution, handed_back = backend.solve_and_handback(lambda item: item.primal.size)

    assert solution.solved
    assert solution.primal_residual == pytest.approx(0.0)
    assert solution.dual_residual == pytest.approx(0.0)
    assert handed_back == problem.structure.n_variables
    assert backend.last_report is not None
    assert backend.last_report.native_primal_residual == 123.0
    assert backend.last_report.canonical_residuals.primal == pytest.approx(0.0)
    assert backend.last_report.warm_start.primal_accepted
    assert backend.last_report.warm_start.dual_discarded
    assert "dual discarded" in backend.last_report.warm_start.reason
    np.testing.assert_allclose(api.starts[-1], 0.0)

    backend.update(problem.values)
    assert backend.update_count == 1
    assert api.update_calls == 1
    backend.close()
    backend.close()
    assert api.cleanup_calls == 1
    with pytest.raises(QOCOAdapterError, match="closed"):
        backend.solve()


def test_qoco_duals_return_to_canonical_row_and_cone_order() -> None:
    problem = _mixed_problem()
    with QOCOGPU(problem, qoco_api=DualQOCOAPI()) as backend:
        solution = backend.solve()

    np.testing.assert_allclose(
        solution.dual,
        [2.0, -1.0, -5.0, 8.0, 9.0, 7.0],
    )


def test_rejected_warm_start_is_cleared_and_reported() -> None:
    problem = _zero_cw_problem()
    api = FakeQOCOAPI()
    backend = QOCOGPU(
        problem,
        qoco_api=api,
        warm_start_primal_tolerance=1.0e-12,
    )
    bad = np.full(problem.structure.n_variables, np.nan)
    backend.warm_start(bad)
    backend.solve()

    assert api.starts[-1] is None
    assert backend.last_report is not None
    assert not backend.last_report.warm_start.primal_qualified
    assert not backend.last_report.warm_start.primal_accepted
    backend.close()


def test_setup_solve_failure_and_ownership_are_explicit() -> None:
    problem = _zero_cw_problem()
    with pytest.raises(QOCOSetupError) as caught:
        QOCOGPU(problem, qoco_api=FakeQOCOAPI(setup_error=5))
    assert caught.value.failure_class == "out_of_memory"

    api = FakeQOCOAPI(status=3)
    with QOCOGPU(problem, qoco_api=api) as backend:
        solution = backend.solve()
        assert not solution.solved
        assert backend.last_report is not None
        assert backend.last_report.failure_class == "numerical_failure"
    assert api.cleanup_calls == 1


@pytest.mark.parametrize(
    ("kind", "classification"),
    [
        (ConeKind.EXPONENTIAL, UnsupportedQOCOClass.UNSUPPORTED_CONE),
        (ConeKind.POWER, UnsupportedQOCOClass.UNSUPPORTED_CONE),
        (ConeKind.POSITIVE_SEMIDEFINITE, UnsupportedQOCOClass.UNSUPPORTED_CONE),
    ],
)
def test_unsupported_native_cones_are_classified(kind, classification) -> None:
    vector_dimension = 2 if kind is ConeKind.POSITIVE_SEMIDEFINITE else 1
    cone = ConeBlock(
        kind,
        0,
        vector_dimension,
        0.5 if kind is ConeKind.POWER else 0.0,
    )
    size = cone.slot_count
    quadratic = sp.eye(size, format="csc")
    affine = sp.eye(size, format="csc")
    scalar = sp.csc_matrix((0, size))
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(quadratic),
        constraint=CSCStructure.from_matrix(scalar),
        affine_cone=CSCStructure.from_matrix(affine),
        affine_cones=(cone,),
    )
    problem = CanonicalCQP(
        structure,
        CQPValues(
            quadratic=structure.quadratic.values_from(quadratic),
            constraint=np.empty(0),
            linear=np.zeros(3),
            lower=np.empty(0),
            upper=np.empty(0),
            affine_cone=structure.affine_cone.values_from(affine),
            affine_offset=np.zeros(3),
            variable_lower=np.full(size, -np.inf),
            variable_upper=np.full(size, np.inf),
        ),
    )
    with pytest.raises(QOCOUnsupportedError) as caught:
        convert_to_qoco(problem)
    assert caught.value.classification is classification


def test_nonconvex_and_nonsymmetric_quadratics_are_classified() -> None:
    problem = _zero_cw_problem()
    values = problem.values.copy()
    values.quadratic[0] = -1.0
    with pytest.raises(QOCOUnsupportedError) as nonconvex:
        convert_to_qoco(CanonicalCQP(problem.structure, values))
    assert nonconvex.value.classification is UnsupportedQOCOClass.NONCONVEX_QUADRATIC

    quadratic = sp.csc_matrix([[1.0, 2.0], [0.0, 1.0]])
    scalar = sp.csc_matrix((0, 2))
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(quadratic),
        constraint=CSCStructure.from_matrix(scalar),
    )
    nonsymmetric_problem = CanonicalCQP(
        structure,
        CQPValues(
            quadratic=structure.quadratic.values_from(quadratic),
            constraint=np.empty(0),
            linear=np.zeros(2),
            lower=np.empty(0),
            upper=np.empty(0),
            affine_cone=np.empty(0),
            affine_offset=np.empty(0),
            variable_lower=np.full(2, -np.inf),
            variable_upper=np.full(2, np.inf),
        ),
    )
    with pytest.raises(QOCOUnsupportedError) as nonsymmetric:
        convert_to_qoco(nonsymmetric_problem)
    assert nonsymmetric.value.classification is UnsupportedQOCOClass.NONSYMMETRIC_QUADRATIC


def test_bound_kind_change_is_rejected_before_native_update() -> None:
    problem = _zero_cw_problem()
    backend = QOCOGPU(problem, qoco_api=FakeQOCOAPI())
    values = problem.values.copy()
    values.upper[0] += 1.0

    with pytest.raises(QOCOUnsupportedError) as caught:
        backend.update(values)
    assert caught.value.classification is UnsupportedQOCOClass.CHANGING_BOUND_STRUCTURE
    backend.close()


class FakePDHCG:
    def __init__(self, problem: CanonicalCQP) -> None:
        self.structure = problem.structure
        self.values = problem.values
        self.warm_start_count = 0

    def update(self, values: CQPValues) -> None:
        self.values = values

    def warm_start(self, primal=None, dual=None) -> None:
        del primal, dual
        self.warm_start_count += 1

    def solve(self, **kwargs) -> CQPSolution:
        del kwargs
        return CQPSolution(
            status="Solved (fake PDHCG)",
            primal=np.zeros(self.structure.n_variables),
            dual=np.zeros(self.structure.n_duals),
            objective=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
            iterations=3,
            solve_seconds=0.004,
        )


def test_hybrid_qualifies_primal_discards_dual_and_hands_back() -> None:
    problem = _zero_cw_problem()
    pdhcg = FakePDHCG(problem)
    qoco = QOCOGPU(
        problem,
        qoco_api=FakeQOCOAPI(),
        warm_start_primal_tolerance=1.0e-12,
    )
    hybrid = PDHCGQOCOHybrid(pdhcg, qoco)
    hybrid.warm_start(
        np.zeros(problem.structure.n_variables),
        np.zeros(problem.structure.n_duals),
    )
    solution, result = hybrid.solve(handback=lambda item: {"size": item.primal.size})

    assert solution.solved
    assert result == {"size": problem.structure.n_variables}
    assert hybrid.last_report is not None
    assert hybrid.last_report.pdhcg_status == "Solved (fake PDHCG)"
    assert hybrid.last_report.warm_start.primal_accepted
    assert hybrid.last_report.warm_start.dual_discarded
    assert hybrid.last_report.polish_seconds >= 0.0
    assert hybrid.warm_start_count == 1
    assert pdhcg.warm_start_count == 1
    qoco.close()


def test_cpu_clarabel_reference_matches_qoco_mapping() -> None:
    model = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=3,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    problem = model.canonical(
        np.array([10.0, -2.0, 1.0, 0.0, 0.0, 0.0]),
        np.zeros(6),
    )
    reference = PersistentClarabel(problem, tolerance=1.0e-9).solve()
    formulation = convert_to_qoco(problem)
    matrix = sp.vstack((formulation.a, formulation.g), format="csc")
    rhs = np.concatenate((formulation.b, formulation.h))
    cones = [
        clarabel.ZeroConeT(formulation.equality_dimension),
        clarabel.NonnegativeConeT(formulation.nonnegative_dimension),
        *(clarabel.SecondOrderConeT(int(size)) for size in formulation.soc_dimensions),
    ]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1.0e-9
    settings.tol_gap_rel = 1.0e-9
    settings.tol_feas = 1.0e-9
    settings.presolve_enable = False
    mapped = clarabel.DefaultSolver(
        formulation.p,
        formulation.c,
        matrix,
        rhs,
        cones,
        settings,
    ).solve()

    assert reference.solved
    assert str(mapped.status).lower().startswith("solved")
    np.testing.assert_allclose(mapped.x, reference.primal, atol=2.0e-7, rtol=2.0e-7)
    mapped_objective = 0.5 * np.asarray(mapped.x) @ (
        (formulation.p + sp.triu(formulation.p, k=1).T) @ np.asarray(mapped.x)
    ) + formulation.c @ np.asarray(mapped.x)
    assert mapped_objective == pytest.approx(reference.objective, abs=1.0e-7, rel=1.0e-7)


def test_settings_overrides_reach_native_update() -> None:
    problem = _zero_cw_problem()
    api = FakeQOCOAPI()
    with QOCOGPU(problem, qoco_api=api, settings=QOCOSettings()) as backend:
        backend.solve(tolerance=2.0e-6, iteration_limit=42)
    assert api.update_calls == 1
