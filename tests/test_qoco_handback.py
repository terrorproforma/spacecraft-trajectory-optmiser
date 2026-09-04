from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from spacepdhcg.backends import (
    QOCOGPU,
    DeviceReplayEvaluation,
    FrozenOuterPolicy,
    HandbackDisposition,
    OuterCandidateContext,
    PDHCGQOCOHybrid,
    QOCOHybridIneligibleError,
    QOCORawSolution,
    QOCOSolverMode,
    TrustDecision,
    canonical_numeric_fingerprint,
    handback_qoco_candidate,
)
from spacepdhcg.cqp import CQPSolution
from spacepdhcg.cqp.fingerprint import structure_fingerprint_hex
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem, ThrustConstraint


@dataclass
class _Handle:
    formulation: object


class _ExactAPI:
    def __init__(self) -> None:
        self.solve_calls = 0
        self.starts: list[np.ndarray | None] = []

    def setup(self, formulation, settings):
        del settings
        return _Handle(formulation)

    def update(self, handle, formulation, settings) -> None:
        del settings
        handle.formulation = formulation

    def set_primal_start(self, handle, primal) -> None:
        del handle
        self.starts.append(None if primal is None else primal.copy())

    def solve(self, handle, formulation) -> QOCORawSolution:
        del handle
        self.solve_calls += 1
        return QOCORawSolution(
            x=np.zeros(formulation.n),
            y=np.zeros(formulation.equality_dimension),
            z=np.zeros(formulation.m),
            objective=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
            gap=0.0,
            iterations=20,
            setup_seconds=0.001,
            solve_seconds=0.002,
            analysis_seconds=0.0001,
            status=1,
        )

    def cleanup(self, handle) -> None:
        del handle


def _problem():
    model = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=2,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )
    return model.canonical(np.zeros(6), np.zeros(6))


class _DeviceOwner:
    def __init__(
        self,
        backend: QOCOGPU,
        *,
        family: str,
        actual_merit: float = 0.2,
        model_merit: float = 0.1,
        residual: float = 1.0e-9,
        permutation: str = "canonical-identity",
        fingerprint: str | None = None,
        cpu_fallback: bool = False,
    ) -> None:
        self.topology_fingerprint = structure_fingerprint_hex(backend.structure)
        self.family = family
        self.actual_merit = actual_merit
        self.model_merit = model_merit
        self.residual = residual
        self.permutation = permutation
        self.expected_fingerprint = canonical_numeric_fingerprint(backend.current_values)
        self.fingerprint = fingerprint or self.expected_fingerprint
        self.cpu_fallback = cpu_fallback
        self.committed = 0
        self.rejected = 0
        self.received_primal: np.ndarray | None = None
        self.received_dual: np.ndarray | None = None

    def evaluate_qoco_candidate(
        self,
        primal,
        dual,
        *,
        family,
        cqp_numeric_fingerprint,
    ) -> DeviceReplayEvaluation:
        assert cqp_numeric_fingerprint == self.expected_fingerprint
        self.received_primal = primal.copy()
        self.received_dual = dual.copy()
        inventories = {
            "P1-C-pd3": {
                "thrust": 0.0,
                "mass": 0.0,
                "altitude": 0.0,
                "glide_slope": 0.0,
            },
            "P1-D-pd6": {
                "thrust": 0.0,
                "torque": 0.0,
                "pointing": 0.0,
                "mass": 0.0,
                "altitude": 0.0,
                "glide_slope": 0.0,
                "angular_rate": 0.0,
                "quaternion": 2.0e-13,
            },
            "P1-E-low-thrust": {
                "thrust": 0.0,
                "mass": 0.0,
                "altitude": 0.0,
            },
            "P1-B-hcw": {},
        }
        return DeviceReplayEvaluation(
            family=self.family if self.family else family,
            model_merit=self.model_merit,
            actual_merit=self.actual_merit,
            objective=0.1,
            dynamics_residual=self.residual,
            path_residual=self.residual,
            terminal_residual=self.residual,
            virtual_control_residual=self.residual,
            trajectory_step=0.9,
            path_inventory=inventories[self.family],
            topology_fingerprint=self.topology_fingerprint,
            cqp_numeric_fingerprint=self.fingerprint,
            permutation=self.permutation,
            device_replay=True,
            hidden_cpu_fallback=self.cpu_fallback,
            transfer_seconds=0.003,
            replay_seconds=0.004,
            token="candidate",
        )

    def commit_qoco_candidate(self, token) -> None:
        assert token == "candidate"
        self.committed += 1

    def reject_qoco_candidate(self, token) -> None:
        assert token == "candidate"
        self.rejected += 1


def _context(backend: QOCOGPU, family: str, *, fingerprint: str | None = None):
    return OuterCandidateContext(
        family=family,
        current_merit=1.0,
        current_residual=1.0e-3,
        trust_radius=1.0,
        cqp_numeric_fingerprint=(
            fingerprint or canonical_numeric_fingerprint(backend.current_values)
        ),
        policy=FrozenOuterPolicy(quality_tolerance=1.0e-8),
    )


@pytest.mark.parametrize(
    "family",
    ["P1-B-hcw", "P1-C-pd3", "P1-D-pd6", "P1-E-low-thrust"],
)
def test_pure_qoco_candidate_uses_device_replay_and_frozen_acceptance(family) -> None:
    backend = QOCOGPU(_problem(), qoco_api=_ExactAPI())
    solution = backend.solve()
    owner = _DeviceOwner(backend, family=family)

    record = handback_qoco_candidate(backend, solution, owner, _context(backend, family))

    assert record.solver_mode is QOCOSolverMode.PURE_GPU_IPM
    assert record.solver_label == "pure-gpu-ipm"
    assert record.disposition is HandbackDisposition.ACCEPTED
    assert record.accepted
    assert record.trust_action is TrustDecision.EXPAND
    assert record.fingerprint_match
    assert record.path_inventory_complete
    assert record.evaluation is not None and record.evaluation.device_replay
    assert not record.hidden_cpu_fallback
    assert owner.committed == 1
    assert owner.received_primal is not None
    assert owner.received_dual is not None
    if family == "P1-D-pd6":
        assert record.evaluation.path_inventory["quaternion"] <= 1.0e-8
        assert record.evaluation.terminal_residual <= 1.0e-8
    backend.close()


@pytest.mark.parametrize(
    ("owner_overrides", "context_fingerprint", "disposition"),
    [
        ({"permutation": "qoco-row-order"}, None, HandbackDisposition.PERMUTATION_MISMATCH),
        ({"cpu_fallback": True}, None, HandbackDisposition.CPU_FALLBACK),
        (
            {"fingerprint": "0" * 64},
            None,
            HandbackDisposition.STALE_CQP,
        ),
        (
            {"actual_merit": 1.5, "model_merit": 0.8, "residual": 2.0e-3},
            None,
            HandbackDisposition.NONLINEAR_REJECTED,
        ),
    ],
)
def test_qoco_handback_negative_dispositions(
    owner_overrides,
    context_fingerprint,
    disposition,
) -> None:
    backend = QOCOGPU(_problem(), qoco_api=_ExactAPI())
    solution = backend.solve()
    owner = _DeviceOwner(backend, family="P1-D-pd6", **owner_overrides)
    context = _context(backend, "P1-D-pd6", fingerprint=context_fingerprint)

    record = handback_qoco_candidate(backend, solution, owner, context)

    assert record.disposition is disposition
    assert not record.accepted
    assert record.trust_action is TrustDecision.SHRINK
    assert owner.rejected == 1
    backend.close()


def test_stale_cqp_is_rejected_before_native_transfer() -> None:
    backend = QOCOGPU(_problem(), qoco_api=_ExactAPI())
    solution = backend.solve()
    owner = _DeviceOwner(backend, family="P1-D-pd6")

    record = handback_qoco_candidate(
        backend,
        solution,
        owner,
        _context(backend, "P1-D-pd6", fingerprint="f" * 64),
    )

    assert record.disposition is HandbackDisposition.STALE_CQP
    assert owner.received_primal is None
    backend.close()


class _UnqualifiedPDHCG:
    def __init__(self, problem) -> None:
        self.structure = problem.structure

    def update(self, values) -> None:
        del values

    def warm_start(self, primal=None, dual=None) -> None:
        del primal, dual

    def solve(self, **kwargs) -> CQPSolution:
        del kwargs
        return CQPSolution(
            status="Maximum iterations (fake PDHCG)",
            primal=np.ones(self.structure.n_variables),
            dual=np.zeros(self.structure.n_duals),
            objective=0.0,
            primal_residual=2.818e-2,
            dual_residual=2.818e-2,
            iterations=300_000,
            solve_seconds=1.0,
        )


class _QualifiedPDHCG(_UnqualifiedPDHCG):
    def __init__(self, problem) -> None:
        super().__init__(problem)
        self.values = problem.values

    def solve(self, **kwargs) -> CQPSolution:
        del kwargs
        return CQPSolution(
            status="Solved (qualified fake PDHCG)",
            primal=np.zeros(self.structure.n_variables),
            dual=np.zeros(self.structure.n_duals),
            objective=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
            iterations=3,
            solve_seconds=0.004,
        )


def test_qualified_hybrid_keeps_distinct_label_timing_and_dual_disposition() -> None:
    problem = _problem()
    api = _ExactAPI()
    qoco = QOCOGPU(problem, qoco_api=api)
    hybrid = PDHCGQOCOHybrid(_QualifiedPDHCG(problem), qoco)
    owner = _DeviceOwner(qoco, family="P1-D-pd6")

    _, record = hybrid.solve_outer_candidate(
        owner,
        _context(qoco, "P1-D-pd6"),
    )

    assert record.solver_mode is QOCOSolverMode.HYBRID_PDHCG_IPM
    assert record.solver_label == "hybrid-pdhcg-ipm"
    assert record.predictor_seconds == pytest.approx(0.004)
    assert record.polish_seconds > 0.0
    assert hybrid.last_report is not None
    assert hybrid.last_report.eligible
    assert hybrid.last_report.fingerprint_match
    assert hybrid.last_report.dual_disposition == ("discarded-unsupported-by-pinned-qoco")
    qoco.close()


def test_unqualified_pdhcg_never_runs_qoco_or_claims_hybrid() -> None:
    problem = _problem()
    api = _ExactAPI()
    qoco = QOCOGPU(problem, qoco_api=api)
    hybrid = PDHCGQOCOHybrid(
        _UnqualifiedPDHCG(problem),
        qoco,
        handoff_tolerance=1.0e-6,
    )

    with pytest.raises(QOCOHybridIneligibleError) as caught:
        hybrid.solve()

    report = caught.value.report
    assert not report.eligible
    assert report.disposition == "ineligible"
    assert report.qoco is None
    assert report.polish_seconds == 0.0
    assert report.dual_disposition == "discarded-unsupported-by-pinned-qoco"
    assert report.reported_primal_residual == pytest.approx(2.818e-2)
    assert report.independent_primal_residual > 1.0e-6
    assert api.solve_calls == 0
    qoco.close()
