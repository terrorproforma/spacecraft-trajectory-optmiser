"""Production nonlinear handback for pure-QOCO and qualified hybrid candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from spacepdhcg.backends.qoco_gpu import QOCOGPU, canonical_numeric_fingerprint
from spacepdhcg.cqp import CQPSolution
from spacepdhcg.cqp.fingerprint import structure_fingerprint_hex
from spacepdhcg.experiments.g4 import PATH_INVENTORY


class QOCOSolverMode(StrEnum):
    PURE_GPU_IPM = "pure-gpu-ipm"
    HYBRID_PDHCG_IPM = "hybrid-pdhcg-ipm"


class HandbackDisposition(StrEnum):
    ACCEPTED = "accepted"
    NONLINEAR_REJECTED = "nonlinear_rejected"
    CQP_UNQUALIFIED = "cqp_unqualified"
    SOLVER_FAILED = "solver_failed"
    PERMUTATION_MISMATCH = "permutation_mismatch"
    STALE_CQP = "stale_cqp"
    CPU_FALLBACK = "cpu_fallback"
    DEVICE_REPLAY_FAILED = "device_replay_failed"


class TrustDecision(StrEnum):
    RETAIN = "retain"
    SHRINK = "shrink"
    EXPAND = "expand"


@dataclass(frozen=True, slots=True)
class FrozenOuterPolicy:
    """Values copied from the frozen G4 trust and quality policy."""

    quality_tolerance: float
    acceptance_threshold: float = 0.05
    restoration_reduction: float = 0.9
    shrink_factor: float = 0.5
    expansion_factor: float = 1.8
    maximum_trust_radius: float = 8.0
    minimum_trust_radius: float = 1.0e-4
    strong_agreement: float = 0.75
    near_boundary_fraction: float = 0.8
    minimum_actual_reduction: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.quality_tolerance <= 0.0:
            raise ValueError("quality_tolerance must be positive")
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must lie in [0, 1]")
        if not 0.0 < self.restoration_reduction < 1.0:
            raise ValueError("restoration_reduction must lie in (0, 1)")
        if not 0.0 < self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must lie in (0, 1)")
        if self.expansion_factor <= 1.0:
            raise ValueError("expansion_factor must exceed one")


@dataclass(frozen=True, slots=True)
class OuterCandidateContext:
    family: str
    current_merit: float
    current_residual: float
    trust_radius: float
    cqp_numeric_fingerprint: str
    policy: FrozenOuterPolicy


@dataclass(frozen=True, slots=True)
class DeviceReplayEvaluation:
    """Metrics returned by the device/native nonlinear trajectory owner."""

    family: str
    model_merit: float
    actual_merit: float
    objective: float
    dynamics_residual: float
    path_residual: float
    terminal_residual: float
    virtual_control_residual: float
    trajectory_step: float
    path_inventory: Mapping[str, float]
    topology_fingerprint: str
    cqp_numeric_fingerprint: str
    permutation: str
    device_replay: bool
    hidden_cpu_fallback: bool
    transfer_seconds: float
    replay_seconds: float
    token: Any = None

    @property
    def maximum_residual(self) -> float:
        return max(
            self.dynamics_residual,
            self.path_residual,
            self.terminal_residual,
            self.virtual_control_residual,
        )


class DeviceNonlinearOwner(Protocol):
    """A native owner that transfers and replays a canonical candidate on-device."""

    topology_fingerprint: str

    def evaluate_qoco_candidate(
        self,
        primal: np.ndarray,
        dual: np.ndarray,
        *,
        family: str,
        cqp_numeric_fingerprint: str,
    ) -> DeviceReplayEvaluation: ...

    def commit_qoco_candidate(self, token: Any) -> None: ...

    def reject_qoco_candidate(self, token: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class OuterHandbackRecord:
    solver_mode: QOCOSolverMode
    solver_label: str
    disposition: HandbackDisposition
    reason: str
    accepted: bool
    restoration_accepted: bool
    trust_action: TrustDecision
    trust_radius_before: float
    trust_radius_after: float
    predicted_reduction: float
    actual_reduction: float
    reduction_ratio: float
    canonical_primal_residual: float
    canonical_dual_residual: float
    topology_fingerprint: str
    cqp_numeric_fingerprint: str
    fingerprint_match: bool
    permutation: str
    path_inventory_complete: bool
    evaluation: DeviceReplayEvaluation | None
    predictor_seconds: float
    conversion_seconds: float
    setup_seconds: float
    polish_seconds: float
    transfer_seconds: float
    replay_seconds: float
    acceptance_seconds: float
    hidden_cpu_fallback: bool


def _rejected_record(
    backend: QOCOGPU,
    context: OuterCandidateContext,
    mode: QOCOSolverMode,
    disposition: HandbackDisposition,
    reason: str,
    solution: CQPSolution,
    *,
    fingerprint_match: bool,
    permutation: str = "unverified",
    evaluation: DeviceReplayEvaluation | None = None,
    acceptance_seconds: float = 0.0,
    predictor_seconds: float = 0.0,
) -> OuterHandbackRecord:
    radius = max(
        context.policy.minimum_trust_radius,
        context.policy.shrink_factor * context.trust_radius,
    )
    report = backend.last_report
    return OuterHandbackRecord(
        solver_mode=mode,
        solver_label=mode.value,
        disposition=disposition,
        reason=reason,
        accepted=False,
        restoration_accepted=False,
        trust_action=TrustDecision.SHRINK,
        trust_radius_before=context.trust_radius,
        trust_radius_after=radius,
        predicted_reduction=0.0,
        actual_reduction=0.0,
        reduction_ratio=float("-inf"),
        canonical_primal_residual=solution.primal_residual,
        canonical_dual_residual=solution.dual_residual,
        topology_fingerprint=structure_fingerprint_hex(backend.structure),
        cqp_numeric_fingerprint=canonical_numeric_fingerprint(backend.current_values),
        fingerprint_match=fingerprint_match,
        permutation=permutation,
        path_inventory_complete=False,
        evaluation=evaluation,
        predictor_seconds=predictor_seconds,
        conversion_seconds=backend.conversion_seconds,
        setup_seconds=backend.setup_seconds,
        polish_seconds=0.0 if report is None else report.solve_seconds,
        transfer_seconds=0.0 if evaluation is None else evaluation.transfer_seconds,
        replay_seconds=0.0 if evaluation is None else evaluation.replay_seconds,
        acceptance_seconds=acceptance_seconds,
        hidden_cpu_fallback=False if evaluation is None else evaluation.hidden_cpu_fallback,
    )


def handback_qoco_candidate(
    backend: QOCOGPU,
    solution: CQPSolution,
    owner: DeviceNonlinearOwner,
    context: OuterCandidateContext,
    *,
    mode: QOCOSolverMode = QOCOSolverMode.PURE_GPU_IPM,
    predictor_seconds: float = 0.0,
) -> OuterHandbackRecord:
    """Validate, replay, decide, and transactionally hand back one QOCO candidate."""

    topology = structure_fingerprint_hex(backend.structure)
    numeric = canonical_numeric_fingerprint(backend.current_values)
    qoco_report = backend.last_report
    if not solution.solved or qoco_report is None:
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.SOLVER_FAILED,
            f"{mode.value} QOCO solve did not succeed",
            solution,
            fingerprint_match=numeric == context.cqp_numeric_fingerprint,
            predictor_seconds=predictor_seconds,
        )
    independent_quality = max(
        qoco_report.canonical_residuals.primal,
        qoco_report.canonical_residuals.dual,
        qoco_report.canonical_residuals.cones,
    )
    if independent_quality > context.policy.quality_tolerance:
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.CQP_UNQUALIFIED,
            "independent canonical QOCO residual exceeds the frozen quality gate",
            solution,
            fingerprint_match=numeric == context.cqp_numeric_fingerprint,
            predictor_seconds=predictor_seconds,
        )
    if numeric != context.cqp_numeric_fingerprint:
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.STALE_CQP,
            "candidate CQP fingerprint differs from the current frozen CQP",
            solution,
            fingerprint_match=False,
            predictor_seconds=predictor_seconds,
        )
    if owner.topology_fingerprint != topology:
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.PERMUTATION_MISMATCH,
            "native owner topology differs from QOCO canonical ordering",
            solution,
            fingerprint_match=True,
            predictor_seconds=predictor_seconds,
        )

    started = perf_counter()
    try:
        evaluation = owner.evaluate_qoco_candidate(
            solution.primal,
            solution.dual,
            family=context.family,
            cqp_numeric_fingerprint=numeric,
        )
    except Exception as error:
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.DEVICE_REPLAY_FAILED,
            f"device/native nonlinear replay failed: {error}",
            solution,
            fingerprint_match=True,
            acceptance_seconds=perf_counter() - started,
            predictor_seconds=predictor_seconds,
        )
    if not evaluation.device_replay or evaluation.hidden_cpu_fallback:
        owner.reject_qoco_candidate(evaluation.token)
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.CPU_FALLBACK,
            "native owner did not prove device replay without CPU fallback",
            solution,
            fingerprint_match=True,
            permutation=evaluation.permutation,
            evaluation=evaluation,
            acceptance_seconds=perf_counter() - started,
            predictor_seconds=predictor_seconds,
        )
    if evaluation.permutation != "canonical-identity" or evaluation.family != context.family:
        owner.reject_qoco_candidate(evaluation.token)
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.PERMUTATION_MISMATCH,
            "candidate permutation or family label differs from native owner",
            solution,
            fingerprint_match=True,
            permutation=evaluation.permutation,
            evaluation=evaluation,
            acceptance_seconds=perf_counter() - started,
            predictor_seconds=predictor_seconds,
        )
    fingerprint_match = evaluation.cqp_numeric_fingerprint == numeric
    if not fingerprint_match:
        owner.reject_qoco_candidate(evaluation.token)
        return _rejected_record(
            backend,
            context,
            mode,
            HandbackDisposition.STALE_CQP,
            "native replay used a stale CQP fingerprint",
            solution,
            fingerprint_match=False,
            permutation=evaluation.permutation,
            evaluation=evaluation,
            acceptance_seconds=perf_counter() - started,
            predictor_seconds=predictor_seconds,
        )

    required_inventory = PATH_INVENTORY.get(context.family, ())
    inventory_complete = all(name in evaluation.path_inventory for name in required_inventory)
    predicted = max(1.0e-12, context.current_merit - evaluation.model_merit)
    actual = context.current_merit - evaluation.actual_merit
    ratio = actual / predicted
    restoration = (
        evaluation.maximum_residual
        <= context.policy.restoration_reduction * context.current_residual
    )
    accepted = bool(
        inventory_complete
        and (
            (
                actual > context.policy.minimum_actual_reduction
                and np.isfinite(ratio)
                and ratio >= context.policy.acceptance_threshold
            )
            or restoration
        )
    )
    radius = context.trust_radius
    action = TrustDecision.RETAIN
    if accepted:
        owner.commit_qoco_candidate(evaluation.token)
        if (
            ratio >= context.policy.strong_agreement
            and evaluation.trajectory_step
            >= context.policy.near_boundary_fraction * max(1.0e-12, radius)
        ):
            radius = min(
                context.policy.maximum_trust_radius,
                context.policy.expansion_factor * radius,
            )
            action = TrustDecision.EXPAND
    else:
        owner.reject_qoco_candidate(evaluation.token)
        radius = max(
            context.policy.minimum_trust_radius,
            context.policy.shrink_factor * radius,
        )
        action = TrustDecision.SHRINK
    acceptance_seconds = perf_counter() - started
    report = backend.last_report
    return OuterHandbackRecord(
        solver_mode=mode,
        solver_label=mode.value,
        disposition=(
            HandbackDisposition.ACCEPTED if accepted else HandbackDisposition.NONLINEAR_REJECTED
        ),
        reason=(
            "candidate committed after device nonlinear replay"
            if accepted
            else "candidate rejected by frozen nonlinear merit/trust policy"
        ),
        accepted=accepted,
        restoration_accepted=bool(accepted and restoration),
        trust_action=action,
        trust_radius_before=context.trust_radius,
        trust_radius_after=radius,
        predicted_reduction=predicted,
        actual_reduction=actual,
        reduction_ratio=float(ratio),
        canonical_primal_residual=solution.primal_residual,
        canonical_dual_residual=solution.dual_residual,
        topology_fingerprint=topology,
        cqp_numeric_fingerprint=numeric,
        fingerprint_match=True,
        permutation=evaluation.permutation,
        path_inventory_complete=inventory_complete,
        evaluation=evaluation,
        predictor_seconds=predictor_seconds,
        conversion_seconds=backend.conversion_seconds,
        setup_seconds=backend.setup_seconds,
        polish_seconds=0.0 if report is None else report.solve_seconds,
        transfer_seconds=evaluation.transfer_seconds,
        replay_seconds=evaluation.replay_seconds,
        acceptance_seconds=acceptance_seconds,
        hidden_cpu_fallback=False,
    )
