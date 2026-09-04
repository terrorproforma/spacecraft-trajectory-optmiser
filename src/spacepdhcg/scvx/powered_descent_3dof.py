"""Executable nonlinear 3-DoF powered-descent successive convexification loop.

This is the transparent CPU reference outer loop.  It uses the fixed-pattern CQP
transcription and a backend factory so the same acceptance, forcing and trust-region
logic can later drive the persistent CUDA workspace without being rewritten.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.backends.base import PersistentCQPBackend
from spacepdhcg.cqp import (
    CQPSolution,
    CanonicalCQP,
    CanonicalResidualAudit,
    CQPValues,
)
from spacepdhcg.models.powered_descent_3dof import (
    CONTROL_DIMENSION,
    STATE_DIMENSION,
    PoweredDescent3DOFModel,
    PoweredDescentPathDiagnostics,
)
from spacepdhcg.transcription.powered_descent_3dof import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
    PoweredDescentSCvxDiagnostics,
)

from .forcing_rule import (
    AdaptiveForcingRule,
    ForcingDecision,
    ForcingRuleConfig,
    OuterResidual,
    SolvePhase,
)
from .trust_region import (
    RadiusAction,
    TrustRegionConfig,
    TrustRegionController,
    TrustRegionUpdate,
)

FloatArray = NDArray[np.float64]

MERIT_MODES = ("single_shooting", "multiple_shooting")


class BackendBuilder(Protocol):
    """Construct a solver workspace for one numerical CQP."""

    def __call__(
        self,
        problem: CanonicalCQP,
        *,
        tolerance: float,
        iteration_limit: int,
    ) -> PersistentCQPBackend: ...


@dataclass(frozen=True, slots=True)
class PoweredDescentOuterConfig:
    """Acceptance, convergence and exact-penalty settings."""

    max_iterations: int = 15
    minimum_iterations: int = 2
    convergence_tolerance: float = 2.0e-4
    step_tolerance: float = 2.0e-2
    acceptance_threshold: float = 0.05
    feasibility_penalty: float = 100.0
    virtual_penalty: float = 100.0
    minimum_actual_reduction: float = 1.0e-10
    minimum_predicted_reduction: float = 1.0e-12
    restoration_reduction: float = 0.9
    max_resolves_per_iteration: int = 1
    # Opt-in: treat a residual-qualifiable inner status (Clarabel ``AlmostSolved``) as solved
    # when every independent residual is below ``almost_solved_residual_bound`` (the nonlinear
    # acceptance test still guards the step), instead of aborting with ``solver_failed``.
    accept_almost_solved: bool = False
    almost_solved_residual_bound: float = 1.0e-3
    # ``single_shooting`` (frozen default) evaluates the actual merit on a full rollout of the
    # candidate controls, so second-order linearisation error accumulates over the whole horizon
    # and is charged at ``feasibility_penalty`` through the terminal error.  ``multiple_shooting``
    # evaluates the actual merit on the decision states with per-interval nonlinear defects
    # penalised at ``virtual_penalty`` (the textbook SCvx exact-penalty merit), which keeps
    # model agreement first-order accurate on long horizons.
    merit_mode: str = "single_shooting"
    # Opt-in objective-stall termination: declare convergence when the merit has moved by less
    # than ``stall_merit_tolerance`` over the last ``stall_iterations`` outer iterations while the
    # nonlinear feasibility residual is below ``stall_feasibility_tolerance``.  ``0.0`` disables it.
    stall_merit_tolerance: float = 0.0
    stall_iterations: int = 3
    stall_feasibility_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 <= self.minimum_iterations <= self.max_iterations:
            raise ValueError("minimum_iterations must lie within the iteration budget")
        for name in (
            "convergence_tolerance",
            "step_tolerance",
            "feasibility_penalty",
            "virtual_penalty",
            "minimum_actual_reduction",
            "minimum_predicted_reduction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.acceptance_threshold < 1.0:
            raise ValueError("acceptance_threshold must lie in [0, 1)")
        if not 0.0 < self.restoration_reduction < 1.0:
            raise ValueError("restoration_reduction must lie strictly between zero and one")
        if self.max_resolves_per_iteration < 0:
            raise ValueError("max_resolves_per_iteration must be non-negative")
        if not np.isfinite(self.almost_solved_residual_bound) or (
            self.almost_solved_residual_bound <= 0.0
        ):
            raise ValueError("almost_solved_residual_bound must be finite and positive")
        if self.merit_mode not in MERIT_MODES:
            raise ValueError(f"merit_mode must be one of {MERIT_MODES}")
        if not np.isfinite(self.stall_merit_tolerance) or self.stall_merit_tolerance < 0.0:
            raise ValueError("stall_merit_tolerance must be finite and non-negative")
        if self.stall_iterations < 1:
            raise ValueError("stall_iterations must be positive")
        if not np.isfinite(self.stall_feasibility_tolerance) or (
            self.stall_feasibility_tolerance <= 0.0
        ):
            raise ValueError("stall_feasibility_tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class SCvxIterationRecord:
    """Machine-readable record for one outer iteration."""

    iteration: int
    phase: SolvePhase
    requested_tolerance: float
    effective_tolerance: float
    iteration_limit: int
    solver_status: str
    solver_iterations: int
    primal_residual: float
    dual_residual: float
    objective: float
    independent_primal_residual: float
    independent_dual_residual: float
    independent_natural_residual: float
    independent_cone_residual: float
    independent_complementarity: float
    setup_seconds: float
    solve_seconds: float
    trust_radius_before: float
    trust_radius_after: float
    trust_action: str
    step_fraction: float
    predicted_reduction: float
    actual_reduction: float
    agreement: float
    accepted: bool
    restoration_accepted: bool
    re_solved: bool
    merit_before: float
    merit_after: float
    residual: OuterResidual
    convex_diagnostics: PoweredDescentSCvxDiagnostics

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        return payload


@dataclass(frozen=True, slots=True)
class PoweredDescentSCvxResult:
    """Final nonlinear trajectory and complete outer-loop evidence."""

    status: str
    states: FloatArray
    controls: FloatArray
    merit: float
    residual: OuterResidual
    path_diagnostics: PoweredDescentPathDiagnostics
    iterations: tuple[SCvxIterationRecord, ...]
    accepted_iterations: int
    total_setup_seconds: float
    total_solve_seconds: float
    termination_reason: str = ""

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    @property
    def outer_iterations(self) -> int:
        return len(self.iterations)


@dataclass(frozen=True, slots=True)
class _Candidate:
    solution: CQPSolution
    independent_audit: CanonicalResidualAudit
    setup_seconds: float
    states: FloatArray
    controls: FloatArray
    rollout: FloatArray | None
    virtual: FloatArray
    convex_diagnostics: PoweredDescentSCvxDiagnostics
    residual: OuterResidual
    model_merit: float
    actual_merit: float
    step_fraction: float
    predicted_reduction: float
    actual_reduction: float
    agreement: float
    accepted: bool
    restoration_accepted: bool
    requested_tolerance: float = 0.0


def clarabel_reference_builder(
    problem: CanonicalCQP,
    *,
    tolerance: float,
    iteration_limit: int,
) -> PersistentClarabel:
    """Default high-transparency CPU reference backend."""

    return PersistentClarabel(
        problem,
        tolerance=tolerance,
        iteration_limit=iteration_limit,
        verbose=False,
    )


def make_dynamics_consistent_reference(
    model: PoweredDescent3DOFModel,
    initial_state: FloatArray,
    target_position: FloatArray,
    target_velocity: FloatArray,
    *,
    intervals: int,
    step_seconds: float,
    method: str = "forward_euler",
    substeps: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Generate a path-feasible discrete reference without solving an optimiser.

    A quadratic velocity sequence is chosen to satisfy the desired terminal velocity
    and the forward-Euler position sum.  Required thrust is then projected into the
    throttle and tilt set before nonlinear rollout.  When no projection is active the
    reference reaches the requested terminal position and velocity to roundoff (with the
    default forward-Euler map; the RK4 map lands within one step's curvature term).
    """

    initial = _vector(initial_state, STATE_DIMENSION, "initial_state")
    target_position_vector = _vector(target_position, 3, "target_position")
    target_velocity_vector = _vector(target_velocity, 3, "target_velocity")
    if intervals < 2:
        raise ValueError("intervals must be at least two")
    if not np.isfinite(step_seconds) or step_seconds <= 0.0:
        raise ValueError("step_seconds must be finite and positive")
    if initial[6] <= model.config.minimum_mass:
        raise ValueError("initial mass must exceed the configured minimum mass")

    node_indices = np.arange(intervals + 1, dtype=np.float64)
    velocities = np.empty((intervals + 1, 3), dtype=np.float64)
    first_moment = intervals * (intervals - 1) / 2.0
    second_moment = (intervals - 1) * intervals * (2 * intervals - 1) / 6.0
    system = np.asarray(
        [
            [intervals, intervals**2],
            [step_seconds * first_moment, step_seconds * second_moment],
        ],
        dtype=np.float64,
    )
    for axis in range(3):
        right_hand_side = np.asarray(
            [
                target_velocity_vector[axis] - initial[3 + axis],
                target_position_vector[axis]
                - initial[axis]
                - step_seconds * intervals * initial[3 + axis],
            ],
            dtype=np.float64,
        )
        linear_term, quadratic_term = np.linalg.solve(system, right_hand_side)
        velocities[:, axis] = (
            initial[3 + axis] + linear_term * node_indices + quadratic_term * node_indices**2
        )

    controls = np.empty((intervals, CONTROL_DIMENSION), dtype=np.float64)
    states = np.empty((intervals + 1, STATE_DIMENSION), dtype=np.float64)
    states[0] = initial
    for interval in range(intervals):
        acceleration = (velocities[interval + 1] - velocities[interval]) / step_seconds
        requested_thrust = states[interval, 6] * (acceleration - model.config.gravity_vector)
        thrust = _project_thrust(model, requested_thrust)
        sigma = float(np.linalg.norm(thrust))
        controls[interval] = np.concatenate((thrust, np.asarray([sigma])))
        states[interval + 1] = model.discrete_step(
            states[interval],
            controls[interval],
            step_seconds,
            method=method,
            substeps=substeps,
        )
        if states[interval + 1, 6] <= model.config.minimum_mass:
            raise ValueError("initial reference consumes the available propellant reserve")
    return states, controls


class PoweredDescentSCvxSolver:
    """Reference implementation of the nonlinear fixed-grid SCvx lifecycle."""

    def __init__(
        self,
        subproblem: PoweredDescent3DOFSubproblem | None = None,
        *,
        outer_config: PoweredDescentOuterConfig | None = None,
        forcing_config: ForcingRuleConfig | None = None,
        trust_config: TrustRegionConfig | None = None,
        backend_builder: BackendBuilder = clarabel_reference_builder,
    ) -> None:
        self.subproblem = subproblem or PoweredDescent3DOFSubproblem()
        self.model = self.subproblem.model
        self.outer_config = outer_config or PoweredDescentOuterConfig()
        self.forcing = AdaptiveForcingRule(forcing_config)
        if trust_config is None:
            initial_radius = self.subproblem.config.trust_radius
            trust_config = TrustRegionConfig(
                initial_radius=initial_radius,
                maximum_radius=max(8.0, initial_radius),
            )
        self.trust = TrustRegionController(trust_config)
        self.backend_builder = backend_builder

    def solve(
        self,
        initial_state: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
        *,
        reference_states: FloatArray | None = None,
        reference_controls: FloatArray | None = None,
    ) -> PoweredDescentSCvxResult:
        initial = _vector(initial_state, STATE_DIMENSION, "initial_state")
        target_position_vector = _vector(target_position, 3, "target_position")
        target_velocity_vector = _vector(target_velocity, 3, "target_velocity")
        if (reference_states is None) != (reference_controls is None):
            raise ValueError("reference states and controls must be supplied together")
        if reference_states is None:
            current_states, current_controls = make_dynamics_consistent_reference(
                self.model,
                initial,
                target_position_vector,
                target_velocity_vector,
                intervals=self.subproblem.layout.intervals,
                step_seconds=self.subproblem.config.step_seconds,
                method=self.subproblem.config.discretisation,
                substeps=self.subproblem.config.integration_substeps,
            )
        else:
            current_states, current_controls = self.subproblem._reference(
                reference_states,
                reference_controls,
            )
            current_states = self._rollout(initial, current_controls)

        current_merit = self._actual_merit(
            current_states,
            current_controls,
            target_position_vector,
            target_velocity_vector,
        )
        current_residual = self._outer_residual(
            decision_states=current_states,
            controls=current_controls,
            rollout=current_states,
            target_position=target_position_vector,
            target_velocity=target_velocity_vector,
            step_fraction=0.0,
        )
        records: list[SCvxIterationRecord] = []
        accepted_streak = 0
        accepted_iterations = 0
        previous_agreement: float | None = None
        warm_primal: FloatArray | None = None
        warm_dual: FloatArray | None = None
        status = "maximum_iterations"
        termination_reason = "outer iteration budget exhausted"
        merit_history: list[float] = [current_merit]
        backend: PersistentCQPBackend | None = None
        backend_tolerance: float | None = None
        backend_iteration_limit: int | None = None

        try:
            for iteration in range(self.outer_config.max_iterations):
                request = self.forcing.request(
                    iteration,
                    current_residual,
                    accepted_streak=accepted_streak,
                    agreement=previous_agreement,
                )
                values = self.subproblem.values(
                    current_states,
                    current_controls,
                    initial,
                    target_position_vector,
                    target_velocity_vector,
                    trust_radius=self.trust.radius,
                )
                problem = CanonicalCQP(self.subproblem.structure, values)
                setup_seconds = 0.0
                rebuild_for_settings = (
                    backend is not None
                    and not getattr(backend, "supports_dynamic_solve_settings", True)
                    and (
                        request.tolerance != backend_tolerance
                        or request.iteration_limit != backend_iteration_limit
                    )
                )
                if rebuild_for_settings:
                    close = getattr(backend, "close", None)
                    if callable(close):
                        close()
                    backend = None
                if backend is None:
                    setup_start = perf_counter()
                    backend = self.backend_builder(
                        problem,
                        tolerance=request.tolerance,
                        iteration_limit=request.iteration_limit,
                    )
                    setup_seconds = float(
                        getattr(backend, "setup_seconds", perf_counter() - setup_start)
                    )
                    if backend.structure != self.subproblem.structure:
                        raise RuntimeError("backend builder returned incompatible CQP structure")
                    backend_tolerance = request.tolerance
                    backend_iteration_limit = request.iteration_limit
                else:
                    backend.update(values)
                candidate = self._solve_candidate(
                    backend,
                    values,
                    request,
                    current_states,
                    current_controls,
                    initial,
                    target_position_vector,
                    target_velocity_vector,
                    current_merit,
                    current_residual,
                    warm_primal,
                    warm_dual,
                    setup_seconds,
                )
                effective_tolerance = request.tolerance
                re_solved = False
                for _ in range(self.outer_config.max_resolves_per_iteration):
                    if not self.forcing.should_resolve(
                        accepted=candidate.accepted,
                        primal_residual=candidate.solution.primal_residual,
                        dual_residual=candidate.solution.dual_residual,
                        requested_tolerance=effective_tolerance,
                    ):
                        break
                    effective_tolerance = self.forcing.refined_tolerance(effective_tolerance)
                    refined_request = ForcingDecision(
                        tolerance=effective_tolerance,
                        raw_target=request.raw_target,
                        iteration_limit=max(
                            request.iteration_limit,
                            self.forcing.config.convergence_iteration_limit,
                        ),
                        phase=request.phase,
                        reason="re-solve rejected candidate before shrinking trust region",
                    )
                    resolve_setup_seconds = 0.0
                    if not getattr(backend, "supports_dynamic_solve_settings", True):
                        close = getattr(backend, "close", None)
                        if callable(close):
                            close()
                        setup_start = perf_counter()
                        backend = self.backend_builder(
                            problem,
                            tolerance=refined_request.tolerance,
                            iteration_limit=refined_request.iteration_limit,
                        )
                        resolve_setup_seconds = float(
                            getattr(
                                backend,
                                "setup_seconds",
                                perf_counter() - setup_start,
                            )
                        )
                        backend_tolerance = refined_request.tolerance
                        backend_iteration_limit = refined_request.iteration_limit
                    candidate = self._solve_candidate(
                        backend,
                        values,
                        refined_request,
                        current_states,
                        current_controls,
                        initial,
                        target_position_vector,
                        target_velocity_vector,
                        current_merit,
                        current_residual,
                        warm_primal,
                        warm_dual,
                        resolve_setup_seconds,
                    )
                    re_solved = True

                retained_converged = (
                    current_residual.feasibility <= self.outer_config.convergence_tolerance
                    and current_residual.step <= self.outer_config.step_tolerance
                )
                trust_update = (
                    TrustRegionUpdate(
                        radius_before=self.trust.radius,
                        radius_after=self.trust.radius,
                        action=RadiusAction.KEEP,
                        reason="retained reference already satisfies outer convergence",
                    )
                    if retained_converged and not candidate.accepted
                    else self.trust.update(
                        accepted=candidate.accepted,
                        agreement=candidate.agreement,
                        step_fraction=candidate.step_fraction,
                    )
                )
                records.append(
                    self._record(
                        iteration,
                        request,
                        effective_tolerance,
                        candidate,
                        trust_update,
                        current_merit,
                        re_solved,
                    )
                )

                if not self._inner_solved(candidate):
                    status = "solver_failed"
                    termination_reason = f"inner solver status {candidate.solution.status}"
                    break
                if retained_converged and not candidate.accepted:
                    status = "converged"
                    termination_reason = "retained reference satisfies feasibility/step tolerances"
                    break
                if candidate.accepted and candidate.rollout is not None:
                    current_states = (
                        candidate.states
                        if self.outer_config.merit_mode == "multiple_shooting"
                        else candidate.rollout
                    )
                    current_controls = candidate.controls
                    current_merit = candidate.actual_merit
                    current_residual = candidate.residual
                    warm_primal = candidate.solution.primal
                    warm_dual = candidate.solution.dual
                    accepted_streak += 1
                    accepted_iterations += 1
                    previous_agreement = candidate.agreement
                    if (
                        iteration + 1 >= self.outer_config.minimum_iterations
                        and current_residual.feasibility <= self.outer_config.convergence_tolerance
                        and candidate.step_fraction <= self.outer_config.step_tolerance
                    ):
                        status = "converged"
                        termination_reason = "feasibility and step tolerances satisfied"
                        break
                else:
                    accepted_streak = 0
                    previous_agreement = None
                    if self.trust.exhausted:
                        status = "trust_region_exhausted"
                        termination_reason = "trust radius reached its minimum"
                        break
                merit_history.append(current_merit)
                if self._merit_stalled(merit_history, current_residual, iteration):
                    status = "converged"
                    termination_reason = (
                        f"merit stalled below {self.outer_config.stall_merit_tolerance:g} over "
                        f"{self.outer_config.stall_iterations} iterations"
                    )
                    break
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()

        path = self.model.path_diagnostics(current_states, current_controls)
        return PoweredDescentSCvxResult(
            status=status,
            states=current_states.copy(),
            controls=current_controls.copy(),
            merit=current_merit,
            residual=current_residual,
            path_diagnostics=path,
            iterations=tuple(records),
            accepted_iterations=accepted_iterations,
            total_setup_seconds=float(sum(record.setup_seconds for record in records)),
            total_solve_seconds=float(sum(record.solve_seconds for record in records)),
            termination_reason=termination_reason,
        )

    def _merit_stalled(
        self,
        merit_history: list[float],
        residual: OuterResidual,
        iteration: int,
    ) -> bool:
        config = self.outer_config
        if config.stall_merit_tolerance <= 0.0:
            return False
        if iteration + 1 < config.minimum_iterations:
            return False
        if len(merit_history) <= config.stall_iterations:
            return False
        if residual.feasibility > config.stall_feasibility_tolerance:
            return False
        window = np.asarray(merit_history[-(config.stall_iterations + 1) :], dtype=np.float64)
        if not np.all(np.isfinite(window)):
            return False
        return bool(np.max(window) - np.min(window) <= config.stall_merit_tolerance)

    def _solve_candidate(
        self,
        backend: PersistentCQPBackend,
        values: CQPValues,
        request: ForcingDecision,
        current_states: FloatArray,
        current_controls: FloatArray,
        initial_state: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
        current_merit: float,
        current_residual: OuterResidual,
        warm_primal: FloatArray | None,
        warm_dual: FloatArray | None,
        setup_seconds: float,
    ) -> _Candidate:
        if warm_primal is not None or warm_dual is not None:
            try:
                backend.warm_start(warm_primal, warm_dual)
            except NotImplementedError:
                pass
        if getattr(backend, "supports_dynamic_solve_settings", True):
            solution = backend.solve(
                tolerance=request.tolerance,
                iteration_limit=request.iteration_limit,
            )
        else:
            solution = backend.solve()
        if solution.primal.shape != (self.subproblem.layout.n_variables,):
            raise RuntimeError("backend returned an invalid primal vector")
        audit_method = getattr(backend, "independent_residuals", None)
        independent_audit = (
            audit_method(solution.primal)
            if callable(audit_method)
            else CanonicalResidualAudit(*(float("inf"),) * 5)
        )
        states, controls, virtual, _ = self.subproblem.decode(solution.primal)
        convex_diagnostics = self.subproblem.diagnostics(solution.primal, values)
        step_fraction = self._step_fraction(
            states,
            controls,
            current_states,
            current_controls,
            self.trust.radius,
        )
        rollout: FloatArray | None
        try:
            rollout = self._rollout(initial_state, controls)
        except ValueError:
            rollout = None

        model_merit = self._model_merit(
            states,
            controls,
            virtual,
            target_position,
            target_velocity,
        )
        if self.outer_config.merit_mode == "multiple_shooting":
            actual_merit = self._multiple_shooting_merit(
                states,
                controls,
                target_position,
                target_velocity,
            )
        else:
            actual_merit = (
                np.inf
                if rollout is None
                else self._actual_merit(
                    rollout,
                    controls,
                    target_position,
                    target_velocity,
                )
            )
        predicted_reduction = current_merit - model_merit
        actual_reduction = current_merit - actual_merit
        agreement = (
            actual_reduction / predicted_reduction
            if predicted_reduction > self.outer_config.minimum_predicted_reduction
            else -np.inf
        )
        residual = (
            OuterResidual(np.inf, np.inf, np.inf, step_fraction)
            if rollout is None
            else self._outer_residual(
                decision_states=states,
                controls=controls,
                rollout=rollout,
                target_position=target_position,
                target_velocity=target_velocity,
                step_fraction=step_fraction,
            )
        )
        restoration = (
            rollout is not None
            and residual.feasibility
            < self.outer_config.restoration_reduction * current_residual.feasibility
        )
        accepted = bool(
            self._solution_usable(solution, independent_audit, request.tolerance)
            and rollout is not None
            and np.isfinite(actual_merit)
            and (
                (
                    actual_reduction > self.outer_config.minimum_actual_reduction
                    and agreement >= self.outer_config.acceptance_threshold
                )
                or restoration
            )
        )
        return _Candidate(
            solution=solution,
            independent_audit=independent_audit,
            setup_seconds=setup_seconds,
            states=states,
            controls=controls,
            rollout=rollout,
            virtual=virtual,
            convex_diagnostics=convex_diagnostics,
            residual=residual,
            model_merit=model_merit,
            actual_merit=float(actual_merit),
            step_fraction=step_fraction,
            predicted_reduction=predicted_reduction,
            actual_reduction=actual_reduction,
            agreement=float(agreement),
            accepted=accepted,
            restoration_accepted=bool(accepted and restoration),
            requested_tolerance=float(request.tolerance),
        )

    def _rollout(self, initial_state: FloatArray, controls: FloatArray) -> FloatArray:
        return self.model.rollout(
            initial_state,
            controls,
            self.subproblem.config.step_seconds,
            method=self.subproblem.config.discretisation,
            substeps=self.subproblem.config.integration_substeps,
        )

    def _solution_usable(
        self,
        solution: CQPSolution,
        audit: CanonicalResidualAudit,
        requested_tolerance: float,
    ) -> bool:
        if solution.solved:
            return True
        if not self.outer_config.accept_almost_solved:
            return False
        if solution.status.lower().replace("_", "") not in {"almostsolved", "solvedinaccurate"}:
            return False
        residuals = np.asarray([audit.primal, audit.dual, audit.cone], dtype=np.float64)
        if not np.all(np.isfinite(residuals)):
            return False
        return bool(np.all(residuals <= self.outer_config.almost_solved_residual_bound))

    def _inner_solved(self, candidate: _Candidate) -> bool:
        return self._solution_usable(
            candidate.solution,
            candidate.independent_audit,
            candidate.requested_tolerance,
        )

    def _record(
        self,
        iteration: int,
        request: ForcingDecision,
        effective_tolerance: float,
        candidate: _Candidate,
        trust_update: TrustRegionUpdate,
        merit_before: float,
        re_solved: bool,
    ) -> SCvxIterationRecord:
        return SCvxIterationRecord(
            iteration=iteration,
            phase=request.phase,
            requested_tolerance=request.tolerance,
            effective_tolerance=effective_tolerance,
            iteration_limit=request.iteration_limit,
            solver_status=candidate.solution.status,
            solver_iterations=candidate.solution.iterations,
            primal_residual=candidate.solution.primal_residual,
            dual_residual=candidate.solution.dual_residual,
            objective=candidate.solution.objective,
            independent_primal_residual=candidate.independent_audit.primal,
            independent_dual_residual=candidate.independent_audit.dual,
            independent_natural_residual=candidate.independent_audit.natural,
            independent_cone_residual=candidate.independent_audit.cone,
            independent_complementarity=candidate.independent_audit.complementarity,
            setup_seconds=candidate.setup_seconds,
            solve_seconds=candidate.solution.solve_seconds,
            trust_radius_before=trust_update.radius_before,
            trust_radius_after=trust_update.radius_after,
            trust_action=trust_update.action.value,
            step_fraction=candidate.step_fraction,
            predicted_reduction=candidate.predicted_reduction,
            actual_reduction=candidate.actual_reduction,
            agreement=candidate.agreement,
            accepted=candidate.accepted,
            restoration_accepted=candidate.restoration_accepted,
            re_solved=re_solved,
            merit_before=merit_before,
            merit_after=candidate.actual_merit,
            residual=candidate.residual,
            convex_diagnostics=candidate.convex_diagnostics,
        )

    def _outer_residual(
        self,
        *,
        decision_states: FloatArray,
        controls: FloatArray,
        rollout: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
        step_fraction: float,
    ) -> OuterResidual:
        state_scales = np.asarray(self.subproblem.config.state_trust_scales)
        dynamics = float(np.max(np.abs((decision_states - rollout) * state_scales)))
        terminal_error = np.concatenate(
            (
                rollout[-1, :3] - target_position,
                rollout[-1, 3:6] - target_velocity,
            )
        )
        terminal = float(np.max(np.abs(terminal_error * state_scales[:6])))
        path = max(self._path_components(rollout, controls))
        return OuterResidual(
            dynamics=dynamics,
            path=path,
            terminal=terminal,
            step=step_fraction,
        )

    def _actual_merit(
        self,
        states: FloatArray,
        controls: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
    ) -> float:
        state_scales = np.asarray(self.subproblem.config.state_trust_scales)
        terminal_error = np.concatenate(
            (
                states[-1, :3] - target_position,
                states[-1, 3:6] - target_velocity,
            )
        )
        terminal_measure = float(np.sum(np.abs(terminal_error * state_scales[:6])))
        path_measure = float(sum(self._path_components(states, controls)))
        return self._normalised_fuel(controls) + self.outer_config.feasibility_penalty * (
            terminal_measure + path_measure
        )

    def _multiple_shooting_merit(
        self,
        states: FloatArray,
        controls: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
    ) -> float:
        """Exact-penalty merit on the decision trajectory with nonlinear interval defects.

        Mirrors :meth:`_model_merit` with the virtual control replaced by the true defect
        ``x_{k+1} - F(x_k, u_k)`` of the configured discrete map, so predicted and actual
        reductions differ only by the local second-order linearisation error of each interval.
        """

        if np.any(states[:, 6] <= 0.0):
            return float(np.inf)
        defects = np.empty((self.subproblem.layout.intervals, STATE_DIMENSION))
        try:
            for interval in range(self.subproblem.layout.intervals):
                defects[interval] = states[interval + 1] - self.model.discrete_step(
                    states[interval],
                    controls[interval],
                    self.subproblem.config.step_seconds,
                    method=self.subproblem.config.discretisation,
                    substeps=self.subproblem.config.integration_substeps,
                )
        except ValueError:
            return float(np.inf)
        return self._actual_merit(
            states,
            controls,
            target_position,
            target_velocity,
        ) + self._consistent_defect_penalty(defects)

    def _consistent_defect_penalty(self, defects: FloatArray) -> float:
        """Penalise defects with the weight the convex subproblem itself uses.

        The CQP minimises ``fuel_weight * dt * sum(sigma) + virtual_l1_weight * sum|nu|`` (plus
        proximal terms that vanish at the reference).  Dividing by ``fuel_weight * dt * N * T_max``
        expresses that objective in the merit's normalised-fuel units, so the merit's defect
        weight is ``virtual_l1_weight / (fuel_weight * dt * N * T_max)`` per unit of summed defect.
        Using the same weight in the model and actual merits guarantees a non-negative predicted
        reduction and makes the agreement ratio measure linearisation error only.
        """

        config = self.subproblem.config
        scale = (
            config.fuel_weight
            * config.step_seconds
            * self.subproblem.layout.intervals
            * self.model.config.maximum_thrust
        )
        return float(config.virtual_l1_weight / scale * np.sum(np.abs(defects)))

    def _model_merit(
        self,
        states: FloatArray,
        controls: FloatArray,
        virtual: FloatArray,
        target_position: FloatArray,
        target_velocity: FloatArray,
    ) -> float:
        state_scales = np.asarray(self.subproblem.config.state_trust_scales)
        terminal_error = np.concatenate(
            (
                states[-1, :3] - target_position,
                states[-1, 3:6] - target_velocity,
            )
        )
        terminal_measure = float(np.sum(np.abs(terminal_error * state_scales[:6])))
        path_measure = float(sum(self._path_components(states, controls)))
        if self.outer_config.merit_mode == "multiple_shooting":
            virtual_term = self._consistent_defect_penalty(virtual)
        else:
            virtual_term = self.outer_config.virtual_penalty * float(
                np.mean(np.abs(virtual) * state_scales)
            )
        return (
            self._normalised_fuel(controls)
            + self.outer_config.feasibility_penalty * (terminal_measure + path_measure)
            + virtual_term
        )

    def _path_components(
        self,
        states: FloatArray,
        controls: FloatArray,
    ) -> tuple[float, ...]:
        diagnostics = self.model.path_diagnostics(states, controls)
        maximum_thrust = self.model.config.maximum_thrust
        position_scale = max(self.subproblem.config.state_trust_scales[:3])
        mass_scale = self.subproblem.config.state_trust_scales[6]
        return (
            diagnostics.thrust_epigraph / maximum_thrust,
            diagnostics.throttle_lower / maximum_thrust,
            diagnostics.throttle_upper / maximum_thrust,
            diagnostics.tilt / maximum_thrust,
            diagnostics.minimum_mass * mass_scale,
            diagnostics.altitude * position_scale,
            diagnostics.glide_slope * position_scale,
        )

    def _normalised_fuel(self, controls: FloatArray) -> float:
        return float(np.mean(controls[:, 3]) / self.model.config.maximum_thrust)

    def _step_fraction(
        self,
        states: FloatArray,
        controls: FloatArray,
        reference_states: FloatArray,
        reference_controls: FloatArray,
        radius: float,
    ) -> float:
        state_scales = np.asarray(self.subproblem.config.state_trust_scales)
        control_scales = np.asarray(self.subproblem.config.control_trust_scales)
        fractions: list[float] = []
        for interval in range(self.subproblem.layout.intervals):
            delta = np.concatenate(
                (
                    (states[interval] - reference_states[interval]) * state_scales,
                    (controls[interval] - reference_controls[interval]) * control_scales,
                )
            )
            fractions.append(float(np.linalg.norm(delta) / radius))
        terminal_delta = (states[-1] - reference_states[-1]) * state_scales
        fractions.append(float(np.linalg.norm(terminal_delta) / radius))
        return max(fractions)


def _project_thrust(model: PoweredDescent3DOFModel, requested: FloatArray) -> FloatArray:
    thrust = _vector(requested, 3, "requested thrust")
    if thrust[2] <= 0.0:
        thrust = np.asarray([0.0, 0.0, max(model.config.minimum_sigma, 1.0)], dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(thrust[:2]))
    maximum_horizontal = thrust[2] * np.tan(model.config.maximum_tilt_radians)
    if horizontal_norm > maximum_horizontal and horizontal_norm > 0.0:
        thrust[:2] *= maximum_horizontal / horizontal_norm
    norm = float(np.linalg.norm(thrust))
    if norm > model.config.maximum_thrust:
        thrust *= model.config.maximum_thrust / norm
        norm = model.config.maximum_thrust
    if norm < model.config.minimum_sigma:
        thrust = np.asarray([0.0, 0.0, model.config.minimum_sigma], dtype=np.float64)
    return thrust


def _vector(values: FloatArray, size: int, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector.copy()
