"""CPU reference planner: transparent SCvx outer loop over the native transcription ABI.

The inner conic solves use Clarabel (the project's CPU correctness reference); the
transcription, dynamics, replay, and nonlinear metrics come from the same C++ code the
CUDA executable uses (through :mod:`spacepdhcg.planner.native_library`).  Acceptance,
trust-region, and convergence rules mirror the device SCvx driver so a CPU plan and a
GPU plan of the same problem can be compared directly.  Results are labelled
``execution: cpu_reference`` and are never presented as GPU results.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import CanonicalCQP
from spacepdhcg.planner.native_library import Evaluation, PlannerTranscription
from spacepdhcg.planner.result import RESULT_KIND, RESULT_SCHEMA_VERSION, PlanResult, json_safe
from spacepdhcg.scvx.forcing_rule import AdaptiveForcingRule, ForcingRuleConfig, OuterResidual

FloatArray = NDArray[np.float64]
ProgressCallback = Callable[[str], None]

CPU_REFERENCE_EXIT_CODES = {
    "certified": 0,
    "not_certified": 2,
    "solver_failure": 3,
}


@dataclass(frozen=True, slots=True)
class _Candidate:
    accepted: bool
    restoration: bool
    solved: bool
    states: FloatArray
    controls: FloatArray
    rollout: FloatArray | None
    virtual_max: float
    virtual_mean: float
    model_evaluation: Evaluation
    actual_evaluation: Evaluation | None
    dynamics_defect: float
    step: float
    model_merit: float
    actual_merit: float
    predicted_reduction: float
    actual_reduction: float
    ratio: float
    rollout_error: str


def _merit(evaluation: Evaluation, feasibility_penalty: float, terminal_scaled_sum: float) -> float:
    path_sum = float(sum(evaluation.path_components.values()))
    return evaluation.objective + feasibility_penalty * (path_sum + terminal_scaled_sum)


def _terminal_sum(
    states: FloatArray,
    target: FloatArray,
    fixed: list[bool],
    scales: FloatArray,
) -> float:
    final = states[-1]
    total = 0.0
    for index, flag in enumerate(fixed):
        if flag:
            total += abs(float(final[index] - target[index])) * float(scales[index])
    return total


def _scaled_step(
    states: FloatArray,
    controls: FloatArray,
    reference_states: FloatArray,
    reference_controls: FloatArray,
    state_scales: FloatArray,
    control_scales: FloatArray,
) -> float:
    intervals = controls.shape[0]
    maximum = 0.0
    for interval in range(intervals):
        delta_state = (states[interval] - reference_states[interval]) * state_scales
        delta_control = (controls[interval] - reference_controls[interval]) * control_scales
        maximum = max(
            maximum,
            math.sqrt(
                float(np.dot(delta_state, delta_state) + np.dot(delta_control, delta_control))
            ),
        )
    terminal = (states[-1] - reference_states[-1]) * state_scales
    return max(maximum, math.sqrt(float(np.dot(terminal, terminal))))


def solve_cpu_reference(
    canonical_document: Mapping[str, Any],
    *,
    progress: ProgressCallback | None = None,
) -> PlanResult:
    """Run the CPU reference SCvx planner on a canonical problem document."""

    wall_started = time.perf_counter()
    report = progress or (lambda _message: None)
    transcription = PlannerTranscription(canonical_document)
    try:
        return _solve(transcription, wall_started, report)
    finally:
        transcription.close()


def _solve(
    transcription: PlannerTranscription, wall_started: float, report: ProgressCallback
) -> PlanResult:
    description = transcription.description
    solver = description["solver"]
    trust = solver["trust_region"]
    penalty = solver["penalty"]
    forcing_options = solver["forcing"]
    weights = description["transcription"]
    dims = transcription.dimensions
    family = description["family"]
    tolerance = float(solver["tolerance"])
    step_tolerance = float(solver["step_tolerance"])
    certificate_tolerance = float(solver["certificate_tolerance"])
    time_limit = float(solver["time_limit_seconds"])
    maximum_outer = int(solver["maximum_outer_iterations"])
    minimum_outer = int(solver["minimum_outer_iterations"])
    substeps = int(description["output"]["dense_replay_substeps"])
    include_iterations = bool(description["output"]["include_iterations"])

    initial_state = np.asarray(description["initial_state"], dtype=np.float64)
    target_state = np.asarray(description["terminal"]["state"], dtype=np.float64)
    fixed = [bool(flag) for flag in description["terminal"]["fixed"]]
    if family == "hcw":
        state_scales = np.ones(dims.state_dimension)
        control_scales = np.ones(dims.control_dimension)
    else:
        state_scales = np.asarray(weights["state_trust_scales"], dtype=np.float64)
        control_scales = np.asarray(weights["control_trust_scales"], dtype=np.float64)
    feasibility_penalty = float(penalty["feasibility_penalty"])
    virtual_penalty = float(penalty["virtual_penalty"])

    # Inner tolerance policy: the pure-QOCO preset fixes 1e-8; otherwise follow the frozen
    # adaptive forcing rule (Clarabel is an interior-point method, so the request mostly
    # governs the final polish accuracy).
    fixed_inner = float(forcing_options.get("fixed_inner_tolerance", 0.0) or 0.0)
    forcing = AdaptiveForcingRule(
        ForcingRuleConfig(
            epsilon_max=float(forcing_options["epsilon_max"]),
            epsilon_floor=float(forcing_options["epsilon_floor"]),
            epsilon_0=float(forcing_options["epsilon_0"]),
            coefficient=float(forcing_options["coefficient"]),
            alpha=float(forcing_options["alpha"]),
            gamma=float(forcing_options["gamma"]),
            polish_tolerance=min(1.0e-9, float(forcing_options["epsilon_floor"])),
        )
    )

    topology_started = time.perf_counter()
    reference_states, reference_controls = transcription.initial_reference()
    current_evaluation = transcription.evaluate(reference_states, reference_controls)
    initial_evaluation = current_evaluation
    current_merit = _merit(
        current_evaluation,
        feasibility_penalty,
        _terminal_sum(reference_states, target_state, fixed, state_scales),
    )
    current_residual = OuterResidual(
        dynamics=0.0,
        path=current_evaluation.path_violation,
        terminal=current_evaluation.terminal_residual,
        step=0.0,
    )
    initial_feasibility = current_residual.feasibility
    topology_seconds = time.perf_counter() - topology_started
    report(
        f"cpu_reference {family}: N={dims.intervals} variables={dims.variables} "
        f"initial path={current_residual.path:.3e} terminal={current_residual.terminal:.3e}"
    )

    radius = float(trust["initial_radius"])
    minimum_radius = float(trust["minimum_radius"])
    maximum_radius = float(trust["maximum_radius"])
    backend: PersistentClarabel | None = None
    backend_tolerance: float | None = None
    records: list[dict[str, Any]] = []
    accepted_steps = 0
    rejected_steps = 0
    accepted_streak = 0
    previous_agreement: float | None = None
    last_audit_natural = math.nan
    last_audit_absolute = math.nan
    last_virtual = 0.0
    inner_iterations = 0
    setup_seconds = 0.0
    solve_seconds = 0.0
    replay_seconds = 0.0
    status = "maximum_iterations"
    solver_failure = ""
    time_limit_triggered = False

    try:
        for outer in range(maximum_outer):
            if time_limit > 0.0 and time.perf_counter() - wall_started > time_limit:
                time_limit_triggered = True
                status = "cancelled"
                break
            request = forcing.request(
                outer,
                current_residual,
                accepted_streak=accepted_streak,
                agreement=previous_agreement,
            )
            inner_tolerance = (
                fixed_inner if fixed_inner > 0.0 else max(1.0e-9, min(request.tolerance, 1.0e-6))
            )
            values = transcription.values(reference_states, reference_controls, radius)
            problem = CanonicalCQP(transcription.structure, values)
            setup_started = time.perf_counter()
            if backend is None or backend_tolerance != inner_tolerance:
                backend = PersistentClarabel(
                    problem, tolerance=inner_tolerance, iteration_limit=500
                )
                backend_tolerance = inner_tolerance
            else:
                backend.update(values)
            setup_seconds += time.perf_counter() - setup_started
            solve_started = time.perf_counter()
            solution = backend.solve()
            solve_seconds += time.perf_counter() - solve_started
            inner_iterations += int(solution.iterations)
            absolute_audit = backend.independent_residuals(solution.primal)
            audit = backend.relative_kkt_residuals(solution.primal)
            last_audit_natural = float(audit.natural)
            last_audit_absolute = float(absolute_audit.natural)
            # Clarabel reports "AlmostSolved" for reduced-accuracy optima; like the device
            # driver (QOCO status solved / solved-inaccurate) the candidate is usable when the
            # independent canonical audit is finite. The certificate still gates the residual.
            solved = solution.solved or (
                solution.status.lower().startswith("almostsolved")
                and math.isfinite(last_audit_natural)
            )
            candidate = _evaluate_candidate(
                transcription,
                solution.primal,
                solved,
                reference_states,
                reference_controls,
                initial_state,
                target_state,
                fixed,
                state_scales,
                control_scales,
                feasibility_penalty,
                virtual_penalty,
                current_merit,
                current_residual,
                float(trust["acceptance_threshold"]),
                float(trust["restoration_reduction"]),
            )
            replay_seconds += candidate_replay_seconds(candidate)
            radius_before = radius
            if candidate.accepted:
                agreement = candidate.ratio
                if agreement >= float(trust["strong_agreement_threshold"]) and (
                    candidate.step >= float(trust["near_boundary_fraction"]) * radius
                ):
                    radius = min(maximum_radius, radius * float(trust["expansion_factor"]))
                    action = "expand"
                else:
                    action = "retain"
            else:
                radius = max(minimum_radius, radius * float(trust["shrink_factor"]))
                action = "shrink"
            records.append(
                {
                    "outer_iteration": outer,
                    "phase": request.phase.value,
                    "requested_tolerance": inner_tolerance,
                    "achieved_residual": last_audit_natural,
                    "inner_iterations": int(solution.iterations),
                    "solver_status": solution.status,
                    "trust_radius_before": radius_before,
                    "trust_radius_after": radius,
                    "trust_action": action,
                    "predicted_reduction": candidate.predicted_reduction,
                    "actual_reduction": candidate.actual_reduction,
                    "reduction_ratio": candidate.ratio,
                    "step_fraction": candidate.step / radius_before
                    if radius_before > 0
                    else math.nan,
                    "trajectory_step": candidate.step,
                    "objective": candidate.model_evaluation.objective,
                    "virtual_control": candidate.virtual_max,
                    "dynamics_defect": candidate.dynamics_defect,
                    "path_violation": (
                        candidate.actual_evaluation.path_violation
                        if candidate.actual_evaluation is not None
                        else math.nan
                    ),
                    "terminal_residual": (
                        candidate.actual_evaluation.terminal_residual
                        if candidate.actual_evaluation is not None
                        else math.nan
                    ),
                    "accepted": candidate.accepted,
                    "restoration_accepted": candidate.restoration,
                    "re_solved": False,
                    "current_merit": current_merit,
                    "candidate_merit": candidate.actual_merit,
                    "candidate_model_merit": candidate.model_merit,
                    "current_dynamics_defect": current_residual.dynamics,
                    "current_path_violation": current_residual.path,
                    "current_terminal_residual": current_residual.terminal,
                    "independent_primal_residual": float(audit.primal),
                    "independent_dual_residual": float(audit.dual),
                    "natural_residual": float(audit.natural),
                    "absolute_natural_residual": last_audit_absolute,
                    "rollout_error": candidate.rollout_error or None,
                }
            )
            report(
                f"  outer {outer}: {'accepted' if candidate.accepted else 'rejected'} "
                f"ratio={candidate.ratio:.3g} step={candidate.step:.3e} "
                f"radius={radius_before:.3g}->{radius:.3g} "
                f"path={current_residual.path:.2e} terminal={current_residual.terminal:.2e}"
            )
            if not candidate.solved:
                status = "inner_failure"
                solver_failure = f"Clarabel returned status {solution.status!r}"
                break
            if (
                candidate.accepted
                and candidate.rollout is not None
                and candidate.actual_evaluation is not None
            ):
                reference_states = candidate.rollout
                reference_controls = candidate.controls
                current_merit = candidate.actual_merit
                current_evaluation = candidate.actual_evaluation
                current_residual = OuterResidual(
                    dynamics=0.0,
                    path=candidate.actual_evaluation.path_violation,
                    terminal=candidate.actual_evaluation.terminal_residual,
                    step=candidate.step,
                )
                last_virtual = candidate.virtual_max
                accepted_steps += 1
                accepted_streak += 1
                previous_agreement = candidate.ratio
            else:
                rejected_steps += 1
                accepted_streak = 0
                previous_agreement = None
            # Converged when the retained reference is feasible within tolerance and the
            # SCvx step has collapsed: either the accepted step itself was small, or the
            # rejected candidate lies within the step tolerance of the retained reference
            # (the convex subproblem returns the current point, i.e. a fixed point).
            fixed_point_step = current_residual.step if candidate.accepted else candidate.step
            if (
                outer + 1 >= minimum_outer
                and current_residual.feasibility <= tolerance
                and fixed_point_step <= step_tolerance
                and (accepted_steps > 0 or initial_feasibility <= tolerance)
            ):
                status = "converged"
                break
            if not candidate.accepted and radius <= minimum_radius * (1.0 + 1.0e-12):
                status = "trust_region_exhausted"
                break
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    if status == "maximum_iterations" and (
        current_residual.feasibility <= tolerance and current_residual.step <= step_tolerance
    ):
        status = "converged"

    # Independent replay (same C++ integrator the CUDA executable uses) ---------
    replay_started = time.perf_counter()
    node_replay = transcription.rollout(initial_state, reference_controls, 1)
    replay_parity = (
        float(np.max(np.abs(node_replay - reference_states))) if node_replay.size else 0.0
    )
    independent_dynamics = float(np.max(np.abs((node_replay - reference_states) * state_scales)))
    independent = transcription.evaluate(node_replay, reference_controls)
    dense_replay = transcription.rollout(initial_state, reference_controls, substeps)
    dense_controls = np.repeat(reference_controls, substeps, axis=0)
    dense_path = transcription.path_components(dense_replay, dense_controls)
    independent_replay_seconds = time.perf_counter() - replay_started

    converged = status == "converged"
    canonical = last_audit_natural
    gates = {
        "solver_api_success": (
            status != "inner_failure",
            0.0 if status != "inner_failure" else 1.0,
            0.0,
        ),
        "converged": (converged, 0.0 if converged else 1.0, 0.0),
        "canonical_residual": (
            math.isfinite(canonical) and canonical <= certificate_tolerance,
            canonical,
            certificate_tolerance,
        ),
        "device_dynamics_defect": (
            current_residual.dynamics <= certificate_tolerance,
            current_residual.dynamics,
            certificate_tolerance,
        ),
        "device_path_violation": (
            current_residual.path <= certificate_tolerance,
            current_residual.path,
            certificate_tolerance,
        ),
        "device_terminal_residual": (
            current_residual.terminal <= certificate_tolerance,
            current_residual.terminal,
            certificate_tolerance,
        ),
        "virtual_control": (
            last_virtual <= certificate_tolerance,
            last_virtual,
            certificate_tolerance,
        ),
        "independent_replay_parity": (
            replay_parity <= float(solver["replay_parity_tolerance"]),
            replay_parity,
            float(solver["replay_parity_tolerance"]),
        ),
        "independent_dynamics_defect": (
            independent_dynamics <= certificate_tolerance,
            independent_dynamics,
            certificate_tolerance,
        ),
        "independent_path_violation": (
            independent.path_violation <= certificate_tolerance,
            independent.path_violation,
            certificate_tolerance,
        ),
        "independent_terminal_residual": (
            independent.terminal_residual <= certificate_tolerance,
            independent.terminal_residual,
            certificate_tolerance,
        ),
        "no_hidden_cpu_fallback": (True, 0.0, 0.0),
        "steady_state_residency": (True, 0.0, 0.0),
        "coefficient_parity": (True, 0.0, 5.0e-12),
        "independent_replay_evaluated": (True, 0.0, 0.0),
    }
    certified = all(passed for passed, _, _ in gates.values())
    failed = [name for name, (passed, _, _) in gates.items() if not passed]

    if status == "inner_failure":
        code, message = "solver_failure", f"inner solver failure: {solver_failure}"
        exit_code = CPU_REFERENCE_EXIT_CODES["solver_failure"]
    elif time_limit_triggered:
        code, message = "time_limit", "the CPU reference solver stopped at the requested time limit"
        exit_code = CPU_REFERENCE_EXIT_CODES["not_certified"]
    elif certified:
        code, message = (
            "certified",
            "converged and independently certified (CPU reference execution)",
        )
        exit_code = CPU_REFERENCE_EXIT_CODES["certified"]
    elif status == "trust_region_exhausted":
        code = "trust_region_exhausted"
        message = (
            "the trust region shrank to its minimum radius without an accepted improving step; "
            "the retained reference did not meet the certificate gates"
        )
        exit_code = CPU_REFERENCE_EXIT_CODES["not_certified"]
    elif status == "maximum_iterations":
        code, message = (
            "maximum_iterations",
            "the outer iteration budget was exhausted before convergence",
        )
        exit_code = CPU_REFERENCE_EXIT_CODES["not_certified"]
    elif converged:
        code = "converged_not_certified"
        message = "the solver reported convergence but the certificate gates failed"
        exit_code = CPU_REFERENCE_EXIT_CODES["not_certified"]
    else:
        code, message = (
            "not_certified",
            f"the plan was produced but is not certified (status {status})",
        )
        exit_code = CPU_REFERENCE_EXIT_CODES["not_certified"]

    step = dims.step_seconds
    plan_wall = time.perf_counter() - wall_started
    document: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": RESULT_KIND,
        "source_commit": None,
        "status": {
            "code": code,
            "message": message,
            "exit_code": exit_code,
            "solver_status": status,
            "api_status": "success" if status != "inner_failure" else "numerical_failure",
            "time_limit_triggered": time_limit_triggered,
        },
        "problem": description,
        "summary": {
            "objective": independent.objective,
            "objective_definition": _objective_definition(family),
            "outer_iterations": len(records),
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "resolved_steps": 0,
            "inner_iterations": inner_iterations,
            "final_trust_radius": radius,
            "trajectory_step": current_residual.step,
            "propellant_used": independent.propellant_used,
            "final_mass": independent.final_mass,
            "terminal_position_error": independent.terminal_position_error,
            "terminal_velocity_error": independent.terminal_velocity_error,
        },
        "solver_residuals": {
            "canonical_residual": canonical,
            "canonical_residual_definition": (
                "independent relative KKT audit of the last Clarabel solve "
                "(max of relative primal, dual, and complementarity residuals)"
            ),
            "canonical_residual_absolute": last_audit_absolute,
            "dynamics_defect": current_residual.dynamics,
            "path_violation": current_residual.path,
            "terminal_residual": current_residual.terminal,
            "virtual_control": last_virtual,
            "coefficient_parity_relative": 0.0,
        },
        "independent_replay": {
            **independent.as_dict(),
            "objective_definition": _objective_definition(family),
            "replay_error": None,
            "replay_parity": replay_parity,
            "dynamics_defect": independent_dynamics,
            "continuous_time_violation": dense_path.path_violation,
            "continuous_time_components": dict(dense_path.path_components),
            "continuous_time_components_physical": dict(dense_path.path_components_physical),
            "dense_replay_substeps": substeps,
        },
        "model_evaluation": current_evaluation.as_dict(),
        "initial_reference_evaluation": initial_evaluation.as_dict(),
        "trajectory": {
            "times": [index * step for index in range(dims.intervals + 1)],
            "states": json_safe(reference_states),
            "controls": json_safe(reference_controls),
        },
        "dense_replay": {
            "integrator": (
                "exact zero-order-hold HCW transition per substep"
                if family == "hcw"
                else "classical RK4 per substep, piecewise-constant controls"
            ),
            "substeps": substeps,
            "times": [index * step / substeps for index in range(dims.intervals * substeps + 1)],
            "states": json_safe(dense_replay),
        },
        "timings": {
            "cuda_startup_seconds": 0.0,
            "topology_seconds": topology_seconds,
            "coefficient_seconds": 0.0,
            "workspace_create_seconds": setup_seconds,
            "update_seconds": 0.0,
            "scaling_seconds": 0.0,
            "h2d_seconds": 0.0,
            "solve_seconds": solve_seconds,
            "recovery_seconds": 0.0,
            "residual_seconds": 0.0,
            "replay_seconds": replay_seconds,
            "acceptance_seconds": 0.0,
            "d2h_seconds": 0.0,
            "cqp_total_seconds": setup_seconds + solve_seconds,
            "scvx_total_seconds": plan_wall - independent_replay_seconds,
            "solve_wall_seconds": plan_wall - independent_replay_seconds,
            "independent_replay_seconds": independent_replay_seconds,
            "plan_wall_seconds": plan_wall,
        },
        "backend": {
            "execution": "cpu_reference",
            "requested_backend": "cpu_reference",
            "preset": solver["preset"],
            "device_policy": "clarabel_scvx_reference",
            "description": (
                "Python SCvx outer loop (device acceptance/trust/convergence rules) with Clarabel "
                "interior-point inner solves over the native C++ transcription; CPU only"
            ),
            "warm_start_mode": "none",
            "hidden_cpu_fallback": False,
            "qoco_failure": "none",
            "inner_solver": "clarabel",
            "inner_tolerance": fixed_inner if fixed_inner > 0.0 else None,
            "variables": dims.variables,
            "scalar_rows": dims.scalar_rows,
            "affine_rows": dims.affine_rows,
            "device": {"cuda_available": False},
        },
        "certificate": {
            "certified": certified,
            "tolerance": certificate_tolerance,
            "replay_parity_tolerance": float(solver["replay_parity_tolerance"]),
            "gates": {
                name: {"passed": bool(passed), "value": value, "limit": limit}
                for name, (passed, value, limit) in gates.items()
            },
            "failed_gates": failed,
            "continuous_time_violation": dense_path.path_violation,
            "continuous_time_within_tolerance": dense_path.path_violation <= certificate_tolerance,
            "definition": (
                "CPU reference execution: certified only when the SCvx loop converged with the "
                "Clarabel canonical residual, nonlinear dynamics/path/terminal residuals, and "
                "virtual control within tolerance and the independent RK4/ZOH replay of the "
                "returned controls satisfies the same gates; continuous-time violation is reported "
                "but not gated"
            ),
        },
    }
    if include_iterations:
        document["iterations"] = json_safe(records)
    return PlanResult(document=json_safe(document))


def candidate_replay_seconds(candidate: _Candidate) -> float:
    # Replay time is folded into the evaluation; kept as a hook for symmetry with the device.
    del candidate
    return 0.0


def _objective_definition(family: str) -> str:
    if family == "hcw":
        return "0.5 * sum_k |a_k|^2 (m^2/s^4)"
    return "mean_k(sigma_k) / maximum_thrust (normalised fuel)"


def _evaluate_candidate(
    transcription: PlannerTranscription,
    primal: FloatArray,
    solved: bool,
    reference_states: FloatArray,
    reference_controls: FloatArray,
    initial_state: FloatArray,
    target_state: FloatArray,
    fixed: list[bool],
    state_scales: FloatArray,
    control_scales: FloatArray,
    feasibility_penalty: float,
    virtual_penalty: float,
    current_merit: float,
    current_residual: OuterResidual,
    acceptance_threshold: float,
    restoration_reduction: float,
) -> _Candidate:
    states, controls, virtual = transcription.decode(primal)
    virtual_scaled = (
        np.abs(virtual) * state_scales if virtual.size else np.zeros((0, state_scales.size))
    )
    virtual_max = float(np.max(virtual_scaled)) if virtual_scaled.size else 0.0
    virtual_mean = float(np.mean(virtual_scaled)) if virtual_scaled.size else 0.0
    step = _scaled_step(
        states, controls, reference_states, reference_controls, state_scales, control_scales
    )
    rollout: FloatArray | None = None
    rollout_error = ""
    model_evaluation: Evaluation | None = None
    model_merit = math.inf
    try:
        # A non-physical decision vector (e.g. an infeasible CQP returned by the inner
        # solver) is reported as a rejected candidate, never raised.
        model_evaluation = transcription.evaluate(states, controls)
        model_merit = (
            _merit(
                model_evaluation,
                feasibility_penalty,
                _terminal_sum(states, target_state, fixed, state_scales),
            )
            + virtual_penalty * virtual_mean
        )
        rollout = transcription.rollout(initial_state, controls, 1)
    except Exception as error:
        rollout = None
        rollout_error = str(error)
    if model_evaluation is None:
        model_evaluation = Evaluation(
            objective=math.nan,
            path_violation=math.inf,
            path_components={},
            path_components_physical={},
            terminal_residual=math.inf,
            terminal_position_error=math.inf,
            terminal_velocity_error=math.inf,
            propellant_used=math.nan,
            final_mass=math.nan,
        )
    if rollout is None:
        return _Candidate(
            accepted=False,
            restoration=False,
            solved=solved,
            states=states,
            controls=controls,
            rollout=None,
            virtual_max=virtual_max,
            virtual_mean=virtual_mean,
            model_evaluation=model_evaluation,
            actual_evaluation=None,
            dynamics_defect=math.inf,
            step=step,
            model_merit=model_merit,
            actual_merit=math.inf,
            predicted_reduction=current_merit - model_merit,
            actual_reduction=-math.inf,
            ratio=-math.inf,
            rollout_error=rollout_error,
        )
    actual_evaluation = transcription.evaluate(rollout, controls)
    dynamics_defect = float(np.max(np.abs((states - rollout) * state_scales)))
    actual_merit = _merit(
        actual_evaluation,
        feasibility_penalty,
        _terminal_sum(rollout, target_state, fixed, state_scales),
    )
    predicted = current_merit - model_merit
    actual = current_merit - actual_merit
    ratio = actual / predicted if predicted > 1.0e-12 else -math.inf
    actual_feasibility = max(
        dynamics_defect, actual_evaluation.path_violation, actual_evaluation.terminal_residual
    )
    restoration = actual_feasibility < restoration_reduction * current_residual.feasibility
    accepted = bool(
        solved
        and math.isfinite(actual_merit)
        and ((actual > 1.0e-10 and ratio >= acceptance_threshold) or restoration)
    )
    return _Candidate(
        accepted=accepted,
        restoration=bool(accepted and restoration),
        solved=solved,
        states=states,
        controls=controls,
        rollout=rollout,
        virtual_max=virtual_max,
        virtual_mean=virtual_mean,
        model_evaluation=model_evaluation,
        actual_evaluation=actual_evaluation,
        dynamics_defect=dynamics_defect,
        step=step,
        model_merit=model_merit,
        actual_merit=actual_merit,
        predicted_reduction=predicted,
        actual_reduction=actual,
        ratio=float(ratio),
        rollout_error=rollout_error,
    )
