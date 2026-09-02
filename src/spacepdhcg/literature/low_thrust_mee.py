"""Modified-equinoctial-element (MEE) formulation for multi-revolution low-thrust transfers.

Cartesian successive convexification of a many-revolution rendezvous fails from geometric
initial guesses because every revolution of position error is a large, oscillatory defect that
the linearised dynamics cannot absorb.  In MEE ``x = [p, f, g, h, k, L, m]`` the five slow
elements vary smoothly under low thrust while the true longitude ``L`` does the winding, so a
guess that interpolates the slow elements and integrates the Keplerian ``L`` rate is close to
dynamically feasible for any revolution count.  The revolution count is explicit:
``L_f = L_target + 2 pi N`` is an equality constraint, and the solver sweeps ``N``.

Dynamics follow Walker, Ireland & Owens (1985, corrected 1986) with the thrust expressed in the
radial-transverse-normal frame; Jacobians are obtained by complex-step differentiation of the
same code path (exact to roundoff), and every converged trajectory is replayed through the
Cartesian two-body dynamics of :mod:`spacepdhcg.literature.low_thrust` for an independent check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.literature.low_thrust import (
    CONTROL_DIMENSION,
    LowThrustProblem,
    cartesian_to_elements,
)
from spacepdhcg.literature.scvx_core import (
    FreeFinalTimeSCvx,
    SCvxOutcome,
    SCvxSettings,
    SCvxVariables,
)

FloatArray = NDArray[np.float64]
MEE_STATE_DIMENSION = 7
TWO_PI = 2.0 * math.pi

#: Trust-weight continuation ``(soft trust weight, iteration budget)``: each stage is
#: warm-started from the previous and accepted only if it stays dynamically consistent.
DEFAULT_TRUST_SCHEDULE: tuple[tuple[float, int], ...] = (
    (1.0e-2, 15),
    (1.0e-3, 40),
    (1.0e-4, 20),
)
#: Fallback for guesses the default schedule cannot hold (highly eccentric orbits): a stiffer
#: first stage plus a per-node hard trust radius keeps the geometric guess inside the
#: linearisation's validity before the lighter stages take over.
ROBUST_TRUST_SCHEDULE: tuple[tuple[float, int], ...] = ((1.0e-1, 15), *DEFAULT_TRUST_SCHEDULE)
ROBUST_HARD_TRUST_RADIUS = 0.5


# --------------------------------------------------------------------------- conversions
def cartesian_to_mee(mu: float, state: FloatArray) -> FloatArray:
    """``[p, f, g, h, k, L]`` from ``[r, v]``; ``L`` is returned in ``[0, 2 pi)``."""

    a, e, inc, raan, argp, nu = cartesian_to_elements(mu, np.asarray(state, dtype=np.float64))
    p = a * (1.0 - e * e)
    tan_half = math.tan(0.5 * inc)
    L = (raan + argp + nu) % TWO_PI
    return np.array(
        [
            p,
            e * math.cos(argp + raan),
            e * math.sin(argp + raan),
            tan_half * math.cos(raan),
            tan_half * math.sin(raan),
            L,
        ]
    )


def mee_to_cartesian(mu: float, mee: FloatArray) -> FloatArray:
    p, f, g, h, k, L = (float(v) for v in np.asarray(mee, dtype=np.float64)[:6])
    cos_l, sin_l = math.cos(L), math.sin(L)
    alpha2 = h * h - k * k
    s2 = 1.0 + h * h + k * k
    w = 1.0 + f * cos_l + g * sin_l
    r = p / w
    sqrt_mu_p = math.sqrt(mu / p)
    position = (r / s2) * np.array(
        [
            cos_l + alpha2 * cos_l + 2.0 * h * k * sin_l,
            sin_l - alpha2 * sin_l + 2.0 * h * k * cos_l,
            2.0 * (h * sin_l - k * cos_l),
        ]
    )
    velocity = (sqrt_mu_p / s2) * np.array(
        [
            -(sin_l + alpha2 * sin_l - 2.0 * h * k * cos_l + g - 2.0 * f * h * k + alpha2 * g),
            -(-cos_l + alpha2 * cos_l + 2.0 * h * k * sin_l - f + 2.0 * g * h * k + alpha2 * f),
            2.0 * (h * cos_l + k * sin_l + f * h + g * k),
        ]
    )
    return np.concatenate([position, velocity])


def rtn_frame(mu: float, mee: FloatArray) -> FloatArray:
    """Rows ``[r_hat, t_hat, n_hat]`` of the radial-transverse-normal frame at an MEE state."""

    state = mee_to_cartesian(mu, mee)
    r = state[:3]
    v = state[3:]
    r_hat = r / np.linalg.norm(r)
    n = np.cross(r, v)
    n_hat = n / np.linalg.norm(n)
    t_hat = np.cross(n_hat, r_hat)
    return np.vstack([r_hat, t_hat, n_hat])


# --------------------------------------------------------------------------- dynamics
class MEEDynamics:
    """``x = [p, f, g, h, k, L, m]``, ``u = [T_r, T_t, T_n, Gamma]`` (thrust in the RTN frame)."""

    def __init__(self, mu: float, exhaust_velocity: float) -> None:
        self.mu = float(mu)
        self.c = float(exhaust_velocity)

    def f(self, x: FloatArray, u: FloatArray) -> FloatArray:
        p, f_, g, h, k, L, m = x
        w = 1.0 + f_ * np.cos(L) + g * np.sin(L)
        s2 = 1.0 + h * h + k * k
        q = np.sqrt(p / self.mu)
        acc_r, acc_t, acc_n = u[0] / m, u[1] / m, u[2] / m
        sin_l, cos_l = np.sin(L), np.cos(L)
        hk = h * sin_l - k * cos_l
        out = np.empty(MEE_STATE_DIMENSION, dtype=np.result_type(x, u))
        out[0] = 2.0 * p * q * acc_t / w
        out[1] = q * (acc_r * sin_l + ((w + 1.0) * cos_l + f_) * acc_t / w - g * hk * acc_n / w)
        out[2] = q * (-acc_r * cos_l + ((w + 1.0) * sin_l + g) * acc_t / w + f_ * hk * acc_n / w)
        out[3] = q * s2 * cos_l * acc_n / (2.0 * w)
        out[4] = q * s2 * sin_l * acc_n / (2.0 * w)
        out[5] = np.sqrt(self.mu * p) * (w / p) ** 2 + q * hk * acc_n / w
        out[6] = -u[3] / self.c
        return out

    def jacobians(self, x: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Complex-step Jacobians (exact to roundoff for this analytic vector field)."""

        step = 1.0e-20
        xc = np.asarray(x, dtype=np.complex128)
        uc = np.asarray(u, dtype=np.complex128)
        A = np.empty((MEE_STATE_DIMENSION, MEE_STATE_DIMENSION))
        B = np.empty((MEE_STATE_DIMENSION, CONTROL_DIMENSION))
        for column in range(MEE_STATE_DIMENSION):
            perturbed = xc.copy()
            perturbed[column] += 1j * step
            A[:, column] = np.imag(self.f(perturbed, uc)) / step
        for column in range(CONTROL_DIMENSION):
            perturbed = uc.copy()
            perturbed[column] += 1j * step
            B[:, column] = np.imag(self.f(xc, perturbed)) / step
        return A, B


# --------------------------------------------------------------------------- initial guess
def natural_revolutions(problem: LowThrustProblem, tof: float, *, steps: int = 4000) -> float:
    """Revolutions a coasting spacecraft on the interpolated slow elements makes in ``tof``."""

    mu = problem.mu
    x0 = cartesian_to_mee(mu, np.asarray(problem.initial_state))
    xf = cartesian_to_mee(mu, np.asarray(problem.final_state))
    L = x0[5]
    h = tof / steps
    for i in range(steps):
        tau = (i + 0.5) / steps
        p = x0[0] + tau * (xf[0] - x0[0])
        f_ = x0[1] + tau * (xf[1] - x0[1])
        g = x0[2] + tau * (xf[2] - x0[2])
        w = 1.0 + f_ * math.cos(L) + g * math.sin(L)
        L += h * math.sqrt(mu * p) * (w / p) ** 2
    return (L - x0[5]) / TWO_PI


def default_revolution_candidates(problem: LowThrustProblem) -> tuple[int, ...]:
    """Revolution counts worth trying: the coasting estimate (over the time-of-flight range for
    free-time problems) plus and minus one, never negative."""

    counts: set[int] = set()
    bounds = (problem.time_of_flight,) if problem.fixed_time else problem.tof_bounds
    for tof in bounds:
        centre = max(0, round(natural_revolutions(problem, tof)))
        counts.update({max(0, centre - 1), centre, centre + 1})
    return tuple(sorted(counts))


def mee_initial_guess(
    problem: LowThrustProblem,
    nodes: int,
    revolutions: int,
    tof: float,
) -> tuple[FloatArray, FloatArray, float]:
    """Interpolate the slow elements, integrate the Keplerian ``L`` rate, rescale to ``N`` revs.

    Returns the node states, zero controls, and the target final longitude ``L_target + 2 pi N``.
    """

    mu = problem.mu
    x0 = cartesian_to_mee(mu, np.asarray(problem.initial_state))
    xf = cartesian_to_mee(mu, np.asarray(problem.final_state))
    delta = (xf[5] - x0[5]) % TWO_PI
    total = delta + TWO_PI * revolutions
    final_longitude = x0[5] + total

    # Integrate the coasting rate on the interpolated elements, then rescale the accumulated
    # angle so the guess ends exactly at the requested longitude.
    fine = 40 * (nodes - 1)
    h = tof / fine
    longitude = np.empty(fine + 1)
    longitude[0] = x0[5]
    for i in range(fine):
        tau = (i + 0.5) / fine
        p = x0[0] + tau * (xf[0] - x0[0])
        f_ = x0[1] + tau * (xf[1] - x0[1])
        g = x0[2] + tau * (xf[2] - x0[2])
        w = 1.0 + f_ * math.cos(longitude[i]) + g * math.sin(longitude[i])
        longitude[i + 1] = longitude[i] + h * math.sqrt(mu * p) * (w / p) ** 2
    accumulated = longitude[-1] - longitude[0]
    scale = total / accumulated if accumulated > 0.0 else 1.0
    longitude = longitude[0] + scale * (longitude - longitude[0])

    taus = np.linspace(0.0, 1.0, nodes)
    states = np.zeros((nodes, MEE_STATE_DIMENSION))
    fuel_fraction = min(0.6, 0.5 * problem.max_thrust * tof / problem.exhaust_velocity)
    for i, tau in enumerate(taus):
        states[i, 0:5] = x0[0:5] + tau * (xf[0:5] - x0[0:5])
        states[i, 5] = longitude[i * 40]
        states[i, 6] = problem.initial_mass * (1.0 - fuel_fraction * tau)
    states[0, 0:6] = x0
    states[-1, 0:5] = xf[0:5]
    states[-1, 5] = final_longitude
    controls = np.zeros((nodes, CONTROL_DIMENSION))
    return states, controls, float(final_longitude)


# --------------------------------------------------------------------------- transcription
def _constraints(problem: LowThrustProblem, x0: FloatArray, xf: FloatArray, final_longitude: float):
    def add(v: SCvxVariables, ref_x: FloatArray, ref_u: FloatArray, ref_sigma: float) -> None:
        b = v.builder
        K = len(v.states)
        for i in range(6):
            b.add_equality(v.x(0, i), float(x0[i]))
        b.add_equality(v.x(0, 6), problem.initial_mass)
        for i in range(5):
            b.add_equality(v.x(K - 1, i), float(xf[i]))
        b.add_equality(v.x(K - 1, 5), final_longitude)
        for k in range(K):
            b.add_soc(v.u(k, 3), [v.u(k, 0), v.u(k, 1), v.u(k, 2)])
            b.add_geq(v.u(k, 3), 0.0)
            b.add_leq(v.u(k, 3), problem.max_thrust)
            b.add_geq(v.x(k, 6), 1.0e-3 * problem.initial_mass)
            b.add_geq(v.x(k, 0), 0.05 * min(x0[0], xf[0]))  # keep p away from the singularity
        del ref_x, ref_u, ref_sigma

    return add


def _objective(v: SCvxVariables, ref_x: FloatArray, ref_u: FloatArray, ref_sigma: float) -> None:
    del ref_x, ref_u, ref_sigma
    v.builder.add_linear_cost(v.states[len(v.states) - 1].start + 6, -1.0)


def _path_check(problem: LowThrustProblem):
    def check(states: FloatArray, controls: FloatArray, sigma: float) -> dict[str, float]:
        del sigma
        thrust = np.linalg.norm(controls[:, 0:3], axis=1)
        return {
            "thrust_epigraph": float(np.max(np.maximum(thrust - controls[:, 3], 0.0))),
            "thrust_max": float(np.max(np.maximum(controls[:, 3] - problem.max_thrust, 0.0))),
            "mass_positive": float(np.max(np.maximum(-states[:, 6], 0.0))),
            "semi_latus_positive": float(np.max(np.maximum(-states[:, 0], 0.0))),
        }

    return check


def cartesian_replay(
    problem: LowThrustProblem,
    mee_states: FloatArray,
    rtn_controls: FloatArray,
    sigma: float,
    *,
    substeps: int = 32,
) -> tuple[FloatArray, float, float]:
    """Independent check: integrate the Cartesian dynamics with the FOH thrust rotated per stage.

    Returns the Cartesian node states, the terminal position error and the terminal velocity
    error against the problem's final Cartesian state (non-dimensional units).
    """

    mu = problem.mu
    c = problem.exhaust_velocity
    K = mee_states.shape[0]
    d_tau = 1.0 / (K - 1)
    state = np.concatenate([mee_to_cartesian(mu, mee_states[0]), [mee_states[0, 6]]])
    nodes = [state.copy()]

    def inertial_thrust(x: FloatArray, u_k: FloatArray, u_next: FloatArray, frac: float):
        u = (1.0 - frac) * u_k + frac * u_next
        r = x[:3]
        v = x[3:6]
        r_hat = r / np.linalg.norm(r)
        n = np.cross(r, v)
        n_hat = n / np.linalg.norm(n)
        t_hat = np.cross(n_hat, r_hat)
        return u[0] * r_hat + u[1] * t_hat + u[2] * n_hat, u[3]

    def f(x: FloatArray, u_k: FloatArray, u_next: FloatArray, frac: float) -> FloatArray:
        thrust, gamma = inertial_thrust(x, u_k, u_next, frac)
        r = x[:3]
        rn = np.linalg.norm(r)
        return sigma * np.concatenate([x[3:6], -mu * r / rn**3 + thrust / x[6], [-gamma / c]])

    h = d_tau / substeps
    for k in range(K - 1):
        u_k, u_next = rtn_controls[k], rtn_controls[k + 1]
        tau = 0.0
        for _ in range(substeps):
            k1 = f(state, u_k, u_next, tau / d_tau)
            k2 = f(state + 0.5 * h * k1, u_k, u_next, (tau + 0.5 * h) / d_tau)
            k3 = f(state + 0.5 * h * k2, u_k, u_next, (tau + 0.5 * h) / d_tau)
            k4 = f(state + h * k3, u_k, u_next, (tau + h) / d_tau)
            state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tau += h
        nodes.append(state.copy())
    array = np.vstack(nodes)
    final = np.asarray(problem.final_state)
    return (
        array,
        float(np.linalg.norm(array[-1, :3] - final[:3])),
        float(np.linalg.norm(array[-1, 3:6] - final[3:6])),
    )


@dataclass(slots=True)
class MEEReproduction:
    outcome: SCvxOutcome
    nodes: int
    revolutions: int
    time_of_flight: float
    final_mass: float
    final_mass_si: float | None
    replay_terminal_position_error: float
    replay_terminal_velocity_error: float
    replay_final_mass: float
    max_path_violation: float
    label: str
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.outcome.status == "converged"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "formulation": "modified equinoctial elements, RTN thrust, FOH SCvx",
            "status": self.outcome.status,
            "nodes": self.nodes,
            "revolutions": self.revolutions,
            "time_of_flight": self.time_of_flight,
            "final_mass": self.final_mass,
            "final_mass_si": self.final_mass_si,
            "replay_final_mass": self.replay_final_mass,
            "replay_defect_inf": self.outcome.replay_defect_inf,
            "replay_terminal_error_inf": self.outcome.replay_terminal_error_inf,
            "cartesian_replay_terminal_position_error": self.replay_terminal_position_error,
            "cartesian_replay_terminal_velocity_error": self.replay_terminal_velocity_error,
            "path_violations": self.outcome.path_violations,
            "max_path_violation": self.max_path_violation,
            "iterations": [asdict(item) for item in self.outcome.iterations],
            "trust_schedule": self.outcome.extras.get("trust_schedule", []),
            "termination_reason": self.outcome.extras.get(
                "termination_reason", self.outcome.status
            ),
            "stages": self.stages,
        }


def solve_low_thrust_mee(
    problem: LowThrustProblem,
    *,
    revolutions: int,
    nodes: int = 300,
    max_iterations: int = 40,
    virtual_weight: float = 1.0e3,
    trust_weight: float = 1.0e-2,
    trust_schedule: tuple[tuple[float, int], ...] | None = DEFAULT_TRUST_SCHEDULE,
    sigma_trust_weight: float = 1.0e-2,
    substeps: int = 8,
    trust_tolerance: float = 1.0e-4,
    virtual_tolerance: float = 1.0e-7,
    initial_guess: tuple[FloatArray, FloatArray] | None = None,
    tof_guess: float | None = None,
    hard_trust_radius: float | None = None,
    max_thrust_override: float | None = None,
    stage_defect_tolerance: float = 1.0e-4,
    stall_tolerance: float = 1.0e-5,
) -> MEEReproduction:
    """Solve one fixed-``N`` MEE transcription.

    ``trust_schedule`` is a tolerance/trust continuation: a sequence of
    ``(trust_weight, iteration_budget)`` stages, each warm-started from the previous.  A stiff
    first stage keeps the geometric guess from diverging; later, lighter stages let the
    linearisation (accurate near the solution) take the long steps a bang-bang mass-optimal
    profile needs.  When ``None`` a single stage ``(trust_weight, max_iterations)`` is run.
    ``max_thrust_override`` supports a thrust homotopy (solve with more thrust, then tighten).
    """

    mu = problem.mu
    x0 = cartesian_to_mee(mu, np.asarray(problem.initial_state))
    xf = cartesian_to_mee(mu, np.asarray(problem.final_state))
    tof = tof_guess if tof_guess is not None else problem.time_of_flight
    max_thrust = problem.max_thrust if max_thrust_override is None else max_thrust_override
    working = LowThrustProblem(
        initial_state=problem.initial_state,
        final_state=problem.final_state,
        initial_mass=problem.initial_mass,
        max_thrust=max_thrust,
        exhaust_velocity=problem.exhaust_velocity,
        tof_bounds=problem.tof_bounds,
        mu=problem.mu,
        minimum_radius=problem.minimum_radius,
        label=problem.label,
        scaling=problem.scaling,
    )
    guess_states, guess_controls, final_longitude = mee_initial_guess(
        working, nodes, revolutions, tof
    )
    if initial_guess is not None:
        guess_states, guess_controls = initial_guess
    dynamics = MEEDynamics(mu, problem.exhaust_velocity)
    schedule = trust_schedule if trust_schedule is not None else ((trust_weight, max_iterations),)
    states, controls, sigma = guess_states, guess_controls, tof
    history = []
    stage_records: list[dict[str, Any]] = []
    outcome: SCvxOutcome | None = None
    for stage_index, (stage_weight, stage_iterations) in enumerate(schedule):
        settings = SCvxSettings(
            nodes=nodes,
            max_iterations=stage_iterations,
            virtual_weight=virtual_weight,
            trust_weight=stage_weight,
            sigma_trust_weight=sigma_trust_weight,
            virtual_tolerance=virtual_tolerance,
            trust_tolerance=trust_tolerance,
            substeps=substeps,
            replay_substeps=32,
            state_scale=np.array([1.0, 0.5, 0.5, 0.5, 0.5, TWO_PI, problem.initial_mass]),
            control_scale=np.full(CONTROL_DIMENSION, max(max_thrust, 1.0e-12)),
            # The hard per-node radius only guards the first stage (the geometric guess); later
            # stages start near a consistent trajectory and would only be slowed by it.
            hard_trust_radius=hard_trust_radius if stage_index == 0 else None,
        )
        solver = FreeFinalTimeSCvx(
            n_x=MEE_STATE_DIMENSION,
            n_u=CONTROL_DIMENSION,
            dynamics=dynamics.f,
            jacobians=dynamics.jacobians,
            constraints=_constraints(working, x0, xf, final_longitude),
            objective=_objective,
            settings=settings,
            free_final_time=not problem.fixed_time,
            sigma_bounds=None if problem.fixed_time else problem.tof_bounds,
            path_check=_path_check(working),
        )
        try:
            with np.errstate(all="ignore"):
                stage = solver.solve(states, controls, sigma)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            # The linearised flow left the domain (p <= 0 or a NaN state); record and stop.
            stage_records.append(
                {
                    "trust_weight": stage_weight,
                    "iterations": 0,
                    "status": "numerical_failure",
                    "error": str(error),
                    "accepted": False,
                }
            )
            if outcome is None:
                outcome = SCvxOutcome(
                    status="numerical_failure",
                    states=states,
                    controls=controls,
                    sigma=sigma,
                    iterations=[],
                    replay_defect_inf=float("inf"),
                    replay_terminal_error_inf=float("inf"),
                    replay_states=states,
                    path_violations={},
                )
            break
        for item in stage.iterations:
            item.iteration = len(history)
            history.append(item)
        # The first stage is the baseline (accepted whenever it produced a finite iterate); a
        # lighter stage is accepted only if it stays dynamically consistent and does not lose
        # objective, otherwise the continuation stops at the previous (stiffer) stage.
        finite = stage.status != "solver_failed" and bool(np.all(np.isfinite(stage.states)))
        if outcome is None:
            accepted = finite
        else:
            accepted = (
                finite
                and stage.replay_defect_inf <= stage_defect_tolerance
                and stage.states[-1, 6] >= outcome.states[-1, 6] - 1.0e-6
            )
        stage_records.append(
            {
                "trust_weight": stage_weight,
                "iterations": len(stage.iterations),
                "status": stage.status,
                "final_mass": float(stage.states[-1, 6]),
                "replay_defect_inf": stage.replay_defect_inf,
                "accepted": bool(accepted),
            }
        )
        if not accepted:
            if outcome is None:
                outcome = stage
            break
        states, controls, sigma = stage.states, stage.controls, stage.sigma
        outcome = stage
    assert outcome is not None
    outcome.iterations = history
    outcome.extras["trust_schedule"] = stage_records
    # Termination bookkeeping across the continuation: the result is "converged" when the
    # accepted trajectory is dynamically consistent and either a stage met the step tolerance
    # or the last two accepted stages agree on the final mass (objective stall).
    accepted_records = [record for record in stage_records if record.get("accepted")]
    if accepted_records and outcome.replay_defect_inf <= stage_defect_tolerance:
        stage_converged = any(record["status"] == "converged" for record in accepted_records)
        stalled = len(accepted_records) >= 2 and (
            abs(accepted_records[-1]["final_mass"] - accepted_records[-2]["final_mass"])
            <= stall_tolerance
        )
        if stage_converged or stalled:
            outcome.status = "converged"
            outcome.extras["termination_reason"] = (
                "stage_converged" if stage_converged else "objective_stall"
            )
    final_mass = float(outcome.states[-1, 6])
    if np.all(np.isfinite(outcome.states)) and np.all(np.isfinite(outcome.controls)):
        with np.errstate(all="ignore"):
            replay, position_error, velocity_error = cartesian_replay(
                problem, outcome.states, outcome.controls, outcome.sigma
            )
    else:
        replay = outcome.states
        position_error = velocity_error = float("inf")
    violations = outcome.path_violations
    return MEEReproduction(
        outcome=outcome,
        nodes=nodes,
        revolutions=revolutions,
        time_of_flight=outcome.sigma,
        final_mass=final_mass,
        final_mass_si=problem.mass_to_si(final_mass) if problem.scaling else None,
        replay_terminal_position_error=position_error,
        replay_terminal_velocity_error=velocity_error,
        replay_final_mass=float(replay[-1, 6]),
        max_path_violation=float(max(violations.values())) if violations else float("nan"),
        label=problem.label,
    )


def _dynamically_consistent(result: MEEReproduction) -> bool:
    return (
        result.outcome.status not in {"solver_failed", "numerical_failure"}
        and math.isfinite(result.outcome.replay_defect_inf)
        and result.outcome.replay_defect_inf <= 1.0e-3
    )


def solve_multirev(
    problem: LowThrustProblem,
    *,
    revolution_candidates: tuple[int, ...] | None = None,
    nodes: int = 300,
    max_iterations: int = 40,
    thrust_factors: tuple[float, ...] = (1.0,),
    robust_fallback: bool = True,
    **kwargs: Any,
) -> tuple[MEEReproduction | None, list[dict[str, Any]]]:
    """Sweep the revolution count (and optionally a thrust homotopy) and keep the best result.

    ``revolution_candidates`` defaults to :func:`default_revolution_candidates`.  The best result
    is the converged run with the largest final mass whose Cartesian replay closes the terminal
    state; ``None`` if nothing converged.  The sweep summary is returned for the report.  With
    ``robust_fallback`` a candidate whose default continuation diverges is retried with
    :data:`ROBUST_TRUST_SCHEDULE`.
    """

    if revolution_candidates is None:
        revolution_candidates = default_revolution_candidates(problem)
    history: list[dict[str, Any]] = []
    best: MEEReproduction | None = None
    for revolutions in revolution_candidates:
        guess = None
        result = None
        stages: list[dict[str, Any]] = []
        for factor in thrust_factors:
            result = solve_low_thrust_mee(
                problem,
                revolutions=revolutions,
                nodes=nodes,
                max_iterations=max_iterations,
                initial_guess=guess,
                max_thrust_override=problem.max_thrust * factor,
                **kwargs,
            )
            if (
                robust_fallback
                and guess is None
                and "trust_schedule" not in kwargs
                and not _dynamically_consistent(result)
            ):
                # The default continuation lost the guess; retry with the stiff first stage.
                stages.append({"thrust_factor": factor, "default_schedule": result.as_dict()})
                result = solve_low_thrust_mee(
                    problem,
                    revolutions=revolutions,
                    nodes=nodes,
                    max_iterations=max_iterations,
                    max_thrust_override=problem.max_thrust * factor,
                    trust_schedule=ROBUST_TRUST_SCHEDULE,
                    hard_trust_radius=ROBUST_HARD_TRUST_RADIUS,
                    **kwargs,
                )
            stages.append(
                {
                    "thrust_factor": factor,
                    "status": result.outcome.status,
                    "iterations": len(result.outcome.iterations),
                    "final_mass": result.final_mass,
                    "replay_defect_inf": result.outcome.replay_defect_inf,
                    "trust_schedule": result.outcome.extras.get("trust_schedule", []),
                }
            )
            controls = result.outcome.controls.copy()
            magnitude = np.linalg.norm(controls[:, 0:3], axis=1)
            scale = np.minimum(1.0, problem.max_thrust / np.maximum(magnitude, 1.0e-300))
            controls[:, 0:3] *= scale[:, None]
            controls[:, 3] = np.minimum(controls[:, 3], problem.max_thrust)
            guess = (result.outcome.states.copy(), controls)
        assert result is not None
        result.stages = stages
        closes = (
            result.replay_terminal_position_error < 1.0e-4
            and result.replay_terminal_velocity_error < 1.0e-4
        )
        history.append(
            {
                "revolutions": revolutions,
                "status": result.outcome.status,
                "final_mass": result.final_mass,
                "final_mass_si": result.final_mass_si,
                "time_of_flight": result.time_of_flight,
                "iterations": len(result.outcome.iterations),
                "replay_defect_inf": result.outcome.replay_defect_inf,
                "cartesian_replay_terminal_position_error": result.replay_terminal_position_error,
                "cartesian_replay_terminal_velocity_error": result.replay_terminal_velocity_error,
                "cartesian_replay_closes": closes,
                "stages": stages,
            }
        )
        if result.converged and closes and (best is None or result.final_mass > best.final_mass):
            best = result
    return best, history
