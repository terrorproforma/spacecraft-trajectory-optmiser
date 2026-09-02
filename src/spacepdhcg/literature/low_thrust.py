"""Low-thrust two-body point-to-point transfers (P1-E and TOPS Cartesian two-body).

The problem is the classic fixed- or free-time minimum-fuel rendezvous:

    r' = v,  v' = -mu r / |r|^3 + T / m,  m' = -Gamma / c,   ||T|| <= Gamma <= T_max,

with both boundary states fixed (zero hyperbolic excess velocity) and the final mass free.
All solves run in non-dimensional units (length unit ``L``, time unit ``sqrt(L^3/mu)``, mass
unit ``m_0``); the SI convenience constructor performs the scaling.  The transcription is the
FOH successive convexification of :mod:`spacepdhcg.literature.scvx_core`, so free-final-time
instances (TOPS ``tof_bounds`` with distinct ends) use the time-dilation variable directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.literature._socp import LinearExpression
from spacepdhcg.literature.scvx_core import (
    FreeFinalTimeSCvx,
    SCvxOutcome,
    SCvxSettings,
    SCvxVariables,
)

FloatArray = NDArray[np.float64]

STANDARD_GRAVITY_SI = 9.80665  # m/s^2
ASTRONOMICAL_UNIT_KM = 149_597_870.7
SUN_MU_KM3_S2 = 132_712_440_018.0  # Tafazzol & Taheri Table 1/3
DAY_SECONDS = 86_400.0

STATE_DIMENSION = 7
CONTROL_DIMENSION = 4


@dataclass(frozen=True, slots=True)
class LowThrustProblem:
    """Non-dimensional point-to-point transfer (``mu = 1``)."""

    initial_state: tuple[float, ...]
    final_state: tuple[float, ...]
    initial_mass: float
    max_thrust: float
    exhaust_velocity: float
    tof_bounds: tuple[float, float]
    mu: float = 1.0
    minimum_radius: float = 0.0
    label: str = ""
    scaling: dict[str, float] = field(default_factory=dict)

    @property
    def fixed_time(self) -> bool:
        return math.isclose(self.tof_bounds[0], self.tof_bounds[1])

    @property
    def time_of_flight(self) -> float:
        return self.tof_bounds[1]

    @classmethod
    def from_si(
        cls,
        *,
        r0_km: tuple[float, float, float],
        v0_km_s: tuple[float, float, float],
        rf_km: tuple[float, float, float],
        vf_km_s: tuple[float, float, float],
        initial_mass_kg: float,
        max_thrust_n: float,
        specific_impulse_s: float,
        time_of_flight_days: float,
        mu_km3_s2: float = SUN_MU_KM3_S2,
        length_unit_km: float = ASTRONOMICAL_UNIT_KM,
        label: str = "",
    ) -> LowThrustProblem:
        L = length_unit_km
        TU = math.sqrt(L**3 / mu_km3_s2)  # seconds
        V = L / TU  # km/s
        ACC = V / TU  # km/s^2
        force_unit_n = initial_mass_kg * ACC * 1000.0  # kg * m/s^2
        r0 = np.asarray(r0_km) / L
        v0 = np.asarray(v0_km_s) / V
        rf = np.asarray(rf_km) / L
        vf = np.asarray(vf_km_s) / V
        tof = time_of_flight_days * DAY_SECONDS / TU
        c = specific_impulse_s * STANDARD_GRAVITY_SI / 1000.0 / V
        return cls(
            initial_state=tuple(float(x) for x in np.concatenate([r0, v0])),
            final_state=tuple(float(x) for x in np.concatenate([rf, vf])),
            initial_mass=1.0,
            max_thrust=max_thrust_n / force_unit_n,
            exhaust_velocity=c,
            tof_bounds=(tof, tof),
            mu=1.0,
            label=label,
            scaling={
                "length_unit_km": L,
                "time_unit_s": TU,
                "velocity_unit_km_s": V,
                "mass_unit_kg": initial_mass_kg,
                "force_unit_n": force_unit_n,
                "time_of_flight_days": time_of_flight_days,
            },
        )

    def mass_to_si(self, nondimensional_mass: float) -> float:
        return nondimensional_mass * self.scaling.get("mass_unit_kg", 1.0)


class LowThrustDynamics:
    def __init__(self, problem: LowThrustProblem) -> None:
        self.mu = problem.mu
        self.c = problem.exhaust_velocity

    def f(self, x: FloatArray, u: FloatArray) -> FloatArray:
        r = x[0:3]
        v = x[3:6]
        m = x[6]
        rn = float(np.linalg.norm(r))
        out = np.empty(STATE_DIMENSION)
        out[0:3] = v
        out[3:6] = -self.mu * r / rn**3 + u[0:3] / m
        out[6] = -u[3] / self.c
        return out

    def jacobians(self, x: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        r = x[0:3]
        m = x[6]
        rn = float(np.linalg.norm(r))
        A = np.zeros((STATE_DIMENSION, STATE_DIMENSION))
        B = np.zeros((STATE_DIMENSION, CONTROL_DIMENSION))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = self.mu * (3.0 * np.outer(r, r) / rn**5 - np.eye(3) / rn**3)
        A[3:6, 6] = -u[0:3] / (m * m)
        B[3:6, 0:3] = np.eye(3) / m
        B[6, 3] = -1.0 / self.c
        return A, B


def kepler_propagate(mu: float, state: FloatArray, dt: float, *, steps: int = 400) -> FloatArray:
    """RK4 ballistic propagation used only to build initial guesses."""

    x = np.array(state, dtype=np.float64)
    h = dt / steps

    def f(s: FloatArray) -> FloatArray:
        r = s[0:3]
        return np.concatenate([s[3:6], -mu * r / np.linalg.norm(r) ** 3])

    for _ in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def blended_initial_guess(
    problem: LowThrustProblem, nodes: int, tof: float
) -> tuple[FloatArray, FloatArray]:
    """Blend forward/backward ballistic arcs; zero thrust; linear mass guess."""

    x0 = np.asarray(problem.initial_state)
    xf = np.asarray(problem.final_state)
    taus = np.linspace(0.0, 1.0, nodes)
    states = np.zeros((nodes, STATE_DIMENSION))
    forward = np.array(x0)
    backward_states = np.zeros((nodes, 6))
    backward = np.array(xf)
    dt = tof / (nodes - 1)
    forward_states = np.zeros((nodes, 6))
    forward_states[0] = x0
    backward_states[-1] = xf
    for k in range(1, nodes):
        forward = kepler_propagate(problem.mu, forward, dt, steps=20)
        forward_states[k] = forward
        backward = kepler_propagate(problem.mu, backward, -dt, steps=20)
        backward_states[nodes - 1 - k] = backward
    for k, tau in enumerate(taus):
        states[k, 0:6] = (1.0 - tau) * forward_states[k] + tau * backward_states[k]
        states[k, 6] = problem.initial_mass * (1.0 - 0.1 * tau)
    controls = np.zeros((nodes, CONTROL_DIMENSION))
    return states, controls


def cartesian_to_elements(
    mu: float, state: FloatArray
) -> tuple[float, float, float, float, float, float]:
    """Classical elements ``(a, e, i, raan, argp, true_anomaly)`` from a Cartesian state."""

    r = np.asarray(state[0:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    rn = float(np.linalg.norm(r))
    vn = float(np.linalg.norm(v))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    n_vec = np.cross([0.0, 0.0, 1.0], h)
    nn = float(np.linalg.norm(n_vec))
    e_vec = ((vn**2 - mu / rn) * r - float(r @ v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    energy = vn**2 / 2.0 - mu / rn
    a = -mu / (2.0 * energy)
    inc = math.acos(max(-1.0, min(1.0, h[2] / hn)))
    raan = math.atan2(n_vec[1], n_vec[0]) if nn > 1.0e-12 else 0.0
    if nn > 1.0e-12 and e > 1.0e-12:
        argp = math.atan2(float(h @ np.cross(n_vec, e_vec)) / hn / nn, float(n_vec @ e_vec) / nn)
    else:
        argp = math.atan2(e_vec[1], e_vec[0]) if e > 1.0e-12 else 0.0
    if e > 1.0e-12:
        nu = math.atan2(float(h @ np.cross(e_vec, r)) / hn / e, float(e_vec @ r) / e)
    else:
        nu = math.atan2(
            float(h @ np.cross(n_vec, r)) / hn / max(nn, 1e-12), float(n_vec @ r) / max(nn, 1e-12)
        )
    return a, e, inc, raan, argp, nu


def elements_to_cartesian(
    mu: float, a: float, e: float, inc: float, raan: float, argp: float, nu: float
) -> FloatArray:
    p = a * (1.0 - e * e)
    rn = p / (1.0 + e * math.cos(nu))
    r_pf = np.array([rn * math.cos(nu), rn * math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    cO, sO = math.cos(raan), math.sin(raan)
    co, so = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    rot = np.array(
        [
            [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si, co * si, ci],
        ]
    )
    return np.concatenate([rot @ r_pf, rot @ v_pf])


def element_interpolation_guess(
    problem: LowThrustProblem, nodes: int, revolutions: int
) -> tuple[FloatArray, FloatArray]:
    """Interpolate classical elements between the boundary orbits with a prescribed rev count.

    The true longitude advances by ``2*pi*revolutions`` plus the boundary offset, so the guess
    encodes the topology (number of heliocentric revolutions) reported by the source paper.
    """

    mu = problem.mu
    a0, e0, i0, O0, w0, nu0 = cartesian_to_elements(mu, np.asarray(problem.initial_state))
    af, ef, if_, Of, wf, nuf = cartesian_to_elements(mu, np.asarray(problem.final_state))
    L0 = O0 + w0 + nu0
    Lf = Of + wf + nuf
    delta = (Lf - L0) % (2.0 * math.pi)
    total = delta + 2.0 * math.pi * revolutions
    taus = np.linspace(0.0, 1.0, nodes)
    states = np.zeros((nodes, STATE_DIMENSION))
    for k, tau in enumerate(taus):
        a = a0 + tau * (af - a0)
        e = e0 + tau * (ef - e0)
        inc = i0 + tau * (if_ - i0)
        raan = O0 + tau * (((Of - O0 + math.pi) % (2 * math.pi)) - math.pi)
        argp = w0 + tau * (((wf - w0 + math.pi) % (2 * math.pi)) - math.pi)
        L = L0 + tau * total
        nu = L - raan - argp
        states[k, 0:6] = elements_to_cartesian(mu, a, e, inc, raan, argp, nu)
        states[k, 6] = problem.initial_mass * (1.0 - 0.3 * tau)
    states[0, 0:6] = problem.initial_state
    states[-1, 0:6] = problem.final_state
    controls = np.zeros((nodes, CONTROL_DIMENSION))
    return states, controls


def _constraints(problem: LowThrustProblem):
    def add(v: SCvxVariables, ref_x: FloatArray, ref_u: FloatArray, ref_sigma: float) -> None:
        b = v.builder
        K = len(v.states)
        for i in range(6):
            b.add_equality(v.x(0, i), problem.initial_state[i])
            b.add_equality(v.x(K - 1, i), problem.final_state[i])
        b.add_equality(v.x(0, 6), problem.initial_mass)
        for k in range(K):
            b.add_soc(v.u(k, 3), [v.u(k, 0), v.u(k, 1), v.u(k, 2)])
            b.add_geq(v.u(k, 3), 0.0)
            b.add_leq(v.u(k, 3), problem.max_thrust)
            b.add_geq(v.x(k, 6), 1.0e-3 * problem.initial_mass)
            if problem.minimum_radius > 0.0:
                # linearised |r| >= r_min about the reference direction
                ref_r = ref_x[k, 0:3]
                norm = float(np.linalg.norm(ref_r))
                if norm > 0.0:
                    direction = ref_r / norm
                    b.add_geq(
                        LinearExpression(
                            [v.states[k].start + i for i in range(3)],
                            [float(direction[i]) for i in range(3)],
                            0.0,
                        ),
                        problem.minimum_radius,
                    )

    return add


def _objective(v: SCvxVariables, ref_x: FloatArray, ref_u: FloatArray, ref_sigma: float) -> None:
    K = len(v.states)
    v.builder.add_linear_cost(v.states[K - 1].start + 6, -1.0)


def _path_check(problem: LowThrustProblem):
    def check(states: FloatArray, controls: FloatArray, sigma: float) -> dict[str, float]:
        thrust = np.linalg.norm(controls[:, 0:3], axis=1)
        return {
            "thrust_epigraph": float(np.max(np.maximum(thrust - controls[:, 3], 0.0))),
            "thrust_max": float(np.max(np.maximum(controls[:, 3] - problem.max_thrust, 0.0))),
            "mass_positive": float(np.max(np.maximum(-states[:, 6], 0.0))),
            "minimum_radius": float(
                np.max(
                    np.maximum(problem.minimum_radius - np.linalg.norm(states[:, 0:3], axis=1), 0.0)
                )
            ),
        }

    return check


@dataclass(slots=True)
class LowThrustReproduction:
    outcome: SCvxOutcome
    nodes: int
    time_of_flight: float
    final_mass: float
    final_mass_si: float | None
    propellant_fraction: float
    max_path_violation: float
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.outcome.status,
            "nodes": self.nodes,
            "time_of_flight": self.time_of_flight,
            "final_mass": self.final_mass,
            "final_mass_si": self.final_mass_si,
            "propellant_fraction": self.propellant_fraction,
            "replay_defect_inf": self.outcome.replay_defect_inf,
            "replay_terminal_error_inf": self.outcome.replay_terminal_error_inf,
            "path_violations": self.outcome.path_violations,
            "max_path_violation": self.max_path_violation,
            "iterations": [asdict(item) for item in self.outcome.iterations],
        }


def solve_low_thrust(
    problem: LowThrustProblem,
    *,
    nodes: int = 200,
    max_iterations: int = 40,
    virtual_weight: float = 1.0e4,
    trust_weight: float = 1.0e-3,
    sigma_trust_weight: float = 1.0e-2,
    substeps: int = 6,
    trust_tolerance: float = 1.0e-4,
    virtual_tolerance: float = 1.0e-8,
    initial_guess: tuple[FloatArray, FloatArray] | None = None,
    tof_guess: float | None = None,
    hard_trust_radius: float | None = None,
    revolutions: int | None = None,
) -> LowThrustReproduction:
    dynamics = LowThrustDynamics(problem)
    settings = SCvxSettings(
        nodes=nodes,
        max_iterations=max_iterations,
        virtual_weight=virtual_weight,
        trust_weight=trust_weight,
        sigma_trust_weight=sigma_trust_weight,
        virtual_tolerance=virtual_tolerance,
        trust_tolerance=trust_tolerance,
        substeps=substeps,
        replay_substeps=32,
        state_scale=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, problem.initial_mass]),
        control_scale=np.full(CONTROL_DIMENSION, max(problem.max_thrust, 1.0e-12)),
        hard_trust_radius=hard_trust_radius,
    )
    solver = FreeFinalTimeSCvx(
        n_x=STATE_DIMENSION,
        n_u=CONTROL_DIMENSION,
        dynamics=dynamics.f,
        jacobians=dynamics.jacobians,
        constraints=_constraints(problem),
        objective=_objective,
        settings=settings,
        free_final_time=not problem.fixed_time,
        sigma_bounds=None if problem.fixed_time else problem.tof_bounds,
        path_check=_path_check(problem),
    )
    tof = tof_guess if tof_guess is not None else problem.time_of_flight
    if initial_guess is None and revolutions is not None:
        states, controls = element_interpolation_guess(problem, nodes, revolutions)
    elif initial_guess is None:
        states, controls = blended_initial_guess(problem, nodes, tof)
    else:
        states, controls = initial_guess
    outcome = solver.solve(states, controls, tof)
    final_mass = float(outcome.states[-1, 6])
    violations = outcome.path_violations
    return LowThrustReproduction(
        outcome=outcome,
        nodes=nodes,
        time_of_flight=outcome.sigma,
        final_mass=final_mass,
        final_mass_si=problem.mass_to_si(final_mass) if problem.scaling else None,
        propellant_fraction=float(1.0 - final_mass / problem.initial_mass),
        max_path_violation=float(max(violations.values())) if violations else float("nan"),
        label=problem.label,
    )


def solve_low_thrust_continuation(
    problem: LowThrustProblem,
    *,
    nodes: int = 400,
    thrust_factors: tuple[float, ...] = (8.0, 4.0, 2.0, 1.4, 1.0),
    max_iterations: int = 30,
    revolutions: int | None = None,
    hard_trust_radius: float | None = 0.5,
    **kwargs: Any,
) -> tuple[LowThrustReproduction, list[dict[str, Any]]]:
    """Thrust-bound homotopy: solve with a relaxed ``T_max`` first, then tighten it.

    Multi-revolution transfers rarely converge from a geometric initial guess because the
    guess needs more than ``T_max`` to be realised.  Relaxing the bound yields a dynamically
    feasible trajectory with the intended topology; each tightening stage is warm-started
    from the previous converged trajectory.  Returns the final-stage reproduction and the
    per-stage summary.
    """

    history: list[dict[str, Any]] = []
    guess: tuple[FloatArray, FloatArray] | None = None
    result: LowThrustReproduction | None = None
    for stage, factor in enumerate(thrust_factors):
        relaxed = LowThrustProblem(
            initial_state=problem.initial_state,
            final_state=problem.final_state,
            initial_mass=problem.initial_mass,
            max_thrust=problem.max_thrust * factor,
            exhaust_velocity=problem.exhaust_velocity,
            tof_bounds=problem.tof_bounds,
            mu=problem.mu,
            minimum_radius=problem.minimum_radius,
            label=problem.label,
            scaling=problem.scaling,
        )
        result = solve_low_thrust(
            relaxed,
            nodes=nodes,
            max_iterations=max_iterations,
            revolutions=revolutions if guess is None else None,
            initial_guess=guess,
            hard_trust_radius=hard_trust_radius,
            **kwargs,
        )
        # Clip the warm start to the next thrust bound so the next stage starts feasibly.
        controls = result.outcome.controls.copy()
        if stage + 1 < len(thrust_factors):
            next_bound = problem.max_thrust * thrust_factors[stage + 1]
            magnitude = np.linalg.norm(controls[:, 0:3], axis=1)
            scale = np.minimum(1.0, next_bound / np.maximum(magnitude, 1.0e-300))
            controls[:, 0:3] *= scale[:, None]
            controls[:, 3] = np.minimum(controls[:, 3], next_bound)
        history.append(
            {
                "stage": stage,
                "thrust_factor": factor,
                "status": result.outcome.status,
                "iterations": len(result.outcome.iterations),
                "final_mass": result.final_mass,
                "replay_defect_inf": result.outcome.replay_defect_inf,
            }
        )
        guess = (result.outcome.states.copy(), controls)
    assert result is not None
    return result, history


# --------------------------------------------------------------------------- P1-E profiles
def problem_from_document(document: dict[str, Any]) -> LowThrustProblem:
    p = document["parameters"]
    return LowThrustProblem.from_si(
        r0_km=tuple(p["departure_position_km"]),
        v0_km_s=tuple(p["departure_velocity_km_s"]),
        rf_km=tuple(p["arrival_position_km"]),
        vf_km_s=tuple(p["arrival_velocity_km_s"]),
        initial_mass_kg=p["initial_mass_kg"],
        max_thrust_n=p["maximum_thrust_n"],
        specific_impulse_s=p["specific_impulse_s"],
        time_of_flight_days=p["time_of_flight_days"],
        mu_km3_s2=p.get("sun_mu_km3_s2", SUN_MU_KM3_S2),
        label=document["id"],
    )


def run_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    problem = problem_from_document(document)
    published = document["published"]
    node_values = tuple(options.get("nodes_values", document.get("envelope_nodes", [200, 400])))
    max_iterations = int(options.get("max_iterations", document.get("max_iterations", 40)))
    runs: dict[str, Any] = {}
    final_masses: list[float] = []
    revolutions = document.get("initial_guess_revolutions")
    for nodes in node_values:
        result = solve_low_thrust(
            problem,
            nodes=int(nodes),
            max_iterations=max_iterations,
            revolutions=int(revolutions) if revolutions is not None else None,
        )
        runs[f"nodes={nodes}"] = result.as_dict()
        if result.outcome.status == "converged" or result.outcome.replay_defect_inf < 1.0e-6:
            final_masses.append(float(result.final_mass_si))
    published_mass = float(published["final_mass_kg"])
    if final_masses:
        best = max(final_masses)
        gap_kg = best - published_mass
        status = (
            "reproduced"
            if abs(gap_kg) <= float(document.get("acceptance_tolerance_kg", 2.0))
            else "gap"
        )
    else:
        best = float("nan")
        gap_kg = float("nan")
        status = "gap"
    return {
        "target_id": document["id"],
        "status": status,
        "published": published,
        "measured": {
            "final_mass_kg_best": best,
            "final_mass_kg_by_nodes": {key: run["final_mass_si"] for key, run in runs.items()},
            "statuses": {key: run["status"] for key, run in runs.items()},
        },
        "gap": {
            "final_mass_minus_published_kg": gap_kg,
            "relative": gap_kg / published_mass if final_masses else float("nan"),
            "acceptance_tolerance_kg": document.get("acceptance_tolerance_kg", 2.0),
        },
        "labels": {
            "published.final_mass_kg": published.get("evidence_label", "published-reference"),
            "measured.final_mass_kg_best": "measured-local",
        },
        "envelope": {
            "discretisation": "FOH successive convexification, RK4 STM, nodes swept",
            "nodes_values": list(node_values),
            "final_mass_spread_kg": (max(final_masses) - min(final_masses))
            if final_masses
            else None,
            "declared_envelope_kg": document.get("declared_envelope_kg"),
        },
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [
            "boundary states are the paper's fixed heliocentric states (zero hyperbolic excess); "
            "no ephemeris model is involved",
        ],
        "details": {"runs": runs, "scaling": problem.scaling},
    }
