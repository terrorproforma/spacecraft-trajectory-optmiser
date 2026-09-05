"""P1-D literature profile: Szmuk & Acikmese (2018) free-final-time 6-DoF landing.

Source: M. Szmuk and B. Acikmese, "Successive Convexification for 6-DoF Mars Rocket Powered
Landing with Free-Final-Time", AIAA GNC 2018, DOI 10.2514/6.2018-0617 (arXiv:1802.03827).

Everything in this module follows the paper's Problem 1/2 and Algorithm 1 with the Table 1/2
parameters.  Two inputs are *not* recoverable from the paper and are recorded as such in the
provenance store:

* ``alpha_mdot = 1/(Isp g0)`` is defined symbolically but Table 1 prints no value.  The value
  ``0.01 UT/UL`` is an assumption shared with public re-implementations.
* The converged time-of-flight is shown only in Figure 2; no digits are printed.  The
  reproduction therefore compares the qualitative published statements (all ten initial
  guesses converge to the same t_f within 0.01 UT; convergence by the sixth iteration) and
  records our value as ``measured-local``.

The state ordering is the paper's: ``x = [m, r_I(3), v_I(3), q_B/I(4), omega_B(3)]`` with an
Up-East-North inertial frame, and ``u = T_B(3)``.
"""

from __future__ import annotations

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

STATE_DIMENSION = 14
CONTROL_DIMENSION = 3


@dataclass(frozen=True, slots=True)
class Szmuk2018Parameters:
    """Table 1 and Table 2 of the paper in non-dimensional units (UM, UL, UT)."""

    gravity: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    wet_mass: float = 2.0
    dry_mass: float = 1.0
    thrust_min: float = 0.3
    thrust_max: float = 5.0
    gimbal_max_deg: float = 20.0
    tilt_max_deg: float = 90.0
    glide_slope_deg: float = 20.0
    omega_max_deg: float = 60.0
    inertia: tuple[float, float, float] = (1.0e-2, 1.0e-2, 1.0e-2)
    thrust_arm: tuple[float, float, float] = (-1.0e-2, 0.0, 0.0)
    alpha_mdot: float = 1.0e-2  # NOT printed in the paper; assumption (see module docstring)
    initial_position: tuple[float, float, float] = (4.0, 4.0, 0.0)
    initial_velocity: tuple[float, float, float] = (0.0, -4.0, 0.0)  # 2-D case, from text
    final_velocity: tuple[float, float, float] = (-1.0e-1, 0.0, 0.0)
    final_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    initial_omega: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Algorithm parameters (Table 2)
    virtual_weight: float = 1.0e5
    trust_weight: float = 1.0e-3
    sigma_trust_weight: float = 1.0e-1
    virtual_tolerance: float = 1.0e-10
    trust_tolerance: float = 1.0e-3
    max_iterations: int = 15
    nodes: int = 50

    @property
    def gravity_vector(self) -> FloatArray:
        return np.asarray(self.gravity, dtype=np.float64)

    @property
    def inertia_matrix(self) -> FloatArray:
        return np.diag(np.asarray(self.inertia, dtype=np.float64))

    @property
    def thrust_arm_vector(self) -> FloatArray:
        return np.asarray(self.thrust_arm, dtype=np.float64)


def skew(v: FloatArray) -> FloatArray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)


def rotate_body_to_inertial(q: FloatArray, v: FloatArray) -> FloatArray:
    """``C_I/B(q) v`` for a scalar-first unit quaternion (paper's DCM transposed)."""

    q0 = q[0]
    qv = q[1:4]
    return v + 2.0 * q0 * np.cross(qv, v) + 2.0 * np.cross(qv, np.cross(qv, v))


def rotation_body_to_inertial(q: FloatArray) -> FloatArray:
    q0 = q[0]
    qv = q[1:4]
    s = skew(qv)
    return np.eye(3) + 2.0 * q0 * s + 2.0 * s @ s


def omega_matrix(w: FloatArray) -> FloatArray:
    return np.array(
        [
            [0.0, -w[0], -w[1], -w[2]],
            [w[0], 0.0, w[2], -w[1]],
            [w[1], -w[2], 0.0, w[0]],
            [w[2], w[1], -w[0], 0.0],
        ],
        dtype=np.float64,
    )


class Szmuk2018Dynamics:
    """Rigid-body dynamics of Problem 1 with analytic Jacobians."""

    def __init__(self, parameters: Szmuk2018Parameters) -> None:
        self.p = parameters
        self.J = parameters.inertia_matrix
        self.J_inv = np.linalg.inv(self.J)
        self.r_T = parameters.thrust_arm_vector
        self.g = parameters.gravity_vector

    def f(self, x: FloatArray, u: FloatArray) -> FloatArray:
        m = x[0]
        v = x[4:7]
        q = x[7:11]
        w = x[11:14]
        thrust_norm = float(np.linalg.norm(u))
        out = np.empty(STATE_DIMENSION)
        out[0] = -self.p.alpha_mdot * thrust_norm
        out[1:4] = v
        out[4:7] = rotate_body_to_inertial(q, u) / m + self.g
        out[7:11] = 0.5 * omega_matrix(w) @ q
        out[11:14] = self.J_inv @ (np.cross(self.r_T, u) - np.cross(w, self.J @ w))
        return out

    def jacobians(self, x: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        m = x[0]
        q = x[7:11]
        w = x[11:14]
        q0 = q[0]
        qv = q[1:4]
        A = np.zeros((STATE_DIMENSION, STATE_DIMENSION))
        B = np.zeros((STATE_DIMENSION, CONTROL_DIMENSION))
        thrust_norm = float(np.linalg.norm(u))
        if thrust_norm > 0.0:
            B[0, :] = -self.p.alpha_mdot * u / thrust_norm
        A[1:4, 4:7] = np.eye(3)
        C = rotation_body_to_inertial(q)
        thrust_inertial = C @ u
        A[4:7, 0] = -thrust_inertial / (m * m)
        # d(C_I/B(q) u)/dq  with C u = u + 2 q0 (qv x u) + 2 qv x (qv x u)
        d_q0 = 2.0 * np.cross(qv, u)
        d_qv = -2.0 * q0 * skew(u) + 2.0 * (
            float(qv @ u) * np.eye(3) + np.outer(qv, u) - 2.0 * np.outer(u, qv)
        )
        A[4:7, 7] = d_q0 / m
        A[4:7, 8:11] = d_qv / m
        B[4:7, :] = C / m
        A[7:11, 7:11] = 0.5 * omega_matrix(w)
        # d(0.5 Omega(w) q)/dw
        A[7, 11:14] = -0.5 * qv
        A[8:11, 11:14] = 0.5 * (q0 * np.eye(3) + skew(qv))
        Jw = self.J @ w
        A[11:14, 11:14] = -self.J_inv @ (skew(w) @ self.J - skew(Jw))
        B[11:14, :] = self.J_inv @ skew(self.r_T)
        return A, B


@dataclass(slots=True)
class Szmuk2018Reproduction:
    outcome: SCvxOutcome
    time_of_flight: float
    fuel_used: float
    final_mass: float
    iterations_to_converge: int
    tf_guess: float
    max_path_violation: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.outcome.status,
            "time_of_flight": self.time_of_flight,
            "fuel_used": self.fuel_used,
            "final_mass": self.final_mass,
            "iterations_to_converge": self.iterations_to_converge,
            "tf_guess": self.tf_guess,
            "replay_defect_inf": self.outcome.replay_defect_inf,
            "replay_terminal_error_inf": self.outcome.replay_terminal_error_inf,
            "path_violations": self.outcome.path_violations,
            "max_path_violation": self.max_path_violation,
            "iterations": [asdict(item) for item in self.outcome.iterations],
        }


def initial_guess(p: Szmuk2018Parameters) -> tuple[FloatArray, FloatArray]:
    """Algorithm 1 initialisation (straight-line, dynamically inconsistent)."""

    K = p.nodes
    states = np.zeros((K, STATE_DIMENSION))
    controls = np.zeros((K, CONTROL_DIMENSION))
    r_i = np.asarray(p.initial_position)
    v_i = np.asarray(p.initial_velocity)
    v_f = np.asarray(p.final_velocity)
    for k in range(K):
        a1 = (K - k) / K
        a2 = k / K
        m_k = a1 * p.wet_mass + a2 * p.dry_mass
        states[k, 0] = m_k
        states[k, 1:4] = a1 * r_i
        states[k, 4:7] = a1 * v_i + a2 * v_f
        states[k, 7:11] = (1.0, 0.0, 0.0, 0.0)
        controls[k] = -m_k * p.gravity_vector
    return states, controls


def _path_violations(p: Szmuk2018Parameters):
    tan_gs = np.tan(np.deg2rad(p.glide_slope_deg))
    cos_tilt = np.cos(np.deg2rad(p.tilt_max_deg))
    cos_gimbal = np.cos(np.deg2rad(p.gimbal_max_deg))
    w_max = np.deg2rad(p.omega_max_deg)

    def check(states: FloatArray, controls: FloatArray, sigma: float) -> dict[str, float]:
        m = states[:, 0]
        r = states[:, 1:4]
        q = states[:, 7:11]
        w = states[:, 11:14]
        thrust = np.linalg.norm(controls, axis=1)
        return {
            "dry_mass": float(np.max(np.maximum(p.dry_mass - m, 0.0))),
            "glide_slope": float(
                np.max(np.maximum(tan_gs * np.linalg.norm(r[:, 1:3], axis=1) - r[:, 0], 0.0))
            ),
            "tilt": float(
                np.max(np.maximum(cos_tilt - (1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)), 0.0))
            ),
            "angular_rate": float(np.max(np.maximum(np.linalg.norm(w, axis=1) - w_max, 0.0))),
            "thrust_min": float(np.max(np.maximum(p.thrust_min - thrust, 0.0))),
            "thrust_max": float(np.max(np.maximum(thrust - p.thrust_max, 0.0))),
            "gimbal": float(np.max(np.maximum(cos_gimbal * thrust - controls[:, 0], 0.0))),
            "quaternion_norm": float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0))),
        }

    return check


def _constraints(p: Szmuk2018Parameters):
    tan_gs = np.tan(np.deg2rad(p.glide_slope_deg))
    cos_tilt = np.cos(np.deg2rad(p.tilt_max_deg))
    cos_gimbal = np.cos(np.deg2rad(p.gimbal_max_deg))
    w_max = np.deg2rad(p.omega_max_deg)
    tilt_bound = float(np.sqrt(max((1.0 - cos_tilt) / 2.0, 0.0)))

    def add(v: SCvxVariables, ref_x: FloatArray, ref_u: FloatArray, ref_sigma: float) -> None:
        b = v.builder
        K = len(v.states)
        # Boundary conditions.
        b.add_equality(v.x(0, 0), p.wet_mass)
        for i in range(3):
            b.add_equality(v.x(0, 1 + i), p.initial_position[i])
            b.add_equality(v.x(0, 4 + i), p.initial_velocity[i])
            b.add_equality(v.x(0, 11 + i), p.initial_omega[i])
            b.add_equality(v.x(K - 1, 1 + i), 0.0)
            b.add_equality(v.x(K - 1, 4 + i), p.final_velocity[i])
            b.add_equality(v.x(K - 1, 11 + i), 0.0)
        for i in range(4):
            b.add_equality(v.x(K - 1, 7 + i), p.final_quaternion[i])
        b.add_equality(v.u(K - 1, 1), 0.0)
        b.add_equality(v.u(K - 1, 2), 0.0)
        for k in range(K):
            b.add_geq(v.x(k, 0), p.dry_mass)
            # tan(gs) ||H23 r|| <= e1.r
            b.add_soc(v.x(k, 1).scaled(1.0 / tan_gs), [v.x(k, 2), v.x(k, 3)])
            # tilt: ||[q2, q3]|| <= sqrt((1 - cos theta_max)/2)
            b.add_soc(LinearExpression.const(tilt_bound), [v.x(k, 9), v.x(k, 10)])
            # angular rate
            b.add_soc(LinearExpression.const(w_max), [v.x(k, 11 + i) for i in range(3)])
            # thrust upper bound
            b.add_soc(LinearExpression.const(p.thrust_max), [v.u(k, i) for i in range(3)])
            # gimbal: cos(delta_max) ||u|| <= e1.u
            b.add_soc(v.u(k, 0).scaled(1.0 / cos_gimbal), [v.u(k, i) for i in range(3)])
            # linearised thrust lower bound: T_min <= (u_hat/||u_hat||) . u
            u_hat = ref_u[k]
            norm = float(np.linalg.norm(u_hat))
            direction = u_hat / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])
            expression = LinearExpression(
                [v.controls[k].start + i for i in range(3)],
                [float(direction[i]) for i in range(3)],
                0.0,
            )
            b.add_geq(expression, p.thrust_min)

    return add


def _objective(v: SCvxVariables, ref_x: FloatArray, ref_u: FloatArray, ref_sigma: float) -> None:
    if v.sigma is None:
        raise RuntimeError("the Szmuk 2018 profile is a free-final-time problem")
    v.builder.add_linear_cost(v.sigma, 1.0)


def _normalise_quaternion(state: FloatArray) -> FloatArray:
    out = np.array(state, dtype=np.float64)
    norm = np.linalg.norm(out[7:11])
    if norm > 0.0:
        out[7:11] /= norm
    return out


def build_solver(
    p: Szmuk2018Parameters,
    *,
    substeps: int = 8,
    project_quaternion: bool = False,
    hard_trust_radius: float | None = None,
) -> FreeFinalTimeSCvx:
    dynamics = Szmuk2018Dynamics(p)
    settings = SCvxSettings(
        nodes=p.nodes,
        max_iterations=p.max_iterations,
        virtual_weight=p.virtual_weight,
        trust_weight=p.trust_weight,
        sigma_trust_weight=p.sigma_trust_weight,
        virtual_tolerance=p.virtual_tolerance,
        trust_tolerance=p.trust_tolerance,
        substeps=substeps,
        hard_trust_radius=hard_trust_radius,
    )
    return FreeFinalTimeSCvx(
        n_x=STATE_DIMENSION,
        n_u=CONTROL_DIMENSION,
        dynamics=dynamics.f,
        jacobians=dynamics.jacobians,
        constraints=_constraints(p),
        objective=_objective,
        settings=settings,
        free_final_time=True,
        sigma_bounds=(1.0e-2, 1.0e3),
        path_check=_path_violations(p),
        state_projection=_normalise_quaternion if project_quaternion else None,
    )


def reproduce(
    p: Szmuk2018Parameters | None = None,
    *,
    tf_guess: float = 5.0,
    substeps: int = 8,
    max_iterations: int | None = None,
    virtual_tolerance: float | None = None,
) -> Szmuk2018Reproduction:
    params = p or Szmuk2018Parameters()
    if max_iterations is not None or virtual_tolerance is not None:
        params = Szmuk2018Parameters(
            **{
                **{f: getattr(params, f) for f in params.__dataclass_fields__},
                **({"max_iterations": max_iterations} if max_iterations is not None else {}),
                **(
                    {"virtual_tolerance": virtual_tolerance}
                    if virtual_tolerance is not None
                    else {}
                ),
            }
        )
    solver = build_solver(params, substeps=substeps)
    states, controls = initial_guess(params)
    outcome = solver.solve(states, controls, tf_guess)
    violations = outcome.path_violations
    return Szmuk2018Reproduction(
        outcome=outcome,
        time_of_flight=outcome.sigma,
        fuel_used=float(params.wet_mass - outcome.states[-1, 0]),
        final_mass=float(outcome.states[-1, 0]),
        iterations_to_converge=len(outcome.iterations),
        tf_guess=tf_guess,
        max_path_violation=float(max(violations.values())) if violations else float("nan"),
        parameters={f: getattr(params, f) for f in params.__dataclass_fields__},
    )


def tf_guess_sweep(
    p: Szmuk2018Parameters | None = None,
    guesses: tuple[float, ...] = tuple(float(v) for v in range(1, 11)),
    **kwargs: Any,
) -> list[Szmuk2018Reproduction]:
    """Section 4.1 robustness experiment: ten t_f guesses from 1.0 to 10.0 UT."""

    return [reproduce(p, tf_guess=guess, **kwargs) for guess in guesses]


def parameters_from_document(document: dict[str, Any]) -> Szmuk2018Parameters:
    p = document["parameters"]
    return Szmuk2018Parameters(
        gravity=tuple(p["gravity"]),
        wet_mass=p["wet_mass"],
        dry_mass=p["dry_mass"],
        thrust_min=p["thrust_min"],
        thrust_max=p["thrust_max"],
        gimbal_max_deg=p["gimbal_max_deg"],
        tilt_max_deg=p["tilt_max_deg"],
        glide_slope_deg=p["glide_slope_deg"],
        omega_max_deg=p["omega_max_deg"],
        inertia=tuple(p["inertia"]),
        thrust_arm=tuple(p["thrust_arm"]),
        alpha_mdot=p["alpha_mdot"],
        initial_position=tuple(p["initial_position"]),
        initial_velocity=tuple(p["initial_velocity"]),
        final_velocity=tuple(p["final_velocity"]),
        final_quaternion=tuple(p["final_quaternion"]),
        initial_omega=tuple(p["initial_omega"]),
        virtual_weight=p["virtual_weight"],
        trust_weight=p["trust_weight"],
        sigma_trust_weight=p["sigma_trust_weight"],
        virtual_tolerance=p["virtual_tolerance"],
        trust_tolerance=p["trust_tolerance"],
        max_iterations=p["max_iterations"],
        nodes=p["nodes"],
    )


#: Declared discretisation envelope between the FOH Python core and the ZOH native pd6_fft.
NATIVE_TF_ENVELOPE_UT = 0.01


def _native_free_time_leg(
    params: Szmuk2018Parameters, core_time_of_flight: float, options: dict[str, Any]
) -> dict[str, Any]:
    """Run the same problem through the native ``pd6_fft`` transcription when it is available."""

    if options.get("skip_native"):
        return {"status": "skipped", "reason": "skip_native option set"}
    try:
        from spacepdhcg.native import native_available
        from spacepdhcg.native._library import load_native_library
    except Exception as error:
        return {"status": "unavailable", "reason": f"native package import failed: {error}"}
    if not native_available():
        return {
            "status": "unavailable",
            "reason": "native library not found (build cpp/ and set SPACEPDHCG_NATIVE_LIBRARY)",
        }
    if not hasattr(load_native_library(), "spacepdhcg_pd6_fft_create"):
        return {"status": "unavailable", "reason": "native library predates the pd6_fft C API"}
    from spacepdhcg.literature.pd6_szmuk_2018_native import reproduce_native

    try:
        result = reproduce_native(params)
    except Exception as error:
        return {"status": "error", "reason": repr(error)}
    gap = float(result.time_of_flight - core_time_of_flight)
    within = result.outcome.converged and abs(gap) <= NATIVE_TF_ENVELOPE_UT
    within = within and result.max_path_violation <= 1.0e-6
    return {
        "status": "reproduced" if within else "gap",
        "time_of_flight": result.time_of_flight,
        "fuel_used": result.fuel_used,
        "converged": result.outcome.converged,
        "termination": result.outcome.termination,
        "iterations": len(result.outcome.iterations),
        "replay_defect_inf": result.outcome.replay_defect_inf,
        "max_path_violation": result.max_path_violation,
        "path_violations": result.path_violations,
        "topology_fingerprint": f"{result.topology_fingerprint:016x}",
        "gap_vs_cpu_core_ut": gap,
        "envelope_ut": NATIVE_TF_ENVELOPE_UT,
        "label": "measured-local",
    }


def run_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    params = parameters_from_document(document)
    guesses = tuple(
        float(v)
        for v in options.get("tf_guesses", document.get("tf_guess_sweep", [1.0, 5.0, 10.0]))
    )
    extended = int(options.get("extended_iterations", document.get("extended_iterations", 30)))
    # Paper protocol: 15 iterations, nu_tol 1e-10 (an interior-point solve leaves ~1e-11..1e-13).
    sweep = [reproduce(params, tf_guess=g) for g in guesses]
    tfs = [r.time_of_flight for r in sweep]
    spread = float(max(tfs) - min(tfs)) if tfs else float("nan")
    # Extended run with the paper's weights but more iterations to reach Delta_tol.
    extended_run = reproduce(params, tf_guess=5.0, max_iterations=extended)
    published = document["published"]
    acceptance_spread = float(document.get("acceptance_spread_ut", 0.01))
    all_feasible = all(
        r.max_path_violation <= 1.0e-6 and r.outcome.replay_defect_inf <= 1.0e-5 for r in sweep
    )
    status = "reproduced" if (spread <= acceptance_spread and all_feasible) else "gap"
    native_leg = _native_free_time_leg(params, extended_run.time_of_flight, options)
    return {
        "target_id": document["id"],
        "status": status,
        "published": published,
        "measured": {
            "time_of_flight_by_guess": {str(r.tf_guess): r.time_of_flight for r in sweep},
            "time_of_flight_spread_ut": spread,
            "iterations_by_guess": {str(r.tf_guess): r.iterations_to_converge for r in sweep},
            "statuses_by_guess": {str(r.tf_guess): r.outcome.status for r in sweep},
            "extended_run": {
                "iterations": extended,
                "status": extended_run.outcome.status,
                "time_of_flight": extended_run.time_of_flight,
                "fuel_used": extended_run.fuel_used,
                "replay_defect_inf": extended_run.outcome.replay_defect_inf,
                "max_path_violation": extended_run.max_path_violation,
                "final_trust_norm": extended_run.outcome.iterations[-1].trust_norm,
                "final_virtual_l1": extended_run.outcome.iterations[-1].virtual_l1,
            },
            "native_pd6_fft": native_leg,
        },
        "gap": {
            "time_of_flight_spread_minus_published_ut": spread
            - float(published["tf_guess_sweep_spread_ut"]),
            "converged_time_of_flight": (
                "descriptive-only: the paper prints no digits (Figure 2 only)"
            ),
        },
        "labels": {
            "published.tf_guess_sweep_spread_ut": "published-reference",
            "published.iterations_to_converge": "published-reference",
            "published.converged_time_of_flight": "descriptive-only",
            "measured.time_of_flight": "measured-local",
            "measured.native_pd6_fft.time_of_flight": "measured-local",
            "parameters.alpha_mdot": "descriptive-only",
        },
        "envelope": {
            "discretisation": (
                "K = 50 nodes, FOH, RK4 STM with 8 substeps per interval; free final time via sigma"
            ),
            "paper_stop_rule": "||Delta||_2 <= 1e-3 and ||nu||_1 <= 1e-10 within 15 iterations",
            "native_pd6_fft": (
                "K = 50 nodes, ZOH, variational RK4 with 4 substeps per interval, sigma column "
                "analytic; hard-trust-region SCvx; declared envelope vs the FOH core 0.01 UT"
            ),
        },
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [
            (
                "alpha_mdot = 0.01 UT/UL and a zero vertical initial velocity are assumptions (not "
                "printed in the paper)"
            ),
        ],
        "details": {"sweep": [r.as_dict() for r in sweep], "extended": extended_run.as_dict()},
    }
