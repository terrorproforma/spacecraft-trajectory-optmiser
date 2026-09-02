"""Szmuk-Acikmese 2018 free-final-time 6-DoF through the NATIVE ``pd6_fft`` transcription.

The Python core (:mod:`spacepdhcg.literature.pd6_szmuk_2018`) reproduces the paper with an
FOH / K-node / soft-trust SCvx.  This module drives the native C++ time-dilated transcription
(``cpp/include/spacepdhcg/transcription/powered_descent_6dof_free_time.hpp``) through the C API
and Clarabel, so the native sigma column, quaternion tangent rule and Szmuk control model are
exercised end to end against the same problem data.

Frame mapping
-------------
The paper uses an *x-up* inertial frame with the body thrust axis along body x; the native
6-DoF model is *z-up* with thrust along body z.  Both frames are related by the same cyclic
permutation ``P: (x, y, z)_S -> (z, x, y)_N`` (native = (S_y, S_z, S_x)), a proper rotation, so
quaternion vector parts permute identically and the identity attitude maps to the identity.

Known model differences (declared, they bound the expected t_f gap):
  * the native transcription is ZOH per interval with RK4 substeps (the Python core is FOH);
  * the native mass flow uses the throttle slack Gamma (= |T| at convergence in linearised
    mode) rather than |T| directly;
  * the native path constraints are the paper's: glide slope, angular-rate cone, gimbal cone
    ``|T| <= T_z / cos(delta_max)``, attitude tilt ``|[q_x, q_y]| <= sqrt((1 - cos theta_max)/2)``
    (theta_max = 90 deg is ACTIVE: without it the native optimum drops to t_f ~ 2.97 UT with the
    vehicle tilted ~120 deg), linearised thrust lower bound and thrust upper bound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.literature.pd6_szmuk_2018 import Szmuk2018Parameters
from spacepdhcg.native.free_time import (
    FreeTimeLoopSettings,
    FreeTimeOutcome,
    NativeFreeTimeTranscription,
    Pd6FreeTimeOptions,
    run_free_time_scvx,
)

FloatArray = NDArray[np.float64]

# Szmuk (x, y, z) -> native (S_y, S_z, S_x)
_PERMUTATION = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def to_native_vector(v: tuple[float, float, float] | FloatArray) -> FloatArray:
    return _PERMUTATION @ np.asarray(v, dtype=np.float64)


def to_native_quaternion(q: tuple[float, float, float, float] | FloatArray) -> FloatArray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate(([q[0]], _PERMUTATION @ q[1:]))


#: Exact-penalty weight on the virtual control.  The paper uses 1e5 with a soft trust region and
#: accepts every step; the native loop is a hard-trust-region SCvx whose merit function must use
#: the SAME weight (so the predicted reduction is non-negative by construction).  The weight only
#: has to exceed the dynamics multipliers of this min-time problem (O(1): d t_f / d defect), so
#: the L1 penalty is exact (nu -> 0 at convergence, replay defect ~1e-7).  Much larger weights
#: (1e2..1e3) make every O(step^2) linearisation defect dominate the sigma gain of a step and
#: force the trust region to crawl (t_f still 3.67 after 150 iterations at 1e2); 10 converges in
#: ~30 iterations to t_f = 3.3925 UT.
NATIVE_VIRTUAL_WEIGHT = 10.0


def native_options(p: Szmuk2018Parameters, *, intervals: int, substeps: int) -> Pd6FreeTimeOptions:
    """Build the native ``pd6_fft`` configuration from the paper's parameters."""

    return Pd6FreeTimeOptions(
        gravity=tuple(to_native_vector(p.gravity)),
        principal_inertia=tuple(to_native_vector(p.inertia)),
        mass_flow_coefficient=p.alpha_mdot,
        minimum_mass=p.dry_mass,
        maximum_thrust=p.thrust_max,
        minimum_sigma=p.thrust_min,
        maximum_torque=1.0e3,  # torque is tied to the thrust arm; keep the cone inactive
        maximum_angular_rate=float(np.deg2rad(p.omega_max_deg)),
        maximum_tilt_radians=float(np.deg2rad(p.gimbal_max_deg)),
        # theta_max = 90 deg IS active on this problem: without it the optimiser tilts the
        # vehicle past horizontal and reaches t_f ~ 2.97 UT (verified by independent replay).
        maximum_attitude_tilt_radians=float(np.deg2rad(p.tilt_max_deg)),
        # The native cone is |r_xy| <= tan(glide_slope) * z (half-angle from the vertical); the
        # paper's gamma_gs is measured from the horizontal: tan(gamma_gs) |r_xy| <= z.
        glide_slope_radians=float(np.deg2rad(90.0 - p.glide_slope_deg)),
        intervals=intervals,
        substeps=substeps,
        sigma_minimum=1.0e-2,
        sigma_maximum=1.0e3,
        trust_radius=1.0,
        sigma_trust_radius=1.0,
        virtual_l1_weight=NATIVE_VIRTUAL_WEIGHT,
        virtual_quadratic_weight=1.0e-8,
        virtual_epigraph_regularisation=1.0e-10,
        fuel_weight=0.0,
        time_weight=1.0,
        # Native quadratic terms are 1/2 w (.)^2; the paper penalises w_tr ||Delta||^2.
        sigma_tracking_weight=2.0 * p.sigma_trust_weight,
        thrust_norm_mode="linearised",
        torque_mode="thrust_arm",
        terminal_thrust_axial=True,
        thrust_arm=tuple(to_native_vector(p.thrust_arm)),
        # initial: r, v, omega, m fixed; attitude free
        initial_fixed=(True,) * 6 + (False,) * 4 + (True,) * 4,
        # terminal: r, v, q, omega fixed; mass free
        terminal_fixed=(True,) * 13 + (False,),
        state_tracking_weights=(2.0 * p.trust_weight,) * 14,
        control_tracking_weights=(2.0 * p.trust_weight,) * 7,
    )


def native_boundary(p: Szmuk2018Parameters) -> tuple[FloatArray, FloatArray]:
    initial = np.zeros(14)
    initial[0:3] = to_native_vector(p.initial_position)
    initial[3:6] = to_native_vector(p.initial_velocity)
    initial[6:10] = (1.0, 0.0, 0.0, 0.0)
    initial[10:13] = to_native_vector(p.initial_omega)
    initial[13] = p.wet_mass
    target = np.zeros(14)
    target[3:6] = to_native_vector(p.final_velocity)
    target[6:10] = to_native_quaternion(p.final_quaternion)
    target[13] = p.dry_mass  # masked out (terminal mass free)
    return initial, target


def native_initial_guess(
    p: Szmuk2018Parameters, transcription: NativeFreeTimeTranscription
) -> tuple[FloatArray, FloatArray]:
    """Algorithm 1 straight-line initialisation, expressed in the native layout."""

    intervals = transcription.layout.intervals
    initial, target = native_boundary(p)
    states = np.zeros((intervals + 1, 14))
    controls = np.zeros((intervals, 7))
    gravity = to_native_vector(p.gravity)
    for node in range(intervals + 1):
        a1 = (intervals - node) / intervals
        a2 = node / intervals
        states[node, 0:3] = a1 * initial[0:3]
        states[node, 3:6] = a1 * initial[3:6] + a2 * target[3:6]
        states[node, 6:10] = (1.0, 0.0, 0.0, 0.0)
        states[node, 13] = a1 * p.wet_mass + a2 * p.dry_mass
    for interval in range(intervals):
        mass = 0.5 * (states[interval, 13] + states[interval + 1, 13])
        thrust = -mass * gravity
        raw = np.concatenate((thrust, np.zeros(3), [np.linalg.norm(thrust)]))
        controls[interval] = transcription.project_control(raw)
    return states, controls


def path_violations(
    p: Szmuk2018Parameters, states: FloatArray, controls: FloatArray
) -> dict[str, float]:
    """Paper path constraints evaluated on the native trajectory (native frame)."""

    tan_gs = np.tan(np.deg2rad(p.glide_slope_deg))
    cos_gimbal = np.cos(np.deg2rad(p.gimbal_max_deg))
    w_max = np.deg2rad(p.omega_max_deg)
    r = states[:, 0:3]
    q = states[:, 6:10]
    w = states[:, 10:13]
    m = states[:, 13]
    thrust = np.linalg.norm(controls[:, 0:3], axis=1)
    cos_tilt = np.cos(np.deg2rad(p.tilt_max_deg))
    return {
        "dry_mass": float(np.max(np.maximum(p.dry_mass - m, 0.0))),
        "glide_slope": float(
            np.max(np.maximum(tan_gs * np.linalg.norm(r[:, 0:2], axis=1) - r[:, 2], 0.0))
        ),
        # native q vector part = P q_S: Szmuk (q2, q3) are native (q_x, q_y)
        "tilt": float(
            np.max(np.maximum(cos_tilt - (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)), 0.0))
        ),
        "angular_rate": float(np.max(np.maximum(np.linalg.norm(w, axis=1) - w_max, 0.0))),
        "thrust_min": float(np.max(np.maximum(p.thrust_min - thrust, 0.0))),
        "thrust_max": float(np.max(np.maximum(thrust - p.thrust_max, 0.0))),
        "gimbal": float(np.max(np.maximum(cos_gimbal * thrust - controls[:, 2], 0.0))),
        "quaternion_norm": float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0))),
    }


@dataclass(slots=True)
class NativeSzmukReproduction:
    outcome: FreeTimeOutcome
    time_of_flight: float
    final_mass: float
    fuel_used: float
    intervals: int
    substeps: int
    tf_guess: float
    max_path_violation: float
    path_violations: dict[str, float]
    topology_fingerprint: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_of_flight": self.time_of_flight,
            "final_mass": self.final_mass,
            "fuel_used": self.fuel_used,
            "intervals": self.intervals,
            "substeps": self.substeps,
            "tf_guess": self.tf_guess,
            "converged": self.outcome.converged,
            "termination": self.outcome.termination,
            "iterations": len(self.outcome.iterations),
            "replay_defect_inf": self.outcome.replay_defect_inf,
            "max_path_violation": self.max_path_violation,
            "path_violations": self.path_violations,
            "topology_fingerprint": f"{self.topology_fingerprint:016x}",
            "trace": [asdict(record) for record in self.outcome.iterations],
        }


def reproduce_native(
    p: Szmuk2018Parameters | None = None,
    *,
    intervals: int = 49,
    substeps: int = 4,
    tf_guess: float = 5.0,
    settings: FreeTimeLoopSettings | None = None,
    backend_builder: Callable[[Any], Any] | None = None,
) -> NativeSzmukReproduction:
    """Solve the paper's problem with the native ``pd6_fft`` CQP and a Clarabel outer loop.

    ``intervals = 49`` matches the paper's K = 50 nodes.  ``backend_builder`` swaps the conic
    solver (e.g. the pure-QOCO GPU backend for the deferred GPU legs).
    """

    params = p or Szmuk2018Parameters()
    transcription = NativeFreeTimeTranscription(
        native_options(params, intervals=intervals, substeps=substeps)
    )
    initial, target = native_boundary(params)
    states, controls = native_initial_guess(params, transcription)
    # Hard-trust-region SCvx around the paper's soft weights; the merit penalty equals the CQP
    # virtual weight so model and nonlinear merit are consistent (predicted reduction >= 0).
    loop = settings or FreeTimeLoopSettings(
        max_iterations=params.max_iterations * 10,
        defect_tolerance=1.0e-6,
        sigma_tolerance=1.0e-4,
        virtual_tolerance=1.0e-6,
        defect_penalty=NATIVE_VIRTUAL_WEIGHT,
        trust_radius=2.0,
        sigma_trust_radius=1.0,
        minimum_trust_radius=1.0e-5,
        shrink=0.5,
        grow=1.5,
        accept_ratio=0.05,
        grow_ratio=0.5,
        clarabel_tolerance=1.0e-9,
        clarabel_iterations=500,
    )
    outcome = run_free_time_scvx(
        transcription,
        states,
        controls,
        tf_guess,
        initial,
        target,
        loop,
        backend_builder=backend_builder,
    )
    violations = path_violations(params, outcome.states, outcome.controls)
    return NativeSzmukReproduction(
        outcome=outcome,
        time_of_flight=outcome.sigma,
        final_mass=float(outcome.states[-1, 13]),
        fuel_used=float(params.wet_mass - outcome.states[-1, 13]),
        intervals=intervals,
        substeps=substeps,
        tf_guess=tf_guess,
        max_path_violation=float(max(violations.values())),
        path_violations=violations,
        topology_fingerprint=transcription.layout.topology_fingerprint,
    )
