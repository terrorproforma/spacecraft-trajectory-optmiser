"""Independent CPU successive-convexification core with free-final-time support.

This module implements the Szmuk-Acikmese (2018, DOI 10.2514/6.2018-0617) transcription:

* normalised time ``tau in [0, 1]`` with a time-dilation variable ``sigma = t_f`` so that
  ``x'(tau) = sigma * f(x, u)``;
* first-order-hold (FOH) control interpolation between nodes;
* discrete dynamics ``x_{k+1} = A_k x_k + B_k u_k + C_k u_{k+1} + S_k sigma + z_k + nu_k``
  obtained by integrating the state-transition and forced-response equations along the
  nonlinear reference trajectory of each interval (multiple-shooting linearisation);
* virtual control ``nu`` with an L1 penalty and quadratic soft trust regions on the state,
  control, and dilation deviations.

The same core solves fixed-final-time problems by freezing ``sigma``.  It is intentionally
independent of the SpacePDHCG persistent backend so it can act as a ``measured-local``
reference implementation for the literature reproductions.  Clarabel is the conic solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.literature._socp import LinearExpression, SOCPBuilder, affine, lin

FloatArray = NDArray[np.float64]
DynamicsFn = Callable[[FloatArray, FloatArray], FloatArray]
JacobianFn = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True, slots=True)
class DiscreteInterval:
    """Affine discrete map for one interval in normalised time."""

    A: FloatArray
    B: FloatArray
    C: FloatArray
    S: FloatArray
    z: FloatArray
    propagated: FloatArray


def foh_control(u_k: FloatArray, u_next: FloatArray, fraction: float) -> FloatArray:
    return (1.0 - fraction) * u_k + fraction * u_next


def discretise_interval(
    dynamics: DynamicsFn,
    jacobians: JacobianFn,
    x_k: FloatArray,
    u_k: FloatArray,
    u_next: FloatArray,
    sigma: float,
    d_tau: float,
    *,
    substeps: int = 8,
    zero_order_hold: bool = False,
) -> DiscreteInterval:
    """Integrate the augmented variational system with fixed-step RK4.

    The integration state is ``[x, Psi (inverse STM), b, c, s, zeta]`` with
    ``Psi' = -Psi A(tau)``, ``b' = Psi B alpha``, ``c' = Psi B beta``, ``s' = Psi f``,
    ``zeta' = Psi (-A x_hat - B u_hat)``.  Multiplying the forced responses by the final STM
    ``Phi = Psi^{-1}`` yields the discrete coefficients.
    """

    n_x = x_k.shape[0]
    n_u = u_k.shape[0]
    if substeps < 1:
        raise ValueError("substeps must be positive")

    def rhs(
        tau_local: float,
        x: FloatArray,
        psi: FloatArray,
        b: FloatArray,
        c: FloatArray,
        s: FloatArray,
        zeta: FloatArray,
    ) -> tuple[FloatArray, ...]:
        fraction = 0.0 if zero_order_hold else tau_local / d_tau
        u = foh_control(u_k, u_next, fraction)
        f = np.asarray(dynamics(x, u), dtype=np.float64)
        a_cont, b_cont = jacobians(x, u)
        a_scaled = sigma * np.asarray(a_cont, dtype=np.float64)
        b_scaled = sigma * np.asarray(b_cont, dtype=np.float64)
        alpha = 1.0 if zero_order_hold else 1.0 - fraction
        beta = 0.0 if zero_order_hold else fraction
        psi_b = psi @ b_scaled
        return (
            sigma * f,
            -psi @ a_scaled,
            psi_b * alpha,
            psi_b * beta,
            psi @ f,
            psi @ (-a_scaled @ x - b_scaled @ u),
        )

    x = np.array(x_k, dtype=np.float64)
    psi = np.eye(n_x)
    b = np.zeros((n_x, n_u))
    c = np.zeros((n_x, n_u))
    s = np.zeros(n_x)
    zeta = np.zeros(n_x)
    h = d_tau / substeps
    tau_local = 0.0
    for _ in range(substeps):
        k1 = rhs(tau_local, x, psi, b, c, s, zeta)
        k2 = rhs(
            tau_local + 0.5 * h,
            *[y + 0.5 * h * k for y, k in zip((x, psi, b, c, s, zeta), k1, strict=True)],
        )
        k3 = rhs(
            tau_local + 0.5 * h,
            *[y + 0.5 * h * k for y, k in zip((x, psi, b, c, s, zeta), k2, strict=True)],
        )
        k4 = rhs(
            tau_local + h,
            *[y + h * k for y, k in zip((x, psi, b, c, s, zeta), k3, strict=True)],
        )
        x, psi, b, c, s, zeta = (
            y + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            for y, a1, a2, a3, a4 in zip((x, psi, b, c, s, zeta), k1, k2, k3, k4, strict=True)
        )
        tau_local += h
    phi = np.linalg.inv(psi)
    return DiscreteInterval(
        A=phi,
        B=phi @ b,
        C=phi @ c,
        S=phi @ s,
        z=phi @ zeta,
        propagated=x,
    )


def propagate_interval(
    dynamics: DynamicsFn,
    x_k: FloatArray,
    u_k: FloatArray,
    u_next: FloatArray,
    sigma: float,
    d_tau: float,
    *,
    substeps: int = 64,
    zero_order_hold: bool = False,
) -> FloatArray:
    """High-resolution RK4 replay of one interval with the FOH control law."""

    x = np.array(x_k, dtype=np.float64)
    h = d_tau / substeps
    tau_local = 0.0

    def f(tau_l: float, state: FloatArray) -> FloatArray:
        fraction = 0.0 if zero_order_hold else tau_l / d_tau
        return sigma * np.asarray(dynamics(state, foh_control(u_k, u_next, fraction)))

    for _ in range(substeps):
        k1 = f(tau_local, x)
        k2 = f(tau_local + 0.5 * h, x + 0.5 * h * k1)
        k3 = f(tau_local + 0.5 * h, x + 0.5 * h * k2)
        k4 = f(tau_local + h, x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tau_local += h
    return x


@dataclass(slots=True)
class SCvxVariables:
    """Variable slices of one convex sub-problem."""

    states: list[slice]
    controls: list[slice]
    sigma: int | None
    virtual: list[slice]
    builder: SOCPBuilder

    def x(self, node: int, component: int) -> LinearExpression:
        return lin(self.states[node].start + component)

    def u(self, node: int, component: int) -> LinearExpression:
        return lin(self.controls[node].start + component)

    def sigma_expression(self, fixed_value: float) -> LinearExpression:
        if self.sigma is None:
            return LinearExpression.const(fixed_value)
        return lin(self.sigma)


ConstraintFn = Callable[[SCvxVariables, FloatArray, FloatArray, float], None]
ObjectiveFn = Callable[[SCvxVariables, FloatArray, FloatArray, float], None]
PathCheckFn = Callable[[FloatArray, FloatArray, float], dict[str, float]]


@dataclass(frozen=True, slots=True)
class SCvxSettings:
    nodes: int = 50
    max_iterations: int = 15
    virtual_weight: float = 1.0e5
    trust_weight: float = 1.0e-3
    sigma_trust_weight: float = 1.0e-1
    virtual_tolerance: float = 1.0e-10
    trust_tolerance: float = 1.0e-3
    substeps: int = 8
    replay_substeps: int = 64
    zero_order_hold: bool = False
    solver_tolerance: float = 1.0e-9
    solver_iterations: int = 500
    state_scale: FloatArray | None = None
    control_scale: FloatArray | None = None
    hard_trust_radius: float | None = None

    def validate(self, n_x: int, n_u: int) -> None:
        if self.nodes < 3:
            raise ValueError("at least three nodes are required")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        for name in ("virtual_weight", "trust_weight", "sigma_trust_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.state_scale is not None and np.asarray(self.state_scale).shape != (n_x,):
            raise ValueError("state_scale must match the state dimension")
        if self.control_scale is not None and np.asarray(self.control_scale).shape != (n_u,):
            raise ValueError("control_scale must match the control dimension")


@dataclass(slots=True)
class SCvxIteration:
    iteration: int
    sigma: float
    objective: float
    virtual_l1: float
    trust_norm: float
    solver_status: str
    solver_iterations: int
    solve_seconds: float
    discretisation_seconds: float
    replay_defect_inf: float


@dataclass(slots=True)
class SCvxOutcome:
    status: str
    states: FloatArray
    controls: FloatArray
    sigma: float
    iterations: list[SCvxIteration]
    replay_defect_inf: float
    replay_terminal_error_inf: float
    replay_states: FloatArray
    path_violations: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sigma": self.sigma,
            "iterations": [asdict(item) for item in self.iterations],
            "replay_defect_inf": self.replay_defect_inf,
            "replay_terminal_error_inf": self.replay_terminal_error_inf,
            "path_violations": self.path_violations,
            **self.extras,
        }


class FreeFinalTimeSCvx:
    """Generic FOH successive convexification with optional time dilation."""

    def __init__(
        self,
        *,
        n_x: int,
        n_u: int,
        dynamics: DynamicsFn,
        jacobians: JacobianFn,
        constraints: ConstraintFn,
        objective: ObjectiveFn,
        settings: SCvxSettings,
        free_final_time: bool,
        sigma_bounds: tuple[float, float] | None = None,
        path_check: PathCheckFn | None = None,
        state_projection: Callable[[FloatArray], FloatArray] | None = None,
    ) -> None:
        settings.validate(n_x, n_u)
        self.n_x = n_x
        self.n_u = n_u
        self.dynamics = dynamics
        self.jacobians = jacobians
        self.constraints = constraints
        self.objective = objective
        self.settings = settings
        self.free_final_time = free_final_time
        self.sigma_bounds = sigma_bounds
        self.path_check = path_check
        self.state_projection = state_projection

    # ------------------------------------------------------------------ helpers
    def discretise(
        self, states: FloatArray, controls: FloatArray, sigma: float
    ) -> list[DiscreteInterval]:
        K = self.settings.nodes
        d_tau = 1.0 / (K - 1)
        return [
            discretise_interval(
                self.dynamics,
                self.jacobians,
                states[k],
                controls[k],
                controls[k + 1],
                sigma,
                d_tau,
                substeps=self.settings.substeps,
                zero_order_hold=self.settings.zero_order_hold,
            )
            for k in range(K - 1)
        ]

    def replay(
        self, states: FloatArray, controls: FloatArray, sigma: float
    ) -> tuple[FloatArray, float]:
        K = self.settings.nodes
        d_tau = 1.0 / (K - 1)
        replayed = np.empty_like(states)
        replayed[0] = states[0]
        defect = 0.0
        for k in range(K - 1):
            propagated = propagate_interval(
                self.dynamics,
                states[k],
                controls[k],
                controls[k + 1],
                sigma,
                d_tau,
                substeps=self.settings.replay_substeps,
                zero_order_hold=self.settings.zero_order_hold,
            )
            defect = max(defect, float(np.max(np.abs(propagated - states[k + 1]))))
            replayed[k + 1] = propagate_interval(
                self.dynamics,
                replayed[k],
                controls[k],
                controls[k + 1],
                sigma,
                d_tau,
                substeps=self.settings.replay_substeps,
                zero_order_hold=self.settings.zero_order_hold,
            )
        return replayed, defect

    # ------------------------------------------------------------------ main loop
    def solve(
        self,
        reference_states: FloatArray,
        reference_controls: FloatArray,
        sigma_guess: float,
    ) -> SCvxOutcome:
        K = self.settings.nodes
        states = np.array(reference_states, dtype=np.float64)
        controls = np.array(reference_controls, dtype=np.float64)
        sigma = float(sigma_guess)
        if states.shape != (K, self.n_x) or controls.shape != (K, self.n_u):
            raise ValueError("reference trajectory has the wrong shape")
        x_scale = (
            np.ones(self.n_x)
            if self.settings.state_scale is None
            else np.asarray(self.settings.state_scale, dtype=np.float64)
        )
        u_scale = (
            np.ones(self.n_u)
            if self.settings.control_scale is None
            else np.asarray(self.settings.control_scale, dtype=np.float64)
        )
        history: list[SCvxIteration] = []
        status = "maximum_iterations"
        for iteration in range(self.settings.max_iterations):
            t0 = perf_counter()
            intervals = self.discretise(states, controls, sigma)
            discretisation_seconds = perf_counter() - t0

            builder = SOCPBuilder()
            x_slices = [builder.add_variables(self.n_x) for _ in range(K)]
            u_slices = [builder.add_variables(self.n_u) for _ in range(K)]
            sigma_index = builder.add_variables(1).start if self.free_final_time else None
            nu_slices = [builder.add_variables(self.n_x) for _ in range(K - 1)]
            nu_abs = [builder.add_variables(self.n_x) for _ in range(K - 1)]
            trust = builder.add_variables(K)  # Delta_k >= 0
            trust_norm = builder.add_variables(1).start
            variables = SCvxVariables(x_slices, u_slices, sigma_index, nu_slices, builder)

            # Dynamics with virtual control.
            for k, interval in enumerate(intervals):
                for row in range(self.n_x):
                    terms: dict[int, float] = {x_slices[k + 1].start + row: -1.0}
                    for col in range(self.n_x):
                        terms[x_slices[k].start + col] = (
                            terms.get(x_slices[k].start + col, 0.0) + interval.A[row, col]
                        )
                    for col in range(self.n_u):
                        terms[u_slices[k].start + col] = (
                            terms.get(u_slices[k].start + col, 0.0) + interval.B[row, col]
                        )
                        terms[u_slices[k + 1].start + col] = (
                            terms.get(u_slices[k + 1].start + col, 0.0) + interval.C[row, col]
                        )
                    terms[nu_slices[k].start + row] = 1.0
                    constant = interval.z[row]
                    if self.free_final_time:
                        terms[sigma_index] = interval.S[row]
                    else:
                        constant += interval.S[row] * sigma
                    builder.add_equality(affine(terms, constant))
            # L1 virtual control epigraph.
            for k in range(K - 1):
                for row in range(self.n_x):
                    nu = lin(nu_slices[k].start + row)
                    e = lin(nu_abs[k].start + row)
                    builder.add_leq(nu.minus(e))
                    builder.add_leq(nu.scaled(-1.0).minus(e))
                    builder.add_linear_cost(nu_abs[k].start + row, self.settings.virtual_weight)
            # Quadratic soft trust regions (rotated SOC): |dx|^2 + |du|^2 <= Delta_k.
            for k in range(K):
                vector = []
                for row in range(self.n_x):
                    vector.append(
                        lin(x_slices[k].start + row, 2.0 / x_scale[row]).plus(
                            -2.0 * states[k, row] / x_scale[row]
                        )
                    )
                for row in range(self.n_u):
                    vector.append(
                        lin(u_slices[k].start + row, 2.0 / u_scale[row]).plus(
                            -2.0 * controls[k, row] / u_scale[row]
                        )
                    )
                delta = lin(trust.start + k)
                vector.append(delta.minus(1.0))
                builder.add_soc(delta.plus(1.0), vector)
                builder.add_geq(delta, 0.0)
                if self.settings.hard_trust_radius is not None:
                    builder.add_leq(delta, self.settings.hard_trust_radius**2)
            builder.add_norm_bound(trust, lin(trust_norm))
            builder.add_linear_cost(trust_norm, self.settings.trust_weight)
            if self.free_final_time:
                sigma_abs = builder.add_variables(1).start
                d_sigma = lin(sigma_index).minus(sigma)
                builder.add_leq(d_sigma.minus(lin(sigma_abs)))
                builder.add_leq(d_sigma.scaled(-1.0).minus(lin(sigma_abs)))
                builder.add_linear_cost(sigma_abs, self.settings.sigma_trust_weight)
                if self.sigma_bounds is not None:
                    builder.add_bounds(slice(sigma_index, sigma_index + 1), *self.sigma_bounds)

            self.constraints(variables, states, controls, sigma)
            self.objective(variables, states, controls, sigma)

            solution = builder.solve(
                tolerance=self.settings.solver_tolerance,
                max_iterations=self.settings.solver_iterations,
                raise_on_failure=False,
            )
            if not solution.solved:
                history.append(
                    SCvxIteration(
                        iteration=iteration,
                        sigma=sigma,
                        objective=float("nan"),
                        virtual_l1=float("nan"),
                        trust_norm=float("nan"),
                        solver_status=solution.status,
                        solver_iterations=solution.iterations,
                        solve_seconds=solution.solve_seconds,
                        discretisation_seconds=discretisation_seconds,
                        replay_defect_inf=float("nan"),
                    )
                )
                status = "solver_failed"
                break
            x = solution.x
            new_states = np.vstack([x[s] for s in x_slices])
            new_controls = np.vstack([x[s] for s in u_slices])
            new_sigma = float(x[sigma_index]) if self.free_final_time else sigma
            virtual_l1 = float(sum(np.sum(np.abs(x[s])) for s in nu_slices))
            trust_norm_value = float(np.linalg.norm(x[trust]))
            if self.state_projection is not None:
                new_states = np.vstack([self.state_projection(row) for row in new_states])
            _, defect = self.replay(new_states, new_controls, new_sigma)
            history.append(
                SCvxIteration(
                    iteration=iteration,
                    sigma=new_sigma,
                    objective=solution.objective,
                    virtual_l1=virtual_l1,
                    trust_norm=trust_norm_value,
                    solver_status=solution.status,
                    solver_iterations=solution.iterations,
                    solve_seconds=solution.solve_seconds,
                    discretisation_seconds=discretisation_seconds,
                    replay_defect_inf=defect,
                )
            )
            states, controls, sigma = new_states, new_controls, new_sigma
            if (
                trust_norm_value <= self.settings.trust_tolerance
                and virtual_l1 <= self.settings.virtual_tolerance
            ):
                status = "converged"
                break

        replayed, defect = self.replay(states, controls, sigma)
        terminal_error = float(np.max(np.abs(replayed[-1] - states[-1])))
        violations = self.path_check(states, controls, sigma) if self.path_check is not None else {}
        return SCvxOutcome(
            status=status,
            states=states,
            controls=controls,
            sigma=sigma,
            iterations=history,
            replay_defect_inf=defect,
            replay_terminal_error_inf=terminal_error,
            replay_states=replayed,
            path_violations=violations,
        )
