"""Heliocentric low-thrust rendezvous arcs by successive convexification (CPU reference).

One leg joins a departure body state at ``t0`` to an arrival body state at ``tf`` with the GTOC12
ship model (``T <= 0.6 N``, ``Isp = 4000 s``, mass flow ``T / (Isp g0)``).  Earth ends may carry a
free hyperbolic excess velocity bounded by 6 km/s.

Transcription
-------------
* Nodes are one day apart (last interval partial) so the emitted samples *are* the solution file.
* Between nodes the thrust is the cubic Lagrange interpolant of the nodal samples over the same
  clamped four-node stencil the official verifier uses; the discrete dynamics therefore depend on
  four nodal controls per interval and the certified rollout reproduces the verifier's model.
* Linearisation integrates the state, the state-transition matrix and the four control
  sensitivities with a batched fixed-step RK4 over every interval simultaneously.
* The convex subproblem (Clarabel) maximises final mass with an L1 exact penalty on virtual control,
  the lossless ``|T| <= Gamma <= T_max`` relaxation, a linearised ``r >= 0.3 AU`` half-space and
  box trust regions updated by the usual SCvx reduction-ratio rule.

Units inside the solver: AU, TU = sqrt(AU^3 / mu) (~58.13 d), mass / m0, thrust / T_max.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from . import constants as C

FloatArray = NDArray[np.float64]

DU_KM = C.AU_KM
TU_S = math.sqrt(C.AU_KM**3 / C.MU_SUN_KM3_S2)
VU_KM_S = DU_KM / TU_S
ACC_UNIT_M_S2 = DU_KM * 1e3 / TU_S**2
STENCIL = C.THRUST_INTERPOLATION_ORDER + 1  # four nodal controls influence one interval
# Segments below this magnitude are emitted as coasts (the file prints 14 decimals of newtons; a
# nanonewton over a day moves the ship by micrometres).
COAST_THRUST_N = 1.0e-9


@dataclass(frozen=True, slots=True)
class LegBoundary:
    """Physical boundary data for one leg (km, km/s, kg, MJD)."""

    departure_epoch: float
    departure_position: FloatArray
    departure_velocity: FloatArray  # body velocity
    arrival_epoch: float
    arrival_position: FloatArray
    arrival_velocity: FloatArray  # body velocity
    initial_mass: float
    free_departure_vinf: bool = False
    free_arrival_vinf: bool = False
    minimum_final_mass: float = C.DRY_MASS_KG

    @property
    def duration_days(self) -> float:
        return self.arrival_epoch - self.departure_epoch


@dataclass(frozen=True, slots=True)
class ScvxSettings:
    node_days: float = 2.0
    hold: str = "zoh"  # "zoh" (piecewise-constant arcs, verifier-exact) or "lagrange"
    substeps: int = 8
    polish_substeps: int = 16
    polish_iterations: int = 4
    max_iterations: int = 40
    virtual_weight: float = 1.0e4
    smoothness_weight: float = 0.0
    initial_trust_state: float = 0.2  # scaled state units (AU, AU/TU, mass fraction)
    initial_trust_control: float = 1.0
    minimum_trust: float = 1.0e-6
    maximum_trust: float = 2.0
    ratio_reject: float = 0.0
    ratio_shrink: float = 0.25
    ratio_grow: float = 0.7
    shrink_factor: float = 0.5
    grow_factor: float = 1.6
    defect_tolerance: float = 5.0e-9  # scaled per-interval nonlinear defect (inf norm)
    step_tolerance: float = 1.0e-7
    objective_tolerance: float = 1.0e-6  # predicted merit reduction below this = converged
    clarabel_tolerance: float = 1.0e-9
    time_limit_s: float = 900.0


@dataclass(slots=True)
class LegSolution:
    status: str  # converged | iteration_limit | infeasible | failed | timeout
    boundary: LegBoundary
    node_epochs_mjd: FloatArray
    thrust_n: FloatArray  # (nodes, 3)
    states_scaled: FloatArray  # (nodes, 7)
    departure_vinf_km_s: FloatArray
    arrival_vinf_km_s: FloatArray
    final_mass_kg: float
    propellant_kg: float
    delta_v_km_s: float
    iterations: int
    accepted_iterations: int
    max_defect: float
    virtual_inf: float
    solve_seconds: float
    hold: str = "zoh"
    history: list[dict[str, float]] = field(default_factory=list)
    diagnostic: str = ""

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    def burn_arcs(self, sample_days: float = 1.0) -> list:
        """Official-format burn arcs for this leg (one arc per constant ZOH segment, or one
        arc with the nodal samples for the cubic hold)."""

        from .solution import make_burn_arc

        if self.hold != "zoh":
            return [make_burn_arc(self.node_epochs_mjd, self.thrust_n)]
        arcs = []
        for k in range(self.node_epochs_mjd.shape[0] - 1):
            thrust = self.thrust_n[k]
            if float(np.linalg.norm(thrust)) <= COAST_THRUST_N:
                continue  # coast segment: the verifier propagates it analytically
            t0, t1 = float(self.node_epochs_mjd[k]), float(self.node_epochs_mjd[k + 1])
            samples = np.arange(t0, t1, sample_days)
            samples = np.unique(np.concatenate((samples, [t1])))
            if samples.shape[0] < 2:
                samples = np.array([t0, t1])
            arcs.append(
                make_burn_arc(samples, np.repeat(thrust[None, :], samples.shape[0], axis=0))
            )
        return arcs

    def departure_ship_velocity_km_s(self) -> FloatArray:
        return self.boundary.departure_velocity + self.departure_vinf_km_s

    def arrival_ship_velocity_km_s(self) -> FloatArray:
        return self.boundary.arrival_velocity + self.arrival_vinf_km_s


class _Model:
    def __init__(self, initial_mass_kg: float) -> None:
        self.kappa = C.THRUST_MAX_N / (initial_mass_kg * ACC_UNIT_M_S2)
        self.lam = C.THRUST_MAX_N * TU_S / (C.ISP_S * C.G0_M_S2 * initial_mass_kg)

    # The verifier's mass flow is proportional to |T(t)| of the *interpolated thrust vector*, so
    # the nonlinear model uses the same quantity; the cone slack Gamma only bounds |T_k| <= T_max.
    THRUST_EPSILON = 1.0e-9

    def f(self, x: FloatArray, u: FloatArray) -> FloatArray:
        r = x[:, 0:3]
        radius = np.linalg.norm(r, axis=1)
        out = np.empty_like(x)
        out[:, 0:3] = x[:, 3:6]
        out[:, 3:6] = -r / radius[:, None] ** 3 + self.kappa * u[:, 0:3] / x[:, 6:7]
        out[:, 6] = -self.lam * np.linalg.norm(u[:, 0:3], axis=1)
        return out

    def jacobians(self, x: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        n = x.shape[0]
        r = x[:, 0:3]
        radius = np.linalg.norm(r, axis=1)
        m = x[:, 6]
        a = np.zeros((n, 7, 7))
        b = np.zeros((n, 7, 4))
        a[:, 0:3, 3:6] = np.eye(3)
        eye = np.eye(3)[None, :, :]
        outer = np.einsum("ni,nj->nij", r, r)
        a[:, 3:6, 0:3] = (
            -eye / radius[:, None, None] ** 3 + 3.0 * outer / radius[:, None, None] ** 5
        )
        a[:, 3:6, 6] = -self.kappa * u[:, 0:3] / (m * m)[:, None]
        b[:, 3:6, 0:3] = self.kappa * eye / m[:, None, None]
        # Linearise the mass flow through the cone slack Gamma (|T| <= Gamma).  This is the
        # lossless convex surrogate: exact wherever Gamma = |T| (true at every optimum) and, unlike
        # the |T| gradient, well defined at a coasting reference.
        b[:, 6, 3] = -self.lam
        return a, b


def _lagrange_weights(times: FloatArray, nodes: FloatArray) -> FloatArray:
    """Basis weights ``(M, S)`` of the S-node Lagrange interpolant at ``times`` (M,)."""

    m, s = nodes.shape
    weights = np.ones((m, s))
    for j in range(s):
        for k in range(s):
            if k != j:
                weights[:, j] *= (times - nodes[:, k]) / (nodes[:, j] - nodes[:, k])
    return weights


def stencil_indices(interval: int, node_count: int) -> np.ndarray:
    start = min(max(interval - 1, 0), node_count - STENCIL)
    return np.arange(start, start + STENCIL)


class _Discretisation:
    """Batched RK4 linearisation of every interval around a reference trajectory.

    ``hold="zoh"`` keeps the control constant over each interval (the emitted file then repeats
    the value at every daily sample, which the verifier's cubic interpolation reproduces exactly);
    ``hold="lagrange"`` models the verifier's four-node cubic stencil directly.
    """

    def __init__(
        self, model: _Model, node_times: FloatArray, substeps: int, hold: str = "zoh"
    ) -> None:
        self.model = model
        self.node_times = node_times
        self.substeps = substeps
        self.hold = hold
        self.intervals = node_times.shape[0] - 1
        if hold == "zoh":
            self.stencils = np.arange(self.intervals)[:, None]
        elif hold == "lagrange":
            self.stencils = np.stack(
                [stencil_indices(k, node_times.shape[0]) for k in range(self.intervals)]
            )
        else:
            raise ValueError("hold must be 'zoh' or 'lagrange'")

    def _controls_at(
        self, times: FloatArray, controls: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        u_nodes = controls[self.stencils]  # (N, S, 4)
        if self.hold == "zoh":
            weights = np.ones((self.intervals, 1))
            return u_nodes[:, 0, :], weights
        nodes = self.node_times[self.stencils]  # (N, S)
        weights = _lagrange_weights(times, nodes)  # (N, S)
        return np.einsum("ns,nsj->nj", weights, u_nodes), weights

    def _augmented_rhs(
        self, tau: FloatArray, x: FloatArray, phi: FloatArray, psi: FloatArray, controls: FloatArray
    ):
        u, weights = self._controls_at(tau, controls)
        f = self.model.f(x, u)
        a, b = self.model.jacobians(x, u)
        dphi = np.einsum("nij,njk->nik", a, phi)
        dpsi = np.einsum("nij,nsjk->nsik", a, psi) + weights[:, :, None, None] * b[:, None, :, :]
        return f, dphi, dpsi

    def linearise(self, states: FloatArray, controls: FloatArray):
        """Return ``A (N,7,7)``, ``B (N,S,7,4)``, ``c (N,7)`` and the propagated ``x_{k+1}``."""

        n = self.intervals
        x = states[:-1].copy()
        phi = np.tile(np.eye(7), (n, 1, 1))
        psi = np.zeros((n, self.stencils.shape[1], 7, 4))
        t = self.node_times[:-1].copy()
        h = (self.node_times[1:] - self.node_times[:-1]) / self.substeps
        for _ in range(self.substeps):
            k1 = self._augmented_rhs(t, x, phi, psi, controls)
            k2 = self._augmented_rhs(
                t + 0.5 * h,
                x + 0.5 * h[:, None] * k1[0],
                phi + 0.5 * h[:, None, None] * k1[1],
                psi + 0.5 * h[:, None, None, None] * k1[2],
                controls,
            )
            k3 = self._augmented_rhs(
                t + 0.5 * h,
                x + 0.5 * h[:, None] * k2[0],
                phi + 0.5 * h[:, None, None] * k2[1],
                psi + 0.5 * h[:, None, None, None] * k2[2],
                controls,
            )
            k4 = self._augmented_rhs(
                t + h,
                x + h[:, None] * k3[0],
                phi + h[:, None, None] * k3[1],
                psi + h[:, None, None, None] * k3[2],
                controls,
            )
            x = x + h[:, None] / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            phi = phi + h[:, None, None] / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            psi = psi + h[:, None, None, None] / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            t = t + h
        u_nodes = controls[self.stencils]  # (N, S, 4)
        c = x - np.einsum("nij,nj->ni", phi, states[:-1]) - np.einsum("nsij,nsj->ni", psi, u_nodes)
        return phi, psi, c, x

    def propagate(self, states: FloatArray, controls: FloatArray) -> FloatArray:
        """Nonlinear RK4 propagation of every interval from its own reference start state."""

        x = states[:-1].copy()
        t = self.node_times[:-1].copy()
        h = (self.node_times[1:] - self.node_times[:-1]) / self.substeps
        for _ in range(self.substeps):
            u1, _ = self._controls_at(t, controls)
            k1 = self.model.f(x, u1)
            u2, _ = self._controls_at(t + 0.5 * h, controls)
            k2 = self.model.f(x + 0.5 * h[:, None] * k1, u2)
            k3 = self.model.f(x + 0.5 * h[:, None] * k2, u2)
            u4, _ = self._controls_at(t + h, controls)
            k4 = self.model.f(x + h[:, None] * k3, u4)
            x = x + h[:, None] / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + h
        return x

    def rollout(self, x0: FloatArray, controls: FloatArray) -> FloatArray:
        """Sequential nonlinear rollout from ``x0`` (returns all node states)."""

        states = np.zeros((self.intervals + 1, 7))
        states[0] = x0
        for k in range(self.intervals):
            states[k + 1] = _propagate_single(self, k, states[k], controls)
        return states


def _propagate_single(
    disc: _Discretisation, k: int, x0: FloatArray, controls: FloatArray
) -> FloatArray:
    x = x0[None, :].copy()
    t = disc.node_times[k : k + 1].copy()
    h = (disc.node_times[k + 1] - disc.node_times[k]) / disc.substeps
    nodes = disc.node_times[disc.stencils[k]][None, :]
    u_nodes = controls[disc.stencils[k]][None, :, :]

    def control(tau):
        if disc.hold == "zoh":
            return u_nodes[:, 0, :]
        w = _lagrange_weights(tau, nodes)
        return np.einsum("ns,nsj->nj", w, u_nodes)

    for _ in range(disc.substeps):
        k1 = disc.model.f(x, control(t))
        um = control(t + 0.5 * h)
        k2 = disc.model.f(x + 0.5 * h * k1, um)
        k3 = disc.model.f(x + 0.5 * h * k2, um)
        k4 = disc.model.f(x + h * k3, control(t + h))
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
    return x[0]


class _ConvexProblem:
    """Fixed-pattern Clarabel subproblem builder for one leg (pattern reused every iteration)."""

    def __init__(self, nodes: int, free_dep: bool, free_arr: bool) -> None:
        self.nodes = nodes
        self.intervals = nodes - 1
        self.free_dep = free_dep
        self.free_arr = free_arr
        n, k = nodes, self.intervals
        self.ix = 0
        self.iu = self.ix + 7 * n
        self.inu = self.iu + 4 * n
        self.isl = self.inu + 7 * k
        self.ivd = self.isl + 7 * k
        self.iva = self.ivd + (3 if free_dep else 0)
        self.n_variables = self.iva + (3 if free_arr else 0)

    def x(self, node: int, component: int | None = None) -> int:
        return self.ix + 7 * node + (component or 0)

    def u(self, node: int, component: int | None = None) -> int:
        return self.iu + 4 * node + (component or 0)

    def nu(self, interval: int, component: int) -> int:
        return self.inu + 7 * interval + component

    def slack(self, interval: int, component: int) -> int:
        return self.isl + 7 * interval + component

    def build(
        self,
        phi: FloatArray,
        psi: FloatArray,
        c: FloatArray,
        stencils: np.ndarray,
        ref_states: FloatArray,
        ref_controls: FloatArray,
        boundary: dict[str, FloatArray],
        trust_state: float,
        trust_control: float,
        virtual_weight: float,
        minimum_mass: float,
        radius_floor: float,
        vinf_max: float,
        fuel_weights: FloatArray,
        smoothness_weight: float,
        zero_last_control: bool = False,
    ):
        n, k = self.nodes, self.intervals
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        b: list[float] = []
        row = 0

        def add(r: int, col: int, value: float) -> None:
            rows.append(r)
            cols.append(col)
            vals.append(value)

        # --- equalities (zero cone) ---
        for interval in range(k):
            for i in range(7):
                add(row, self.x(interval + 1, i), 1.0)
                for j in range(7):
                    add(row, self.x(interval, j), -phi[interval, i, j])
                for s_index, node in enumerate(stencils[interval]):
                    for j in range(4):
                        value = psi[interval, s_index, i, j]
                        if value != 0.0:
                            add(row, self.u(int(node), j), -value)
                add(row, self.nu(interval, i), -1.0)
                b.append(c[interval, i])
                row += 1
        for i in range(3):
            add(row, self.x(0, i), 1.0)
            b.append(boundary["r0"][i])
            row += 1
        for i in range(3):
            add(row, self.x(0, 3 + i), 1.0)
            if self.free_dep:
                add(row, self.ivd + i, -1.0)
            b.append(boundary["v0"][i])
            row += 1
        add(row, self.x(0, 6), 1.0)
        b.append(1.0)
        row += 1
        for i in range(3):
            add(row, self.x(n - 1, i), 1.0)
            b.append(boundary["rf"][i])
            row += 1
        for i in range(3):
            add(row, self.x(n - 1, 3 + i), 1.0)
            if self.free_arr:
                add(row, self.iva + i, -1.0)
            b.append(boundary["vf"][i])
            row += 1
        if zero_last_control:  # ZOH: the control at the final node drives nothing
            for i in range(4):
                add(row, self.u(n - 1, i), 1.0)
                b.append(0.0)
                row += 1
        n_equalities = row

        # --- inequalities (nonnegative cone): b - A x >= 0 ---
        for node in range(n):
            add(row, self.u(node, 3), -1.0)  # Gamma >= 0
            b.append(0.0)
            row += 1
            add(row, self.u(node, 3), 1.0)  # Gamma <= 1
            b.append(1.0)
            row += 1
            add(row, self.x(node, 6), -1.0)  # m >= m_min
            b.append(-minimum_mass)
            row += 1
            # linearised minimum solar distance: rhat . r >= radius_floor
            ref_r = ref_states[node, 0:3]
            norm = float(np.linalg.norm(ref_r))
            for i in range(3):
                add(row, self.x(node, i), -ref_r[i] / norm)
            b.append(-radius_floor)
            row += 1
            for i in range(7):  # trust region on states
                add(row, self.x(node, i), 1.0)
                b.append(ref_states[node, i] + trust_state)
                row += 1
                add(row, self.x(node, i), -1.0)
                b.append(-ref_states[node, i] + trust_state)
                row += 1
            for i in range(4):  # trust region on controls
                add(row, self.u(node, i), 1.0)
                b.append(ref_controls[node, i] + trust_control)
                row += 1
                add(row, self.u(node, i), -1.0)
                b.append(-ref_controls[node, i] + trust_control)
                row += 1
        for interval in range(k):
            for i in range(7):  # slack >= |nu|
                add(row, self.nu(interval, i), 1.0)
                add(row, self.slack(interval, i), -1.0)
                b.append(0.0)
                row += 1
                add(row, self.nu(interval, i), -1.0)
                add(row, self.slack(interval, i), -1.0)
                b.append(0.0)
                row += 1
        n_inequalities = row - n_equalities

        # --- second-order cones ---
        cones_soc: list[int] = []
        for node in range(n):
            add(row, self.u(node, 3), -1.0)
            b.append(0.0)
            row += 1
            for i in range(3):
                add(row, self.u(node, i), -1.0)
                b.append(0.0)
                row += 1
            cones_soc.append(4)
        for flag, offset in ((self.free_dep, self.ivd), (self.free_arr, self.iva)):
            if flag:
                b.append(vinf_max)
                row += 1
                for i in range(3):
                    add(row, offset + i, -1.0)
                    b.append(0.0)
                    row += 1
                cones_soc.append(4)

        a_matrix = sp.csc_matrix(
            (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
            shape=(row, self.n_variables),
        )
        # Convex fuel objective (lossless form): propellant = lam * sum Gamma_k * w_k with
        # trapezoidal node weights; the exact L1 penalty on virtual control keeps it feasible.
        q = np.zeros(self.n_variables)
        for node in range(n):
            q[self.u(node, 3)] = fuel_weights[node]
        q[self.isl : self.isl + 7 * k] = virtual_weight
        # Quadratic control-smoothness term: damps cubic-interpolation overshoot at switches so the
        # verifier's |T(t)| stays smooth and the emitted profile is robust to its stencil.
        d_rows: list[int] = []
        d_cols: list[int] = []
        d_vals: list[float] = []
        for node in range(n - 1):
            for i in range(3):
                d_rows.extend([3 * node + i, 3 * node + i])
                d_cols.extend([self.u(node + 1, i), self.u(node, i)])
                d_vals.extend([1.0, -1.0])
        difference = sp.csc_matrix(
            (d_vals, (d_rows, d_cols)), shape=(3 * (n - 1), self.n_variables)
        )
        p_matrix = (2.0 * smoothness_weight) * (difference.T @ difference)
        import clarabel

        cones = [clarabel.ZeroConeT(n_equalities), clarabel.NonnegativeConeT(n_inequalities)]
        cones.extend(clarabel.SecondOrderConeT(size) for size in cones_soc)
        return a_matrix, np.asarray(b), q, cones, sp.triu(p_matrix).tocsc()


def _clarabel_solve(a_matrix, b, q, cones, tolerance: float, p_matrix):
    import clarabel

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = tolerance
    settings.tol_gap_rel = tolerance
    settings.tol_feas = tolerance
    settings.max_iter = 200
    solver = clarabel.DefaultSolver(p_matrix, q, a_matrix, b, cones, settings)
    result = solver.solve()
    status = str(result.status)
    ok = status in {"Solved", "AlmostSolved"}
    return ok, status, np.asarray(result.x, dtype=np.float64)


def _ballistic_reference(
    boundary: LegBoundary, node_times: FloatArray, scale_mass: float
) -> tuple[FloatArray, FloatArray]:
    """Initial guess: zero-revolution Lambert arc sampled at the nodes, zero thrust."""

    from .ephemeris import propagate_kepler
    from .lambert import lambert_batch

    tof_s = boundary.duration_days * C.DAY_S
    best = None
    for long_way in (False, True):
        result = lambert_batch(
            boundary.departure_position,
            boundary.arrival_position,
            np.array([tof_s]),
            long_way=long_way,
        )
        if not result.feasible[0]:
            continue
        cost = np.linalg.norm(
            result.departure_velocity[0] - boundary.departure_velocity
        ) + np.linalg.norm(result.arrival_velocity[0] - boundary.arrival_velocity)
        if best is None or cost < best[0]:
            best = (cost, result.departure_velocity[0])
    states = np.zeros((node_times.shape[0], 7))
    if best is None:
        # straight interpolation fallback in position/velocity space
        alpha = (node_times - node_times[0]) / (node_times[-1] - node_times[0])
        states[:, 0:3] = (1 - alpha)[:, None] * boundary.departure_position / DU_KM + alpha[
            :, None
        ] * boundary.arrival_position / DU_KM
        states[:, 3:6] = (1 - alpha)[:, None] * boundary.departure_velocity / VU_KM_S + alpha[
            :, None
        ] * boundary.arrival_velocity / VU_KM_S
    else:
        dt = (node_times - node_times[0]) * TU_S
        r, v = propagate_kepler(
            np.repeat(boundary.departure_position[None, :], node_times.shape[0], 0),
            np.repeat(best[1][None, :], node_times.shape[0], 0),
            dt,
        )
        states[:, 0:3] = r / DU_KM
        states[:, 3:6] = v / VU_KM_S
    states[:, 6] = 1.0
    # Make the reference satisfy the boundary equalities exactly (the first convex subproblem
    # otherwise has to bridge the Lambert velocity mismatch inside one trust region).  Free Earth
    # ends keep the Lambert v-infinity direction clipped to the 6 km/s allowance.
    vinf_cap = C.MAX_VINF_EARTH_KM_S / VU_KM_S

    def clip(reference: FloatArray, body: FloatArray, free: bool) -> FloatArray:
        if not free:
            return body
        offset = reference - body
        norm = float(np.linalg.norm(offset))
        if norm <= vinf_cap or norm == 0.0:
            return reference
        return body + offset * (0.98 * vinf_cap / norm)

    states[0, 3:6] = clip(
        states[0, 3:6], boundary.departure_velocity / VU_KM_S, boundary.free_departure_vinf
    )
    states[-1, 3:6] = clip(
        states[-1, 3:6], boundary.arrival_velocity / VU_KM_S, boundary.free_arrival_vinf
    )
    states[0, 0:3] = boundary.departure_position / DU_KM
    states[-1, 0:3] = boundary.arrival_position / DU_KM
    controls = np.zeros((node_times.shape[0], 4))
    controls[:, 3] = 1.0e-3
    return states, controls


def solve_leg(boundary: LegBoundary, settings: ScvxSettings | None = None) -> LegSolution:
    """Run SCvx on one leg and return the nodal thrust samples plus diagnostics."""

    settings = settings or ScvxSettings()
    started = time.perf_counter()
    duration_days = boundary.duration_days
    if duration_days <= 0.0:
        raise ValueError("leg duration must be positive")
    whole = math.floor(duration_days / settings.node_days + 1e-9)
    node_days = np.arange(whole + 1) * settings.node_days
    if duration_days - node_days[-1] > 1e-9:
        node_days = np.append(node_days, duration_days)
    if node_days.shape[0] < STENCIL:
        node_days = np.linspace(0.0, duration_days, STENCIL)
    node_times = node_days * C.DAY_S / TU_S
    nodes = node_times.shape[0]
    model = _Model(boundary.initial_mass)
    disc = _Discretisation(model, node_times, settings.substeps, settings.hold)
    zoh = settings.hold == "zoh"
    problem = _ConvexProblem(nodes, boundary.free_departure_vinf, boundary.free_arrival_vinf)
    bnd = {
        "r0": boundary.departure_position / DU_KM,
        "v0": boundary.departure_velocity / VU_KM_S,
        "rf": boundary.arrival_position / DU_KM,
        "vf": boundary.arrival_velocity / VU_KM_S,
    }
    minimum_mass = boundary.minimum_final_mass / boundary.initial_mass
    vinf_max = C.MAX_VINF_EARTH_KM_S / VU_KM_S
    radius_floor = C.MIN_SUN_DISTANCE_AU

    states, controls = _ballistic_reference(boundary, node_times, boundary.initial_mass)
    trust_state = settings.initial_trust_state
    trust_control = settings.initial_trust_control
    history: list[dict[str, float]] = []
    # weights so that lam * sum(w_k Gamma_k) is the propellant fraction: interval lengths for
    # ZOH (the last node drives nothing), trapezoidal for the cubic hold
    interval_lengths = np.diff(node_times)
    fuel_weights = np.zeros(nodes)
    if zoh:
        fuel_weights[:-1] = interval_lengths
    else:
        fuel_weights[:-1] += 0.5 * interval_lengths
        fuel_weights[1:] += 0.5 * interval_lengths
    fuel_weights *= model.lam

    def fuel(ct: FloatArray) -> float:
        return float(np.dot(fuel_weights, ct[:, 3]))

    def merit(st: FloatArray, ct: FloatArray) -> tuple[float, float]:
        propagated = disc.propagate(st, ct)
        defect = st[1:] - propagated
        # Defects below the convex solver tolerance are numerical noise; a dead zone keeps the
        # exact-penalty merit from rejecting genuine improvements near convergence.
        penalty = float(np.sum(np.maximum(np.abs(defect) - settings.clarabel_tolerance, 0.0)))
        return fuel(ct) + settings.virtual_weight * penalty, float(np.max(np.abs(defect)))

    current_merit, current_defect = merit(states, controls)
    status = "iteration_limit"
    accepted = 0
    iterations = 0
    diagnostic = ""
    vinf_dep = np.zeros(3)
    vinf_arr = np.zeros(3)
    last_virtual = math.inf
    polishing = False
    polish_left = settings.polish_iterations
    total_budget = settings.max_iterations + settings.polish_iterations
    for iteration in range(total_budget):
        if time.perf_counter() - started > settings.time_limit_s:
            status = "timeout"
            break
        if polishing and polish_left <= 0:
            break
        iterations = iteration + 1
        phi, psi, c, _ = disc.linearise(states, controls)
        a_matrix, b, q, cones, p_matrix = problem.build(
            phi,
            psi,
            c,
            disc.stencils,
            states,
            controls,
            bnd,
            trust_state,
            trust_control,
            settings.virtual_weight,
            minimum_mass,
            radius_floor,
            vinf_max,
            fuel_weights,
            settings.smoothness_weight,
            zero_last_control=zoh,
        )
        ok, solver_status, x = _clarabel_solve(
            a_matrix, b, q, cones, settings.clarabel_tolerance, p_matrix
        )
        if not ok or not np.all(np.isfinite(x)):
            trust_state *= settings.shrink_factor
            trust_control *= settings.shrink_factor
            history.append({"iteration": iterations, "solver": solver_status, "accepted": 0.0})
            if trust_state < settings.minimum_trust:
                status = "failed"
                diagnostic = f"convex subproblem {solver_status}"
                break
            continue
        new_states = x[problem.ix : problem.iu].reshape(nodes, 7)
        new_controls = x[problem.iu : problem.inu].reshape(nodes, 4)
        virtual = x[problem.inu : problem.isl].reshape(nodes - 1, 7)
        linear_merit = fuel(new_controls) + settings.virtual_weight * float(np.sum(np.abs(virtual)))
        new_merit, new_defect = merit(new_states, new_controls)
        if polishing:
            polish_left -= 1
        predicted = current_merit - linear_merit
        actual = current_merit - new_merit
        ratio = actual / predicted if predicted > 1e-15 else (1.0 if actual >= 0.0 else -1.0)
        step = float(
            max(np.max(np.abs(new_states - states)), np.max(np.abs(new_controls - controls)))
        )
        record = {
            "iteration": iterations,
            "merit": new_merit,
            "final_mass_fraction": float(new_states[-1, 6]),
            "max_defect": new_defect,
            "virtual_inf": float(np.max(np.abs(virtual))),
            "ratio": ratio,
            "step": step,
            "trust_state": trust_state,
            "trust_control": trust_control,
        }
        if ratio < settings.ratio_reject:
            trust_state = max(trust_state * settings.shrink_factor, settings.minimum_trust)
            trust_control = max(trust_control * settings.shrink_factor, settings.minimum_trust)
            record["accepted"] = 0.0
            history.append(record)
            if trust_state <= settings.minimum_trust and trust_control <= settings.minimum_trust:
                status = "failed"
                diagnostic = "trust region collapsed"
                break
            continue
        # accept
        accepted += 1
        states, controls = new_states, new_controls
        current_merit, current_defect = new_merit, new_defect
        last_virtual = float(np.max(np.abs(virtual)))
        if problem.free_dep:
            vinf_dep = x[problem.ivd : problem.ivd + 3] * VU_KM_S
        if problem.free_arr:
            vinf_arr = x[problem.iva : problem.iva + 3] * VU_KM_S
        record["accepted"] = 1.0
        history.append(record)
        if ratio < settings.ratio_shrink:
            trust_state = max(trust_state * settings.shrink_factor, settings.minimum_trust)
            trust_control = max(trust_control * settings.shrink_factor, settings.minimum_trust)
        elif ratio > settings.ratio_grow:
            trust_state = min(trust_state * settings.grow_factor, settings.maximum_trust)
            trust_control = min(trust_control * settings.grow_factor, settings.maximum_trust)
        feasible_now = (
            current_defect <= settings.defect_tolerance
            and last_virtual <= settings.defect_tolerance * 10.0
        )
        if feasible_now and (
            step <= settings.step_tolerance or predicted <= settings.objective_tolerance
        ):
            if (
                polishing
                or settings.polish_iterations == 0
                or settings.polish_substeps <= settings.substeps
            ):
                status = "converged"
                break
            # Polish: re-linearise with a finer integrator so the discrete model matches the
            # verifier's adaptive RKF78 through the kinks of |T(t)| at thrust switches.
            polishing = True
            disc = _Discretisation(model, node_times, settings.polish_substeps, settings.hold)
            current_merit, current_defect = merit(states, controls)
            trust_state = max(trust_state, 1e-3)
            trust_control = max(trust_control, 1e-2)
            continue
        if iterations >= settings.max_iterations and not polishing:
            break
    if (
        polishing
        and status == "iteration_limit"
        and current_defect <= settings.defect_tolerance * 10.0
        and last_virtual <= settings.defect_tolerance * 100.0
    ):
        status = "converged"
        diagnostic = "polish budget reached with converged defects"
    if (
        status == "iteration_limit"
        and current_defect <= settings.defect_tolerance * 10.0
        and last_virtual <= settings.defect_tolerance * 100.0
    ):
        status = "converged"
        diagnostic = "iteration budget reached with converged defects"
    if (
        status == "failed"
        and current_defect <= settings.defect_tolerance * 10.0
        and last_virtual <= settings.defect_tolerance * 100.0
    ):
        status = "converged"
        diagnostic = "trust region exhausted at a feasible point"
    if status in {"iteration_limit", "failed"} and last_virtual > 1e-4:
        status = "infeasible"
        diagnostic = f"virtual control remains {last_virtual:.3e}"
    thrust_n = controls[:, 0:3] * C.THRUST_MAX_N
    if zoh:
        thrust_n[-1] = 0.0
    final_mass = float(states[-1, 6]) * boundary.initial_mass
    delta_v = (
        C.ISP_S * C.G0_M_S2 * 1e-3 * math.log(boundary.initial_mass / final_mass)
        if final_mass > 0
        else math.inf
    )
    return LegSolution(
        status=status,
        boundary=boundary,
        node_epochs_mjd=boundary.departure_epoch + node_days,
        thrust_n=thrust_n,
        hold=settings.hold,
        states_scaled=states,
        departure_vinf_km_s=vinf_dep,
        arrival_vinf_km_s=vinf_arr,
        final_mass_kg=final_mass,
        propellant_kg=boundary.initial_mass - final_mass,
        delta_v_km_s=delta_v,
        iterations=iterations,
        accepted_iterations=accepted,
        max_defect=current_defect,
        virtual_inf=last_virtual,
        solve_seconds=time.perf_counter() - started,
        history=history,
        diagnostic=diagnostic,
    )


@dataclass(frozen=True, slots=True)
class LegCertificate:
    """Verifier-model rollout of a leg: endpoint errors against the arrival body state."""

    position_error_km: float
    velocity_error_km_s: float
    final_mass_kg: float
    minimum_sun_distance_au: float
    maximum_thrust_n: float
    rk4_vs_dop853_km: float

    @property
    def within_tolerance(self) -> bool:
        return (
            self.position_error_km <= C.TOLERANCE_POSITION_KM
            and self.velocity_error_km_s <= C.TOLERANCE_VELOCITY_KM_S
            and self.minimum_sun_distance_au >= C.MIN_SUN_DISTANCE_AU
            and self.maximum_thrust_n <= C.THRUST_MAX_N + 1e-9
        )


def certify_leg(solution: LegSolution) -> LegCertificate:
    """Independently propagate the emitted samples with the verifier model (DOP853, cubic)."""

    from .verifier import propagate_burn, propagate_coast

    boundary = solution.boundary
    r = boundary.departure_position.copy()
    v = solution.departure_ship_velocity_km_s().copy()
    m = boundary.initial_mass
    epoch = boundary.departure_epoch
    min_radius = float(np.linalg.norm(r))
    for arc in solution.burn_arcs():
        if arc.start > epoch:
            r, v, radius = propagate_coast(epoch, r, v, m, arc.start)
            min_radius = min(min_radius, radius)
            epoch = arc.start
        r, v, m, radius = propagate_burn(epoch, r, v, m, arc, rtol=1e-12)
        min_radius = min(min_radius, radius)
        epoch = arc.end
    if boundary.arrival_epoch > epoch:
        r, v, radius = propagate_coast(epoch, r, v, m, boundary.arrival_epoch)
        min_radius = min(min_radius, radius)
    target_v = solution.arrival_ship_velocity_km_s()
    rk4_end = solution.states_scaled[-1, 0:3] * DU_KM
    return LegCertificate(
        position_error_km=float(np.linalg.norm(r - boundary.arrival_position)),
        velocity_error_km_s=float(np.linalg.norm(v - target_v)),
        final_mass_kg=m,
        minimum_sun_distance_au=min_radius / C.AU_KM,
        maximum_thrust_n=float(np.max(np.linalg.norm(solution.thrust_n, axis=1))),
        rk4_vs_dop853_km=float(np.linalg.norm(r - rk4_end)),
    )
