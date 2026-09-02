"""P1-C literature profile: Acikmese & Ploen (2007) Mars 3-DoF powered descent.

Primary source: B. Acikmese and S. R. Ploen, "Convex Programming Approach to Powered Descent
Guidance for Mars Landing", JGCD 30(5), 2007, DOI 10.2514/1.27553.  The journal text is not
open access; the numerical example is pinned through two independent open secondary sources
that reproduce it (see ``pinned_values.py``):

* A. Wenzel, DLR master thesis (elib.dlr.de/118732): parameter table and the 2007 fuel
  values 387.9 kg (t_f = 72 s, no glide slope) and 399.5 kg (t_f = 81 s, 4 deg glide slope);
* Blackmore, Acikmese, Scharf, JGCD 33(4) 2010, DOI 10.2514/1.47202 (open PDF): the same
  vehicle constants (eq. 72) and a fully specified case 1 with 399.4 kg at t_f = 78.4 s.

Three reproductions are run:

1. ``lossless`` - an independent implementation of the paper's own lossless-convexification
   SOCP (log-mass change of variables, exact zero-order-hold discretisation) with Clarabel;
2. ``scvx_cpu`` - the repository CPU reference SCvx (forward-Euler transcription, Clarabel);
3. ``scvx_qoco_gpu`` - the same SCvx driven by the persistent pure-QOCO GPU backend when a
   library is available.

Every trajectory is replayed independently with the nonlinear mass-varying dynamics.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.literature._socp import LinearExpression, SOCPBuilder, lin

FloatArray = NDArray[np.float64]

STANDARD_GRAVITY = 9.807  # m/s^2, value used by the 2007 paper (g_e)


@dataclass(frozen=True, slots=True)
class MarsDescentProfile:
    """Physical constants and boundary conditions (paper frame: x up, y/z horizontal)."""

    gravity_mars: float = 3.7114
    wet_mass: float = 1905.0
    dry_mass: float = 1505.0
    specific_impulse: float = 225.0
    thruster_count: int = 6
    thruster_max_thrust: float = 3100.0
    cant_angle_deg: float = 27.0
    throttle_min: float = 0.3
    throttle_max: float = 0.8
    glide_slope_deg: float | None = 4.0
    initial_position: tuple[float, float, float] = (1500.0, 0.0, 2000.0)
    initial_velocity: tuple[float, float, float] = (-75.0, 0.0, 100.0)
    time_of_flight: float = 81.0
    alpha_convention: str = "cant-corrected"  # or "isp-only" (Blackmore 2010 eq. 72)
    final_thrust_vertical: bool = False

    @property
    def cos_cant(self) -> float:
        return math.cos(math.radians(self.cant_angle_deg))

    @property
    def rho_min(self) -> float:
        return self.throttle_min * self.thruster_count * self.thruster_max_thrust * self.cos_cant

    @property
    def rho_max(self) -> float:
        return self.throttle_max * self.thruster_count * self.thruster_max_thrust * self.cos_cant

    @property
    def alpha(self) -> float:
        base = 1.0 / (self.specific_impulse * STANDARD_GRAVITY)
        if self.alpha_convention == "cant-corrected":
            return base / self.cos_cant
        if self.alpha_convention == "isp-only":
            return base
        raise ValueError(f"unknown alpha convention {self.alpha_convention!r}")

    def replace(self, **changes: Any) -> MarsDescentProfile:
        payload = {f: getattr(self, f) for f in self.__dataclass_fields__}
        payload.update(changes)
        return MarsDescentProfile(**payload)


BLACKMORE_2010_CASE1 = MarsDescentProfile(
    initial_position=(1500.0, 500.0, 2000.0),
    initial_velocity=(-75.0, 0.0, 100.0),
    time_of_flight=78.4,
    alpha_convention="isp-only",
)


# --------------------------------------------------------------------------- replay
def replay_zoh(
    profile: MarsDescentProfile,
    controls: FloatArray,
    dt: float,
    *,
    substeps: int = 200,
    hold: str = "thrust",
) -> tuple[FloatArray, float]:
    """RK4 replay of a zero-order-hold control through the nonlinear mass-varying dynamics.

    ``hold="thrust"`` keeps the thrust vector ``T`` constant over each interval (repository SCvx
    convention).  ``hold="acceleration"`` keeps the mass-normalised thrust ``u = T/m`` constant,
    which is the control parametrisation of the lossless-convexification SOCP; the realised
    thrust then decays with the mass.  Returns the node states ``[r(3), v(3), m]`` and the
    propellant used.
    """

    if hold not in {"thrust", "acceleration"}:
        raise ValueError("hold must be 'thrust' or 'acceleration'")
    g = np.array([-profile.gravity_mars, 0.0, 0.0])
    alpha = profile.alpha
    state = np.concatenate([profile.initial_position, profile.initial_velocity, [profile.wet_mass]])
    states = [state.copy()]

    def f(x: FloatArray, c: FloatArray) -> FloatArray:
        thrust = c if hold == "thrust" else c * x[6]
        return np.concatenate([x[3:6], g + thrust / x[6], [-alpha * np.linalg.norm(thrust)]])

    h = dt / substeps
    for c in controls:
        for _ in range(substeps):
            k1 = f(state, c)
            k2 = f(state + 0.5 * h * k1, c)
            k3 = f(state + 0.5 * h * k2, c)
            k4 = f(state + h * k3, c)
            state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(state.copy())
    array = np.vstack(states)
    return array, float(profile.wet_mass - array[-1, 6])


# --------------------------------------------------------------------------- lossless SOCP
@dataclass(slots=True)
class LosslessResult:
    dt: float
    intervals: int
    fuel_used: float
    final_mass: float
    objective_final_log_mass: float
    thrust: FloatArray
    positions: FloatArray
    velocities: FloatArray
    log_mass: FloatArray
    replay_fuel_used: float
    replay_terminal_position_error: float
    replay_terminal_velocity_error: float
    max_thrust_epigraph_gap: float
    min_throttle_violation: float
    max_throttle_violation: float
    glide_slope_violation: float
    solver_status: str
    solver_iterations: int
    solve_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"thrust", "positions", "velocities", "log_mass"}
        }


def solve_lossless_convexification(
    profile: MarsDescentProfile,
    *,
    dt: float = 1.0,
    tolerance: float = 1.0e-9,
) -> LosslessResult:
    """Acikmese-Ploen Problem 3 (relaxed, convex) with exact ZOH double-integrator discretisation.

    Variables per node ``k = 0..N``: position ``r_k`` (3), velocity ``v_k`` (3), log-mass ``z_k``;
    per interval ``k = 0..N-1``: mass-normalised thrust ``u_k`` (3) and slack ``s_k``.  The
    control is held constant over ``[t_k, t_{k+1})`` so the translational states propagate exactly
    and the log mass integrates exactly as ``z_{k+1} = z_k - alpha s_k dt``.  The non-convex
    throttle bounds are replaced by the paper's convex bounds on ``s_k`` (second-order Taylor
    lower bound, linear upper bound), which is lossless per Lemma 2 of the paper.
    """

    tf = profile.time_of_flight
    N = round(tf / dt)
    if abs(N * dt - tf) > 1.0e-9:
        raise ValueError("time of flight must be an integer number of steps")
    alpha = profile.alpha
    rho1 = profile.rho_min
    rho2 = profile.rho_max
    g = np.array([-profile.gravity_mars, 0.0, 0.0])

    b = SOCPBuilder()
    r = [b.add_variables(3) for _ in range(N + 1)]
    v = [b.add_variables(3) for _ in range(N + 1)]
    z = [b.add_variables(1).start for _ in range(N + 1)]
    u = [b.add_variables(3) for _ in range(N)]
    s = [b.add_variables(1).start for _ in range(N)]

    def vec(block: slice, i: int) -> LinearExpression:
        return lin(block.start + i)

    for i in range(3):
        b.add_equality(vec(r[0], i), profile.initial_position[i])
        b.add_equality(vec(v[0], i), profile.initial_velocity[i])
        b.add_equality(vec(r[N], i), 0.0)
        b.add_equality(vec(v[N], i), 0.0)
    b.add_equality(lin(z[0]), math.log(profile.wet_mass))

    for k in range(N):
        for i in range(3):
            # r_{k+1} = r_k + dt v_k + dt^2/2 (g + u_k)
            expr = (
                vec(r[k + 1], i)
                .minus(vec(r[k], i))
                .minus(vec(v[k], i).scaled(dt))
                .minus(vec(u[k], i).scaled(0.5 * dt * dt))
            )
            b.add_equality(expr, 0.5 * dt * dt * g[i])
            # v_{k+1} = v_k + dt (g + u_k)
            expr_v = vec(v[k + 1], i).minus(vec(v[k], i)).minus(vec(u[k], i).scaled(dt))
            b.add_equality(expr_v, dt * g[i])
        # z_{k+1} = z_k - alpha dt s_k
        b.add_equality(lin(z[k + 1]).minus(lin(z[k])).plus(lin(s[k], alpha * dt)), 0.0)
        # ||u_k|| <= s_k
        b.add_soc(lin(s[k]), [vec(u[k], i) for i in range(3)])
        # throttle bounds around z0_k = ln(m_wet - alpha rho2 t_k)
        t_k = k * dt
        z0 = math.log(profile.wet_mass - alpha * rho2 * t_k)
        z1 = math.log(profile.wet_mass - alpha * rho1 * t_k)
        mu1 = rho1 * math.exp(-z0)
        mu2 = rho2 * math.exp(-z0)
        dz = lin(z[k]).minus(z0)
        # upper: s_k <= mu2 (1 - dz)
        b.add_leq(lin(s[k]).plus(dz.scaled(mu2)), mu2)
        # lower: s_k >= mu1 (1 - dz + dz^2/2)  <=>  (mu1/2) dz^2 <= s_k - mu1 + mu1 dz
        t_expr = lin(s[k]).plus(dz.scaled(mu1)).minus(mu1)
        a = 0.5 * mu1
        b.add_soc(t_expr.plus(1.0), [dz.scaled(2.0 * math.sqrt(a)), t_expr.minus(1.0)])
        b.add_geq(lin(z[k]), z0)
        b.add_leq(lin(z[k]), z1)
        if profile.final_thrust_vertical and k == N - 1:
            b.add_equality(vec(u[k], 1), 0.0)
            b.add_equality(vec(u[k], 2), 0.0)
    if profile.glide_slope_deg is not None:
        tan_gs = math.tan(math.radians(profile.glide_slope_deg))
        for k in range(N + 1):
            b.add_soc(vec(r[k], 0).scaled(1.0 / tan_gs), [vec(r[k], 1), vec(r[k], 2)])
    # Maximise final log-mass (minimum fuel).
    b.add_linear_cost(z[N], -1.0)

    solution = b.solve(tolerance=tolerance, max_iterations=1000)
    x = solution.x
    positions = np.vstack([x[block] for block in r])
    velocities = np.vstack([x[block] for block in v])
    log_mass = np.array([x[index] for index in z])
    accelerations = np.vstack([x[block] for block in u])
    slack = np.array([x[index] for index in s])
    masses = np.exp(log_mass)
    thrust = accelerations * masses[:-1, None]
    thrust_norm = np.linalg.norm(thrust, axis=1)
    replay_states, replay_fuel = replay_zoh(profile, accelerations, dt, hold="acceleration")
    gs_violation = 0.0
    if profile.glide_slope_deg is not None:
        tan_gs = math.tan(math.radians(profile.glide_slope_deg))
        gs_violation = float(
            np.max(
                np.maximum(tan_gs * np.linalg.norm(positions[:, 1:3], axis=1) - positions[:, 0], 0)
            )
        )
    return LosslessResult(
        dt=dt,
        intervals=N,
        fuel_used=float(profile.wet_mass - masses[-1]),
        final_mass=float(masses[-1]),
        objective_final_log_mass=float(log_mass[-1]),
        thrust=thrust,
        positions=positions,
        velocities=velocities,
        log_mass=log_mass,
        replay_fuel_used=replay_fuel,
        replay_terminal_position_error=float(np.linalg.norm(replay_states[-1, :3])),
        replay_terminal_velocity_error=float(np.linalg.norm(replay_states[-1, 3:6])),
        max_thrust_epigraph_gap=float(
            np.max(np.abs(np.linalg.norm(accelerations, axis=1) - slack))
        ),
        min_throttle_violation=float(np.max(np.maximum(rho1 - thrust_norm, 0.0))),
        max_throttle_violation=float(np.max(np.maximum(thrust_norm - rho2, 0.0))),
        glide_slope_violation=gs_violation,
        solver_status=solution.status,
        solver_iterations=solution.iterations,
        solve_seconds=solution.solve_seconds,
    )


# --------------------------------------------------------------------------- repository SCvx
def _repository_model(profile: MarsDescentProfile):
    from spacepdhcg.models import PoweredDescent3DOFConfig, PoweredDescent3DOFModel

    glide_from_vertical = (
        math.radians(90.0 - profile.glide_slope_deg)
        if profile.glide_slope_deg is not None
        else math.radians(89.999)
    )
    # The 2007 convex problem bounds the mass from below by the maximum-throttle depletion
    # m_wet - alpha rho2 t, not by the dry mass (the published solution uses 399.5 of the 400 kg
    # of propellant, so a hard dry-mass bound would make the discretised problem infeasible).
    minimum_mass = profile.wet_mass - profile.alpha * profile.rho_max * profile.time_of_flight
    config = PoweredDescent3DOFConfig(
        gravity=(0.0, 0.0, -profile.gravity_mars),
        mass_flow_coefficient=profile.alpha,
        minimum_mass=max(minimum_mass, 1.0),
        maximum_thrust=profile.rho_max,
        minimum_sigma=profile.rho_min,
        maximum_tilt_radians=math.radians(89.999),
        glide_slope_radians=glide_from_vertical,
    )
    return PoweredDescent3DOFModel(config)


def paper_to_repository_state(position, velocity, mass: float) -> FloatArray:
    """Map the paper's x-up frame onto the repository's z-up frame."""

    px, py, pz = position
    vx, vy, vz = velocity
    return np.array([pz, py, px, vz, vy, vx, mass], dtype=np.float64)


def repository_to_paper_thrust(controls: FloatArray) -> FloatArray:
    return np.column_stack([controls[:, 2], controls[:, 1], controls[:, 0]])


@dataclass(slots=True)
class RepositorySCvxResult:
    backend: str
    dt: float
    intervals: int
    status: str
    fuel_used: float
    final_mass: float
    outer_iterations: int
    accepted_iterations: int
    residual_dynamics: float
    residual_path: float
    residual_terminal: float
    path_max_violation: float
    replay_fuel_used: float
    replay_terminal_position_error: float
    replay_terminal_velocity_error: float
    total_setup_seconds: float
    total_solve_seconds: float
    wall_seconds: float
    workspace_creations: int | None = None
    numeric_updates: int | None = None
    solves: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def solve_repository_scvx(
    profile: MarsDescentProfile,
    *,
    dt: float = 1.0,
    backend: str = "clarabel",
    qoco_library: Path | None = None,
    max_iterations: int = 60,
    warm_start_from_lossless: bool = True,
) -> RepositorySCvxResult:
    """Run the repository's fixed-grid 3-DoF SCvx on the literature profile."""

    from spacepdhcg.scvx import (
        PoweredDescentOuterConfig,
        PoweredDescentSCvxSolver,
        TrustRegionConfig,
    )
    from spacepdhcg.transcription import (
        PoweredDescent3DOFSubproblem,
        PoweredDescentSCvxConfig,
    )

    tf = profile.time_of_flight
    intervals = round(tf / dt)
    model = _repository_model(profile)
    config = PoweredDescentSCvxConfig(
        intervals=intervals,
        step_seconds=dt,
        trust_radius=8.0,
        # Tuned on this profile: the repository default virtual weight (1e5) lets the
        # trust-region loop stall at ~424 kg; 1e3 reaches ~406 kg within 60 outer iterations.
        virtual_l1_weight=1.0e3,
        virtual_quadratic_weight=1.0e-6,
        virtual_epigraph_regularisation=1.0e-8,
        fuel_weight=1.0e-3,
        control_trust_scales=(
            1.0 / profile.rho_max,
            1.0 / profile.rho_max,
            1.0 / profile.rho_max,
            1.0 / profile.rho_max,
        ),
    )
    subproblem = PoweredDescent3DOFSubproblem(model, config)
    initial = paper_to_repository_state(
        profile.initial_position, profile.initial_velocity, profile.wet_mass
    )
    target_position = np.zeros(3)
    target_velocity = np.zeros(3)

    reference_states = reference_controls = None
    if warm_start_from_lossless:
        lossless = solve_lossless_convexification(profile, dt=dt)
        controls = np.column_stack(
            [
                lossless.thrust[:, 2],
                lossless.thrust[:, 1],
                lossless.thrust[:, 0],
                np.linalg.norm(lossless.thrust, axis=1),
            ]
        )
        reference_controls = controls
        reference_states = model.rollout(initial, controls, dt)

    workspaces: list[Any] = []
    if backend == "clarabel":
        from spacepdhcg.scvx.powered_descent_3dof import clarabel_reference_builder

        builder = clarabel_reference_builder
    elif backend == "qoco-gpu":
        from spacepdhcg.backends import QOCOGPU

        if qoco_library is None:
            raise FileNotFoundError("a QOCO GPU library path is required for backend qoco-gpu")

        def builder(problem, **settings):
            workspace = QOCOGPU(problem, library_path=qoco_library, **settings)
            workspaces.append(workspace)
            return workspace

    else:
        raise ValueError(f"unknown backend {backend!r}")

    solver = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(
            max_iterations=max_iterations,
            minimum_iterations=1,
            convergence_tolerance=1.0e-6,
            step_tolerance=1.0e-3,
        ),
        trust_config=TrustRegionConfig(
            initial_radius=8.0,
            minimum_radius=1.0e-5,
            maximum_radius=64.0,
        ),
        backend_builder=builder,
    )
    start = perf_counter()
    result = solver.solve(
        initial,
        target_position,
        target_velocity,
        reference_states=reference_states,
        reference_controls=reference_controls,
    )
    wall = perf_counter() - start
    thrust = repository_to_paper_thrust(result.controls)
    replay_states, replay_fuel = replay_zoh(profile, thrust, dt, hold="thrust")
    return RepositorySCvxResult(
        backend=backend,
        dt=dt,
        intervals=intervals,
        status=result.status,
        fuel_used=float(profile.wet_mass - result.states[-1, 6]),
        final_mass=float(result.states[-1, 6]),
        outer_iterations=result.outer_iterations,
        accepted_iterations=result.accepted_iterations,
        residual_dynamics=float(result.residual.dynamics),
        residual_path=float(result.residual.path),
        residual_terminal=float(result.residual.terminal),
        path_max_violation=float(result.path_diagnostics.maximum_violation),
        replay_fuel_used=replay_fuel,
        replay_terminal_position_error=float(np.linalg.norm(replay_states[-1, :3])),
        replay_terminal_velocity_error=float(np.linalg.norm(replay_states[-1, 3:6])),
        total_setup_seconds=float(result.total_setup_seconds),
        total_solve_seconds=float(result.total_solve_seconds),
        wall_seconds=float(wall),
        workspace_creations=len(workspaces) if workspaces else None,
        numeric_updates=sum(w.update_count for w in workspaces) if workspaces else None,
        solves=sum(w.solve_count for w in workspaces) if workspaces else None,
    )


# --------------------------------------------------------------------------- target runner
@dataclass(slots=True)
class ReproductionRecord:
    target_id: str
    status: str
    published: dict[str, Any]
    measured: dict[str, Any]
    gap: dict[str, Any]
    labels: dict[str, str]
    envelope: dict[str, Any]
    commands: list[str]
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_from_document(document: dict[str, Any]) -> MarsDescentProfile:
    p = document["parameters"]
    return MarsDescentProfile(
        gravity_mars=p["gravity_mars_m_s2"],
        wet_mass=p["wet_mass_kg"],
        dry_mass=p["dry_mass_kg"],
        specific_impulse=p["specific_impulse_s"],
        thruster_count=p["thruster_count"],
        thruster_max_thrust=p["thruster_max_thrust_n"],
        cant_angle_deg=p["cant_angle_deg"],
        throttle_min=p["throttle_min"],
        throttle_max=p["throttle_max"],
        glide_slope_deg=p.get("glide_slope_deg"),
        initial_position=tuple(p["initial_position_m"]),
        initial_velocity=tuple(p["initial_velocity_m_s"]),
        time_of_flight=p["time_of_flight_s"],
        alpha_convention=p.get("alpha_convention", "cant-corrected"),
        final_thrust_vertical=bool(p.get("final_thrust_vertical", False)),
    )


def run_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    profile = profile_from_document(document)
    published = document["published"]
    envelope_steps = tuple(options.get("dt_values", document.get("envelope_dt_values", [1.0, 0.5])))
    run_scvx = bool(options.get("run_repository_scvx", True))
    qoco_library = options.get("qoco_library") or os.environ.get("SPACEPDHCG_QOCO_LIBRARY")
    run_gpu = bool(options.get("run_gpu", bool(qoco_library)))

    lossless_runs = {}
    for dt in envelope_steps:
        lossless_runs[f"dt={dt}"] = solve_lossless_convexification(profile, dt=dt).as_dict()
    # Alternative mass-flow convention (Blackmore 2010 eq. 72 omits cos(phi)).
    alternative = solve_lossless_convexification(
        profile.replace(alpha_convention="isp-only"), dt=envelope_steps[0]
    ).as_dict()

    primary = lossless_runs[f"dt={envelope_steps[0]}"]
    fuel_values = [run["fuel_used"] for run in lossless_runs.values()]
    measured: dict[str, Any] = {
        "lossless_fuel_used_kg": primary["fuel_used"],
        "lossless_fuel_used_kg_by_dt": {k: v["fuel_used"] for k, v in lossless_runs.items()},
        "lossless_replay_fuel_used_kg": primary["replay_fuel_used"],
        "lossless_isp_only_alpha_fuel_used_kg": alternative["fuel_used"],
    }
    details: dict[str, Any] = {"lossless": lossless_runs, "lossless_isp_only_alpha": alternative}
    commands = [
        f"spacepdhcg literature run {document['id']}",
    ]
    notes: list[str] = []
    scvx_records: dict[str, Any] = {}
    if run_scvx:
        try:
            cpu = solve_repository_scvx(profile, dt=envelope_steps[0], backend="clarabel")
            scvx_records["clarabel"] = cpu.as_dict()
            measured["scvx_cpu_fuel_used_kg"] = cpu.fuel_used
            measured["scvx_cpu_replay_fuel_used_kg"] = cpu.replay_fuel_used
            measured["scvx_cpu_status"] = cpu.status
        except Exception as error:
            scvx_records["clarabel"] = {"error": repr(error)}
            notes.append(f"repository CPU SCvx failed: {error!r}")
        if run_gpu:
            if not qoco_library:
                scvx_records["qoco-gpu"] = {"error": "no QOCO GPU library configured"}
                notes.append("GPU pure-QOCO SCvx skipped: SPACEPDHCG_QOCO_LIBRARY not set")
            else:
                try:
                    gpu = solve_repository_scvx(
                        profile,
                        dt=envelope_steps[0],
                        backend="qoco-gpu",
                        qoco_library=Path(qoco_library),
                    )
                    scvx_records["qoco-gpu"] = gpu.as_dict()
                    measured["scvx_qoco_gpu_fuel_used_kg"] = gpu.fuel_used
                    measured["scvx_qoco_gpu_replay_fuel_used_kg"] = gpu.replay_fuel_used
                    measured["scvx_qoco_gpu_status"] = gpu.status
                except Exception as error:
                    scvx_records["qoco-gpu"] = {"error": repr(error)}
                    notes.append(f"GPU pure-QOCO SCvx failed: {error!r}")
        else:
            notes.append(
                str(
                    options.get(
                        "gpu_note",
                        "GPU pure-QOCO SCvx not run (pass run_gpu=true with "
                        "SPACEPDHCG_QOCO_LIBRARY set)",
                    )
                )
            )
    details["repository_scvx"] = scvx_records

    published_fuel = float(published["fuel_used_kg"])
    envelope_width = float(max(fuel_values) - min(fuel_values))
    gap_kg = float(primary["fuel_used"] - published_fuel)
    tolerance_kg = float(document.get("acceptance_tolerance_kg", 1.5))
    status = "reproduced" if abs(gap_kg) <= tolerance_kg else "gap"
    gap: dict[str, Any] = {
        "lossless_minus_published_kg": gap_kg,
        "lossless_relative": gap_kg / published_fuel,
        "acceptance_tolerance_kg": tolerance_kg,
    }
    if "scvx_cpu_fuel_used_kg" in measured:
        gap["scvx_cpu_minus_published_kg"] = measured["scvx_cpu_fuel_used_kg"] - published_fuel
        gap["scvx_cpu_minus_lossless_kg"] = measured["scvx_cpu_fuel_used_kg"] - primary["fuel_used"]
    if "scvx_qoco_gpu_fuel_used_kg" in measured:
        gap["scvx_qoco_gpu_minus_lossless_kg"] = (
            measured["scvx_qoco_gpu_fuel_used_kg"] - primary["fuel_used"]
        )
    labels = {
        "published.fuel_used_kg": published.get("evidence_label", "published-reference"),
        "measured.lossless_fuel_used_kg": "measured-local",
    }
    if "scvx_cpu_fuel_used_kg" in measured:
        labels["measured.scvx_cpu_fuel_used_kg"] = "measured-local"
    if "scvx_qoco_gpu_fuel_used_kg" in measured:
        labels["measured.scvx_qoco_gpu_fuel_used_kg"] = "measured-local"
    return ReproductionRecord(
        target_id=document["id"],
        status=status,
        published=published,
        measured=measured,
        gap=gap,
        labels=labels,
        envelope={
            "discretisation": "zero-order hold, exact double-integrator map, exact log-mass step",
            "dt_values_s": list(envelope_steps),
            "fuel_spread_kg": envelope_width,
            "declared_envelope_kg": document.get("declared_envelope_kg"),
        },
        commands=commands,
        notes=notes,
        details=details,
    ).as_dict()
