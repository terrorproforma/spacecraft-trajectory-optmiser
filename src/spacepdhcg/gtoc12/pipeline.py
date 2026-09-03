"""OrbitWeaver pipeline for GTOC12: plan -> refined certified arcs -> official solution file.

The refinement stage drives the CPU SCvx leg solver through the existing G7 contracts: every leg
becomes an :class:`ArcRequest`, the :class:`G3TrajectoryOracleAdapter` owns one
:class:`Gtoc12ScvxDriver` per topology group, the :class:`BoundedScheduler` batches and orders the
work deterministically, and an :class:`IndependentCertifier` re-propagates each emitted arc with the
verifier model (DOP853 + cubic Lagrange thrust) before it may enter the route master.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.orbitweaver.adapters import (
    G3Solve,
    G3Status,
    G3TrajectoryOracleAdapter,
    RouteDefinition,
    build_route_columns,
    solve_certified_route_master,
)
from spacepdhcg.orbitweaver.g7 import (
    ArcFidelity,
    ArcRequest,
    ArcResult,
    BoundedScheduler,
    CertificationChecks,
    IndependentCertifier,
    Ownership,
    SchedulerConfig,
    TopologyKey,
)

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .low_thrust import (
    COAST_THRUST_N,
    LegBoundary,
    LegCertificate,
    LegSolution,
    ScvxSettings,
    certify_leg,
    solve_leg,
)
from .search import EARTH_ID, PlannedLeg, RoutePlan
from .solution import Event, ShipTrajectory, Solution, StateLine

FloatArray = NDArray[np.float64]
MODEL_IDENTIFIER = "gtoc12-heliocentric-low-thrust-v1"


def body_state(
    catalogue: AsteroidCatalogue, body: int, epoch: float
) -> tuple[FloatArray, FloatArray]:
    if body == EARTH_ID:
        return earth_state(epoch)
    return asteroid_state(catalogue, body, epoch)


def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") | 1


@dataclass(slots=True)
class LegRecord:
    request: ArcRequest
    boundary: LegBoundary
    solution: LegSolution | None = None
    certificate: LegCertificate | None = None
    result: ArcResult | None = None


class LegRegistry:
    """Side table linking deterministic arc IDs to their emitted solutions and certificates."""

    def __init__(self) -> None:
        self.records: dict[int, LegRecord] = {}

    def register(self, request: ArcRequest, boundary: LegBoundary) -> None:
        self.records[request.deterministic_id] = LegRecord(request, boundary)


class Gtoc12ScvxDriver:
    """CPU SCvx driver implementing the public G3 persistent-driver protocol."""

    def __init__(
        self, topology: TopologyKey, owner: Ownership, registry: LegRegistry, settings: ScvxSettings
    ) -> None:
        self.topology = topology
        self.owner = owner
        self.registry = registry
        self.settings = settings
        self._request: ArcRequest | None = None
        self._cancelled = False
        self.numeric_updates = 0

    @property
    def topology_fingerprint(self) -> int:
        return self.topology.topology_fingerprint

    @property
    def intervals(self) -> int:
        return self.topology.intervals

    @property
    def scenario_count(self) -> int:
        return self.topology.scenario_count

    def update_numeric_in_place(self, request: ArcRequest) -> None:
        self._request = request
        self.numeric_updates += 1

    def import_warm_state(self, state: object, source_intervals: int) -> bool:
        del state, source_intervals
        return False  # cold starts only; warm remeshing is not implemented for this driver

    def solve(self, request: ArcRequest, cancelled: threading.Event) -> G3Solve:
        if cancelled.is_set() or self._cancelled:
            return G3Solve(G3Status.CANCELLED, diagnostic="cancelled before solve")
        record = self.registry.records[request.deterministic_id]
        try:
            solution = solve_leg(record.boundary, self.settings)
        except Exception as error:
            return G3Solve(
                G3Status.NUMERICAL_FAILURE, diagnostic=f"{type(error).__name__}: {error}"
            )
        record.solution = solution
        if solution.status == "timeout":
            return G3Solve(G3Status.TIMEOUT, diagnostic=solution.diagnostic)
        if solution.status in {"infeasible"}:
            return G3Solve(G3Status.INFEASIBLE, diagnostic=solution.diagnostic)
        if solution.status == "failed":
            return G3Solve(G3Status.NUMERICAL_FAILURE, diagnostic=solution.diagnostic)
        clamp_thrust(solution)
        certificate = certify_leg(solution)
        record.certificate = certificate
        status = G3Status.CONVERGED if solution.status == "converged" else G3Status.ITERATION_LIMIT
        return G3Solve(
            status=status,
            objective=solution.propellant_kg,
            lower_bound=0.0,
            duration=solution.boundary.duration_days,
            delta_v=solution.delta_v_km_s,
            propellant=solution.propellant_kg,
            final_mass=solution.final_mass_kg,
            canonical_residual=solution.max_defect,
            replay_residual=certificate.rk4_vs_dop853_km / C.TOLERANCE_POSITION_KM,
            path_violation=max(0.0, certificate.maximum_thrust_n - C.THRUST_MAX_N) / C.THRUST_MAX_N
            + max(0.0, C.MIN_SUN_DISTANCE_AU - certificate.minimum_sun_distance_au),
            terminal_error=max(
                certificate.position_error_km / C.TOLERANCE_POSITION_KM,
                certificate.velocity_error_km_s / C.TOLERANCE_VELOCITY_KM_S,
            ),
            uncertainty_violation=0.0,
            warm_state=None,
            certifiable_candidate=True,
            diagnostic=solution.diagnostic or solution.status,
        )

    def cancel(self) -> None:
        self._cancelled = True

    def close(self) -> None:
        self._request = None


def clamp_thrust(solution: LegSolution) -> None:
    """Scale any node whose magnitude exceeds T_max down to T_max (1 - 1e-9)."""

    limit = C.THRUST_MAX_N * (1.0 - 1e-9)
    magnitude = np.linalg.norm(solution.thrust_n, axis=1)
    over = magnitude > limit
    if np.any(over):
        solution.thrust_n[over] *= (limit / magnitude[over])[:, None]
    solution.thrust_n[magnitude <= COAST_THRUST_N] = 0.0


def certification_callback(registry: LegRegistry):
    def callback(result: ArcResult) -> CertificationChecks:
        record = registry.records[result.deterministic_id]
        certificate = record.certificate
        if certificate is None:
            return CertificationChecks(math.inf, math.inf, math.inf, math.inf, math.inf)
        return CertificationChecks(
            dynamics_defect=certificate.position_error_km / C.TOLERANCE_POSITION_KM,
            path_violation=max(0.0, certificate.maximum_thrust_n - C.THRUST_MAX_N) / C.THRUST_MAX_N
            + max(0.0, C.MIN_SUN_DISTANCE_AU - certificate.minimum_sun_distance_au),
            terminal_error=certificate.velocity_error_km_s / C.TOLERANCE_VELOCITY_KM_S,
            uncertainty_violation=0.0,
            integration_error=certificate.rk4_vs_dop853_km / C.TOLERANCE_POSITION_KM,
        )

    return callback


@dataclass(slots=True)
class RefinedLeg:
    planned: PlannedLeg
    request: ArcRequest
    result: ArcResult
    solution: LegSolution | None
    certificate: LegCertificate | None
    certified: bool
    mass_before: float
    mass_after_leg: float


@dataclass(slots=True)
class RefinedRoute:
    plan: RoutePlan
    legs: list[RefinedLeg]
    collected_mass: dict[int, float]
    final_mass_kg: float
    certified: bool
    master_certified: bool
    passes: int
    wall_seconds: float
    scheduler_telemetry: dict[str, Any]
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_collected_kg(self) -> float:
        return sum(self.collected_mass.values())

    @property
    def refined_arc_count(self) -> int:
        return sum(1 for leg in self.legs if leg.solution is not None)

    def summary(self) -> dict[str, Any]:
        return {
            "asteroids": list(self.plan.asteroids),
            "collected_mass_kg": dict(self.collected_mass),
            "total_collected_kg": self.total_collected_kg,
            "final_mass_kg": self.final_mass_kg,
            "certified": self.certified,
            "master_certified": self.master_certified,
            "passes": self.passes,
            "wall_seconds": self.wall_seconds,
            "refined_arcs": self.refined_arc_count,
            "scheduler": self.scheduler_telemetry,
            "plan": self.plan.summary(),  # reloadable (``RoutePlan.from_summary``) for archives
            "legs": [
                {
                    "from": leg.planned.from_id,
                    "to": leg.planned.to_id,
                    "t0": leg.planned.departure_epoch,
                    "tf": leg.planned.arrival_epoch,
                    "status": leg.result.status.value,
                    "certified": leg.certified,
                    "mass_before": leg.mass_before,
                    "mass_after": leg.mass_after_leg,
                    "propellant_kg": None if leg.solution is None else leg.solution.propellant_kg,
                    "delta_v_km_s": None if leg.solution is None else leg.solution.delta_v_km_s,
                    "scvx_iterations": None if leg.solution is None else leg.solution.iterations,
                    "solve_seconds": None if leg.solution is None else leg.solution.solve_seconds,
                    "position_error_km": None
                    if leg.certificate is None
                    else leg.certificate.position_error_km,
                    "velocity_error_m_s": None
                    if leg.certificate is None
                    else leg.certificate.velocity_error_km_s * 1e3,
                }
                for leg in self.legs
            ],
            "failures": list(self.failures),
        }


def refine_route(
    plan: RoutePlan,
    catalogue: AsteroidCatalogue,
    *,
    scvx: ScvxSettings | None = None,
    certification_tolerance: float = 0.5,
    max_passes: int = 3,
    scheduler_config: SchedulerConfig | None = None,
) -> RefinedRoute:
    """Refine every planned leg with certified SCvx arcs; collected masses are sized to fit."""

    started = time.perf_counter()
    scvx = scvx or ScvxSettings()
    registry = LegRegistry()
    adapter = G3TrajectoryOracleAdapter(
        lambda topology, owner: Gtoc12ScvxDriver(topology, owner, registry, scvx)
    )
    scheduler = BoundedScheduler(
        adapter,
        config=scheduler_config or SchedulerConfig(maximum_batch_size=8, maximum_buffered_arcs=64),
    )
    certifier = IndependentCertifier(
        certification_callback(registry),
        backend_identifier="gtoc12-verifier-model-dop853",
        tolerance=certification_tolerance,
    )
    deploy = dict(plan.deploy_epochs)
    collect = dict(plan.collect_epochs)
    # cooperative plans collect miners another ship deployed (plan.foreign_deploy_epochs) and may
    # leave their own miners for another ship: the mass is mined from the deployer's epoch
    collected = {
        asteroid: C.maximum_collected_mass(collect[asteroid] - plan.deploy_epoch_of(asteroid))
        for asteroid in collect
    }
    legs_to_fly = [leg for leg in plan.legs if leg.role != "camp"]
    refined: list[RefinedLeg] = []
    failures: list[dict[str, Any]] = []
    final_mass = math.nan
    passes = 0
    telemetry: dict[str, Any] = {}
    while passes < max_passes:
        passes += 1
        refined = []
        mass = C.MAX_INITIAL_MASS_KG
        route_ok = True
        for index, leg in enumerate(legs_to_fly):
            r0, v0 = body_state(catalogue, leg.from_id, leg.departure_epoch)
            rf, vf = body_state(catalogue, leg.to_id, leg.arrival_epoch)
            carried = sum(collected[a] for a in collect if collect[a] <= leg.departure_epoch)
            boundary = LegBoundary(
                leg.departure_epoch,
                r0,
                v0,
                leg.arrival_epoch,
                rf,
                vf,
                mass,
                free_departure_vinf=leg.from_id == EARTH_ID,
                free_arrival_vinf=leg.to_id == EARTH_ID,
                minimum_final_mass=C.DRY_MASS_KG + carried,
            )
            node_count = math.floor(leg.tof_days + 1e-9) + 2
            topology = TopologyKey(
                _hash_int(f"{MODEL_IDENTIFIER}:{node_count}"),
                ArcFidelity.REFINED_SCVX,
                node_count,
                1,
            )
            request = ArcRequest(
                deterministic_id=passes * 1000 + index,
                from_target=leg.from_id if leg.from_id != EARTH_ID else -3,
                to_target=leg.to_id if leg.to_id != EARTH_ID else -3,
                departure_epoch=leg.departure_epoch,
                arrival_epoch=leg.arrival_epoch,
                initial_mass=mass,
                spacecraft=1,
                scenario_count=1,
                fidelity=ArcFidelity.REFINED_SCVX,
                requested_tolerance=C.TOLERANCE_POSITION_KM,
                model_identifier=MODEL_IDENTIFIER,
                topology=topology,
                route_index=0,
                trajectory_arc_index=index,
            )
            if request.from_target == request.to_target:
                # Earth -> Earth legs are not part of the plan; guard the contract anyway
                raise ValueError("leg endpoints coincide")
            registry.register(request, boundary)
            results = scheduler.run([request])
            result = results[0]
            record = registry.records[request.deterministic_id]
            certification = certifier.certify(result)
            certified = certification.accepted
            mass_after = (
                record.certificate.final_mass_kg if record.certificate is not None else math.nan
            )
            refined.append(
                RefinedLeg(
                    leg,
                    request,
                    result,
                    record.solution,
                    record.certificate,
                    certified,
                    mass,
                    mass_after,
                )
            )
            if not result.feasible or not certified:
                failures.append(
                    {
                        "pass": passes,
                        "leg": index,
                        "from": leg.from_id,
                        "to": leg.to_id,
                        "status": result.status.value,
                        "diagnostic": result.diagnostic,
                        "certification": certification.diagnostic,
                    }
                )
                route_ok = False
                break
            mass = mass_after
            # event mass changes at the arrival body
            if leg.to_id != EARTH_ID:
                if leg.to_id in deploy and abs(deploy[leg.to_id] - leg.arrival_epoch) < 1e-6:
                    mass -= C.MINER_MASS_KG
                elif leg.to_id in collect and abs(collect[leg.to_id] - leg.arrival_epoch) < 1e-6:
                    mass += collected[leg.to_id]
            # a collection scheduled at the *departure* of the next leg (after camping) adds mass
            # then; when it coincides with this arrival it was already counted above
            next_leg = legs_to_fly[index + 1] if index + 1 < len(legs_to_fly) else None
            if (
                next_leg is not None
                and next_leg.from_id == leg.to_id
                and next_leg.from_id in collect
                and abs(collect[next_leg.from_id] - next_leg.departure_epoch) < 1e-6
                and abs(collect[next_leg.from_id] - leg.arrival_epoch) >= 1e-6
            ):
                mass += collected[next_leg.from_id]
        telemetry = {
            "submitted": scheduler.telemetry.submitted,
            "completed": scheduler.telemetry.completed,
            "feasible": scheduler.telemetry.feasible,
            "failed": scheduler.telemetry.failed,
            "batches": scheduler.telemetry.batches,
            "workspace_creations": adapter.workspace_creations,
            "numeric_updates": adapter.numeric_updates,
        }
        if not route_ok:
            final_mass = mass
            break
        final_mass = mass - sum(collected.values())  # unloaded at Earth
        required = C.DRY_MASS_KG
        if final_mass >= required - 1e-9:
            break
        # Not enough propellant margin: reduce the collected masses proportionally and re-fly
        deficit = required - final_mass
        total = sum(collected.values())
        if total <= 0.0:
            break
        scale = max(0.0, (total - 1.05 * deficit) / total)
        collected = {asteroid: value * scale for asteroid, value in collected.items()}
    all_certified = (
        bool(refined)
        and len(refined) == len(legs_to_fly)
        and all(item.certified for item in refined)
    )
    definitions = [
        RouteDefinition(
            0,
            1,
            tuple(plan.asteroids),
            tuple((0, item.request.deterministic_id) for item in refined),
        )
    ]
    arc_results = {(0, item.request.deterministic_id): item.result for item in refined}
    columns = build_route_columns(definitions, arc_results, certifier)
    master = solve_certified_route_master(columns, plan.asteroids)
    adapter.close()
    return RefinedRoute(
        plan=plan,
        legs=refined,
        collected_mass=collected if all_certified else {a: 0.0 for a in collected},
        final_mass_kg=final_mass,
        certified=all_certified,
        master_certified=master.certified,
        passes=passes,
        wall_seconds=time.perf_counter() - started,
        scheduler_telemetry=telemetry,
        failures=failures,
    )


def emit_solution(
    route: RefinedRoute, catalogue: AsteroidCatalogue, *, ship_id: int = 1
) -> Solution:
    """Write the refined route as one official-format ship section."""

    if not route.certified:
        raise ValueError("only fully certified routes are emitted")
    ship = ShipTrajectory(ship_id)
    plan = route.plan
    deploy = plan.deploy_epochs
    collect = plan.collect_epochs
    first = route.legs[0]
    r_e, v_e = earth_state(first.planned.departure_epoch)
    launch_velocity = first.solution.departure_ship_velocity_km_s()
    ship.items.append(
        Event(
            C.EVENT_LAUNCH,
            StateLine(first.planned.departure_epoch, r_e.copy(), v_e.copy(), first.mass_before),
            StateLine(
                first.planned.departure_epoch, r_e.copy(), launch_velocity.copy(), first.mass_before
            ),
        )
    )
    carried = 0.0
    for index, leg in enumerate(route.legs):
        solution = leg.solution
        ship.items.extend(solution.burn_arcs())
        epoch = leg.planned.arrival_epoch
        mass_before = leg.certificate.final_mass_kg
        if leg.planned.to_id == EARTH_ID:
            r, v = earth_state(epoch)
            v_ship = solution.arrival_ship_velocity_km_s()
            ship.items.append(
                Event(
                    C.EVENT_EARTH_FLYBY,
                    StateLine(epoch, r.copy(), v_ship.copy(), mass_before),
                    StateLine(epoch, r.copy(), v_ship.copy(), mass_before - carried),
                )
            )
            carried = 0.0
            continue
        asteroid = leg.planned.to_id
        r, v = asteroid_state(catalogue, asteroid, epoch)
        next_leg = route.legs[index + 1] if index + 1 < len(route.legs) else None
        collects_at_departure = (
            next_leg is not None
            and next_leg.planned.from_id == asteroid
            and asteroid in collect
            and abs(collect[asteroid] - next_leg.planned.departure_epoch) < 1e-6
            and abs(collect[asteroid] - epoch) >= 1e-6
        )
        if asteroid in deploy and abs(deploy[asteroid] - epoch) < 1e-6:
            mass_after = mass_before - C.MINER_MASS_KG
            ship.items.append(
                Event(
                    asteroid,
                    StateLine(epoch, r.copy(), v.copy(), mass_before),
                    StateLine(epoch, r.copy(), v.copy(), mass_after),
                )
            )
        elif asteroid in collect and abs(collect[asteroid] - epoch) < 1e-6:
            gained = route.collected_mass[asteroid]
            mass_after = mass_before + gained
            carried += gained
            ship.items.append(
                Event(
                    asteroid,
                    StateLine(epoch, r.copy(), v.copy(), mass_before),
                    StateLine(epoch, r.copy(), v.copy(), mass_after),
                )
            )
        elif collects_at_departure:
            # Arrive, coast on the asteroid's orbit (no event: each asteroid may be rendezvoused
            # twice only), then collect and depart together.
            mass_after = mass_before
        else:
            raise ValueError(f"leg arrives at asteroid {asteroid} without a deploy or collect")
        if collects_at_departure:
            depart = next_leg.planned.departure_epoch
            r2, v2 = asteroid_state(catalogue, asteroid, depart)
            gained = route.collected_mass[asteroid]
            ship.items.append(
                Event(
                    asteroid,
                    StateLine(depart, r2.copy(), v2.copy(), mass_after),
                    StateLine(depart, r2.copy(), v2.copy(), mass_after + gained),
                )
            )
            carried += gained
    return Solution([ship])


def write_route_artifacts(
    route: RefinedRoute, catalogue: AsteroidCatalogue, directory: Path
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    solution = emit_solution(route, catalogue)
    solution_path = directory / "Result.txt"
    solution.write(solution_path)
    summary = route.summary()
    (directory / "route_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    return {"solution": str(solution_path), "summary": summary}


def plan_from_route_summary(
    summary: dict[str, Any], *, deployers: dict[int, float] | None = None
) -> RoutePlan:
    """Rebuild the :class:`RoutePlan` an archived ``route_summary.json`` was refined from.

    Recent archives carry the plan verbatim (``summary["plan"]``).  Older ones only carry the
    flown legs and the collected masses, from which the schedule is reconstructed: a leg towards
    an unvisited asteroid deploys a miner at its arrival, a later visit collects it (at the
    arrival or the departure — whichever reproduces the archived collected mass), and an
    asteroid collected without an own deploy is a foreign miner whose deploy epoch is taken from
    ``deployers`` (the other ships of the same archived bundle) or backed out of the mass.
    """

    if "plan" in summary:
        plan = RoutePlan.from_summary(summary["plan"])
        if deployers:
            foreign = {
                a: deployers.get(a, epoch) for a, epoch in plan.foreign_deploy_epochs.items()
            }
            plan = RoutePlan(
                plan.legs,
                plan.deploy_epochs,
                plan.collect_epochs,
                plan.collected_mass,
                plan.propellant_proxy_kg,
                plan.final_mass_proxy_kg,
                foreign,
            )
        return plan
    collected = {int(k): float(v) for k, v in summary["collected_mass_kg"].items()}
    rate_per_day = C.MINING_RATE_KG_PER_YEAR / C.YEAR_DAYS
    deploy: dict[int, float] = {}
    collect: dict[int, float] = {}
    foreign: dict[int, float] = {}
    legs: list[PlannedLeg] = []
    raw = [
        (int(item["from"]), int(item["to"]), float(item["t0"]), float(item["tf"]), item)
        for item in summary["legs"]
    ]
    for index, (source, target, t0, tf, item) in enumerate(raw):
        departure = raw[index + 1][2] if index + 1 < len(raw) else tf  # leaving ``target``
        if source == EARTH_ID:
            role = "earth_out"
        elif target == EARTH_ID:
            role = "earth_return"
        elif target in deploy or target in collect:
            role = "collect_hop"  # revisit: the miner deployed earlier is collected now
        elif target not in collected:
            role = "deploy_hop"  # deployed and left for another ship (or never collected)
        else:
            # first visit to an asteroid this ship collects: either the ship deploys now and
            # collects on a later visit / when it leaves after camping, or it collects a foreign
            # miner right away.  The archived mass tells the cases apart.
            mass_days = collected[target] / rate_per_day
            revisit = any(leg[1] == target for leg in raw[index + 1 :])
            camp_matches = abs((departure - tf) - mass_days) < 1e-6
            role = "deploy_hop" if revisit or camp_matches else "collect_hop"
        if role in ("earth_out", "deploy_hop"):
            deploy[target] = tf
        elif target != EARTH_ID and target in collected and target not in collect:
            # collect at the arrival or the departure (camping) - whichever reproduces the mass
            if target in deploy:
                mined_from = deploy[target]
            elif deployers and target in deployers:
                mined_from = deployers[target]
            else:
                mined_from = None
            if mined_from is None:
                collect[target] = tf
                foreign[target] = tf - collected[target] / rate_per_day
            else:
                expected = mined_from + collected[target] / rate_per_day
                collect[target] = (
                    tf if abs(tf - expected) <= abs(departure - expected) else departure
                )
                if target not in deploy:
                    foreign[target] = mined_from
        legs.append(
            PlannedLeg(
                source,
                target,
                t0,
                tf,
                float(item.get("delta_v_km_s") or 0.0),
                1.0,
                role,
            )
        )
    # an own miner collected when the ship leaves the asteroid (camp) has no revisit leg
    for asteroid in collected:
        if asteroid not in collect and asteroid in deploy:
            departures = [leg.departure_epoch for leg in legs if leg.from_id == asteroid]
            if departures:
                collect[asteroid] = departures[-1]
    total = sum(collected.values())
    final_mass = float(summary.get("final_mass_kg", C.DRY_MASS_KG + total))
    return RoutePlan(
        tuple(legs),
        deploy,
        collect,
        collected,
        C.MAX_INITIAL_MASS_KG - final_mass - C.MINER_MASS_KG * len(deploy),
        final_mass + total,
        foreign,
    )
