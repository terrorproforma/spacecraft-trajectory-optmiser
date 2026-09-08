"""Opt-in fixed-cargo refinement, extracted from the v606/v611 truth-set runner.

The low-thrust driver, mass events, independent leg certifier and route master
are unchanged. Full-fleet verification remains a separate mandatory step.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict

from . import pipeline as p
from .refinement_admission import FixedCargoRequest, promotion_blockers


def validate_prescription(plan, cargo):
    """Check the physical schedule/inventory, independently of the rejected proxy mass."""
    from spacepdhcg.gtoc12.retiming import visits_of

    # visits_of labels actions but does not check every flown endpoint. Apply
    # the same immutable event/cargo contract to direct callers as to the queue.
    FixedCargoRequest.from_summary(plan.summary(), cargo)
    visits, arrivals, departures = visits_of(plan)
    if plan.foreign_deploy_epochs:
        raise ValueError("truth set is limited to independent own-miner routes")
    if not 0 < len(plan.deploy_epochs) <= p.C.MAX_MINERS_PER_SHIP:
        raise ValueError("invalid miner count")
    if set(cargo) != set(plan.collect_epochs) or not set(cargo) <= set(plan.deploy_epochs):
        raise ValueError("prescribed cargo must match own collection inventory")
    if visits[0].body != 0 or visits[-1].body != 0:
        raise ValueError("Earth departure and return are required")
    if arrivals[0] < p.C.MISSION_START_MJD or arrivals[-1] > p.C.MISSION_END_MJD:
        raise ValueError("schedule outside mission window")
    if any(not math.isfinite(float(t)) for t in (*arrivals, *departures)):
        raise ValueError("nonfinite schedule")
    if any(departures[i] < arrivals[i] for i in range(len(visits))):
        raise ValueError("negative dwell")
    if any(arrivals[i + 1] <= departures[i] for i in range(len(visits) - 1)):
        raise ValueError("nonpositive flight duration")
    deploy_counts = Counter(v.body for v in visits if v.deploy)
    collect_counts = Counter(v.body for v in visits if v.collect)
    if deploy_counts != Counter({a: 1 for a in plan.deploy_epochs}):
        raise ValueError("deployment actions do not match plan")
    if collect_counts != Counter({a: 1 for a in plan.collect_epochs}):
        raise ValueError("collection actions do not match plan")
    for body, mass in cargo.items():
        stay = plan.collect_epochs[body] - plan.deploy_epochs[body]
        if stay < p.C.MIN_MINING_STAY_YEARS * p.C.YEAR_DAYS - 1e-6:
            raise ValueError("minimum mining stay violated")
        if not math.isfinite(mass) or mass < 0:
            raise ValueError("cargo must be finite and nonnegative")
        if mass > p.C.maximum_collected_mass(stay) + 1e-7:
            raise ValueError("prescribed cargo exceeds mining production")
    # No plan.feasible gate here: this diagnostic explicitly tests rejected
    # proxy predictions. All actual leg, final mass and fleet gates remain.


class FrozenLegRunner:
    """The stock scheduler/driver path, including clamp_thrust and DOP853 certification."""

    def __init__(self, settings):
        self.registry = p.LegRegistry()
        self.adapter = p.G3TrajectoryOracleAdapter(
            lambda topology, owner: p.Gtoc12ScvxDriver(topology, owner, self.registry, settings)
        )
        self.scheduler = p.BoundedScheduler(
            self.adapter, config=p.SchedulerConfig(maximum_batch_size=8, maximum_buffered_arcs=64)
        )
        self.certifier = p.IndependentCertifier(
            p.certification_callback(self.registry),
            backend_identifier="gtoc12-verifier-model-dop853",
            tolerance=0.5,
        )

    def solve(self, leg, boundary, index):
        node_count = math.floor(leg.tof_days + 1e-9) + 2
        topology = p.TopologyKey(
            p._hash_int(f"{p.MODEL_IDENTIFIER}:{node_count}"),
            p.ArcFidelity.REFINED_SCVX,
            node_count,
            1,
        )
        request = p.ArcRequest(
            deterministic_id=1000 + index,
            from_target=leg.from_id if leg.from_id != p.EARTH_ID else -3,
            to_target=leg.to_id if leg.to_id != p.EARTH_ID else -3,
            departure_epoch=leg.departure_epoch,
            arrival_epoch=leg.arrival_epoch,
            initial_mass=boundary.initial_mass,
            spacecraft=1,
            scenario_count=1,
            fidelity=p.ArcFidelity.REFINED_SCVX,
            requested_tolerance=p.C.TOLERANCE_POSITION_KM,
            model_identifier=p.MODEL_IDENTIFIER,
            topology=topology,
            route_index=0,
            trajectory_arc_index=index,
        )
        if request.from_target == request.to_target:
            raise ValueError("leg endpoints coincide")
        self.registry.register(request, boundary)
        result = self.scheduler.run([request])[0]
        record = self.registry.records[request.deterministic_id]
        certification = self.certifier.certify(result)
        after = record.certificate.final_mass_kg if record.certificate is not None else math.nan
        refined = p.RefinedLeg(
            leg,
            request,
            result,
            record.solution,
            record.certificate,
            bool(result.feasible and certification.accepted),
            boundary.initial_mass,
            after,
            record.certification_backend,
        )
        return refined, {
            "leg": index,
            "from": leg.from_id,
            "to": leg.to_id,
            "departure": leg.departure_epoch,
            "arrival": leg.arrival_epoch,
            "initial_mass_kg": boundary.initial_mass,
            "minimum_final_mass_kg": boundary.minimum_final_mass,
            "status": result.status.value,
            "diagnostic": result.diagnostic,
            "certification": certification.diagnostic,
            "certification_backend": record.certification_backend,
            "certified": refined.certified,
            "certificate": None if record.certificate is None else asdict(record.certificate),
            "solution": None
            if record.solution is None
            else {
                "status": record.solution.status,
                "iterations": record.solution.iterations,
                "accepted_iterations": record.solution.accepted_iterations,
                "propellant_kg": record.solution.propellant_kg,
                "max_defect": record.solution.max_defect,
                "solve_seconds": record.solution.solve_seconds,
                "diagnostic": record.solution.diagnostic,
                "solver_reports": record.solution.solver_reports,
                "history": record.solution.history,
                "outer_transfer_bytes": record.solution.outer_transfer_bytes,
                "seed_backend": record.solution.seed_backend,
            },
        }

    def certify_route(self, plan, refined):
        definitions = [
            p.RouteDefinition(
                0,
                1,
                tuple(plan.asteroids),
                tuple((0, item.request.deterministic_id) for item in refined),
            )
        ]
        arc_results = {(0, item.request.deterministic_id): item.result for item in refined}
        columns = p.build_route_columns(definitions, arc_results, self.certifier)
        return p.solve_certified_route_master(columns, plan.asteroids).certified

    @property
    def telemetry(self):
        return {
            key: getattr(self.scheduler.telemetry, key)
            for key in ("submitted", "completed", "feasible", "failed", "batches")
        } | {
            "workspace_creations": self.adapter.workspace_creations,
            "numeric_updates": self.adapter.numeric_updates,
        }

    def close(self):
        self.adapter.close()


def refine_fixed(
    plan,
    catalogue,
    prescribed_cargo,
    *,
    settings,
    on_leg=None,
    deadline=math.inf,
    runner_factory=FrozenLegRunner,
):
    """One sequential native refinement, no cargo resizing or epoch search on any outcome."""
    validate_prescription(plan, prescribed_cargo)
    cargo = {int(body): float(mass) for body, mass in prescribed_cargo.items()}
    original_cargo = dict(cargo)
    original_plan = plan.summary()
    flown = [leg for leg in plan.legs if leg.role != "camp"]
    started = time.perf_counter()
    mass = p.C.MAX_INITIAL_MASS_KG
    refined, failures = [], []
    runner = runner_factory(settings)
    telemetry, master_ok = {}, False
    try:
        for index, leg in enumerate(flown):
            if time.perf_counter() >= deadline:
                failures.append({"leg": index, "status": "campaign_deadline_before_leg"})
                break
            r0, v0 = p.body_state(catalogue, leg.from_id, leg.departure_epoch)
            rf, vf = p.body_state(catalogue, leg.to_id, leg.arrival_epoch)
            carried = sum(cargo[a] for a in cargo if plan.collect_epochs[a] <= leg.departure_epoch)
            boundary = p.LegBoundary(
                leg.departure_epoch,
                r0,
                v0,
                leg.arrival_epoch,
                rf,
                vf,
                mass,
                free_departure_vinf=leg.from_id == p.EARTH_ID,
                free_arrival_vinf=leg.to_id == p.EARTH_ID,
                minimum_final_mass=p.C.DRY_MASS_KG + carried,
            )
            try:
                item, record = runner.solve(leg, boundary, index)
            except Exception as error:
                failures.append(
                    {"leg": index, "status": "solver_path_exception", "error": repr(error)}
                )
                break
            refined.append(item)
            if on_leg is not None:
                on_leg(index, item, record)
            if not item.certified or not math.isfinite(item.mass_after_leg):
                failures.append(record)
                break
            mass = item.mass_after_leg
            # Same event ordering as the validated pipeline: deploy at arrival;
            # collect at arrival, or collect when leaving a camp before the next leg.
            if leg.to_id != p.EARTH_ID:
                if (
                    leg.to_id in plan.deploy_epochs
                    and abs(plan.deploy_epochs[leg.to_id] - leg.arrival_epoch) < 1e-6
                ):
                    mass -= p.C.MINER_MASS_KG
                elif (
                    leg.to_id in plan.collect_epochs
                    and abs(plan.collect_epochs[leg.to_id] - leg.arrival_epoch) < 1e-6
                ):
                    mass += cargo[leg.to_id]
            next_leg = flown[index + 1] if index + 1 < len(flown) else None
            if (
                next_leg is not None
                and next_leg.from_id == leg.to_id
                and next_leg.from_id in plan.collect_epochs
                and abs(plan.collect_epochs[next_leg.from_id] - next_leg.departure_epoch) < 1e-6
                and abs(plan.collect_epochs[next_leg.from_id] - leg.arrival_epoch) >= 1e-6
            ):
                mass += cargo[next_leg.from_id]
        all_legs = len(refined) == len(flown) and all(item.certified for item in refined)
        final_dry = mass - sum(cargo.values()) if all_legs else math.nan
        physical_mass_ok = (
            all_legs and math.isfinite(final_dry) and final_dry >= p.C.DRY_MASS_KG - 1e-9
        )
        if all_legs and not physical_mass_ok:
            failures.append(
                {
                    "status": "fixed_cargo_mass_deficit",
                    "dry_mass_kg": final_dry,
                    "required_dry_mass_kg": p.C.DRY_MASS_KG,
                    "cargo_resized": False,
                }
            )
        if physical_mass_ok:
            master_ok = runner.certify_route(plan, refined)
            if not master_ok:
                failures.append({"status": "independent_route_master_rejected"})
        telemetry = runner.telemetry
    finally:
        runner.close()
    if (
        cargo != original_cargo
        or plan.summary() != original_plan
        or dict(prescribed_cargo) != original_cargo
    ):
        raise AssertionError("fixed schedule/cargo mutated during refinement")
    return p.RefinedRoute(
        plan=plan,
        legs=refined,
        collected_mass=cargo,
        final_mass_kg=final_dry,
        certified=bool(physical_mass_ok and master_ok),
        master_certified=bool(master_ok),
        passes=1,
        wall_seconds=time.perf_counter() - started,
        scheduler_telemetry=telemetry,
        failures=failures,
    )


def run_refinement_queue(
    queue,
    plan_lookup,
    *,
    catalogue,
    settings,
    verify_full_fleet,
    incumbent_weighted_kg,
    deadline,
    on_result,
    refine=refine_fixed,
):
    """Run the claimed finite jobs; require explicit full-fleet emission/check context.

    ``plan_lookup`` is a mapping or callable keyed by request SHA. Values may be
    captured summary JSON bytes, a summary dictionary, or an original RoutePlan.
    ``verify_full_fleet(refined, request_sha256)`` must emit that route into its
    supplied fleet context and run both checkers. It returns ``request_sha256``,
    ``result_sha256``, ``independent`` and ``official`` reports, each bound to
    those Result bytes.
    The callback's independent report supplies verified raw and weighted scores.

    Every outcome is sent to ``on_result`` before the next claim. Claimed failures
    consume budget. A ``verified_gain`` outcome is a candidate for the caller to
    persist; this function never mutates an incumbent or a route master. The outer
    process supervisor must enforce a hard timeout for a stuck native call.
    """
    if settings.outer_loop_backend != "cuda":
        raise ValueError("the admission queue requires explicit native CUDA refinement settings")
    if not math.isfinite(deadline) or not math.isfinite(incumbent_weighted_kg):
        raise ValueError("a finite deadline and incumbent weighted score are required")
    outcomes = []
    while time.perf_counter() < deadline:
        candidate = queue.claim_next()
        if candidate is None:
            break
        request = candidate.request
        outcome = {
            "request_sha256": request.sha256,
            "completion_readback_sha256": candidate.observation.readback_sha256,
            "uncertain_proxy": candidate.uncertain,
            "claimed_index": queue.claimed,
            "status": "failed",
            "refinement_started": False,
            "accepted": False,
        }
        try:
            # The caller's retained plan cannot be mutated through an injected refiner.
            captured = (
                plan_lookup[request.sha256]
                if isinstance(plan_lookup, Mapping)
                else plan_lookup(request.sha256)
            )
            if isinstance(captured, bytes):
                captured = json.loads(captured)
            plan = (
                p.RoutePlan.from_summary(deepcopy(captured))
                if isinstance(captured, dict)
                else deepcopy(captured)
            )
            cargo = dict(plan.collected_mass)
            if FixedCargoRequest.from_summary(plan.summary(), cargo) != request:
                raise ValueError("captured plan does not match the queued prescription")
            outcome["refinement_started"] = True
            route = refine(plan, catalogue, cargo, settings=settings, deadline=deadline)
            outcome["route"] = route
            returned = FixedCargoRequest.from_summary(route.plan.summary(), route.collected_mass)
            if returned != request:
                outcome["blockers"] = ["prescribed_events_or_cargo_changed"]
            elif (
                not route.certified
                or not route.master_certified
                or len(route.legs) != sum(leg.role != "camp" for leg in plan.legs)
                or not all(x.certified for x in route.legs)
            ):
                outcome["blockers"] = ["uncertified_route"]
            elif time.perf_counter() >= deadline:
                outcome["blockers"] = ["deadline_before_full_fleet_verification"]
            else:
                checked = verify_full_fleet(route, request.sha256)
                outcome["fleet_verification"] = checked
                if checked.get("request_sha256") != request.sha256:
                    raise ValueError("full-fleet verification is not bound to this request")
                blockers = promotion_blockers(
                    request,
                    returned,
                    all_legs_certified=bool(route.legs) and all(x.certified for x in route.legs),
                    result_sha256=checked["result_sha256"],
                    independent=checked["independent"],
                    official=checked["official"],
                    incumbent_weighted_kg=incumbent_weighted_kg,
                )
                outcome["blockers"] = list(blockers)
                outcome["status"] = "verified_gain" if not blockers else "verification_rejected"
                outcome["accepted"] = not blockers
        except Exception as error:
            outcome["error"] = repr(error)
        outcomes.append(outcome)
        on_result(outcome)
    return outcomes
