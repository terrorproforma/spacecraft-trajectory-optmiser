"""Import known certified prefixes, retain their exact samples/events, and append one return."""

# Frozen source activation must precede every production import.
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import support

support.activate()
from fixed_refine import FrozenLegRunner, validate_prescription

from spacepdhcg.gtoc12 import pipeline as p
from spacepdhcg.gtoc12.solution import Event, StateLine, format_solution
from spacepdhcg.gtoc12.verifier import propagate_coast


def request_for(leg, boundary, index):
    nodes = math.floor(leg.tof_days + 1e-9) + 2
    topology = p.TopologyKey(
        p._hash_int(f"{p.MODEL_IDENTIFIER}:{nodes}"), p.ArcFidelity.REFINED_SCVX, nodes, 1
    )
    return p.ArcRequest(
        deterministic_id=1000 + index,
        from_target=leg.from_id if leg.from_id else -3,
        to_target=leg.to_id if leg.to_id else -3,
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


def saved_result(runner, request, solution, certificate):
    status = p.G3Status.CONVERGED if solution.status == "converged" else p.G3Status.ITERATION_LIMIT
    solved = p.G3Solve(
        status=status,
        objective=solution.propellant_kg,
        lower_bound=0,
        duration=solution.boundary.duration_days,
        delta_v=solution.delta_v_km_s,
        propellant=solution.propellant_kg,
        final_mass=solution.final_mass_kg,
        canonical_residual=solution.max_defect,
        replay_residual=certificate.rk4_vs_dop853_km / p.C.TOLERANCE_POSITION_KM,
        path_violation=max(0, certificate.maximum_thrust_n - p.C.THRUST_MAX_N) / p.C.THRUST_MAX_N
        + max(0, p.C.MIN_SUN_DISTANCE_AU - certificate.minimum_sun_distance_au),
        terminal_error=max(
            certificate.position_error_km / p.C.TOLERANCE_POSITION_KM,
            certificate.velocity_error_km_s / p.C.TOLERANCE_VELOCITY_KM_S,
        ),
        uncertainty_violation=0,
        certifiable_candidate=True,
        diagnostic="replayed known solution; no new native solve",
    )
    result = runner.adapter._convert(request, solved)
    result.validate(request)
    return result


@dataclass
class Prefix:
    name: str
    plan: p.RoutePlan
    legs: list[p.RefinedLeg]
    fresh_certificates: list[p.LegCertificate]
    cargo: dict[int, float]
    after_mass: float
    audit: dict
    expected_events: list[Event]

    @property
    def collection_epoch(self):
        return self.plan.collect_epochs[59653]

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([self.plan.summary(), self.cargo], sort_keys=True).encode())
        for leg in self.legs:
            h.update(
                json.dumps(
                    [
                        dataclasses.asdict(leg.planned),
                        dataclasses.asdict(leg.certificate),
                        leg.mass_before,
                        leg.mass_after_leg,
                    ],
                    sort_keys=True,
                ).encode()
            )
            for array in (
                leg.solution.node_epochs_mjd,
                leg.solution.thrust_n,
                leg.solution.states_scaled,
                leg.solution.departure_vinf_km_s,
                leg.solution.arrival_vinf_km_s,
            ):
                h.update(array.tobytes())
        return h.hexdigest()

    def attach(self, runner):
        for leg, fresh in zip(self.legs, self.fresh_certificates, strict=True):
            runner.registry.register(leg.request, leg.solution.boundary)
            record = runner.registry.records[leg.request.deterministic_id]
            record.solution, record.certificate = leg.solution, fresh
            record.result = leg.result
            if not runner.certifier.certify(leg.result).accepted:
                raise ValueError("restored prefix failed independent certification")


def load_prefix(name, catalogue, *, include_archived_return=False, recertify=True):
    if name not in ("control", "probe") or (include_archived_return and name != "control"):
        raise ValueError("only the previously certified control may include an archived return")
    directory = support.ROOT / "inputs" / name
    summary = support.read(directory / "refinement.json")
    prescription = support.read(directory / "prescription.json")
    plan = p.RoutePlan.from_summary(summary["plan"])
    cargo = {int(body): float(mass) for body, mass in summary["collected_mass_kg"].items()}
    if cargo != {
        int(body): float(mass) for body, mass in prescription["prescribed_cargo_kg"].items()
    }:
        raise ValueError("saved cargo differs from prescription")
    validate_prescription(plan, cargo)
    if name == "control":
        checked = support.read(directory / "verification.json")
        if not (checked["ok"] and checked["independent"]["ok"] and checked["official"]["ok"]):
            raise ValueError("original control lacks both full-fleet certificates")
    count = 17 if include_archived_return else 16
    planned = [leg for leg in plan.legs if leg.role != "camp"]
    if len(planned) != 17 or planned[-1].from_id != 59653 or planned[-1].to_id != 0:
        raise ValueError("unexpected final return topology")
    settings = p.ScvxSettings(**support.read(support.ROOT / "preparation.json")["scvx_settings"])
    runner = FrozenLegRunner(settings)
    legs, fresh_certificates, audits = [], [], []
    mass = p.C.MAX_INITIAL_MASS_KG
    try:
        for index in range(count):
            detail = support.read(directory / f"leg-{index:02d}.json")
            old = detail["certificate"]
            s = detail["solution"]
            if (
                not detail["certified"]
                or detail["status"] != "feasible"
                or old is None
                or s is None
            ):
                raise ValueError("uncertified prefix leg cannot be imported")
            if s["status"] not in ("converged", "iteration_limit"):
                raise ValueError("invalid archived solver outcome")
            leg = planned[index]
            if (detail["from"], detail["to"], detail["departure"], detail["arrival"]) != (
                leg.from_id,
                leg.to_id,
                leg.departure_epoch,
                leg.arrival_epoch,
            ):
                raise ValueError("saved leg epochs or bodies changed")
            if abs(mass - detail["initial_mass_kg"]) > 1e-7:
                raise ValueError("prefix event mass ledger changed")
            arrays_path = directory / f"leg-{index:02d}-solution.npz"
            if support.sha(arrays_path) != detail["solution_arrays"]["sha256"]:
                raise ValueError("prefix solution array hash mismatch")
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            for array in arrays.values():
                if not np.isfinite(array).all():
                    raise ValueError("nonfinite archived trajectory array")
                array.flags.writeable = False
            if (
                arrays["node_epochs"][0] != leg.departure_epoch
                or arrays["node_epochs"][-1] != leg.arrival_epoch
            ):
                raise ValueError("saved solution grid has different endpoints")
            r0, v0 = p.body_state(catalogue, leg.from_id, leg.departure_epoch)
            rf, vf = p.body_state(catalogue, leg.to_id, leg.arrival_epoch)
            carried = sum(
                cargo[body] for body in cargo if plan.collect_epochs[body] <= leg.departure_epoch
            )
            if detail["minimum_final_mass_kg"] != p.C.DRY_MASS_KG + carried:
                raise ValueError("archived minimum mass differs from fixed cargo")
            boundary = p.LegBoundary(
                leg.departure_epoch,
                r0,
                v0,
                leg.arrival_epoch,
                rf,
                vf,
                detail["initial_mass_kg"],
                leg.from_id == 0,
                leg.to_id == 0,
                detail["minimum_final_mass_kg"],
            )
            solution = p.LegSolution(
                status=s["status"],
                boundary=boundary,
                node_epochs_mjd=arrays["node_epochs"],
                thrust_n=arrays["thrust_n"],
                states_scaled=arrays["states_scaled"],
                departure_vinf_km_s=arrays["departure_vinf"],
                arrival_vinf_km_s=arrays["arrival_vinf"],
                final_mass_kg=boundary.initial_mass - s["propellant_kg"],
                propellant_kg=s["propellant_kg"],
                delta_v_km_s=summary["legs"][index]["delta_v_km_s"],
                iterations=s["iterations"],
                accepted_iterations=s["accepted_iterations"],
                max_defect=s["max_defect"],
                virtual_inf=math.nan,
                solve_seconds=s["solve_seconds"],
                hold="zoh",
                history=s["history"],
                solver_reports=s["solver_reports"],
                discretisation_backend="cuda",
                assembly_backend="cuda",
                convex_solver_backend="qoco",
                outer_loop_backend="cuda",
                seed_backend="cuda",
            )
            archived = p.LegCertificate(**old)
            fresh = p.certify_leg(solution) if recertify else archived
            if abs(fresh.final_mass_kg - archived.final_mass_kg) > 1e-7:
                raise ValueError("fresh prefix rollout changed certified event mass")
            request = request_for(leg, boundary, index)
            runner.registry.register(request, boundary)
            record = runner.registry.records[request.deterministic_id]
            record.solution, record.certificate = solution, fresh
            result = saved_result(runner, request, solution, fresh)
            if not runner.certifier.certify(result).accepted:
                raise ValueError("fresh prefix certificate rejected")
            legs.append(
                p.RefinedLeg(
                    leg,
                    request,
                    result,
                    solution,
                    archived,
                    True,
                    boundary.initial_mass,
                    archived.final_mass_kg,
                )
            )
            fresh_certificates.append(fresh)
            audits.append(
                {
                    "index": index,
                    "fresh_rollout": recertify,
                    "fresh_certificate": dataclasses.asdict(fresh),
                    "preserved_event_mass_kg": archived.final_mass_kg,
                }
            )
            mass = archived.final_mass_kg
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
            next_leg = planned[index + 1] if index + 1 < len(planned) else None
            if (
                next_leg
                and next_leg.from_id in plan.collect_epochs
                and abs(plan.collect_epochs[next_leg.from_id] - next_leg.departure_epoch) < 1e-6
                and abs(plan.collect_epochs[next_leg.from_id] - leg.arrival_epoch) >= 1e-6
            ):
                mass += cargo[next_leg.from_id]
    finally:
        runner.close()
    if not include_archived_return:
        saved_return = support.read(directory / "leg-16.json")
        if mass != saved_return["initial_mass_kg"]:
            raise ValueError("return-start mass differs from saved prefix")
    prefix = Prefix(
        name,
        plan,
        legs,
        fresh_certificates,
        cargo,
        mass,
        {
            "fresh_rollout": recertify,
            "certified_legs": count,
            "legs": audits,
            "missing_archived_virtual_inf": "not used in replay, emission or certification",
        },
        [],
    )
    prefix.expected_events = prefix_events(prefix, catalogue)
    return prefix


def prefix_events(prefix, catalogue):
    """Event ledger of an incomplete prefix, without asserting whole-route certification."""
    first = prefix.legs[0]
    epoch = first.planned.departure_epoch
    r, v = p.body_state(catalogue, 0, epoch)
    events = [
        Event(
            0,
            StateLine(epoch, r.copy(), v.copy(), first.mass_before),
            StateLine(
                epoch, r.copy(), first.solution.departure_ship_velocity_km_s(), first.mass_before
            ),
        )
    ]
    flights = [leg for leg in prefix.plan.legs if leg.role != "camp"]
    for index, leg in enumerate(prefix.legs):
        planned = leg.planned
        if planned.to_id == 0:
            continue
        body, epoch = planned.to_id, planned.arrival_epoch
        before = leg.mass_after_leg
        next_leg = flights[index + 1] if index + 1 < len(flights) else None
        at_departure = (
            next_leg
            and body in prefix.plan.collect_epochs
            and abs(prefix.plan.collect_epochs[body] - next_leg.departure_epoch) < 1e-6
            and abs(prefix.plan.collect_epochs[body] - epoch) >= 1e-6
        )
        if (
            body in prefix.plan.deploy_epochs
            and abs(prefix.plan.deploy_epochs[body] - epoch) < 1e-6
        ):
            after = before - p.C.MINER_MASS_KG
        elif (
            body in prefix.plan.collect_epochs
            and abs(prefix.plan.collect_epochs[body] - epoch) < 1e-6
        ):
            after = before + prefix.cargo[body]
        elif at_departure:
            after = before
        else:
            raise ValueError("prefix arrival lacks an original event")
        r, v = p.body_state(catalogue, body, epoch)
        if not at_departure or before != after:
            events.append(
                Event(
                    body,
                    StateLine(epoch, r.copy(), v.copy(), before),
                    StateLine(epoch, r.copy(), v.copy(), after),
                )
            )
        if at_departure:
            epoch = next_leg.departure_epoch
            r, v = p.body_state(catalogue, body, epoch)
            events.append(
                Event(
                    body,
                    StateLine(epoch, r.copy(), v.copy(), after),
                    StateLine(epoch, r.copy(), v.copy(), after + prefix.cargo[body]),
                )
            )
    return events


def return_plan(prefix, departure, arrival, delta_v=0.0):
    if len(prefix.legs) != 16 or not all(
        math.isfinite(value) for value in (departure, arrival, delta_v)
    ):
        raise ValueError("invalid prefix or nonfinite return window")
    if not prefix.collection_epoch <= departure < arrival <= p.C.MISSION_END_MJD:
        raise ValueError("return window violates fixed collection or mission limits")
    old = prefix.plan.legs[-1]
    if old.from_id != 59653 or old.to_id != 0 or old.role != "earth_return":
        raise ValueError("original route does not end with expected Earth return")
    legs = list(prefix.plan.legs[:-1])
    if departure > prefix.collection_epoch:
        legs.append(p.PlannedLeg(59653, 59653, prefix.collection_epoch, departure, 0, 1, "camp"))
    last = p.PlannedLeg(59653, 0, departure, arrival, delta_v, 1, "earth_return")
    legs.append(last)
    plan = dataclasses.replace(
        prefix.plan,
        legs=tuple(legs),
        deploy_epochs=dict(prefix.plan.deploy_epochs),
        collect_epochs=dict(prefix.plan.collect_epochs),
        collected_mass=dict(prefix.cargo),
    )
    validate_prescription(plan, prefix.cargo)
    return plan, last


def certify_wait(prefix, catalogue, departure):
    if not prefix.collection_epoch <= departure < p.C.MISSION_END_MJD:
        raise ValueError("invalid waiting interval")
    r0, v0 = p.body_state(catalogue, 59653, prefix.collection_epoch)
    rf, vf = p.body_state(catalogue, 59653, departure)
    r, v, minimum = propagate_coast(prefix.collection_epoch, r0, v0, prefix.after_mass, departure)
    position = float(np.linalg.norm(r - rf))
    velocity = float(np.linalg.norm(v - vf))
    index = catalogue.index_of(59653)
    perihelion = (
        catalogue.semi_major_axis_km[index] * (1 - catalogue.eccentricity[index]) / p.C.AU_KM
    )
    accepted = (
        position / p.C.TOLERANCE_POSITION_KM <= 0.5
        and velocity / p.C.TOLERANCE_VELOCITY_KM_S <= 0.5
        and min(minimum / p.C.AU_KM, perihelion) >= p.C.MIN_SUN_DISTANCE_AU
    )
    return {
        "accepted": accepted,
        "departure_mjd": prefix.collection_epoch,
        "arrival_mjd": departure,
        "position_error_km": position,
        "velocity_error_km_s": velocity,
        "minimum_sun_distance_au": minimum / p.C.AU_KM,
        "analytic_perihelion_lower_bound_au": float(perihelion),
        "mass_preserved_kg": prefix.after_mass,
        "additional_asteroid_events": 0,
    }


def solve_return(
    prefix,
    catalogue,
    departure,
    arrival,
    settings,
    *,
    delta_v=0,
    runner_factory=FrozenLegRunner,
    on_result=None,
):
    before = prefix.fingerprint()
    plan, last = return_plan(prefix, departure, arrival, delta_v)
    coast = certify_wait(prefix, catalogue, departure)
    if not coast["accepted"]:
        return None, {"status": "independent_wait_rejected", "coast": coast}
    r0, v0 = p.body_state(catalogue, 59653, departure)
    rf, vf = p.body_state(catalogue, 0, arrival)
    boundary = p.LegBoundary(
        departure,
        r0,
        v0,
        arrival,
        rf,
        vf,
        prefix.after_mass,
        free_departure_vinf=False,
        free_arrival_vinf=True,
        minimum_final_mass=500 + sum(prefix.cargo.values()),
    )
    runner = runner_factory(settings)
    try:
        prefix.attach(runner)
        item, details = runner.solve(last, boundary, 16)
        details["coast"] = coast
        if on_result is not None:
            on_result(item, details)
        if not item.certified or not math.isfinite(item.mass_after_leg):
            return None, details | {"status": "native_return_uncertified"}
        if item.planned != last or item.mass_before != prefix.after_mass:
            raise ValueError("return solver changed boundary or schedule")
        if (
            item.solution.node_epochs_mjd[0] != departure
            or item.solution.node_epochs_mjd[-1] != arrival
        ):
            raise ValueError("return solver grid changed prescribed epochs")
        dry = item.mass_after_leg - sum(prefix.cargo.values())
        if dry < 500 - 1e-9:
            return None, details | {"status": "fixed_cargo_mass_deficit", "final_dry_mass_kg": dry}
        joined = [*prefix.legs, item]
        master_ok = runner.certify_route(plan, joined)
        if not master_ok:
            return None, details | {"status": "independent_route_master_rejected"}
        route = p.RefinedRoute(
            plan,
            joined,
            dict(prefix.cargo),
            dry,
            True,
            True,
            1,
            item.solution.solve_seconds,
            runner.telemetry,
            [],
        )
        return route, details | {"status": "return_and_route_certified", "final_dry_mass_kg": dry}
    except Exception as error:
        return None, {"status": "return_path_exception", "error": repr(error), "coast": coast}
    finally:
        runner.close()
        if prefix.fingerprint() != before:
            raise AssertionError("immutable prefix changed during return solve")


def assert_prefix_emission(prefix, solution):
    """Check event inventory/cargo and byte-identical prefix burn samples before fleet checking."""
    ship = solution.ships[0]
    events = ship.events[: len(prefix.expected_events)]
    if len(events) != len(prefix.expected_events):
        raise ValueError("emission dropped a prefix event")
    for actual_event, expected_event in zip(events, prefix.expected_events, strict=True):
        if actual_event.event_id != expected_event.event_id:
            raise ValueError("emission changed prefix event identity")
        for actual_state, expected_state in (
            (actual_event.before, expected_event.before),
            (actual_event.after, expected_event.after),
        ):
            if (
                actual_state.epoch != expected_state.epoch
                or actual_state.mass != expected_state.mass
                or not np.array_equal(actual_state.position, expected_state.position)
                or not np.array_equal(actual_state.velocity, expected_state.velocity)
            ):
                raise ValueError("emission changed a preserved prefix state")
    actual = [event for event in ship.events if event.is_asteroid]
    expected = sorted(
        (body, epoch)
        for phase in (prefix.plan.deploy_epochs, prefix.plan.collect_epochs)
        for body, epoch in phase.items()
    )
    if sorted((event.event_id, event.epoch) for event in actual) != expected:
        raise ValueError("emission changed collection/deployment events or added a visit")
    for event in actual:
        if event.epoch == prefix.plan.collect_epochs.get(event.event_id):
            if abs(event.after.mass - event.before.mass - prefix.cargo[event.event_id]) > 1e-10:
                raise ValueError("emission changed prescribed cargo")
    emitted = [arc for arc in ship.burns if arc.end <= prefix.collection_epoch]
    expected_arcs = [arc for leg in prefix.legs for arc in leg.solution.burn_arcs()]
    if len(emitted) != len(expected_arcs):
        raise ValueError("emission changed prefix burn count")
    for left, right in zip(emitted, expected_arcs, strict=True):
        for a, b in zip(left.samples, right.samples, strict=True):
            if a.epoch != b.epoch or not np.array_equal(a.thrust, b.thrust):
                raise ValueError("emission changed certified prefix samples")
    return {
        "asteroid_events": len(actual),
        "prefix_burns_preserved": len(expected_arcs),
        "solution_sha256": hashlib.sha256(format_solution(solution).encode()).hexdigest(),
    }
