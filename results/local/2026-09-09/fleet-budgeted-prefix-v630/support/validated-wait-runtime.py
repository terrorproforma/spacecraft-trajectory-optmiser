"""Factory wiring, explicit wait certificates and exact fleet section replacement."""

from __future__ import annotations

import dataclasses
import math
import time

import common as C
import numpy as np
from observe import save_solution


def next_mass(plan, previous, current, certified_mass, cargo):
    mass = certified_mass
    body = previous.to_id
    if body != 0:
        if (
            body in plan.deploy_epochs
            and abs(plan.deploy_epochs[body] - previous.arrival_epoch) < 1e-6
        ):
            mass -= 40.0
        elif (
            body in plan.collect_epochs
            and abs(plan.collect_epochs[body] - previous.arrival_epoch) < 1e-6
        ):
            mass += cargo[body]
    if (
        current is not None
        and current.from_id == body
        and body in plan.collect_epochs
        and abs(plan.collect_epochs[body] - current.departure_epoch) < 1e-6
        and abs(plan.collect_epochs[body] - previous.arrival_epoch) >= 1e-6
    ):
        mass += cargo[body]
    return mass


def wait_metrics(raw, initial, target):
    from spacepdhcg.gtoc12 import constants as K

    final = np.asarray(raw["final_state"])
    values = {
        "native_status": int(raw["status"]),
        "final_state": final,
        "position_error_km": float(np.linalg.norm(final[:3] - target[:3])),
        "velocity_error_km_s": float(np.linalg.norm(final[3:6] - target[3:6])),
        "mass_difference_kg": float(final[6] - initial[6]),
        "minimum_sun_distance_au": float(raw["minimum_radius_km"] / K.AU_KM),
        "position_limit_km": 0.5 * K.TOLERANCE_POSITION_KM,
        "velocity_limit_km_s": 0.5 * K.TOLERANCE_VELOCITY_KM_S,
        "mass_limit_kg": K.TOLERANCE_MASS_KG,
    }
    values["passed"] = bool(
        np.isfinite(final).all()
        and np.isfinite(raw["minimum_radius_km"])
        and int(raw["status"]) == 0
        and values["position_error_km"] <= values["position_limit_km"]
        and values["velocity_error_km_s"] <= values["velocity_limit_km_s"]
        and abs(values["mass_difference_kg"]) <= values["mass_limit_kg"]
        and values["minimum_sun_distance_au"] >= K.MIN_SUN_DISTANCE_AU
    )
    return values


def gpu_wait(descriptor, initial, observer):
    from spacepdhcg.gtoc12.gpu_verifier import ARC, LEG, SAMPLE, GpuVerifier

    legs = np.zeros(1, dtype=LEG)
    legs[0]["initial"] = initial
    legs[0]["duration_s"] = (descriptor["tf"] - descriptor["t0"]) * 86400
    with observer.wait_scope(descriptor["id"]), GpuVerifier(1, 0, 0) as gpu:
        output = gpu.propagate(legs, np.zeros(0, ARC), np.zeros(0, SAMPLE))
    return output[0]


class Coordinator:
    def __init__(self, plan, output, observer, deadline, *, wait_runner=gpu_wait):
        self.plan, self.output, self.observer, self.deadline = plan, output, observer, deadline
        self.cargo = dict(plan.collected_mass)
        self.flown = [leg for leg in plan.legs if leg.role != "camp"]
        self.boundaries = C.read(C.KIT / "inputs/boundaries.json")
        self.waits = C.read(C.KIT / "inputs/waits.json")
        self.grids = np.load(C.KIT / "inputs/grids.npz")
        self.wait_runner = wait_runner
        self.items, self.completed_waits, self.effective = {}, {}, {}
        (output / "legs").mkdir()
        (output / "waits").mkdir()

    def close(self):
        self.grids.close()

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("fixed route deadline before next operation")

    def boundary_factory(self, leg, boundary, index):
        self.check_time()
        self.observer.index = index
        record = self.boundaries[index]
        assert (leg.from_id, leg.to_id, leg.departure_epoch, leg.arrival_epoch) == (
            record["from"],
            record["to"],
            record["t0"],
            record["tf"],
        )
        for name, expected in record["generated"].items():
            assert np.array_equal(getattr(boundary, name), np.asarray(expected)), (index, name)
        result = dataclasses.replace(
            boundary, **{name: np.asarray(value) for name, value in record["candidate"].items()}
        )
        self.effective[index] = result
        directory = self.output / "legs" / f"{index:02d}"
        directory.mkdir()
        C.write(
            directory / "boundary.json",
            {"generated": boundary, "candidate": result, "archive_representation": record},
        )
        return result

    def run_wait(self, name, initial, provenance):
        self.check_time()
        if name in self.completed_waits:
            raise RuntimeError("each fixed wait may run once")
        descriptor = next(row for row in self.waits if row["id"] == name)
        self.completed_waits[name] = {"passed": False, "started": True}
        directory = self.output / "waits"
        C.write(
            directory / (name + "-input.json"),
            {"wait": descriptor, "initial_state": initial, "provenance": provenance},
        )
        try:
            raw = self.wait_runner(descriptor, initial, self.observer)
            values = wait_metrics(raw, initial, np.asarray(descriptor["target_rv"]))
            self.completed_waits[name] = values
            C.write(directory / (name + "-result.json"), values)
            if not values["passed"]:
                raise RuntimeError("independent waiting-coast certificate failed: " + name)
        except BaseException as error:
            C.write(directory / (name + "-failure.json"), {"error": repr(error)})
            raise

    def seed_factory(self, leg, boundary, index):
        from spacepdhcg.gtoc12.trajectory_seed import ZohTrajectorySeed

        self.check_time()
        if index == 0:
            expected_mass = 3000.0
            predecessor = None
        else:
            previous = self.items[index - 1]
            assert previous.certified and previous.certificate is not None
            expected_mass = next_mass(
                self.plan,
                self.flown[index - 1],
                leg,
                previous.certificate.final_mass_kg,
                self.cargo,
            )
            predecessor = C.sha(self.output / "legs" / f"{index - 1:02d}" / "leg-result.json")
        assert boundary.initial_mass == expected_mass, (
            "mass must come from previous actual certificate"
        )
        if index == 9:
            descriptor = self.waits[0]
            self.run_wait(
                descriptor["id"],
                np.r_[descriptor["initial_body_rv"], boundary.initial_mass],
                {
                    "origin": "real submitted deployment-after event at67075",
                    "preceding_leg_certificate_sha256": predecessor,
                },
            )
        row = self.boundaries[index]
        initial = np.r_[row["seed_initial_position_velocity"], boundary.initial_mass]
        key = f"leg_{index:02d}"
        epochs, thrust = self.grids[key + "_node_epochs_mjd"], self.grids[key + "_thrust_n"]
        seed = ZohTrajectorySeed(epochs, initial, thrust, C.RESULT_SHA)
        directory = self.output / "legs" / f"{index:02d}"
        np.savez_compressed(
            directory / "seed.npz",
            initial_state=seed.initial_state,
            thrust_n=seed.thrust_n,
            node_epochs_mjd=seed.node_epochs_mjd,
        )
        C.write(
            directory / "seed.json",
            {
                "source_result_sha256": C.RESULT_SHA,
                "archive_grid_file_sha256": C.sha(C.KIT / "inputs/grids.npz"),
                "payload_sha256": C.sha(directory / "seed.npz"),
                "source_initial_mass_kg": row["archived_initial_mass_kg"],
                "actual_initial_mass_kg": boundary.initial_mass,
                "mass_delta_kg": boundary.initial_mass - row["archived_initial_mass_kg"],
                "initial_rv_origin": row["initial_rv_origin"],
                "predecessor_certificate_sha256": predecessor,
                "trajectory_payload_bytes": (7 + 3 * len(epochs)) * 8,
                "time_array_bytes_separate": len(epochs) * 8,
            },
        )
        return seed

    def on_leg(self, index, item, details):
        directory = self.output / "legs" / f"{index:02d}"
        if item.solution is not None:
            save_solution(item.solution, directory, "post-clamp")
        C.write(directory / "leg-result.json", details)
        self.items[index] = item
        if not item.certified:
            return
        assert item.certificate is not None and math.isfinite(item.mass_after_leg)
        assert item.mass_after_leg == item.certificate.final_mass_kg
        assert item.solution is not None and item.solution.iterations <= 44
        if index == 9:
            captured = self.observer.raw[("flight", 9)]
            assert captured["status"] == 0 and captured["rows"][0]["status"] == 0
            actual = captured["rows"][0]["final_state"].copy()
            assert actual[6] == item.certificate.final_mass_kg
            self.run_wait(
                "after-leg09",
                actual,
                {
                    "origin": (
                        "actual GPU flight certificate final_state at67418; no event/body reset"
                    ),
                    "raw_GPU_certificate_sha256": captured["sha256"],
                },
            )

    def all_waits_passed(self):
        return set(self.completed_waits) == {"before-leg09", "after-leg09"} and all(
            row["passed"] for row in self.completed_waits.values()
        )


def emit_fleet(route, coordinator):
    from spacepdhcg.gtoc12.solution import (
        Event,
        ShipTrajectory,
        Solution,
        StateLine,
        format_solution,
    )

    if not route.certified or len(route.legs) != 19 or not coordinator.all_waits_passed():
        raise ValueError("complete route and both waits must be independently certified")
    assert dict(route.collected_mass) == coordinator.cargo
    templates = C.read(C.KIT / "inputs/events.json")
    ship = ShipTrajectory(23)

    def line(template, mass, *, velocity=None):
        return StateLine(
            template["epoch"],
            np.asarray(template["position"]),
            np.asarray(template["velocity"]) if velocity is None else velocity,
            mass,
        )

    first = route.legs[0]
    launch = templates[0]
    ship.items.append(
        Event(
            0,
            line(launch["before"], first.mass_before),
            line(
                launch["after"],
                first.mass_before,
                velocity=first.solution.departure_ship_velocity_km_s(),
            ),
        )
    )
    carried = 0.0
    for index, item in enumerate(route.legs):
        assert item.certified and item.certificate is not None
        ship.items.extend(item.solution.burn_arcs())
        template = templates[index + 1]
        body, mass = template["event_id"], item.certificate.final_mass_kg
        if index < 9:
            after = mass - 40.0
        elif index < 18:
            gained = coordinator.cargo[body]
            after = mass + gained
            carried += gained
        else:
            after = mass - carried
        velocity = item.solution.arrival_ship_velocity_km_s() if index == 18 else None
        ship.items.append(
            Event(
                body,
                line(template["before"], mass, velocity=velocity),
                line(template["after"], after, velocity=velocity),
            )
        )
    assert len(ship.events) == 20
    assert [(x.event_id, x.epoch) for x in ship.events] == [
        (x["event_id"], x["before"]["epoch"]) for x in templates
    ]
    assert abs(ship.events[-1].after.mass - route.final_mass_kg) <= 1e-9
    original = (C.PRIOR / "inputs/incumbent.txt").read_bytes()
    assert C.digest(original) == C.RESULT_SHA
    replacement = format_solution(Solution([ship])).encode()
    fleet = C.splice_ship(original, replacement)
    assert C.ship_byte_parts(original)[0] == C.ship_byte_parts(fleet)[0]
    return fleet
