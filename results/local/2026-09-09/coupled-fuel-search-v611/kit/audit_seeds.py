"""CPU reference seed checks; native loading prohibited and no optimization loop."""

import ctypes
import json
import os
from unittest.mock import patch

import common


def main():
    for key in list(os.environ):
        if key.startswith(("SPACEPDHCG_", "QOCO_")):
            del os.environ[key]
    os.environ["SPACEPDHCG_GTOC12_DATA"] = common.read(common.ROOT / "profile.json")["data"]
    os.environ["SPACEPDHCG_TEST_GTOC12_JOINT_BATCH"] = "0"
    common.activate()
    from domain import bounds, joint_for, physical_plan, routes, seeds

    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.jointopt import weighted_mass
    from spacepdhcg.gtoc12.solution import Solution

    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    control, probe = routes()
    fleet = Solution.read(common.ROOT / "inputs/Result.txt")
    others = {
        e.event_id for ship in fleet.ships if ship.ship_id != 7 for e in ship.asteroid_visits()
    }
    if set(probe.plan.asteroids) & others:
        raise AssertionError("Probe asteroid conflicts with retained other ships")
    rows = []
    for seed in seeds(probe.plan):
        joint, _ = joint_for(catalogue, weights, 0.05)
        plan = physical_plan(seed["visits"], seed["arrivals"], seed["departures"])
        ev = joint.evaluate(seed["visits"], seed["arrivals"], seed["departures"])
        rows.append(
            {
                "id": seed["id"],
                "arrivals": seed["arrivals"].tolist(),
                "departures": seed["departures"].tolist(),
                "physical_inventory_ok": True,
                "maximum_raw_cargo": plan.total_collected_kg,
                "maximum_weighted_cargo": weighted_mass(plan.collected_mass, weights),
                "initial_CPU_reference_feasible": ev.feasible,
                "failure": ev.failure,
                "evaluated_weighted_kg": ev.weighted_kg,
                "evaluated_raw_kg": ev.collected_kg,
                "estimated_spare_kg": ev.spare_kg,
                "estimated_propellant_kg": ev.propellant_kg,
                "measured_legs": ev.measured_legs,
            }
        )
    result = {
        "GPU_calls": 0,
        "CPU_reference_evaluations": len(rows),
        "original": {
            "raw_kg": control.total_collected_kg,
            "weighted_kg": weighted_mass(control.collected_mass, weights),
        },
        "bounds": bounds(),
        "seeds": rows,
        "other_ship_footprint_disjoint": True,
    }
    destination = common.ROOT / "seed-audit-01.json"
    if destination.exists():
        raise FileExistsError("Preserve prior CPU audit")
    common.write(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("CPU preflight cannot load native libraries")
    ):
        main()
