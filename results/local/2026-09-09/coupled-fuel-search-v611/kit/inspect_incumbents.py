"""CPU-only reference inventory; identify feasible measured-cost starting routes."""

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
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
    )
    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.jointopt import (
        JointItinerary,
        JointSettings,
        route_from_summary,
        weighted_mass,
    )
    from spacepdhcg.gtoc12.retiming import Retimer, visits_of
    from spacepdhcg.gtoc12.solution import Solution

    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    fleet = Solution.read(common.ROOT / "inputs/Result.txt")
    source = common.ROOT.parent / "incident-window-substitution-v604/inputs"
    report = []
    for ship in fleet.ships:
        path = source / f"ship-{ship.ship_id:02d}.json"
        route = route_from_summary(common.read(path))
        events = ship.asteroid_visits()
        actual_collected = {}
        for event in events:
            if event.after.mass > event.before.mass:
                actual_collected[event.event_id] = event.after.mass - event.before.mass
        same = set(actual_collected) == set(route.collected_mass) and all(
            abs(actual_collected[a] - m) < 1e-8 for a, m in route.collected_mass.items()
        )
        independent = not route.plan.foreign_deploy_epochs and not route.plan.orphaned
        row = {
            "ship": ship.ship_id,
            "path": str(path),
            "sha256": common.sha(path),
            "independent_inventory": independent,
            "cargo_matches_retained_Result": same,
            "flight_legs": len(route.legs),
            "raw_kg": route.total_collected_kg,
            "weighted_kg": weighted_mass(route.collected_mass, weights),
            "certified_hop_propellant_kg": sum(
                leg.mass_before - leg.mass_after_leg
                for leg in route.legs
                if leg.planned.role not in ("earth_out", "earth_return")
            ),
            "maximum_hop_propellant_kg": max(
                leg.mass_before - leg.mass_after_leg
                for leg in route.legs
                if leg.planned.role not in ("earth_out", "earth_return")
            ),
        }
        if same and independent and len(route.legs) == 17:
            policy = ClusterPricingSettings()
            retimer = Retimer(
                catalogue,
                cluster_search_settings(policy, len(catalogue.ids)),
                cluster_retime_settings(policy, last=True),
                weights=weights,
            )
            retimer.protect_earth_leg(route.plan)
            joint = JointItinerary(
                catalogue,
                retimer,
                weights=weights,
                settings=JointSettings(insert=False, earth_leg=False),
            )
            joint.learn(route)
            visits, a, d = visits_of(route.plan)
            ev = joint.evaluate(visits, a, d)
            row.update(
                CPU_initial_feasible=ev.feasible,
                failure=ev.failure,
                visits=len(visits),
                measured_legs=ev.measured_legs,
            )
        report.append(row)
    result = {
        "GPU_calls": 0,
        "CPU_reference_evaluations": sum("CPU_initial_feasible" in r for r in report),
        "routes": report,
    }
    common.write(common.ROOT / "incumbent-inventory-01.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("CPU inventory cannot load native libraries")
    ):
        main()
