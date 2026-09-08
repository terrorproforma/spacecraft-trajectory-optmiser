"""CPU-only comparison of retained ship 23 with the saved finite generation."""

import ctypes
import dataclasses
import json
import math
import os
import sys
from collections import Counter
from unittest.mock import patch

from audit import KIT, ROOT, canonical, close, lines, read, sha


def nearest(value, options):
    result = min(options, key=lambda x: (abs(x - value), x))
    return {"exact": abs(result - value) < 1e-8, "nearest": result, "offset_days": result - value}


def main():
    ready = read(KIT / "ready-manifest.json")
    for name, digest in ready["files"].items():
        assert sha(KIT / name) == digest
    source = KIT / "source/src"
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(source))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "profile.json")["data"]
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("No native loading")):
        import numpy as np

        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.collectdp import CollectDPSettings
        from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
        from spacepdhcg.gtoc12.screening import propellant_for_delta_v, thrust_authority_km_s
        from spacepdhcg.gtoc12.search import RoutePlan, RouteSearch, SearchSettings

        saved = read(KIT / "output/report.json")
        values = saved["settings"]
        defaults = {field.name: field.default for field in dataclasses.fields(SearchSettings)}
        values = {
            key: (
                defaults[key]
                if value is None
                and isinstance(defaults[key], float)
                and not math.isfinite(defaults[key])
                else value
            )
            for key, value in values.items()
        }
        settings = SearchSettings(**values)
        inputs = read(KIT / "generation-input.json")
        catalogue, bonus = load_catalogue(), load_bonus_table()
        weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
        archive = read(KIT / "inputs/ship-23.json")
        original = RoutePlan.from_summary(archive["plan"])
        measured = archive["legs"]
        search = RouteSearch(
            catalogue,
            np.asarray(inputs["allowed_ids"]),
            settings,
            weights=weights,
            excluded=set(inputs["excluded_other_fleet_ids"]),
        )
        assert len(search.ids) == 62 and settings.neighbours == 61
        generated = read(KIT / "output/candidate-pool.json")
        audit = read(ROOT / "report-v2.json")
        events = lines(KIT / "output/generation-events.jsonl")
        completion_starts = [
            item["detail"]
            for item in events
            if item["stage"] == "completion_attempts" and item["event"] == "started"
        ]
        original_order = sorted(
            original.deploy_epochs, key=lambda a: (original.deploy_epochs[a], a)
        )
        collect_order = sorted(
            original.collect_epochs, key=lambda a: (original.collect_epochs[a], a)
        )
        closest = []
        for partial in completion_starts:
            order = [a for a, _epoch in partial["deployed"]]
            length = 0
            for a, b in zip(order, original_order, strict=False):
                if a != b:
                    break
                length += 1
            closest.append(
                {
                    "depth": partial["depth"],
                    "prefix_length": length,
                    "deployed": partial["deployed"],
                }
            )

        dp = CollectDPSettings()
        deploy_legs = [leg for leg in original.legs if leg.role == "deploy_hop"]
        collect_legs = [leg for leg in original.legs if leg.role == "collect_hop"]
        ret = next(leg for leg in original.legs if leg.role == "earth_return")
        flight_measured = {(leg["from"], leg["to"], leg["t0"], leg["tf"]): leg for leg in measured}
        grid_rows = []
        for leg in deploy_legs + collect_legs + [ret]:
            role = leg.role
            if role == "deploy_hop":
                previous_arrival = original.deploy_epochs[leg.from_id]
                row = {
                    "wait_grid": nearest(
                        leg.departure_epoch - previous_arrival, settings.deploy_wait_days
                    ),
                    "tof_grid": nearest(leg.tof_days, settings.hop_tofs),
                }
            else:
                options = dp.tofs if role == "collect_hop" else dp.return_tofs
                step = settings.collect_dp_step_days
                nearest_epoch = (
                    C.MISSION_START_MJD
                    + round((leg.departure_epoch - C.MISSION_START_MJD) / step) * step
                )
                row = {
                    "dp_tof_grid": nearest(leg.tof_days, options),
                    "dp_departure_grid": nearest(leg.departure_epoch, [nearest_epoch]),
                }
                if role == "collect_hop":
                    row["heuristic_tof_grid"] = nearest(leg.tof_days, settings.collect_hop_tofs)
            grid_rows.append(
                {
                    "role": role,
                    "from": leg.from_id,
                    "to": leg.to_id,
                    "departure": leg.departure_epoch,
                    "arrival": leg.arrival_epoch,
                    "tof_days": leg.tof_days,
                    **row,
                }
            )

        # This is not another beam run or a fresh Lambert evaluation. Frozen scalar
        # formulas are evaluated at sequential mass using already saved Lambert values.
        # It is not a trace of _finish: return model mass estimates can differ by builder.
        mass = settings.initial_mass - inputs["seed"]["propellant_kg"] - C.MINER_MASS_KG
        hop_spent = 0.0
        deploy_checks = []
        for depth, leg in enumerate(deploy_legs, start=2):
            actual = flight_measured[
                (leg.from_id, leg.to_id, leg.departure_epoch, leg.arrival_epoch)
            ]
            in_pool = leg.to_id in search.band_pool(leg.from_id)
            limit = settings.hop_authority_ratio * float(
                thrust_authority_km_s(mass, leg.tof_days, 1.0)
            )
            predicted = float(
                propellant_for_delta_v(mass, leg.delta_v_proxy_km_s * settings.hop_inflation)
            )
            mass -= predicted + C.MINER_MASS_KG
            hop_spent += predicted
            reserve = settings.reserve_fraction * hop_spent + settings.return_reserve_kg
            deploy_checks.append(
                {
                    "depth": depth,
                    "from": leg.from_id,
                    "to": leg.to_id,
                    "candidate_pool_contains_target": in_pool,
                    "saved_lambert_dv": leg.delta_v_proxy_km_s,
                    "authority_limit_at_sequential_proxy_mass": limit,
                    "authority_gate_passes": leg.delta_v_proxy_km_s <= limit,
                    "proxy_propellant_kg": predicted,
                    "measured_propellant_kg": actual["propellant_kg"],
                    "sequential_proxy_mass_after_deploy_kg": mass,
                    "measured_mass_after_deploy_kg": actual["mass_after"] - C.MINER_MASS_KG,
                    "reserve_kg": reserve,
                    "reserve_margin_kg": mass - C.DRY_MASS_KG - reserve,
                    "reserve_gate_passes": mass >= C.DRY_MASS_KG + reserve,
                }
            )
        collect_checks = []
        counted = set()
        for leg in [*collect_legs, ret]:
            a = leg.from_id
            if abs(original.collect_epochs[a] - leg.departure_epoch) < 1e-6:
                assert a not in counted
                mass += original.collected_mass[a]
                counted.add(a)
            actual = flight_measured[
                (leg.from_id, leg.to_id, leg.departure_epoch, leg.arrival_epoch)
            ]
            limit_ratio = (
                settings.hop_authority_ratio
                if leg.role == "collect_hop"
                else settings.earth_return_authority_ratio
            )
            limit = limit_ratio * float(thrust_authority_km_s(mass, leg.tof_days, 1.0))
            inflation = (
                settings.hop_inflation
                if leg.role == "collect_hop"
                else search.return_inflation_for(leg.delta_v_proxy_km_s, mass, leg.tof_days)
            )
            predicted = float(propellant_for_delta_v(mass, leg.delta_v_proxy_km_s * inflation))
            collect_checks.append(
                {
                    "role": leg.role,
                    "from": leg.from_id,
                    "to": leg.to_id,
                    "saved_lambert_dv": leg.delta_v_proxy_km_s,
                    "sequential_proxy_mass_before_kg": mass,
                    "authority_limit_at_sequential_proxy_mass": limit,
                    "authority_gate_passes": leg.delta_v_proxy_km_s <= limit,
                    "applied_proxy_inflation": inflation,
                    "proxy_propellant_kg": predicted,
                    "measured_propellant_kg": actual["propellant_kg"],
                }
            )
            mass -= predicted
        close(sum(original.collected_mass.values()), audit["ship_23_raw_kg"])
        close(
            sum(weights[a] * m for a, m in original.collected_mass.items()),
            audit["ship_23_weighted_kg"],
        )
        original_summary = original.summary()
        assert counted == set(original.collected_mass)
        originals_same_order = [
            row["index"]
            for row in generated
            if sorted(
                map(int, row["plan"]["deploy_epochs"]),
                key=lambda a: (row["plan"]["deploy_epochs"][str(a)], a),
            )
            == original_order
        ]
        original_set = set(original_order)
        all_cargo_max = True
        for row in generated:
            plan = RoutePlan.from_summary(row["plan"])
            for a, cargo in plan.collected_mass.items():
                all_cargo_max &= (
                    abs(
                        cargo
                        - C.maximum_collected_mass(plan.collect_epochs[a] - plan.deploy_epochs[a])
                    )
                    <= 1e-8
                )
        best_by_depth = {}
        for row in generated:
            key = row["deploy_depth"]
            if key not in best_by_depth or row["raw_proxy_kg"] > best_by_depth[key]["raw_kg"]:
                best_by_depth[key] = {
                    "index": row["index"],
                    "raw_kg": row["raw_proxy_kg"],
                    "weighted_kg": row["weighted_proxy_kg"],
                    "deploy_order": sorted(
                        map(int, row["plan"]["deploy_epochs"]),
                        key=lambda a: (row["plan"]["deploy_epochs"][str(a)], a),
                    ),
                    "deploy_end": max(row["plan"]["deploy_epochs"].values()),
                    "collect_start": min(row["plan"]["collect_epochs"].values()),
                    "collect_end": max(row["plan"]["collect_epochs"].values()),
                    "earth_return": row["plan"]["earth_return_epoch"],
                }
        report = {
            "GPU_calls": 0,
            "generations": 0,
            "refinements": 0,
            "fresh_lambert_evaluations": 0,
            "source_commit": saved["source_commit"],
            "input_sha256": sha(KIT / "inputs/ship-23.json"),
            "original_exact_plan_sha256": canonical(original_summary),
            "original_deploy_order": original_order,
            "original_collect_order": collect_order,
            "original_raw_kg": audit["ship_23_raw_kg"],
            "original_weighted_kg": audit["ship_23_weighted_kg"],
            "original_deploy_end": max(original.deploy_epochs.values()),
            "original_collect_start": min(original.collect_epochs.values()),
            "original_collect_end": max(original.collect_epochs.values()),
            "original_return_epoch": ret.arrival_epoch,
            "grid_comparison": grid_rows,
            "original_deploy_tofs_off_grid": sum(
                not row["tof_grid"]["exact"] for row in grid_rows if row["role"] == "deploy_hop"
            ),
            "original_collect_tofs_off_dp_grid": sum(
                not row["dp_tof_grid"]["exact"] for row in grid_rows if row["role"] == "collect_hop"
            ),
            "original_collect_departures_off_dp_grid": sum(
                not row["dp_departure_grid"]["exact"]
                for row in grid_rows
                if row["role"] == "collect_hop"
            ),
            "original_order_present_in_generated_candidates": originals_same_order,
            "original_deployed_set_present_in_generated_candidates": [
                row["index"]
                for row in generated
                if set(map(int, row["plan"]["deploy_epochs"])) == original_set
            ],
            "maximum_original_order_prefix_reaching_completion": max(
                row["prefix_length"] for row in closest
            ),
            "closest_completion_prefixes": sorted(
                closest, key=lambda row: (-row["prefix_length"], -row["depth"])
            )[:5],
            "incumbent_saved_dv_scalar_proxy_deploy_checks": deploy_checks,
            "incumbent_saved_dv_scalar_proxy_collect_checks": collect_checks,
            "incumbent_saved_dv_sequential_proxy_final_cargo_margin_kg": mass
            - C.DRY_MASS_KG
            - sum(original.collected_mass.values()),
            "all_generated_cargo_equals_mining_at_saved_epochs": all_cargo_max,
            "generated_cargo_shrink_detected": not all_cargo_max,
            "best_raw_by_depth": best_by_depth,
            "failed_completion_depths": audit["failed_completion_depths"],
            "nine_ten_failure_strings": dict(
                Counter(row["failure"] for row in audit["failed_completions"] if row["depth"] >= 9)
            ),
            "finite_conclusion": (
                "The exact incumbent schedule is outside the generation grid. The retained beam "
                "did not recover its topology/quality. Proxy checks are estimates on saved "
                "Lambert values; they are not actual trace evidence that an ungenerated "
                "incumbent branch was pruned. "
                "No conclusion of physical infeasibility or global family exhaustion follows."
            ),
        }
        with (ROOT / "diagnosis-v2.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "original_deploy_tofs_off_grid",
                        "original_collect_tofs_off_dp_grid",
                        "original_collect_departures_off_dp_grid",
                        "maximum_original_order_prefix_reaching_completion",
                        "original_order_present_in_generated_candidates",
                        "incumbent_saved_dv_sequential_proxy_final_cargo_margin_kg",
                        "generated_cargo_shrink_detected",
                        "failed_completion_depths",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
