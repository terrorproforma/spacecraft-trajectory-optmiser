"""Preserve inadequate saved pools and freeze one fresh generation input; CPU only."""

import ctypes
import dataclasses
import math
import os
import shutil
from collections import Counter, defaultdict
from unittest.mock import patch

import common

REPO = common.ROOT.parents[2]
PRIOR = REPO / "build/performance/coupled-fuel-search-v611"


def main():
    if (common.ROOT / "source").exists():
        raise FileExistsError("Fresh preparation only")
    source = common.read(PRIOR / "source-sha256.json")
    for name, digest in source.items():
        origin, target = PRIOR / "source" / name, common.ROOT / "source" / name
        assert common.sha(origin) == digest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    common.write(common.ROOT / "source-sha256.json", source)
    origins = {
        "Result.txt": PRIOR / "inputs/Result.txt",
        "ship-23.json": REPO
        / "build/performance/incident-window-substitution-v604/inputs/ship-23.json",
        "incumbent-inventory.json": PRIOR / "incumbent-inventory-01.json",
        "historical-family-report.json": REPO
        / "results/gtoc12/runs/cluster_fleet_h100_v2/run_report.json",
        "old-generated-pool.json": REPO / "build/performance/fleet-recovery-v379/candidates.json",
        "old-generation-report.json": REPO / "build/performance/fleet-recovery-v379/report.json",
    }
    inputs = {}
    for name, origin in origins.items():
        target = common.ROOT / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        inputs[name] = {"origin": str(origin), "sha256": common.sha(target)}
    common.write(common.ROOT / "inputs-sha256.json", inputs)
    profile = common.read(PRIOR / "profile.json")
    profile["environment"] = {
        key: value for key, value in profile["environment"].items() if "JOINT" not in key
    }
    profile["experiment_scope"] = (
        "One candidate generation; CUDA Lambert/geometry/collection DP, "
        "CPU beam and DP input construction"
    )
    common.write(common.ROOT / "profile.json", profile)
    _, env = common.environment()
    common.runtime_check(env)
    os.environ["SPACEPDHCG_GTOC12_DATA"] = profile["data"]
    common.activate()
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("CPU-only preparation")):
        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, refine_candidates
        from spacepdhcg.gtoc12.clusters import ClusterBands, ComovingClusters
        from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
        from spacepdhcg.gtoc12.search import RoutePlan
        from spacepdhcg.gtoc12.solution import Solution

        catalogue, bonus = load_catalogue(), load_bonus_table()
        fleet = Solution.read(common.ROOT / "inputs/Result.txt")
        old_pool = common.read(common.ROOT / "inputs/old-generated-pool.json")
        plans = [RoutePlan.from_summary(row) for row in old_pool]
        actual_selected = refine_candidates(plans, set(), ClusterPricingSettings(refine_top=2))
        selected_indices = [
            next(i for i, item in enumerate(plans) if item is plan) for plan in actual_selected
        ]
        retained = set()
        groups = defaultdict(list)
        for i, plan in enumerate(plans):
            leg = plan.legs[0]
            groups[(leg.to_id, leg.departure_epoch, leg.tof_days)].append(i)
        retained = {indices[0] for indices in groups.values()}
        assert set(selected_indices).issubset(retained)
        saved = []
        for relative in (common.ROOT / "search-inventory-paths.txt").read_text().splitlines():
            path = REPO / relative.replace("\\", "/")
            item = common.read(path)
            rows = item["search"]["top_candidates"]
            by_earth = defaultdict(set)
            for row in rows:
                leg = row["legs"][0]
                by_earth[(leg["to"], leg["t0"], leg["tf"])].add(len(row["deploy_epochs"]))
            saved.append(
                {
                    "path": relative,
                    "sha256": common.sha(path),
                    "total_generated": item["search"].get("candidates"),
                    "saved": len(rows),
                    "depths": sorted({len(row["deploy_epochs"]) for row in rows}),
                    "same_earth_depth_contrast": any(len(x) > 1 for x in by_earth.values()),
                    "best_raw_kg": max((row["total_collected_kg"] for row in rows), default=0),
                }
            )
        best_raw = max(row["total_collected_kg"] for row in old_pool)
        min_current = min(
            row["raw_kg"]
            for row in common.read(common.ROOT / "inputs/incumbent-inventory.json")["routes"]
        )
        lower_raw = 23 * math.log(23 / 2) / 0.004
        best_replacement_raw = 14051.854893908598 - min_current + best_raw
        common.write(
            common.ROOT / "saved-pool-audit.json",
            {
                "GPU_calls": 0,
                "search_exports": saved,
                "search_export_count": len(saved),
                "mixed_depth_exports": sum(row["same_earth_depth_contrast"] for row in saved),
                "saved_nine_or_ten_depth_exports": sum(
                    any(d >= 9 for d in row["depths"]) for row in saved
                ),
                "old_full_pool_count": len(plans),
                "old_full_pool_depths": dict(Counter(len(p.deploy_epochs) for p in plans)),
                "old_full_pool_best_raw_kg": best_raw,
                "old_full_pool_distinct_earth_legs": len(groups),
                "actual_frozen_refine_candidates_selected_indices": selected_indices,
                "actual_selected_depths": [len(plans[i].deploy_epochs) for i in selected_indices],
                "earth_dedup_discarded_by_depth": dict(
                    Counter(len(p.deploy_epochs) for i, p in enumerate(plans) if i not in retained)
                ),
                "same_earth_depth_groups": [
                    {
                        "earth": list(key),
                        "indices": indices,
                        "depths": sorted({len(plans[i].deploy_epochs) for i in indices}),
                    }
                    for key, indices in groups.items()
                    if len({len(plans[i].deploy_epochs) for i in indices}) > 1
                ],
                "best_raw_one_for_one_fleet_kg": best_replacement_raw,
                "required_raw_fleet_kg": lower_raw,
                "raw_shortfall_even_before_weighted_or_conflict_checks_kg": lower_raw
                - best_replacement_raw,
                "adequate_saved_pool": False,
                "reason": (
                    "Available generated depth alternatives do not reach 9/10 and cannot pass "
                    "current fleet raw replacement gate"
                ),
                "certified_archive_routes_used_as_generated_candidates": False,
            },
        )
        historical = common.read(common.ROOT / "inputs/historical-family-report.json")
        partition = next(
            x for x in historical["instance"]["partitions"] if x["name"] == "phasing_r1.75"
        )
        poolfilter = historical["instance"]["pool_filter"]
        a = catalogue.semi_major_axis_km / C.AU_KM
        import numpy as np

        universe = catalogue.ids[
            (a >= poolfilter["a_au"][0])
            & (a <= poolfilter["a_au"][1])
            & (catalogue.eccentricity <= poolfilter["e_max"])
            & (np.rad2deg(catalogue.inclination_rad) <= poolfilter["i_max_deg"])
        ]
        bands = ClusterBands(
            radius=partition["radius"],
            phase_deg=partition["phase_deg"],
            visit_epochs=tuple(partition["visit_epochs"]),
            phase_weights=tuple(partition["phase_weights"]),
        )
        clusters = ComovingClusters(catalogue, universe, bands)
        original = common.read(common.ROOT / "inputs/ship-23.json")
        first = original["plan"]["legs"][0]
        label = clusters.label_of(first["to"])
        family = set(map(int, clusters.cluster_members(label)))
        excluded = {
            event.event_id
            for ship in fleet.ships
            if ship.ship_id != 23
            for event in ship.asteroid_visits()
        }
        allowed = sorted(family - excluded)
        assert first["to"] in allowed and len(allowed) >= 10
        assert set(map(int, original["plan"]["deploy_epochs"])).issubset(allowed)
        measured = original["legs"][0]
        assert measured["certified"] and original["certified"]
        seed = {
            "target": first["to"],
            "launch_epoch": first["t0"],
            "tof_days": first["tf"] - first["t0"],
            "delta_v_km_s": first["dv_proxy_km_s"],
            "propellant_kg": measured["propellant_kg"],
            "certified": True,
        }
        common.write(
            common.ROOT / "generation-input.json",
            {
                "catalogue_sha256": catalogue.source_sha256,
                "bonus_sha256": bonus.source_sha256,
                "ship": 23,
                "seed": seed,
                "universe_count": len(universe),
                "partition": partition,
                "bands": dataclasses.asdict(bands),
                "label": label,
                "family_ids": sorted(family),
                "excluded_other_fleet_ids": sorted(excluded),
                "allowed_ids": allowed,
                "original_deploy_ids": sorted(map(int, original["plan"]["deploy_epochs"])),
                "source_commit": "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7",
                "GPU_calls_during_preparation": 0,
            },
        )
        print(
            {
                "saved_exports": len(saved),
                "raw_shortfall_kg": lower_raw - best_replacement_raw,
                "old_filter_selected": selected_indices,
                "family_size": len(family),
                "allowed_size": len(allowed),
                "seed": seed,
            }
        )


if __name__ == "__main__":
    main()
