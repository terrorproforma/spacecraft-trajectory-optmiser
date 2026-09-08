"""Bind 20 existing v619 cases plus explicit numerical/gate controls to ABI v1."""

from __future__ import annotations

import copy
import ctypes
import json
import math
import shutil
import sys
from unittest.mock import patch

from fixture_oracle import (
    FAILURES,
    KIT,
    MODELS,
    ROLES,
    STAGES,
    C,
    encode,
    evaluate,
    leg,
    policy,
    read,
    sha,
    source_check,
)

from spacepdhcg.gtoc12.search import RoutePlan


def historical():
    original = read(KIT / "inputs/completion-oracle-v619.json")
    features = read(KIT / "features.json")["features"]
    fit = read(KIT / "inputs/hop_inflation_fit.json")
    settings = read(KIT / "inputs/v616-report.json")["settings"]
    cases = []
    for number, old in enumerate(original["rows"]):
        ship = old["ship"]
        archive_path = KIT / "inputs" / f"ship-{ship:02}.json"
        archive = read(archive_path)
        plan = RoutePlan.from_summary(archive["plan"])
        ordered = sorted(plan.deploy_epochs.items(), key=lambda pair: (pair[1], pair[0]))
        ids = [a for a, _ in ordered]
        deploys = [
            {
                "deploy_epoch": epoch,
                "collect_epoch": plan.collect_epochs[a],
                "has_collect": 1,
                "reserved": 0,
            }
            for a, epoch in ordered
        ]
        split = max(i for i, row in enumerate(plan.legs) if row.role == "deploy_hop") + 1
        epoch = plan.legs[split - 1].arrival_epoch
        location = plan.legs[split - 1].to_id
        rows, provenance = [], []
        for index in range(split, len(plan.legs)):
            saved = plan.legs[index]
            if saved.role == "camp":
                continue
            assert saved.from_id == location and saved.departure_epoch >= epoch - 1e-6
            if saved.departure_epoch > epoch + 1e-6:
                rows.append(
                    leg(
                        "camp",
                        source_deploy=-1,
                        departure=epoch,
                        arrival=saved.departure_epoch,
                        flat=1.0,
                    )
                )
                provenance.append({"kind": "builder_inserted_camp", "at_asteroid": location})
            model = (
                "table_return"
                if saved.role == "earth_return"
                else "fit5"
                if old["model"] == "existing_fit_no_refit"
                else "flat"
            )
            row = leg(
                saved.role,
                model,
                ids.index(saved.from_id),
                departure=saved.departure_epoch,
                arrival=saved.arrival_epoch,
                dv=saved.delta_v_proxy_km_s,
                authority_ratio=settings["earth_return_authority_ratio"]
                if saved.role == "earth_return"
                else settings["hop_authority_ratio"],
                flat=settings["hop_inflation"],
                floor=fit["floor"],
                fit=fit["coefficients"],
            )
            if saved.role == "collect_hop":
                f = features[f"{ship}:{index}"]
                assert (f["from_id"], f["to_id"], f["departure_epoch"]) == (
                    saved.from_id,
                    saved.to_id,
                    saved.departure_epoch,
                )
                row.update(delta_a_au=f["delta_a_au"], delta_longitude_rad=f["delta_longitude_rad"])
            rows.append(row)
            provenance.append(
                {
                    "kind": "saved_flight",
                    "plan_leg_index": index,
                    "from_id": saved.from_id,
                    "to_id": saved.to_id,
                }
            )
            epoch, location = saved.arrival_epoch, saved.to_id
        case = {
            "id": f"historical-{number:02}-ship{ship:02}-{old['model']}-{old['prefix']}",
            "partial_mass": old["prefix_mass_kg"],
            "deploys": deploys,
            "legs": rows,
            "provenance": {
                "v619_row": number,
                "ship": ship,
                "prefix": old["prefix"],
                "model": old["model"],
                "route_sha256": sha(archive_path),
                "deploy_asteroid_ids": ids,
                "legs": provenance,
                "old_left_associated_margin_kg": old["proxy_final_cargo_margin_kg"],
                "weighted_cargo_kg": old["weighted_cargo_kg"],
                "archived_dry_margin_kg": old["archived_final_dry_margin_kg"],
            },
        }
        expected = evaluate(case)
        assert (expected["failure_name"] == "ok") == old["accepted_by_scalar_bridge"]
        assert expected["result"]["final_mass"] == old["proxy_final_mass_kg"]
        assert expected["result"]["collected"] == old["cargo_kg"]
        flights = [
            d
            for row, d in zip(rows, expected["leg_results"], strict=True)
            if row["role"] != ROLES["camp"]
        ]
        assert len(flights) == len(old["trace"])
        for d, saved in zip(flights, old["trace"], strict=True):
            assert d["departure_mass"] == saved["mass_before_kg"]
            assert d["inflation"] == saved["inflation_spent"]
            assert d["propellant"] == saved["propellant_kg"]
            assert d["mass_after"] == saved["mass_after_kg"]
        expected["frozen_source_check"] = source_check(case, expected)
        case["expected"] = expected
        cases.append(case)
    assert len(cases) == 20
    return cases


def synthetic():
    base = {
        "id": "",
        "partial_mass": 600.0,
        "deploys": [
            {"deploy_epoch": 0.0, "collect_epoch": 1000.0, "has_collect": 1, "reserved": 0}
        ],
        "legs": [leg()],
        "provenance": {"synthetic": True, "mission_candidate": False},
    }
    cases = []

    def add(name, change, expected_failure):
        case = copy.deepcopy(base)
        case["id"] = name
        change(case)
        expected = evaluate(case)
        assert expected["failure_name"] == expected_failure, (name, expected)
        expected["frozen_source_check"] = source_check(case, expected)
        case["expected"] = expected
        cases.append(case)

    add(
        "missing_collect_before_bad_leg",
        lambda c: (
            c["deploys"][0].update(has_collect=0),
            c["legs"][0].update(dv=1e99, flat=math.nan),
        ),
        "uncollected",
    )
    add(
        "short_stay_before_bad_leg",
        lambda c: (c["deploys"][0].update(collect_epoch=1.0), c["legs"][0].update(dv=1e99)),
        "stay_too_short",
    )
    add(
        "nan_deploy_raises_mining_before_authority",
        lambda c: (
            c["deploys"][0].update(deploy_epoch=math.nan),
            c["legs"][0].update(dv=1e99, flat=math.nan),
        ),
        "invalid_mining_stay",
    )
    add(
        "authority_before_nan_inflation",
        lambda c: c["legs"][0].update(dv=1e99, flat=math.nan),
        "leg_authority",
    )
    add(
        "nan_inflation_after_authority",
        lambda c: c["legs"][0].update(flat=math.nan),
        "invalid_inflation",
    )
    add(
        "negative_inflation_after_authority",
        lambda c: c["legs"][0].update(flat=-1.0),
        "invalid_inflation",
    )
    add(
        "certified_cell_still_checks_authority",
        lambda c: c["legs"][0].update(
            role=ROLES["earth_return"], model=MODELS["certified_flat"], flat=0.8, dv=1e99
        ),
        "leg_authority",
    )
    add(
        "certified_cell_retains_measured_cost",
        lambda c: c["legs"][0].update(
            role=ROLES["earth_return"], model=MODELS["certified_flat"], flat=0.8, dv=0.5
        ),
        "ok",
    )
    add(
        "camp_skips_pickup_bad_dv_and_model",
        lambda c: c["legs"].insert(0, leg("camp", source_deploy=-1, dv=math.nan, flat=math.nan)),
        "ok",
    )
    add(
        "reposition_does_not_collect_early",
        lambda c: c["legs"].insert(0, leg(departure=900.0, arrival=950.0)),
        "ok",
    )
    add(
        "pickup_inside_epoch_tolerance",
        lambda c: c["legs"][0].update(departure=1000.0 + 0.5e-6),
        "ok",
    )
    add(
        "pickup_outside_epoch_tolerance",
        lambda c: c["legs"][0].update(departure=1000.0 + 2e-6),
        "ok",
    )
    add("repeated_pickup_changes_mass_once_dictionary", lambda c: c["legs"].append(leg()), "ok")
    add("exact_dry_plus_cargo_equality", lambda c: c.update(partial_mass=500.0), "ok")
    add(
        "mass_below_dry_plus_cargo",
        lambda c: c.update(partial_mass=499.0),
        "mass_below_dry_plus_collected",
    )
    add(
        "ratio_model_prices_current_mass",
        lambda c: c["legs"][0].update(model=MODELS["ratio"], floor=1.05, slope=0.65, dv=0.5),
        "ok",
    )
    add(
        "fit_model_prices_all_five_features",
        lambda c: c["legs"][0].update(
            model=MODELS["fit5"],
            fit=[0.9, 0.6, 0.2, -0.1, 0.3],
            floor=0.7,
            delta_a_au=-0.03,
            delta_longitude_rad=-0.4,
            dv=0.5,
        ),
        "ok",
    )
    add(
        "generic_return_current_mass",
        lambda c: c["legs"][0].update(role=ROLES["earth_return"], model=MODELS["return"], dv=0.5),
        "ok",
    )
    add(
        "table_return_current_mass",
        lambda c: c["legs"][0].update(
            role=ROLES["earth_return"], model=MODELS["table_return"], dv=0.5
        ),
        "ok",
    )
    add(
        "negative_infinite_dv_generic_return",
        lambda c: c["legs"][0].update(
            role=ROLES["earth_return"], model=MODELS["return"], dv=-math.inf
        ),
        "ok",
    )
    add(
        "negative_infinite_dv_table_return",
        lambda c: c["legs"][0].update(
            role=ROLES["earth_return"], model=MODELS["table_return"], dv=-math.inf
        ),
        "ok",
    )

    def compensated(c):
        huge_stay = float(2**58) * C.YEAR_DAYS / C.MINING_RATE_KG_PER_YEAR
        c["deploys"] = [
            {"deploy_epoch": -huge_stay, "collect_epoch": 0.0, "has_collect": 1, "reserved": 0},
            {
                "deploy_epoch": -2 * C.YEAR_DAYS,
                "collect_epoch": 0.0,
                "has_collect": 1,
                "reserved": 0,
            },
            {
                "deploy_epoch": -2 * C.YEAR_DAYS,
                "collect_epoch": 0.0,
                "has_collect": 1,
                "reserved": 0,
            },
        ]
        c["legs"] = [leg(source_deploy=i, departure=0.0, arrival=200.0) for i in [0, 2, 1, 2]]

    add("cpython312_compensated_sum_pickup_order", compensated, "ok")
    return cases


def flatten(cases):
    candidates, deployments, flights = [], [], []
    for row in cases:
        candidates.append(
            {
                "deploy_begin": len(deployments),
                "deploy_count": len(row["deploys"]),
                "leg_begin": len(flights),
                "leg_count": len(row["legs"]),
                "partial_mass": row["partial_mass"],
            }
        )
        deployments.extend(row["deploys"])
        flights.extend(row["legs"])
    return {"policy": policy(), "candidates": candidates, "deploys": deployments, "legs": flights}


def main():
    assert sys.version_info[:2] == (3, 12), "Oracle requires CPython 3.12 float sum semantics"
    destination = KIT / "fixtures.json"
    assert not destination.exists(), "Never overwrite recorded fixture outputs"
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("Native loading forbidden")):
        old, controls = historical(), synthetic()
    header = KIT.parents[2] / "cpp/cuda/include/spacepdhcg/cuda/gtoc12_completion_c_api.h"
    assert sha(header) == "e176bb0d5848981a140e264a71d10ac744ce1babc946d5edf5caf01dc3f9dc76"
    shutil.copyfile(header, KIT / header.name)
    document = {
        "schema": "completion-abi-v1-fixtures",
        "python": sys.version,
        "header_sha256": sha(header),
        "source_search_sha256": sha(KIT / "source/src/spacepdhcg/gtoc12/search.py"),
        "historical_oracle_sha256": sha(KIT / "inputs/completion-oracle-v619.json"),
        "features_sha256": sha(KIT / "features.json"),
        "roles": ROLES,
        "models": MODELS,
        "failures": FAILURES,
        "stages": STAGES,
        "historical": old,
        "synthetic": controls,
        "historical_batch": flatten(old),
        "synthetic_batch": flatten(controls),
        "work": {
            "historical_source_finish_calls": len(old),
            "synthetic_source_finish_calls": len(controls),
            "GPU_calls": 0,
            "Lambert_calls": 0,
            "feature_arithmetic_calls": 0,
            "refinements": 0,
            "new_physics_certifications": 0,
        },
        "qualification_comparison": (
            "failure,failed_leg,failed_deploy,processed_legs,stage,pickup are exact; "
            "numerical mass/cost comparison never changes qualification"
        ),
        "detail_scope": (
            "Per-leg/early-failure numeric details follow ABI; _finish returns only "
            "None/RoutePlan or raises. Both return kind and successful totals are "
            "independently cross-checked against frozen source."
        ),
        "nonfinite_encoding": (
            "A sole float64 key encodes nan,+inf,-inf; decode before native packing."
        ),
    }
    with destination.open("x") as stream:
        json.dump(encode(document), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "historical": len(old),
                "synthetic": len(controls),
                "historical_accepts": sum(c["expected"]["failure_name"] == "ok" for c in old),
                "sha256": sha(destination),
                "GPU_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
