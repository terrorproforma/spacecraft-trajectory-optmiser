"""Freeze historical controls and evaluate fixed models at archived burn masses."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
V619 = ROOT / "build/performance/incumbent-admission-v619"
ARCHIVE = ROOT / "build/performance/return-mass-bias-v619"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    assert sha(source) == sha(dest)


def frozen_inputs():
    assert not (KIT / "inputs").exists(), "Never overwrite frozen controls"
    index = read(V619 / "final-source-sha256.json")
    assert (
        index["src/spacepdhcg/gtoc12/search.py"]
        == "9c5d64e37ceea7e5fd8c680556cd41bfbbb1e5a282aacba5d3ede0cbe0797d18"
    )
    for name, digest in index.items():
        source = V619 / "source/candidate-final" / name
        assert sha(source) == digest
        copy(source, KIT / "source" / name)
    copy(V619 / "final-source-sha256.json", KIT / "source-sha256.json")
    inventory = read(ARCHIVE / "inputs/incumbent-inventory.json")
    assert len(inventory["routes"]) == 23
    for row in inventory["routes"]:
        source = ARCHIVE / "inputs" / f"ship-{row['ship']:02}.json"
        assert sha(source) == row["sha256"]
        copy(source, KIT / "inputs" / source.name)
    for source, name in (
        (ARCHIVE / "inputs/incumbent-inventory.json", "incumbent-inventory.json"),
        (ARCHIVE / "report.json", "return-components-v619.json"),
        (V619 / "output/candidate-final.json", "completion-oracle-v619.json"),
        (V619 / "selection.json", "selection-v619.json"),
        (V619 / "inputs/hop_inflation_fit.json", "hop_inflation_fit.json"),
        (V619 / "inputs/v616-report.json", "v616-report.json"),
        (V619 / "inputs/v616-profile.json", "v616-profile.json"),
    ):
        copy(source, KIT / "inputs" / name)
    write(
        KIT / "input-provenance.json",
        {
            "historical_fleet": "v595/v616; not the newer fleet in 2ccb93c1",
            "source_base": "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7 plus final v619 search.py",
            "source_index_sha256": sha(KIT / "source-sha256.json"),
            "source_files": len(index),
            "inputs": {p.name: sha(p) for p in sorted((KIT / "inputs").iterdir())},
            "selection_fixed_before_v619_outcomes": [23, 1, 4, 7, 10],
            "all_23_archival_route_flags_only": (
                "Historical certifications/inventory bindings retained; "
                "no new physics verification."
            ),
        },
    )


def summary(values):
    values = list(values)
    return {
        "count": len(values),
        "sum_kg": sum(values),
        "mean_kg": statistics.mean(values),
        "median_kg": statistics.median(values),
        "minimum_kg": min(values),
        "maximum_kg": max(values),
        "overpredicted_count": sum(x > 0 for x in values),
    }


def diagnose():
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(KIT / "source/src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "inputs/v616-profile.json")["data"]
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("No native loading in v621 preparation")
    ):
        import numpy as np

        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.collectdp import CollectPairTable
        from spacepdhcg.gtoc12.data import load_catalogue
        from spacepdhcg.gtoc12.hopcalib import InflationFit
        from spacepdhcg.gtoc12.screening import (
            propellant_for_delta_v,
            return_inflation_model,
            thrust_authority_km_s,
        )
        from spacepdhcg.gtoc12.search import RoutePlan

        catalogue = load_catalogue()
        assert (
            catalogue.source_sha256
            == "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675"
        )
        fit_record = read(KIT / "inputs/hop_inflation_fit.json")
        fit = InflationFit.from_summary(fit_record)
        table = CollectPairTable(catalogue)
        inventory = read(KIT / "inputs/incumbent-inventory.json")
        rows, route_audits, features = [], [], {}
        feature_calls = 0
        for info in inventory["routes"]:
            ship = info["ship"]
            path = KIT / "inputs" / f"ship-{ship:02}.json"
            raw = read(path)
            plan = RoutePlan.from_summary(raw["plan"])
            assert raw["certified"] and not raw["failures"]
            actual = {(x["from"], x["to"], x["t0"], x["tf"]): x for x in raw["legs"]}
            assert len(actual) == len(raw["legs"])
            stored_cargo = {int(k): float(v) for k, v in raw["collected_mass_kg"].items()}
            assert stored_cargo == plan.collected_mass
            cargo_checks = []
            for asteroid, amount in stored_cargo.items():
                deploy = plan.deploy_epoch_of(asteroid)
                collect = plan.collect_epochs[asteroid]
                maximum = C.maximum_collected_mass(collect - deploy)
                cargo_checks.append(
                    {
                        "asteroid": asteroid,
                        "deployment_epoch": deploy,
                        "collection_epoch": collect,
                        "stored_cargo_kg": amount,
                        "rule_maximum_kg": maximum,
                        "difference_from_maximum_kg": amount - maximum,
                        "minimum_stay_pass": collect - deploy
                        >= C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS - 1e-6,
                        "within_rule_mass": -1e-9 <= amount <= maximum + 1e-8,
                        "foreign_deployer": asteroid not in plan.deploy_epochs,
                    }
                )
            role_counts = Counter()
            route_rows = []
            for ordinal, leg in enumerate(plan.legs):
                if leg.role == "camp":
                    continue
                key = (leg.from_id, leg.to_id, leg.departure_epoch, leg.arrival_epoch)
                measured = actual[key]
                assert measured["certified"] and measured["status"] == "feasible"
                mass, dv, tof = measured["mass_before"], leg.delta_v_proxy_km_s, leg.tof_days
                authority = float(thrust_authority_km_s(mass, tof, 1.0))
                ratio = dv / authority
                is_hop = leg.role in ("deploy_hop", "collect_hop")
                feature = None
                if is_hop:
                    da, dl = table.pair_geometry(
                        leg.from_id, leg.to_id, np.asarray([leg.departure_epoch])
                    )
                    feature = {"delta_a_au": float(da), "delta_longitude_rad": float(dl[0])}
                    features[f"{ship}:{ordinal}"] = {
                        "ship": ship,
                        "plan_leg_index": ordinal,
                        "from_id": leg.from_id,
                        "to_id": leg.to_id,
                        "departure_epoch": leg.departure_epoch,
                        **feature,
                    }
                    feature_calls += 1
                flat_factor = (
                    1.2
                    if is_hop
                    else float(return_inflation_model(tof, ratio))
                    if leg.role == "earth_return"
                    else None
                )
                # The DP fit only governs collection; deployment keeps the flat beam model.
                dp_factor = (
                    float(
                        fit.inflation(
                            np.asarray([dv]),
                            mass,
                            np.asarray([tof]),
                            feature["delta_a_au"],
                            np.asarray([feature["delta_longitude_rad"]]),
                        )[0]
                    )
                    if leg.role == "collect_hop"
                    else flat_factor
                )
                models = {}
                for name, factor in (
                    ("v616_no_fit", flat_factor),
                    ("existing_fit_no_refit", dp_factor),
                ):
                    predicted = (
                        measured["propellant_kg"]
                        if factor is None
                        else float(propellant_for_delta_v(mass, dv * factor))
                    )
                    models[name] = {
                        "inflation": factor,
                        "policy": "measured_Earth_seed"
                        if factor is None
                        else "existing_DP_five_feature_fit"
                        if name == "existing_fit_no_refit" and leg.role == "collect_hop"
                        else "generic_return_model"
                        if leg.role == "earth_return"
                        else "flat_hop_1.2",
                        "predicted_fuel_at_measured_mass_kg": predicted,
                        "error_kg": predicted - measured["propellant_kg"],
                    }
                row = {
                    "ship": ship,
                    "plan_leg_index": ordinal,
                    "role": leg.role,
                    "from_id": leg.from_id,
                    "to_id": leg.to_id,
                    "departure_epoch": leg.departure_epoch,
                    "arrival_epoch": leg.arrival_epoch,
                    "tof_days": tof,
                    "saved_lambert_dv_km_s": dv,
                    "measured_burn_mass_before_kg": mass,
                    "measured_burn_mass_after_kg": measured["mass_after"],
                    "measured_propellant_kg": measured["propellant_kg"],
                    "measured_delta_v_km_s": measured["delta_v_km_s"],
                    "saved_planned_inflation": leg.inflation,
                    "authority_ratio_at_measured_mass": ratio,
                    "recorded_mass_delta_minus_propellant_kg": mass
                    - measured["mass_after"]
                    - measured["propellant_kg"],
                    "input_sha256": sha(path),
                    "archival_certified": True,
                    "certified_now": False,
                    "features": feature,
                    "models": models,
                }
                rows.append(row)
                route_rows.append(row)
                role_counts[leg.role] += 1
            assert len(route_rows) == len(actual)
            route_audits.append(
                {
                    "ship": ship,
                    "input_sha256": sha(path),
                    "independent_inventory": info["independent_inventory"],
                    "archived_cargo_matches_retained_Result": info["cargo_matches_retained_Result"],
                    "raw_cargo_kg": sum(stored_cargo.values()),
                    "weighted_cargo_kg": info["weighted_kg"],
                    "archived_final_dry_margin_kg": raw["final_mass_kg"] - C.DRY_MASS_KG,
                    "roles": dict(role_counts),
                    "cargo_checks": cargo_checks,
                    "all_stored_cargo_within_rule": all(
                        x["within_rule_mass"] and x["minimum_stay_pass"] for x in cargo_checks
                    ),
                    "local_error_sums_by_model_and_role": {
                        model: {
                            role: sum(
                                x["models"][model]["error_kg"]
                                for x in route_rows
                                if x["role"] == role
                            )
                            for role in role_counts
                        }
                        for model in ("v616_no_fit", "existing_fit_no_refit")
                    },
                }
            )
        assert table.lambert_evaluations == 0
        groups = defaultdict(list)
        for row in rows:
            for model in row["models"]:
                groups[(model, row["role"])].append(row["models"][model]["error_kg"])
        summaries = {
            model: {role: summary(values) for (m, role), values in groups.items() if m == model}
            for model in ("v616_no_fit", "existing_fit_no_refit")
        }
        ranked = {
            model: [
                {
                    key: row[key]
                    for key in (
                        "ship",
                        "plan_leg_index",
                        "role",
                        "from_id",
                        "to_id",
                        "input_sha256",
                    )
                }
                | {"error_kg": row["models"][model]["error_kg"]}
                for row in sorted(
                    rows,
                    key=lambda row: (
                        -row["models"][model]["error_kg"],
                        row["ship"],
                        row["plan_leg_index"],
                    ),
                )
            ]
            for model in ("v616_no_fit", "existing_fit_no_refit")
        }
        write(
            KIT / "features.json",
            {
                "catalogue_sha256": catalogue.source_sha256,
                "feature_source_sha256": sha(KIT / "source/src/spacepdhcg/gtoc12/collectdp.py"),
                "feature_function": (
                    "CollectPairTable.pair_geometry (deterministic mean-longitude arithmetic)"
                ),
                "feature_arithmetic_calls": feature_calls,
                "Lambert_calls": 0,
                "features": features,
            },
        )
        write(
            KIT / "residuals.json",
            {
                "complete": True,
                "historical_controls": True,
                "routes": route_audits,
                "legs": rows,
                "summaries": summaries,
                "ranked_positive_errors": ranked,
                "GPU_calls": 0,
                "Lambert_calls": 0,
                "trajectory_solves": 0,
                "refinements": 0,
                "feature_arithmetic_calls": feature_calls,
                "fit_sha256": sha(KIT / "inputs/hop_inflation_fit.json"),
                "fit_refitted": False,
                "new_certificates": 0,
                "scope": (
                    "Each component is priced at its own archived measured burn mass. "
                    "Summed local errors are not a fresh sequential route replay. "
                    "Existing fit governs collection only; deployment remains flat and "
                    "injected Earth seed cost is measured by construction."
                ),
            },
        )
        print(
            json.dumps(
                {
                    "routes": len(route_audits),
                    "flight_legs": len(rows),
                    "feature_arithmetic_calls": feature_calls,
                    "all_cargo_within_rule": all(
                        x["all_stored_cargo_within_rule"] for x in route_audits
                    ),
                    "summaries": summaries,
                },
                indent=2,
            )
        )


def main():
    frozen_inputs()
    diagnose()


if __name__ == "__main__":
    main()
