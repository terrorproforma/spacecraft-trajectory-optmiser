"""Matched CPU bridge replay on archived flight epochs and stored Lambert values."""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

KIT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(leg):
    return (leg.from_id, leg.to_id, leg.departure_epoch, leg.arrival_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("baseline", "candidate-final"))
    args = parser.parse_args()
    source = KIT / "source" / args.arm
    index = (
        read(KIT / "source-sha256.json")["baseline"]
        if args.arm == "baseline"
        else read(KIT / "final-source-sha256.json")
    )
    for name, digest in index.items():
        assert sha(source / name) == digest, name
    provenance = read(KIT / "provenance.json")
    for name, digest in provenance["inputs"].items():
        assert sha(KIT / "inputs" / name) == digest, name
    assert sha(KIT / "selection.json") == provenance["selection_sha256"]
    selection = read(KIT / "selection.json")
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(source / "src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "inputs/v616-profile.json")["data"]
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("CPU replay forbids native loading")
    ):
        import numpy as np

        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.collectdp import CollectTour
        from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
        from spacepdhcg.gtoc12.search import RoutePlan, RouteSearch, SearchSettings, _Partial
        import spacepdhcg.gtoc12.search as search_module

        assert Path(search_module.__file__).resolve() == source / "src/spacepdhcg/gtoc12/search.py"

        class TracedSearch(RouteSearch):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.trace = []

            def run(self, *args, **kwargs):
                raise AssertionError("Generation forbidden")

            def _feasible(self, mass, dv, tof, role):
                accepted = super()._feasible(mass, dv, tof, role)
                self.trace.append(
                    {
                        "role": role,
                        "mass_before_kg": mass,
                        "dv_proxy_km_s": dv,
                        "tof_days": tof,
                        "authority_pass": bool(accepted),
                    }
                )
                return accepted

            def _propellant(self, mass, dv, inflation):
                value = super()._propellant(mass, dv, inflation)
                self.trace[-1].update(
                    inflation_spent=inflation, propellant_kg=value, mass_after_kg=mass - value
                )
                return value

        defaults = {f.name: f.default for f in dataclasses.fields(SearchSettings)}
        base_values = read(KIT / "inputs/v616-report.json")["settings"]
        base_values = {
            k: defaults[k]
            if v is None and isinstance(defaults[k], float) and not math.isfinite(defaults[k])
            else v
            for k, v in base_values.items()
        }
        assert base_values["collect_dp_inflation_fit"] == ""
        catalogue, bonus = load_catalogue(), load_bonus_table()
        rows = []
        for chosen in selection["routes"]:
            ship = chosen["ship"]
            archive = read(KIT / "inputs" / f"ship-{ship:02}.json")
            archived_plan = RoutePlan.from_summary(archive["plan"])
            assert archive["certified"] and not archive["failures"]
            assert archived_plan.self_cleaning and not archived_plan.foreign_deploy_epochs
            assert all(leg["certified"] and leg["status"] == "feasible" for leg in archive["legs"])
            cargo = {int(k): v for k, v in archive["collected_mass_kg"].items()}
            assert cargo == archived_plan.collected_mass
            assert math.isclose(sum(cargo.values()), chosen["raw_kg"], abs_tol=1e-9)
            weighted = sum(bonus.for_asteroid(a) * value for a, value in cargo.items())
            assert math.isclose(weighted, chosen["weighted_kg"], abs_tol=1e-9)
            original_bytes = json.dumps(archive, sort_keys=True)
            split = (
                max(i for i, leg in enumerate(archived_plan.legs) if leg.role == "deploy_hop") + 1
            )
            last_deploy = archived_plan.legs[split - 1]
            actual = {(x["from"], x["to"], x["t0"], x["tf"]): x for x in archive["legs"]}
            measured_prefix = actual[key(last_deploy)]["mass_after"] - C.MINER_MASS_KG
            initial_mass = actual[key(archived_plan.legs[0])]["mass_before"]
            proxy_mass = (
                initial_mass - actual[key(archived_plan.legs[0])]["propellant_kg"] - C.MINER_MASS_KG
            )
            proxy_search = RouteSearch(
                catalogue, np.asarray(sorted(cargo)), SearchSettings(**base_values)
            )
            for leg in archived_plan.legs[1:split]:
                if leg.role == "camp":
                    continue
                assert leg.role == "deploy_hop"
                inflation = proxy_search.hop_inflation_for(
                    leg.delta_v_proxy_km_s, proxy_mass, leg.tof_days
                )
                proxy_mass -= (
                    proxy_search._propellant(proxy_mass, leg.delta_v_proxy_km_s, inflation)
                    + C.MINER_MASS_KG
                )
            saved_forward = archived_plan.legs[split:]
            return_leg = saved_forward[-1]
            assert return_leg.role == "earth_return"
            deploy = sorted(archived_plan.deploy_epochs.items(), key=lambda row: (row[1], row[0]))
            collect = dict(archived_plan.collect_epochs)
            tour = CollectTour(
                order=tuple(
                    a for a, _ in sorted(collect.items(), key=lambda row: (row[1], row[0]))
                ),
                collect_epochs=collect,
                hops=[
                    (x.from_id, x.to_id, x.departure_epoch, x.tof_days, x.delta_v_proxy_km_s)
                    for x in saved_forward
                    if x.role == "collect_hop"
                ],
                reposition=collect[last_deploy.to_id]
                > next(x.departure_epoch for x in saved_forward if x.role != "camp"),
                objective_kg=0.0,
                collected_proxy_kg=sum(cargo.values()),
                propellant_proxy_kg=0.0,
                return_departure=return_leg.departure_epoch,
                return_tof=return_leg.tof_days,
                return_dv=return_leg.delta_v_proxy_km_s,
            )
            for model in selection["models"]:
                values = {**base_values, "initial_mass": initial_mass}
                if model == "existing_fit_no_refit":
                    values["collect_dp_inflation_fit"] = str(KIT / "inputs/hop_inflation_fit.json")
                for prefix_name, mass in [
                    ("flat_deployment_proxy", proxy_mass),
                    ("measured_deployment_prefix", measured_prefix),
                ]:
                    search = TracedSearch(
                        catalogue, np.asarray(sorted(cargo)), SearchSettings(**values)
                    )
                    partial = _Partial(
                        list(archived_plan.legs[:split]),
                        last_deploy.to_id,
                        last_deploy.arrival_epoch,
                        mass,
                        deploy,
                    )
                    result = search._plan_from_tour(partial, tour)
                    assert search.collect_table.lambert_evaluations == 0
                    assert not search.collect_table.return_sweeps
                    flights = [leg for leg in saved_forward if leg.role != "camp"]
                    assert len(search.trace) <= len(flights)
                    for leg, trace in zip(flights, search.trace):
                        trace.update(
                            from_id=leg.from_id,
                            to_id=leg.to_id,
                            departure_epoch=leg.departure_epoch,
                            arrival_epoch=leg.arrival_epoch,
                            archived_measured_propellant_kg=actual[key(leg)]["propellant_kg"],
                            archived_measured_mass_before_kg=actual[key(leg)]["mass_before"],
                        )
                    final_mass = search.trace[-1].get("mass_after_kg")
                    row = {
                        "ship": ship,
                        "purpose": chosen["purpose"],
                        "model": model,
                        "prefix": prefix_name,
                        "prefix_mass_kg": mass,
                        "accepted_by_scalar_bridge": result is not None,
                        "failure": search.last_failure if result is None else None,
                        "cargo_kg": sum(cargo.values()),
                        "weighted_cargo_kg": weighted,
                        "proxy_final_mass_kg": final_mass,
                        "proxy_final_cargo_margin_kg": None
                        if final_mass is None
                        else final_mass - C.DRY_MASS_KG - sum(cargo.values()),
                        "archived_final_dry_margin_kg": archive["final_mass_kg"] - C.DRY_MASS_KG,
                        "trace": search.trace,
                        "complete_forward_flight_count": len(flights),
                        "prescribed_inputs_unchanged": True,
                        "trajectory_certified_now": False,
                        "generated_candidate": False,
                        "emitted_fleet": False,
                    }
                    if result is not None:
                        assert (
                            result.deploy_epochs == archived_plan.deploy_epochs
                            and result.collect_epochs == collect
                        )
                        assert result.legs[:split] == archived_plan.legs[:split]
                        assert [key(x) for x in result.legs if x.role != "camp"] == [
                            key(x) for x in archived_plan.legs if x.role != "camp"
                        ]
                        assert result.collected_mass == cargo
                        assert math.isclose(result.final_mass_proxy_kg, final_mass, abs_tol=1e-9)
                        returned_flights = [x for x in result.legs[split:] if x.role != "camp"]
                        errors = [
                            x.inflation - t["inflation_spent"]
                            for x, t in zip(returned_flights, search.trace)
                        ]
                        row["returned_leg_minus_spent_inflation"] = errors
                        if args.arm == "candidate-final":
                            assert all(value == 0 for value in errors)
                        row["returned_plan"] = result.summary()
                    assert original_bytes == json.dumps(archive, sort_keys=True)
                    rows.append(row)
        assert len(rows) == 20
        report = {
            "arm": args.arm,
            "source_search_sha256": index["src/spacepdhcg/gtoc12/search.py"],
            "selection_sha256": sha(KIT / "selection.json"),
            "catalogue_sha256": catalogue.source_sha256,
            "bonus_sha256": bonus.source_sha256,
            "scalar_bridge_calls": len(rows),
            "GPU_calls": 0,
            "Lambert_calls": 0,
            "DP_solves": 0,
            "refinements": 0,
            "physics_certifications": 0,
            "fleet_promotions": 0,
            "scope": selection["scope"],
            "rows": rows,
        }
        output = KIT / "output"
        output.mkdir(exist_ok=True)
        with (output / f"{args.arm}.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "checks": 20,
                    "accepted": sum(x["accepted_by_scalar_bridge"] for x in rows),
                    "GPU_calls": 0,
                    "refinements": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
