"""34 fixed CPU completions plus queue replay of 20 saved development readbacks."""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

KIT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    prep = read(KIT / "preparation.json")
    assert sha(KIT / "selection.json") == prep["selection_sha256"]
    assert sha(KIT / "source-sha256.json") == prep["source_sha256"]
    assert sha(KIT / "input-provenance.json") == prep["input_provenance_sha256"]
    sources = read(KIT / "source-sha256.json")
    for name, value in sources.items():
        assert sha(KIT / "source" / name) == value, name
    for name, info in read(KIT / "input-provenance.json").items():
        assert sha(KIT / "inputs" / name) == info["sha256"], name
    output = KIT / "output"
    output.mkdir(exist_ok=False)
    split = read(KIT / "selection.json")
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(KIT / "source/src"))
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("v622 forbids native loading")):
        import numpy as np

        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.collectdp import CollectTour
        from spacepdhcg.gtoc12.refinement_admission import (
            AdmissionCandidate,
            CompletionObservation,
            FixedCargoRequest,
            RefinementAdmissionQueue,
        )
        from spacepdhcg.gtoc12.search import RoutePlan, RouteSearch, SearchSettings, _Partial

        class TracedSearch(RouteSearch):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.trace = []

            def run(self, *args, **kwargs):
                raise AssertionError("v622 forbids generation")

            def _feasible(self, mass, dv, tof, role):
                ok = super()._feasible(mass, dv, tof, role)
                self.trace.append(
                    {
                        "role": role,
                        "mass_before_kg": mass,
                        "dv_proxy_km_s": dv,
                        "tof_days": tof,
                        "authority_pass": bool(ok),
                    }
                )
                return ok

            def _propellant(self, mass, dv, inflation):
                value = super()._propellant(mass, dv, inflation)
                self.trace[-1].update(
                    inflation_spent=inflation, propellant_kg=value, mass_after_kg=mass - value
                )
                return value

        defaults = {f.name: f.default for f in dataclasses.fields(SearchSettings)}
        values = {
            k: defaults[k]
            if v is None and isinstance(defaults[k], float) and not math.isfinite(defaults[k])
            else v
            for k, v in read(KIT / "inputs/v616-report.json")["settings"].items()
        }
        assert values["collect_dp_inflation_fit"] == ""
        inventory = {x["ship"]: x for x in read(KIT / "inputs/incumbent-inventory.json")["routes"]}
        baseline = read(KIT / "inputs/v604-fresh-verification.json")["independent"]
        residuals = read(KIT / "inputs/residuals.json")
        rows, candidates = [], {}

        def record(ship, model, prefix, observed, provenance):
            archive = read(KIT / "inputs" / f"ship-{ship:02}.json")
            request = FixedCargoRequest.from_summary(archive["plan"], archive["collected_mass_kg"])
            body = observed["result"]
            observation = CompletionObservation(
                request.sha256,
                provenance["producer_sha256"],
                digest(observed),
                "" if observed["failure_name"] == "ok" else observed["failure_name"],
                len(observed["leg_results"]),
                body["processed_legs"],
                tuple(x["stage"] for x in observed["leg_results"]),
                body["final_mass"],
                body["collected"],
                all(
                    all(
                        math.isfinite(x.get(key, 0)) and x.get(key, 0) >= 0
                        for key in ("inflation", "propellant")
                    )
                    for x in observed["leg_results"]
                    if x["stage"] == 4
                ),
            )
            candidate = AdmissionCandidate(request, observation, 0.0, baseline["total_mass_kg"], 23)
            identifier = f"ship{ship:02}-{model}-{prefix}"
            candidates[identifier] = candidate
            row = {
                "id": identifier,
                "ship": ship,
                "model": model,
                "prefix": prefix,
                "request_sha256": request.sha256,
                "readback_sha256": observation.readback_sha256,
                "input_route_sha256": inventory[ship]["sha256"],
                "provenance": provenance,
                "proxy_failure": observation.failure,
                "proxy_deficit_kg": observation.deficit_kg,
                "admission_blocker": candidate.blocker(replay_controls=True),
                "archived_final_dry_margin_kg": archive["final_mass_kg"] - C.DRY_MASS_KG,
                "raw_cargo_kg": inventory[ship]["raw_kg"],
                "weighted_cargo_kg": inventory[ship]["weighted_kg"],
                "fixed_cargo_events_unchanged": True,
                "new_certificate": False,
                "proxy_readback": observed,
            }
            write(output / f"{identifier}.json", row)
            rows.append(row)

        old = read(KIT / "inputs/fixtures.json")
        for case in old["historical"]:
            p = case["provenance"]
            assert p["ship"] in split["development_ships"]
            record(
                p["ship"],
                p["model"],
                p["prefix"],
                case["expected"],
                {
                    "kind": "saved_v619_CPU_readback_validated_in_v621",
                    "case_id": case["id"],
                    "producer_sha256": old["source_search_sha256"],
                    "fixture_file_sha256": sha(KIT / "inputs/fixtures.json"),
                },
            )
        calls = 0
        features = read(KIT / "inputs/features.json")["features"]
        for ship in split["queue_held_out_ships"]:
            archive = read(KIT / "inputs" / f"ship-{ship:02}.json")
            assert archive["certified"] and archive["master_certified"] and not archive["failures"]
            assert all(x["certified"] and x["status"] == "feasible" for x in archive["legs"])
            assert inventory[ship]["independent_inventory"]
            plan = RoutePlan.from_summary(archive["plan"])
            assert plan.collected_mass == {
                int(k): v for k, v in archive["collected_mass_kg"].items()
            }
            original = digest(archive)
            at = max(i for i, leg in enumerate(plan.legs) if leg.role == "deploy_hop") + 1
            last = plan.legs[at - 1]
            measured = {(x["from"], x["to"], x["t0"], x["tf"]): x for x in archive["legs"]}
            prefix_mass = (
                measured[(last.from_id, last.to_id, last.departure_epoch, last.arrival_epoch)][
                    "mass_after"
                ]
                - C.MINER_MASS_KG
            )
            forward = plan.legs[at:]
            end = forward[-1]
            tour = CollectTour(
                order=tuple(
                    a for a, _ in sorted(plan.collect_epochs.items(), key=lambda x: (x[1], x[0]))
                ),
                collect_epochs=dict(plan.collect_epochs),
                hops=[
                    (x.from_id, x.to_id, x.departure_epoch, x.tof_days, x.delta_v_proxy_km_s)
                    for x in forward
                    if x.role == "collect_hop"
                ],
                reposition=plan.collect_epochs[last.to_id]
                > next(x.departure_epoch for x in forward if x.role != "camp"),
                objective_kg=0,
                collected_proxy_kg=plan.total_collected_kg,
                propellant_proxy_kg=0,
                return_departure=end.departure_epoch,
                return_tof=end.tof_days,
                return_dv=end.delta_v_proxy_km_s,
            )
            for model in split["models"]:
                settings = {**values}
                if model == "existing_fit_no_refit":
                    settings["collect_dp_inflation_fit"] = str(
                        KIT / "inputs/hop_inflation_fit.json"
                    )
                search = TracedSearch(
                    None, np.asarray([], dtype=np.int64), SearchSettings(**settings)
                )
                table = search.collect_table
                for feature in features.values():
                    table._geometry[
                        (feature["from_id"], feature["to_id"], feature["departure_epoch"], 1)
                    ] = (feature["delta_a_au"], np.asarray([feature["delta_longitude_rad"]]))
                cached_geometry = table.pair_geometry

                def cached_only(
                    source, target, epochs, table=table, cached_geometry=cached_geometry
                ):
                    assert (source, target, float(epochs[0]), len(epochs)) in table._geometry
                    return cached_geometry(source, target, epochs)

                table.pair_geometry = cached_only
                partial = _Partial(
                    list(plan.legs[:at]),
                    last.to_id,
                    last.arrival_epoch,
                    prefix_mass,
                    sorted(plan.deploy_epochs.items(), key=lambda x: (x[1], x[0])),
                )
                legs = search._tour_forward_legs(partial, tour)
                assert legs is not None
                calls += 1
                result = search._finish_cpu(
                    partial,
                    dict(partial.deployed),
                    dict(plan.collect_epochs),
                    legs,
                    use_collect_table=True,
                )
                assert table.lambert_evaluations == 0 and not table.return_sweeps
                if result is not None:
                    assert FixedCargoRequest.from_summary(
                        result.summary()
                    ) == FixedCargoRequest.from_summary(archive["plan"])
                trace, j, stopped, stages = search.trace, 0, False, []
                for leg in legs:
                    if stopped:
                        stages.append({"stage": 0})
                    elif leg.role == "camp":
                        stages.append({"stage": 1})
                    else:
                        t = trace[j]
                        stages.append(
                            {
                                "stage": 4 if "propellant_kg" in t else 2,
                                "inflation": t.get("inflation_spent", 0),
                                "propellant": t.get("propellant_kg", 0),
                                "trace": t,
                            }
                        )
                        j += 1
                        stopped = "propellant_kg" not in t
                assert j == len(trace)
                last_mass = trace[-1].get("mass_after_kg", trace[-1]["mass_before_kg"])
                observed = {
                    "failure_name": "ok" if result is not None else search.last_failure,
                    "result": {
                        "processed_legs": sum(x["stage"] != 0 for x in stages),
                        "final_mass": last_mass,
                        "collected": plan.total_collected_kg,
                    },
                    "leg_results": stages,
                    "forward_trace": trace,
                    "prefix_mass_kg": prefix_mass,
                    "margin_kg": last_mass - (C.DRY_MASS_KG + plan.total_collected_kg),
                }
                assert original == digest(archive)
                record(
                    ship,
                    model,
                    "measured_deployment_prefix",
                    observed,
                    {
                        "kind": "fresh_v622_scalar_finish_cpu_saved_DVs_cached_features",
                        "producer_sha256": sources["src/spacepdhcg/gtoc12/search.py"],
                        "features_sha256": sha(KIT / "inputs/features.json"),
                    },
                )
        assert calls == 34 and len(rows) == 54
        groups = []
        for group, ships in (
            ("development", split["development_ships"]),
            ("queue_held_out", split["queue_held_out_ships"]),
        ):
            for model in split["models"]:
                prefixes = (
                    ["flat_deployment_proxy", "measured_deployment_prefix"]
                    if group == "development"
                    else ["measured_deployment_prefix"]
                )
                for prefix in prefixes:
                    relevant = [
                        r
                        for r in rows
                        if r["ship"] in ships and r["model"] == model and r["prefix"] == prefix
                    ]
                    queue = RefinementAdmissionQueue(replay_controls=True)
                    dispositions = [
                        {"id": r["id"], "disposition": queue.offer(candidates[r["id"]])}
                        for r in relevant
                    ]
                    selected = {x.request.sha256 for x in queue.shortlist()}
                    groups.append(
                        {
                            "group": group,
                            "model": model,
                            "prefix": prefix,
                            "requests": len(relevant),
                            "dispositions": dispositions,
                            "selected": [
                                r["id"] for r in relevant if r["request_sha256"] in selected
                            ],
                            "proxy_accepted": sum(not r["proxy_failure"] for r in relevant),
                            "uncertain_eligible": sum(
                                bool(r["proxy_failure"]) and not r["admission_blocker"]
                                for r in relevant
                            ),
                            "earlier_or_other_rejected": sum(
                                bool(r["admission_blocker"]) for r in relevant
                            ),
                            "actual_refinements": 0,
                        }
                    )
        counterexamples = [
            {
                k: r[k]
                for k in (
                    "id",
                    "input_route_sha256",
                    "proxy_failure",
                    "proxy_deficit_kg",
                    "admission_blocker",
                    "archived_final_dry_margin_kg",
                    "request_sha256",
                    "readback_sha256",
                )
            }
            for r in rows
            if r["proxy_failure"]
        ]
        write(output / "counterexamples.json", counterexamples)
        write(
            output / "fuel-residuals.json",
            {
                "source_sha256": sha(KIT / "inputs/residuals.json"),
                "scope": (
                    "Existing actual-burn-mass local component errors; "
                    "not summed as sequential completion error or refitted."
                ),
                "routes": [x for x in residuals["routes"] if x["ship"] != 3],
                "legs": [x for x in residuals["legs"] if x["ship"] != 3],
            },
        )
        report = {
            "complete": True,
            "selection_sha256": sha(KIT / "selection.json"),
            "saved_development_trace_count": 20,
            "fresh_scalar_finish_cpu_calls": calls,
            "unique_routes": 22,
            "requests": len(rows),
            "queue_groups": groups,
            "GPU_calls": 0,
            "Lambert_calls": 0,
            "new_geometry_calls": 0,
            "new_feature_arithmetic_calls": 0,
            "refinements": 0,
            "model_refits": 0,
            "fresh_physics_certifications": 0,
            "fleet_promotions": 0,
            "scope": (
                "Historical positive controls test admission only. Every admission remains "
                "uncertified; held-out is relative to this queue policy, not earlier audits "
                "or the old fit."
            ),
        }
        write(output / "report.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
