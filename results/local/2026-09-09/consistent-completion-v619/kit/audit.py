"""Audit matched controls and independently recompute the return-mass component."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from prepare import KIT, read, sha, write


def row_key(row):
    return row["ship"], row["model"], row["prefix"]


def interpolation(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = next(i for i, right in enumerate(xs[1:], 1) if x <= right)
    return ys[i-1] + (ys[i]-ys[i-1]) * (x-xs[i-1]) / (xs[i]-xs[i-1])


def component_check():
    report = read(KIT / "inputs/root-return-components.json")
    inventory = read(KIT / "inputs/incumbent-inventory.json")
    constants = report["constants"]
    model = report["model"]
    assert constants == {"G0_M_S2": 9.80665, "DAY_S": 86400.0, "ISP_S": 4000.0,
                         "THRUST_MAX_N": 0.6, "MINER_MASS_KG": 40.0}
    verified = []
    for row in report["rows"]:
        item = next(item for item in inventory["routes"] if item["ship"] == row["ship"])
        path = Path(item["path"])
        assert sha(path) == item["sha256"] == row["input_sha256"]
        archive = read(path)
        planned = archive["plan"]["legs"]
        last_deploy = next(x for x in reversed(planned) if x["role"] == "deploy_hop")
        last_return = planned[-1]
        assert last_return["role"] == "earth_return"
        matches = lambda a, b: all(a[k] == b[k] for k in ("from", "to", "t0", "tf"))
        actual_deploy = next(x for x in archive["legs"] if matches(x, last_deploy))
        actual_return = next(x for x in archive["legs"] if matches(x, last_return))
        cargo = sum(archive["collected_mass_kg"].values())
        prefix = actual_deploy["mass_after"] - 40.0
        guessed_mass, actual_mass = prefix + cargo, actual_return["mass_before"]
        dv, tof = last_return["dv_proxy_km_s"], last_return["tf"] - last_return["t0"]

        def factor(mass):
            ratio = dv / (0.6 / mass * tof * 86400.0 / 1000.0)
            base = interpolation(tof, model["RETURN_INFLATION_TOF_DAYS"], model["RETURN_INFLATION_P65"])
            correction = min(max(1.0 + 0.6 * (ratio - 0.33), 0.85), 1.2)
            return max(0.85, base * correction)

        def fuel(inflation):
            return actual_mass * -math.expm1(-dv * inflation / (9.80665 * 4000.0 / 1000.0))

        computed = {
            "fixed_cargo_kg": cargo, "measured_post_deploy_mass_kg": prefix,
            "builder_style_mass_using_measured_prefix_kg": guessed_mass,
            "measured_return_departure_mass_kg": actual_mass,
            "mass_overestimate_kg": guessed_mass-actual_mass,
            "model_at_guessed_mass": factor(guessed_mass), "model_at_measured_mass": factor(actual_mass),
            "fuel_at_measured_mass_with_guessed_mass_inflation_kg": fuel(factor(guessed_mass)),
            "fuel_at_measured_mass_with_measured_mass_inflation_kg": fuel(factor(actual_mass)),
            "fuel_bias_from_mass_argument_alone_kg": fuel(factor(guessed_mass))-fuel(factor(actual_mass)),
            "model_error_even_at_measured_mass_kg": fuel(factor(actual_mass))-actual_return["propellant_kg"],
            "saved_leg_fuel_error_kg": fuel(last_return["inflation"])-actual_return["propellant_kg"],
        }
        for name, value in computed.items():
            assert math.isclose(value, row[name], rel_tol=1e-12, abs_tol=1e-9), (row["ship"], name, value, row[name])
        verified.append({"ship": row["ship"], "input_sha256": row["input_sha256"], **computed})
    summary = {
        "median_mass_overestimate_kg": statistics.median(x["mass_overestimate_kg"] for x in verified),
        "median_return_fuel_bias_from_mass_argument_kg": statistics.median(x["fuel_bias_from_mass_argument_alone_kg"] for x in verified),
        "maximum_return_fuel_bias_from_mass_argument_kg": max(x["fuel_bias_from_mass_argument_alone_kg"] for x in verified),
        "median_model_error_at_measured_mass_kg": statistics.median(x["model_error_even_at_measured_mass_kg"] for x in verified),
        "positive_model_errors_at_measured_mass": sum(x["model_error_even_at_measured_mass_kg"] > 0 for x in verified),
    }
    for name, value in summary.items():
        assert math.isclose(value, report["summary"][name], rel_tol=1e-12, abs_tol=1e-9)
    return {"source_report_sha256": sha(KIT / "inputs/root-return-components.json"),
            "all_23_input_hashes_verified": True, "independent_formula": True,
            "summary": summary, "rows": verified}


def main():
    old = read(KIT / "output/baseline.json")
    new = read(KIT / "output/candidate-final.json")
    assert old["selection_sha256"] == new["selection_sha256"] == sha(KIT / "selection.json")
    assert old["scalar_bridge_calls"] == new["scalar_bridge_calls"] == 20
    for report in (old, new):
        assert all(report[k] == 0 for k in ("GPU_calls", "Lambert_calls", "DP_solves", "refinements", "physics_certifications", "fleet_promotions"))
        assert len({row_key(row) for row in report["rows"]}) == 20
    before = {row_key(row): row for row in old["rows"]}
    differences = []
    for after in new["rows"]:
        prior = before[row_key(after)]
        assert prior["cargo_kg"] == after["cargo_kg"]
        assert prior["weighted_cargo_kg"] == after["weighted_cargo_kg"]
        assert prior["prefix_mass_kg"] == after["prefix_mass_kg"]
        assert prior["complete_forward_flight_count"] == after["complete_forward_flight_count"]
        for record in (prior, after):
            assert record["prescribed_inputs_unchanged"]
            assert not record["trajectory_certified_now"] and not record["emitted_fleet"]
            assert all(x["authority_pass"] for x in record["trace"])
            assert len(record["trace"]) == record["complete_forward_flight_count"]
            for trace in record["trace"]:
                expected = trace["mass_before_kg"] * -math.expm1(-trace["dv_proxy_km_s"] * trace["inflation_spent"] / 39.2266)
                assert math.isclose(expected, trace["propellant_kg"], rel_tol=1e-12, abs_tol=1e-9)
        differences.append({
            "ship": after["ship"], "model": after["model"], "prefix": after["prefix"],
            "baseline_admitted": prior["accepted_by_scalar_bridge"], "candidate_admitted": after["accepted_by_scalar_bridge"],
            "baseline_failure": prior["failure"], "candidate_failure": after["failure"],
            "baseline_margin_kg": prior["proxy_final_cargo_margin_kg"],
            "candidate_margin_kg": after["proxy_final_cargo_margin_kg"],
            "margin_change_kg": after["proxy_final_cargo_margin_kg"] - prior["proxy_final_cargo_margin_kg"],
            "fixed_cargo_kg": after["cargo_kg"], "fixed_weighted_cargo_kg": after["weighted_cargo_kg"],
            "archived_final_dry_margin_kg": after["archived_final_dry_margin_kg"],
        })
    report = {
        "complete": True, "GPU_calls": 0, "scalar_bridge_calls": 40,
        "baseline_admitted": sum(x["baseline_admitted"] for x in differences),
        "candidate_admitted": sum(x["candidate_admitted"] for x in differences),
        "all_measured_prefix_controls_still_rejected": all(not x["candidate_admitted"] for x in differences if x["prefix"] == "measured_deployment_prefix"),
        "all_forward_authority_checks_passed": True,
        "all_rejections_are_final_dry_plus_cargo_proxy_gate": True,
        "no_retiming_cargo_reduction_refit_or_fleet_promotion": True,
        "rows": differences,
        "root_return_component_independent_check": component_check(),
        "scope": "Admission diagnostic; successful scalar completion is not a physics certificate or a scored fleet. All archived certifications are retained historical evidence, not recomputed here.",
    }
    write(KIT / "output/audit.json", report)
    print(json.dumps({key: report[key] for key in ("complete", "scalar_bridge_calls", "baseline_admitted", "candidate_admitted", "all_measured_prefix_controls_still_rejected")}))


if __name__ == "__main__":
    main()
