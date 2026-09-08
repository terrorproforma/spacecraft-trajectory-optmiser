"""Independent scalar arithmetic checks of the archived-mass residual table."""

from __future__ import annotations

import math

from prepare import KIT, read, sha, write


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = next(i for i in range(1, len(xs)) if x <= xs[i])
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])


def main():
    report = read(KIT / "residuals.json")
    source = read(KIT / "source-sha256.json")
    for name, digest in source.items():
        assert sha(KIT / "source" / name) == digest
    for name, digest in read(KIT / "input-provenance.json")["inputs"].items():
        assert sha(KIT / "inputs" / name) == digest
    fit = read(KIT / "inputs/hop_inflation_fit.json")
    returns = read(KIT / "inputs/return-components-v619.json")
    c = returns["constants"]
    r = returns["model"]
    coefficients = fit["coefficients"]
    maximum_error = 0.0
    root_returns = {x["ship"]: x for x in returns["rows"]}
    for leg in report["legs"]:
        mass, dv, tof = (
            leg["measured_burn_mass_before_kg"],
            leg["saved_lambert_dv_km_s"],
            leg["tof_days"],
        )
        ratio = dv / (c["THRUST_MAX_N"] / mass * tof * c["DAY_S"] / 1000.0)
        assert math.isclose(ratio, leg["authority_ratio_at_measured_mass"], abs_tol=1e-14)
        for name, values in leg["models"].items():
            role = leg["role"]
            if role == "earth_out":
                predicted = leg["measured_propellant_kg"]
            else:
                factor = 1.2
                if role == "earth_return":
                    base = interp(tof, r["RETURN_INFLATION_TOF_DAYS"], r["RETURN_INFLATION_P65"])
                    correction = min(max(1.0 + 0.6 * (ratio - 0.33), 0.85), 1.2)
                    factor = max(0.85, base * correction)
                elif role == "collect_hop" and name == "existing_fit_no_refit":
                    f = leg["features"]
                    vector = [
                        1.0,
                        ratio,
                        tof / 365.25,
                        abs(f["delta_a_au"]) / 0.1,
                        abs(f["delta_longitude_rad"]) / math.pi,
                    ]
                    factor = max(
                        fit["floor"], sum(a * b for a, b in zip(coefficients, vector, strict=True))
                    )
                assert math.isclose(factor, values["inflation"], rel_tol=1e-13, abs_tol=1e-13)
                predicted = mass * -math.expm1(-dv * factor / 39.2266)
            delta = abs(predicted - values["predicted_fuel_at_measured_mass_kg"])
            maximum_error = max(maximum_error, delta)
            assert delta <= 1e-9
            assert math.isclose(
                predicted - leg["measured_propellant_kg"], values["error_kg"], abs_tol=1e-9
            )
        if leg["role"] == "earth_return":
            assert math.isclose(
                leg["models"]["v616_no_fit"]["error_kg"],
                root_returns[leg["ship"]]["model_error_even_at_measured_mass_kg"],
                abs_tol=1e-9,
            )
    assert len(report["legs"]) == 404 and len(report["routes"]) == 23
    assert all(route["all_stored_cargo_within_rule"] for route in report["routes"])
    assert all(route["archived_final_dry_margin_kg"] > 0 for route in report["routes"])
    # The final v619 cases reuse exactly the same saved DVs and immutable cargo/events.
    oracle = read(KIT / "inputs/completion-oracle-v619.json")
    by_leg = {
        (x["ship"], x["from_id"], x["to_id"], x["departure_epoch"], x["arrival_epoch"]): x
        for x in report["legs"]
    }
    for case in oracle["rows"]:
        for step in case["trace"]:
            leg = by_leg[
                (
                    case["ship"],
                    step["from_id"],
                    step["to_id"],
                    step["departure_epoch"],
                    step["arrival_epoch"],
                )
            ]
            assert step["dv_proxy_km_s"] == leg["saved_lambert_dv_km_s"]
            assert step["archived_measured_mass_before_kg"] == leg["measured_burn_mass_before_kg"]
            assert step["archived_measured_propellant_kg"] == leg["measured_propellant_kg"]
    out = {
        "passed": True,
        "leg_model_checks": 808,
        "archived_return_components_matched": 23,
        "maximum_independent_fuel_difference_kg": maximum_error,
        "all_v619_saved_DVs_and_measured_masses_match": True,
        "all_stored_cargo_within_rule": True,
        "archived_positive_dry_margin_routes": 23,
        "new_certificates": 0,
        "GPU_calls": 0,
        "Lambert_calls": 0,
        "residual_report_sha256": sha(KIT / "residuals.json"),
        "feature_table_sha256": sha(KIT / "features.json"),
        "inference": (
            "The archived positive controls and valid fixed cargo support proxy false-negatives "
            "in the v619 measured-prefix cases. This arithmetic check does not establish fresh "
            "trajectory feasibility or replace either full-fleet verifier."
        ),
    }
    write(KIT / "residual-audit.json", out)
    print(out)


if __name__ == "__main__":
    main()
