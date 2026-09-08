"""Independent standard-library replay of saved completion readbacks; no CUDA."""
from __future__ import annotations
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"float64"}:
            return float(value["float64"])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def read(path):
    return decode(json.loads(path.read_text()))


def maximum(a, b):
    return math.nan if math.isnan(a) or math.isnan(b) else max(a, b)


def factor(leg, authority, year):
    model, dv = leg["model"], leg["dv"]
    if model in (0, 4):
        return leg["flat"]
    if model in (2, 5) and not math.isfinite(dv):
        dv = 0.0
    ratio = dv / maximum(authority, 1e-12)
    if model == 1:
        return leg["floor"] + leg["slope"] * ratio
    tof = leg["arrival"] - leg["departure"]
    if model == 2:
        features = (1.0, ratio, tof / year, abs(leg["delta_a_au"]) / 0.1, abs(leg["delta_longitude_rad"]) / math.pi)
        value = 0.0
        for x, coefficient in zip(features, leg["fit"], strict=True):
            value += x * coefficient
        return maximum(value, leg["floor"])
    assert model in (3, 5)
    days = (352., 420., 450., 480., 510., 540., 578., 630., 690., 810.)
    values = (1.323, 1.383, 1.295, 1.195, 1.099, .977, .885, .930, .932, 1.014)
    if math.isnan(tof):
        base = math.nan
    elif tof <= days[0]:
        base = values[0]
    elif tof >= days[-1]:
        base = values[-1]
    else:
        i = next(i for i in range(1, len(days)) if tof <= days[i])
        base = values[i] if tof == days[i] else values[i-1] + (values[i]-values[i-1]) / (days[i]-days[i-1]) * (tof-days[i-1])
    correction = 1.0 + .6 * (ratio - .33)
    clipped = math.nan if math.isnan(correction) else min(1.2, max(.85, correction))
    return maximum(base * clipped, .85)


def replay(case, policy):
    mass = case["partial_mass"]
    fuel = (policy["initial_mass"] - mass) - policy["miner_mass"] * len(case["deploys"])
    cargo = {}
    failure, failed_leg, failed_deploy, processed = 0, -1, -1, 0
    keys = ("mass_before", "gained", "departure_mass", "mass_after", "authority", "inflation", "propellant", "tof")
    details = [{"stage": 0, "pickup": 0, **dict.fromkeys(keys, 0.0)} for _ in case["legs"]]
    for i, deploy in enumerate(case["deploys"]):
        if not deploy["has_collect"]:
            failure, failed_deploy = 1, i
            break
        if deploy["collect_epoch"] - deploy["deploy_epoch"] < policy["minimum_stay"] - 1e-6:
            failure, failed_deploy = 2, i
            break
    for i, leg in enumerate(case["legs"]):
        if failure:
            break
        detail = details[i]
        tof = leg["arrival"] - leg["departure"]
        detail.update(mass_before=mass, departure_mass=mass, mass_after=mass, tof=tof)
        if leg["role"] == 0:
            detail.update(stage=1, inflation=leg["flat"])
            processed += 1
            continue
        if leg["role"] in (3, 4):
            source = leg["source_deploy"]
            deploy = case["deploys"][source]
            if abs(deploy["collect_epoch"] - leg["departure"]) < 1e-6:
                detail["pickup"] = 1
                stay = deploy["collect_epoch"] - deploy["deploy_epoch"]
                if not math.isfinite(stay) or stay < 0:
                    failure, failed_leg, failed_deploy = 6, i, source
                    detail["stage"] = 5
                    break
                gain = (policy["mining_rate"] * stay) / policy["year_days"]
                cargo[source] = gain
                mass += gain
                detail.update(gained=gain, departure_mass=mass, mass_after=mass)
        authority = ((policy["thrust"] / mass * 1e-3) * tof) * 86400.0
        detail["authority"] = authority
        if not leg["dv"] <= leg["authority_ratio"] * authority:
            failure, failed_leg, detail["stage"] = 3, i, 2
            break
        inflation = factor(leg, authority, policy["year_days"])
        detail["inflation"] = inflation
        if not math.isfinite(inflation) or inflation < 0:
            failure, failed_leg, detail["stage"] = 4, i, 3
            break
        try:
            exponential = math.exp(-(leg["dv"] * inflation) / policy["exhaust"])
        except OverflowError:
            exponential = math.inf
        spent = mass * (1.0 - exponential)
        fuel += spent
        mass -= spent
        detail.update(stage=4, propellant=spent, mass_after=mass)
        processed += 1
    collected = sum(cargo.values())
    margin = mass - (policy["dry_mass"] + collected)
    if not failure and mass < policy["dry_mass"] + collected:
        failure = 5
    return {
        "result": dict(failure=failure, failed_leg=failed_leg, failed_deploy=failed_deploy, processed_legs=processed,
                       propellant=fuel, final_mass=mass, collected=collected, margin=margin),
        "leg_results": details,
        "collected_by_deploy": [cargo.get(i, 0.0) for i in range(len(case["deploys"]))],
        "pickup_insertion_order": list(cargo),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, default=Path("build/performance/completion-native-controls-v621"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("native-readback-findings.json"))
    args = parser.parse_args()
    kit, run = args.kit, args.kit / "gpu-output"
    document, raw = read(kit / "fixtures.json"), read(run / "batch-readback.json")
    profile, report = read(kit / "gpu-profile.json"), read(run / "report.json")
    assert report["passed"] and report["complete"] and report["failures"] == []
    assert len(report["processes"]) == 2 and all(p["exit_code"] == 0 and not p.get("timed_out") for p in report["processes"])
    assert report["pins"] == profile
    assert raw["fixtures_sha256"] == sha(kit / "fixtures.json") == profile["fixtures_sha256"]
    assert raw["library_sha256"] == profile["library"]["sha256"]
    assert raw["api_status"] == 0 and raw["outputs_valid"] and raw["evaluate_calls"] == 1
    assert (report["native_kernel_launches"], report["fixture_kernel_launches"], report["actual_case_evaluations"]) == (2, 1, 304)
    events = [json.loads(line) for line in (run / "batch-events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["create_requested", "create_returned", "evaluate_requested", "evaluate_returned", "destroy_requested", "destroy_returned"]
    assert all(e["status"] == 0 for e in events if "status" in e) and events[-1]["cleared"]
    cases = document["historical"] + document["synthetic"]
    assert len(cases) == 42 and len(raw["leg_results"]) == 260 and len(raw["collected_by_deploy"]) == 196
    assert raw["candidate_ids"] == [c["id"] for c in cases]
    assert document["historical_batch"]["policy"] == document["synthetic_batch"]["policy"]
    policy = document["historical_batch"]["policy"]
    counts = {"frozen_fields": 0, "independent_formula_fields": 0}
    worst = {name: {"absolute": 0.0, "path": None} for name in counts}
    decimal_worst, decimal_count = Decimal(0), 0
    strict = set(profile["comparison"]["exact_result_cases"])
    outcomes, li, di = [], 0, 0

    def check(group, path, observed, expected, exact=False):
        counts[group] += 1
        if isinstance(expected, int) or exact:
            assert observed == expected, (group, path, observed, expected)
        elif math.isnan(expected):
            assert math.isnan(observed), (group, path)
        elif math.isinf(expected):
            assert observed == expected, (group, path)
        else:
            difference = abs(observed - expected)
            assert math.isfinite(observed) and math.isclose(observed, expected, rel_tol=2e-13, abs_tol=2e-10), (group, path, observed, expected)
            if difference > worst[group]["absolute"]:
                worst[group] = {"absolute": difference, "path": path}

    for index, case in enumerate(cases):
        for group, expected in (("frozen_fields", case["expected"]), ("independent_formula_fields", replay(case, policy))):
            for key, value in expected["result"].items():
                check(group, f"{case['id']}.result.{key}", raw["results"][index][key], value, key == "collected" or case["id"] in strict)
            for j, detail in enumerate(expected["leg_results"]):
                for key, value in detail.items():
                    check(group, f"{case['id']}.leg{j}.{key}", raw["leg_results"][li+j][key], value, key == "gained")
            for j, value in enumerate(expected["collected_by_deploy"]):
                check(group, f"{case['id']}.cargo{j}", raw["collected_by_deploy"][di+j], value, True)
            insertion = []
            for j, leg in enumerate(case["legs"]):
                detail = raw["leg_results"][li+j]
                if detail["pickup"] and detail["stage"] != 5 and leg["source_deploy"] not in insertion:
                    insertion.append(leg["source_deploy"])
            assert insertion == expected["pickup_insertion_order"]
        for j, leg in enumerate(case["legs"]):
            detail = raw["leg_results"][li+j]
            if detail["stage"] == 4 and all(math.isfinite(v) for v in (detail["departure_mass"], detail["inflation"], detail["propellant"], leg["dv"])):
                with localcontext() as context:
                    context.prec = 65
                    mass, inflation, dv, exhaust = map(Decimal.from_float, (detail["departure_mass"], detail["inflation"], leg["dv"], policy["exhaust"]))
                    exact = mass * (Decimal(1) - (-dv * inflation / exhaust).exp())
                    decimal_worst = max(decimal_worst, abs(Decimal.from_float(detail["propellant"]) - exact))
                    decimal_count += 1
        outcomes.append({"id": case["id"], "failure": raw["results"][index]["failure"]})
        li += len(case["legs"])
        di += len(case["deploys"])
    result = {
        "passed": True, "GPU_calls_by_reviewer": 0, "native_loads_by_reviewer": 0,
        "input_hashes": {str(p): sha(p) for p in (kit/"fixtures.json", kit/"gpu-profile.json", run/"report.json", run/"batch-readback.json", run/"batch-comparison.json", run/"native-test.log", run/"batch-events.jsonl")},
        "script_sha256": sha(Path(__file__)), "counts": counts, "maximum_finite_differences": worst,
        "Decimal65_costed_finite_legs": decimal_count, "Decimal65_max_absolute_fuel_error_kg": str(decimal_worst),
        "recorded_kernel_launches": 3, "recorded_candidate_evaluations": 304,
        "historical_successes": sum(o["failure"] == 0 for o in outcomes[:20]),
        "synthetic_successes": sum(o["failure"] == 0 for o in outcomes[20:]),
        "readback_stats": raw["stats"], "recorded_API_seconds": raw["wall_seconds_including_API"],
        "timing_scope": "One first-use 42-case correctness batch; not steady-state throughput, comparative speedup or mission speed.",
        "outcomes": outcomes,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_hashes", "outcomes")}, indent=2))


if __name__ == "__main__":
    main()
