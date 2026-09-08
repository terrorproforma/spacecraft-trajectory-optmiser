"""CPU-only audit of frozen production-adapter calls and saved C API vectors."""
from __future__ import annotations
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "build/performance/completion-adapter-gpu-v621"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    helper = Path(__file__).with_name("review_native_readback.py")
    assert sha(helper) == "86d63e1be44bd2222e8bd7103b489a2bf05a049e82979a3c1c56b0fd661aa88a"
    oracle = types.ModuleType("saved_completion_oracle")
    exec(compile(helper.read_bytes(), str(helper), "exec"), oracle.__dict__)
    read = oracle.read
    run = KIT / "gpu-output"
    profile, ready, report, child = [read(p) for p in
        (KIT/"profile.json", KIT/"ready.json", run/"report.json", run/"child/report.json")]
    assert sha(KIT/"ready.json") == "c420836cce0aa04c819609f09bef9f95261342292fe12fa0927ed2538b67e09f"
    for name, digest in ready["files"].items():
        assert sha(KIT/name) == digest, name
    for name, digest in report["output_sha256"].items():
        assert sha(run/name) == digest, name
    assert report["pins"] == profile and report["ready_sha256"] == sha(KIT/"ready.json")
    assert report["complete"] and report["passed"] and not report["failures"]
    assert len(report["processes"]) == 1 and report["processes"][0]["exit_code"] == 0
    assert not report["processes"][0].get("timed_out", False)
    assert child["complete"] and child["passed"] and not child["collect_only"] and not child["failures"]
    assert child["profile_sha256"] == sha(KIT/"profile.json")
    assert child["actual_evaluation_calls"] == child["actual_completed_evaluation_calls"] == 8
    assert child["actual_candidates"] == 1048 and child["lambert_requests"] == 0
    assert not child["forbidden_gpu_requests"] and len(child["gpu_scopes"]) == 4
    assert len(child["pytest_reports"]) == 12 and all(x["outcome"] == "passed" for x in child["pytest_reports"])
    host_path = ROOT / "build/performance/completion-adapter-cpu-v621b/report.json"
    host = read(host_path)
    assert sha(host_path) == profile["host_report_sha256"]
    source = host_path.parent/"source"
    for name, digest in host["source_sha256"].items():
        assert sha(source/name) == digest, name
    for module, binding in child["source_bindings"].items():
        name = "src/" + module.replace(".", "/") + ".py"
        assert binding["sha256"] == host["source_sha256"][name]
    native_path = ROOT / "build/performance/completion-fullcore-v621/manifest.json"
    assert sha(native_path) == profile["native_manifest_sha256"]
    assert read(native_path)["library"] == profile["library"]
    prior_path = ROOT / "build/performance/completion-native-controls-v621/gpu-output/report.json"
    assert sha(prior_path) == profile["native_controls_report_sha256"] and read(prior_path)["passed"]

    count, max_error, worst_path = 0, 0.0, None
    decimal_count, decimal_worst = 0, Decimal(0)
    outcomes, inputs, outputs = [], [], []

    def check(path, got, expected, exact=False):
        nonlocal count, max_error, worst_path
        count += 1
        if exact or isinstance(expected, int):
            assert got == expected, (path, got, expected)
        elif math.isnan(expected):
            assert math.isnan(got), path
        elif math.isinf(expected):
            assert got == expected, path
        else:
            error = abs(got-expected)
            assert math.isfinite(got) and math.isclose(got, expected, rel_tol=2e-14, abs_tol=2e-11), (path, got, expected)
            if error > max_error:
                max_error, worst_path = error, path

    for index, call in enumerate(child["calls"]):
        inp_path = run/"child"/f"call-{index:02d}-input.json"
        out_path = run/"child"/f"call-{index:02d}-output.json"
        assert sha(inp_path) == call["input_sha256"] and sha(out_path) == call["output_sha256"]
        inp, out = read(inp_path), read(out_path)
        inputs.append(inp); outputs.append(out)
        assert call["native_status"] == 0 and call["entered_native"]
        assert call["index"] == index and call["candidates"] == (259 if index % 2 == 0 else 3)
        assert call["test"].endswith(profile["test_suffixes"][index//2])
        policy, candidates, deploys, legs = [a["rows"] for a in inp["arrays"]]
        policy = policy[0]
        assert policy["abi_version"] == policy["sum_mode"] == 1
        assert (len(candidates), len(deploys), len(legs)) == (call["candidates"], call["deploy_slots"], call["leg_slots"])
        stats = out["stats"]["rows"][0]
        assert (stats["candidates"], stats["deploy_slots"], stats["leg_slots"]) == (len(candidates), len(deploys), len(legs))
        failures = Counter()
        for j, candidate in enumerate(candidates):
            db, dn, lb, ln = [candidate[k] for k in ("deploy_begin", "deploy_count", "leg_begin", "leg_count")]
            case = {"partial_mass": candidate["partial_mass"], "deploys": deploys[db:db+dn], "legs": legs[lb:lb+ln]}
            expected = oracle.replay(case, policy)
            result = out["results"]["rows"][j]
            failures[result["failure"]] += 1
            for key, value in expected["result"].items():
                check(f"{index}/{j}/result/{key}", result[key], value, key == "collected")
            for k, detail in enumerate(expected["leg_results"]):
                observed = out["leg_results"]["rows"][lb+k]
                for key, value in detail.items():
                    check(f"{index}/{j}/leg{k}/{key}", observed[key], value, key == "gained")
                leg = case["legs"][k]
                if observed["stage"] == 4:
                    with localcontext() as context:
                        context.prec = 65
                        mass, dv, inflation, exhaust = map(Decimal.from_float, (observed["departure_mass"], leg["dv"], observed["inflation"], policy["exhaust"]))
                        exact = mass*(Decimal(1)-(-dv*inflation/exhaust).exp())
                        decimal_worst = max(decimal_worst, abs(Decimal.from_float(observed["propellant"])-exact))
                        decimal_count += 1
            for k, value in enumerate(expected["collected_by_deploy"]):
                check(f"{index}/{j}/cargo{k}", out["collected"]["rows"][db+k], value, True)
        outcomes.append({"index": index, "model": profile["test_suffixes"][index//2].split("[")[-1][:-1],
            "candidates": len(candidates), "failure_counts": dict(sorted(failures.items()))})
        if index % 2:
            for a, b in zip(inputs[index-1]["arrays"], inp["arrays"], strict=True):
                assert a["rows"][:len(b["rows"])] == b["rows"]
            for name in ("results", "leg_results", "collected"):
                assert outputs[index-1][name]["rows"][:len(out[name]["rows"])] == out[name]["rows"]
    for i, scope in enumerate(child["gpu_scopes"]):
        assert scope["test"].endswith(profile["test_suffixes"][i])
        assert scope["closed"] and scope["lambert_batches"] == scope["lambert_evaluations"] == 0
        assert scope["telemetry"]["completion_batches"] == 2 and scope["telemetry"]["completion_candidates"] == 262
    findings = {"passed": True, "scope": "Saved production adapter parity, source/budget/lifecycle audit; no new native loads, GPU calls, optimization or timing experiment.",
        "script_sha256": sha(Path(__file__)), "oracle_sha256": sha(helper),
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (KIT/"ready.json", KIT/"profile.json", run/"report.json", run/"child/report.json", host_path, native_path, prior_path)},
        "recorded_evaluation_calls": 8, "recorded_candidate_evaluations": 1048, "recorded_Lambert_requests": 0,
        "closed_owner_scopes": 4, "passed_pytest_phases": 12, "independently_replayed_fields": count,
        "max_finite_field_absolute_error": max_error, "worst_field": worst_path,
        "Decimal65_costed_legs": decimal_count, "Decimal65_max_fuel_error_kg": str(decimal_worst),
        "first_three_repeat_readbacks_exact": True, "outcomes": outcomes,
        "qualification_scope": "Original scalar completion success/failure and all native details; route completion remains proxy costing, not full trajectory certification.",
        "timing_scope": "Instrumentation serializes full input/output arrays inside native-call timing. No throughput or speedup can be derived from those totals."}
    output = Path(__file__).with_name("adapter-readback-findings.json")
    output.write_text(json.dumps(findings, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in findings.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
