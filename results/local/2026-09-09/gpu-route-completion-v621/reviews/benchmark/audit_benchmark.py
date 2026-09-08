"""Independent stdlib-only audit of saved benchmark counts, timing and NPZ vectors."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import types
import zipfile

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "build/performance/completion-benchmark-v621"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def npy(data):
    assert data[:6] == b"\x93NUMPY"
    version = tuple(data[6:8])
    assert version in ((1, 0), (2, 0))
    length = struct.unpack_from("<H" if version == (1, 0) else "<I", data, 8)[0]
    offset = 10 if version == (1, 0) else 12
    header = ast.literal_eval(data[offset:offset+length].decode("ascii"))
    assert not header["fortran_order"] and len(header["shape"]) == 1
    count, descriptor = header["shape"][0], header["descr"]
    payload = data[offset+length:]
    if descriptor == "<f8":
        assert len(payload) == count*8
        return [row[0] for row in struct.iter_unpack("<d", payload)], payload
    assert isinstance(descriptor, list)
    formats = {"<i4": "i", "<u8": "Q", "<f8": "d"}
    assert all(len(row) == 2 and row[1] in formats for row in descriptor)
    fmt = "<" + "".join(formats[kind] for _, kind in descriptor)
    assert len(payload) == count*struct.calcsize(fmt)
    return [dict(zip((name for name, _ in descriptor), values, strict=True))
            for values in struct.iter_unpack(fmt, payload)], payload


def main():
    output = KIT/"output"
    ready, plan, report, launch = [read(p) for p in
        (KIT/"ready.json", KIT/"plan.json", output/"benchmark-report.json", output/"launch-report.json")]
    assert sha(KIT/"ready.json") == "6227c91680ab4406d5f3fd3bdd7ac17317d78a2c4b90e8b143c24fb25ad23f2f"
    for name, digest in ready["files"].items():
        assert sha(KIT/name) == digest, name
    assert report["complete"] and report["passed"] and not report["failures"]
    assert launch["complete"] and launch["passed"] and not launch["failures"]
    assert launch["exit_code"] == 0 and not launch.get("timed_out", False)
    assert launch["ready_sha256"] == sha(KIT/"ready.json")
    assert launch["fullcore_sha256"] == read(KIT/"inputs/native-manifest.json")["library"]["sha256"]
    assert report["run_sha256"] == sha(KIT/"run.py") and report["plan_sha256"] == sha(KIT/"plan.json")
    assert len(plan["groups"]) == len(report["groups"]) == 12
    assert plan["unique_historical_controls"] == 20
    assert report["GPU_evaluate_calls"] == plan["gpu_evaluate_calls"] == 60
    assert report["GPU_candidate_evaluations"] == report["CPU_candidate_evaluations"] == 14200
    helper = ROOT/"build/performance/completion-integration-review-v621/review_native_readback.py"
    assert sha(helper) == "86d63e1be44bd2222e8bd7103b489a2bf05a049e82979a3c1c56b0fd661aa88a"
    oracle = types.ModuleType("frozen_independent_completion")
    exec(compile(helper.read_bytes(), str(helper), "exec"), oracle.__dict__)
    fixtures = oracle.decode(read(KIT/"inputs/fixtures.json"))
    originals = fixtures["historical"]
    assert len(originals) == 20
    policy = fixtures["historical_batch"]["policy"]
    independent = [oracle.replay(case, policy) for case in originals]
    events = [read_line(line) for line in (output/"events.jsonl").read_text().splitlines()]
    assert len(events) == 240
    counters = {"saved_GPU_calls": 0, "timed_CPU_batches": 0, "GPU_candidates": 0, "CPU_candidates": 0,
                "raw_fields_vs_frozen": 0, "raw_fields_vs_independent": 0, "compact_outcome_fields": 0}
    worst = {name: {"absolute": 0.0, "path": None} for name in
        ("raw_fields_vs_frozen", "raw_fields_vs_independent", "compact_outcome_fields")}

    def check(group, path, actual, expected, exact=False):
        counters[group] += 1
        if exact or isinstance(expected, int):
            assert actual == expected, (path, actual, expected)
        elif math.isnan(expected):
            assert math.isnan(actual), path
        elif math.isinf(expected):
            assert actual == expected, path
        else:
            error = abs(actual-expected)
            assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-10), (path, actual, expected)
            if error > worst[group]["absolute"]:
                worst[group] = {"absolute": error, "path": path}

    summaries = []
    for gi, (group, saved) in enumerate(zip(plan["groups"], report["groups"], strict=True)):
        assert group == saved["group"]
        folder = output/group["id"]
        assert read(folder/"summary.json") == saved
        cases = [originals[i] for i in group["fixture_indices"]]
        assert len(cases) == group["size"] and len(set(group["fixture_indices"])) == group["unique_control_count"]
        assert all(case["provenance"]["model"] == group["model"] for case in cases)
        sequence = (["cpu", "gpu"] if gi % 2 == 0 else ["gpu", "cpu"]) + plan["warm_order"]
        assert len(saved["measurements"]) == len(sequence) == 10
        expected_success = sum(case["expected"]["failure_name"] == "ok" for case in cases)
        assert expected_success == group["expected_accepted"]
        compact_hashes = {}
        for backend in ("cpu", "gpu"):
            compact = read(folder/f"first-{backend}-outcomes.json")
            assert len(compact) == group["size"]
            compact_hashes[backend] = hashlib.sha256(json.dumps(compact, sort_keys=True, allow_nan=False).encode()).hexdigest()
            for ci, (case, item) in enumerate(zip(cases, compact, strict=True)):
                expected = case["expected"]
                reason = "" if expected["failure_name"] == "ok" else expected["failure_name"]
                assert item["reason"] == reason
                if reason:
                    assert set(item) == {"reason"}
                    continue
                for name, field in (("mass", "final_mass"), ("fuel", "propellant")):
                    check("compact_outcome_fields", f"{group['id']}/{backend}/{ci}/{name}", item[name], expected["result"][field])
                ids = case["provenance"]["deploy_asteroid_ids"]
                cargo = [[ids[i], expected["collected_by_deploy"][i]] for i in expected["pickup_insertion_order"]]
                assert item["cargo_in_order"] == cargo
                for index, (value, detail) in enumerate(zip(item["inflation"][-len(case["legs"]):], expected["leg_results"], strict=True)):
                    check("compact_outcome_fields", f"{group['id']}/{backend}/{ci}/inflation{index}", value, detail["inflation"])
        initial_payloads = None
        for trial, (backend, measurement) in enumerate(zip(sequence, saved["measurements"], strict=True)):
            assert measurement["trial"] == trial and measurement["backend"] == backend
            assert measurement["first_call"] == (trial < 2)
            assert measurement["plan_validation"] == {"passed": True, "accepted": expected_success, "rejected": group["size"]-expected_success}
            assert measurement["raw_outcome_sha256"] == compact_hashes[backend]
            duration = measurement["method_seconds"]
            assert duration > 0 and math.isfinite(duration)
            assert group["size"]/duration == measurement["proxy_evaluations_per_second"]
            requested, completed = events[2*(gi*10+trial):2*(gi*10+trial)+2]
            assert requested["event"] == "method_requested" and completed["event"] == "method_completed"
            assert requested["group"] == completed["group"] == group["id"]
            assert requested["trial"] == trial and requested["backend"] == backend
            assert requested["GPU_evaluate_calls_before"] == counters["saved_GPU_calls"]
            assert {k:v for k,v in completed.items() if k not in ("event", "group")} == measurement
            counters["saved_GPU_calls" if backend == "gpu" else "timed_CPU_batches"] += 1
            counters[backend.upper()+"_candidates"] += group["size"]
            if backend == "cpu":
                continue
            stem = f"{trial:02}-{backend}"
            assert read(folder/f"{stem}-parity.json")["passed"]
            with zipfile.ZipFile(folder/f"{stem}-readback.npz") as archive:
                assert set(archive.namelist()) == {"results.npy", "leg_results.npy", "collected_by_deploy.npy", "stats.npy"}
                arrays, payloads = {}, {}
                for member in archive.namelist():
                    arrays[member[:-4]], payloads[member] = npy(archive.read(member))
            if initial_payloads is None:
                initial_payloads = payloads
            else:
                assert all(payloads[k] == initial_payloads[k] for k in payloads if k != "stats.npy")
            stats = arrays["stats"][0]
            assert stats["candidates"] == group["size"]
            assert len(arrays["results"]) == group["size"]
            assert len(arrays["leg_results"]) == stats["leg_slots"] == sum(len(c["legs"]) for c in cases)
            assert len(arrays["collected_by_deploy"]) == stats["deploy_slots"] == sum(len(c["deploys"]) for c in cases)
            delta = measurement["telemetry_delta"]
            assert delta["completion_batches"] == 1 and delta["completion_candidates"] == group["size"]
            assert delta["completion_leg_slots"] == stats["leg_slots"]
            assert math.isclose(delta["completion_kernel_seconds"], stats["kernel_ms"]*1e-3, rel_tol=1e-10, abs_tol=1e-15)
            assert 0 <= delta["completion_pack_seconds"] <= delta["completion_total_seconds"] <= duration
            assert 0 <= delta["completion_native_call_seconds"] <= delta["completion_total_seconds"]
            assert delta["completion_pack_seconds"] + delta["completion_native_call_seconds"] <= delta["completion_total_seconds"] + 1e-12
            li = di = 0
            for ci, (case, fixture_i) in enumerate(zip(cases, group["fixture_indices"], strict=True)):
                for category, expected in (("raw_fields_vs_frozen", case["expected"]), ("raw_fields_vs_independent", independent[fixture_i])):
                    for key, value in expected["result"].items():
                        check(category, f"{group['id']}/{trial}/{ci}/result/{key}", arrays["results"][ci][key], value, key == "collected")
                    for j, detail in enumerate(expected["leg_results"]):
                        for key, value in detail.items():
                            check(category, f"{group['id']}/{trial}/{ci}/leg{j}/{key}", arrays["leg_results"][li+j][key], value, key == "gained")
                    for j, value in enumerate(expected["collected_by_deploy"]):
                        check(category, f"{group['id']}/{trial}/{ci}/cargo{j}", arrays["collected_by_deploy"][di+j], value, True)
                li += len(case["legs"]); di += len(case["deploys"])
        assert saved["telemetry"]["completion_batches"] == 5
        assert saved["telemetry"]["completion_candidates"] == 5*group["size"]
        assert saved["telemetry"]["completed_branch_requests"] == saved["telemetry"]["completed_batches"] == 0
        for backend in ("cpu", "gpu"):
            samples = [m["method_seconds"] for m in saved["measurements"] if m["backend"] == backend and not m["first_call"]]
            assert len(samples) == 4 and samples == saved[backend+"_warm"]["samples_seconds"]
            median = statistics.median(samples)
            assert median == saved[backend+"_warm"]["median_seconds"]
            assert group["size"]/median == saved[backend+"_warm"]["proxy_evaluations_per_second"]
        ratio = saved["cpu_warm"]["median_seconds"]/saved["gpu_warm"]["median_seconds"]
        assert ratio == saved["warm_host_inclusive_speed_ratio_cpu_over_gpu"]
        warm = [m for m in saved["measurements"] if m["backend"] == "gpu" and not m["first_call"]]
        summaries.append({"group": group["id"], "size": group["size"], "accepted_per_call": expected_success,
            "CPU_median_seconds": saved["cpu_warm"]["median_seconds"], "GPU_median_seconds": saved["gpu_warm"]["median_seconds"],
            "CPU_over_GPU_speed_ratio": ratio, "GPU_proxy_evaluations_per_second": saved["gpu_warm"]["proxy_evaluations_per_second"],
            "median_paired_pack_share_of_GPU_method": statistics.median(m["telemetry_delta"]["completion_pack_seconds"]/m["method_seconds"] for m in warm),
            "median_paired_native_share_of_GPU_method": statistics.median(m["telemetry_delta"]["completion_native_call_seconds"]/m["method_seconds"] for m in warm),
            "first_GPU_method_seconds": next(m["method_seconds"] for m in saved["measurements"] if m["backend"] == "gpu"),
            "all_five_GPU_raw_numerical_readbacks_byte_equal": True})
    assert counters["saved_GPU_calls"] == counters["timed_CPU_batches"] == 60
    assert counters["GPU_candidates"] == counters["CPU_candidates"] == 14200
    output_index = {str(p.relative_to(output)).replace("\\", "/"): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    output_tree = hashlib.sha256("".join(k+":"+v+"\n" for k,v in sorted(output_index.items())).encode()).hexdigest()
    findings = {"passed": True, "scope": "Read-only standard-library audit of saved completion benchmark, no GPU/native loading/new costing experiment.",
        "script_sha256": sha(Path(__file__)), "independent_formula_helper_sha256": sha(helper),
        "ready_sha256": sha(KIT/"ready.json"), "launch_report_sha256": sha(output/"launch-report.json"),
        "benchmark_report_sha256": sha(output/"benchmark-report.json"), "output_file_count": len(output_index), "output_tree_sha256": output_tree,
        "counts": counters, "max_finite_differences": worst, "paired_share_definition": "Median of four warm per-sample pack/method or native/method fractions, not ratio of unrelated medians.",
        "unique_historical_controls": 20, "groups": summaries,
        "interpretation": "Warm proxy completion on repeated saved requests with cached geometry. Setup/construction/capture/validation are separately retained; method timing includes complete CPU loop or GPU pack/native/readout. Four warm samples per group, single device/launch. No distinct-solution, SOTA, current-fleet, physics-certification or full-mission speed claim.",
        "next_measured_bottleneck": "Host packing/model metadata dominates large GPU batches; preserve scalar parity when considering retained numeric metadata. Small no-fit batches do not justify automatic GPU routing."}
    out = Path(__file__).with_name("findings.json")
    out.write_text(json.dumps(findings, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    Path(__file__).with_name("output-sha256.json").write_text(json.dumps(output_index, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in findings.items() if k != "groups"}, indent=2))


def read_line(text):
    return json.loads(text)


if __name__ == "__main__":
    main()
