"""Independent stdlib audit of saved finite benchmark arrays and timing arithmetic."""
from pathlib import Path
import ast
import hashlib
import json
import math
import statistics
import struct
import types
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
KIT = ROOT / "build/performance/completion-pack-benchmark-v622"
DATA = KIT / "output"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())

parser_path = ROOT / "build/performance/completion-model-review-v622/readback_review.py"
assert sha(parser_path) == "0817f1d872e426b37d684102b114f78122f51caaa8108da9ca56cab9a9c9bd8b"
nodes = [n for n in ast.parse(parser_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ("npy", "compare")]
context = dict(ast=ast, struct=struct, math=math)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(parser_path), "exec"), context)
npy, compare = context["npy"], context["compare"]
helper_path = ROOT / "build/performance/completion-integration-review-v621/review_native_readback.py"
assert sha(helper_path) == "86d63e1be44bd2222e8bd7103b489a2bf05a049e82979a3c1c56b0fd661aa88a"
helper = types.ModuleType("independent_formula")
helper.__file__ = str(helper_path)
exec(compile(helper_path.read_text(), str(helper_path), "exec"), helper.__dict__)


def load_npz(path):
    arrays, payloads = {}, {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            arrays[Path(name).stem] = npy(data)
            off = 10 if tuple(data[6:8]) == (1, 0) else 12
            length = struct.unpack_from("<H" if off == 10 else "<I", data, 8)[0]
            payloads[Path(name).stem] = data[off + length:]
    assert set(arrays) == {"results", "leg_results", "collected_by_deploy", "stats"}
    numeric = hashlib.sha256(b"".join(payloads[k] for k in sorted(payloads) if k != "stats")).hexdigest()
    return arrays, numeric


def main():
    preparation = read(OUT / "preparation-findings.json")
    report, launch, plan = read(DATA / "benchmark-report.json"), read(DATA / "launch-report.json"), read(KIT / "plan.json")
    assert launch["ready_sha256"] == preparation["ready_sha256"] == sha(KIT / "ready.json")
    assert launch["complete"] and launch["passed"] and launch["exit_code"] == 0 and not launch["failures"]
    assert launch["core_sha256"] == preparation["core_sha256"]
    assert report["complete"] and report["passed"] and not report["failures"]
    assert report["attempted_method_calls"] == report["native_calls"] == 80
    assert report["worker_sha256"] == sha(KIT / "run.py") and report["plan_sha256"] == sha(KIT / "plan.json")
    assert report["candidates"] == {"ordinary": 13520, "compact": 13520}
    for key in ("fresh_Lambert_solves", "fresh_SCvx_refinements", "fresh_CPU_finish_calls"):
        assert report[key] == 0
    assert report["captures_enabled"] is False
    doc = helper.read(KIT / "inputs/fixtures.json")
    cases, policy = doc["historical"], doc["historical_batch"]["policy"]
    unique = {i for group in plan["groups"] for i in group["fixture_indices"]}
    oracles = {i: helper.replay(cases[i], policy) for i in unique}
    rows, raw_index, seen = [], {}, 0
    grand_error, paired_error = 0.0, 0.0
    for group_number, (group, summary) in enumerate(zip(plan["groups"], report["groups"], strict=True)):
        assert group == summary["group"]
        folder = DATA / group["id"]
        assert read(folder / "summary.json") == summary
        first = {}
        hashes = {}
        measures = summary["measurements"]
        order = (["ordinary", "compact"] if group_number % 2 == 0 else ["compact", "ordinary"]) + plan["warm_order"]
        assert [x["backend"] for x in measures] == order
        assert len(measures) == 10
        for trial, measurement in enumerate(measures):
            backend = measurement["backend"]
            label = f"{trial:02}-{backend}"
            paths = [folder / f"{label}.npz", folder / f"{label}-call.json", folder / f"{label}-comparison.json"]
            for path in paths:
                raw_index[path.relative_to(DATA).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
            arrays, digest = load_npz(paths[0])
            assert digest == measurement["numeric_readback_sha256"]
            call, comparison = read(paths[1]), read(paths[2])
            assert call["run_returned"] is True and call["raw_readback_saved"] is True and call["native_evaluation_count"] == 1
            assert call["method_seconds"] == measurement["method_seconds"] > 0
            assert comparison["passed"] and not comparison["failures"]
            if backend in hashes:
                assert hashes[backend] == digest and not measurement["first_call"]
            else:
                hashes[backend] = digest
                first[backend] = arrays
                assert measurement["first_call"]
            delta = measurement["telemetry_delta"]
            assert delta["completion_batches"] == 1 and delta["completion_candidates"] == group["size"]
            assert math.isclose(measurement["proxy_completions_per_second"], group["size"] / measurement["method_seconds"], rel_tol=1e-15)
            assert delta["completion_total_seconds"] <= measurement["method_seconds"]
            stats = arrays["stats"][0]
            assert stats["candidates"] == group["size"]
            assert math.isclose(delta["completion_kernel_seconds"], stats["kernel_ms"] / 1000.0, abs_tol=2e-18)
            assert 0 <= delta["completion_pack_seconds"] < measurement["method_seconds"]
            assert 0 <= delta["completion_native_call_seconds"] < measurement["method_seconds"]
            assert len(arrays["results"]) == group["size"]
            li = di = accepted = 0
            local_error = 0.0
            for row, index in enumerate(group["fixture_indices"]):
                case, answer = cases[index], oracles[index]
                nd, nl = len(case["deploys"]), len(case["legs"])
                actual = arrays["results"][row]
                local_error = max(local_error, compare(actual, answer["result"], atol=2e-10, rtol=2e-13))
                local_error = max(local_error, compare(arrays["leg_results"][li:li + nl], answer["leg_results"], atol=2e-10, rtol=2e-13))
                compare(arrays["collected_by_deploy"][di:di + nd], answer["collected_by_deploy"], strict=True)
                assert actual["collected"] == answer["result"]["collected"]
                for observed, expected in zip(arrays["leg_results"][li:li + nl], answer["leg_results"], strict=True):
                    assert observed["gained"] == expected["gained"]
                accepted += actual["failure"] == 0
                li += nl
                di += nd
            assert li == len(arrays["leg_results"]) == stats["leg_slots"] == delta["completion_leg_slots"]
            assert di == len(arrays["collected_by_deploy"]) == stats["deploy_slots"]
            assert accepted == group["expected_accepted"] == measurement["validation"]["accepted"]
            if backend == "compact":
                assert measurement["native_model_rebuilds"] == 1 and measurement["ordinary_geometry_cache_pairs"] == 0
            grand_error = max(grand_error, local_error)
            seen += group["size"]
        for key in ("results", "leg_results"):
            paired_error = max(paired_error, compare(first["compact"][key], first["ordinary"][key], atol=2e-10, rtol=2e-13))
        compare(first["compact"]["collected_by_deploy"], first["ordinary"]["collected_by_deploy"], strict=True)
        timing = {}
        for backend in ("ordinary", "compact"):
            warm = [x for x in measures if x["backend"] == backend and not x["first_call"]]
            assert len(warm) == 4
            median = statistics.median(x["method_seconds"] for x in warm)
            assert median == summary[backend + "_warm_median_seconds"]
            timing[backend] = {
                "first_seconds": next(x["method_seconds"] for x in measures if x["backend"] == backend and x["first_call"]),
                "warm_median_seconds": median,
                "warm_candidates_per_second": group["size"] / median,
                "warm_pack_share_median": statistics.median(x["telemetry_delta"]["completion_pack_seconds"] / x["method_seconds"] for x in warm),
                "warm_pack_seconds_median": statistics.median(x["telemetry_delta"]["completion_pack_seconds"] for x in warm),
                "warm_native_seconds_median": statistics.median(x["telemetry_delta"]["completion_native_call_seconds"] for x in warm),
                "warm_kernel_seconds_median": statistics.median(x["telemetry_delta"]["completion_kernel_seconds"] for x in warm),
            }
        ratio = timing["ordinary"]["warm_median_seconds"] / timing["compact"]["warm_median_seconds"]
        assert ratio == summary["ordinary_over_compact_warm_ratio"]
        rows.append({"id": group["id"], "size": group["size"], "accepted_per_batch": group["expected_accepted"], "ordinary_over_compact_warm_ratio": ratio, "timing": timing, "same_arm_five_readbacks_bitwise_equal": True})
    assert seen == 27040
    events = [json.loads(line) for line in (DATA / "events.jsonl").read_text().splitlines()]
    assert len(events) == 160
    assert [e["event"] for e in events] == [kind for _ in range(80) for kind in ("requested", "completed")]
    for path in [DATA / name for name in ("events.jsonl", "worker.log", "benchmark-report.json", "launch-report.json", "prerequisite.json")]:
        raw_index[path.relative_to(DATA).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    findings = {
        "scope": "Saved-array and timing arithmetic audit only; no native loading or GPU execution by reviewer",
        "ready_sha256": sha(KIT / "ready.json"), "worker_sha256": sha(KIT / "run.py"),
        "report_sha256": sha(DATA / "benchmark-report.json"), "launch_report_sha256": sha(DATA / "launch-report.json"),
        "core_sha256": launch["core_sha256"], "attempted_and_returned_methods": 80,
        "candidate_evaluations": seen, "per_arm": report["candidates"], "unique_historical_controls": len(unique),
        "kernels_from_frozen_source": 160, "total_worker_seconds": report["total_worker_seconds"],
        "all_calls_classification_and_cargo_agree": True, "all_80_saved_arrays_audited": True,
        "maximum_independent_historical_formula_error": grand_error, "maximum_paired_arm_error": paired_error,
        "formula_scope": "Forward replay of the 20 frozen historical coefficient records, compared with actual outputs. Runtime catalogue/model metadata are checked by prior compact-g correctness and current frozen-source identity, not reconstructed from these output-only NPZ files.",
        "rows": rows,
        "limitations": ["Repeated proxy candidates including rejects; not fresh routes or verified solutions", "One component library, one process, four warm samples per arm/group", "No new Lambert/SCvx solve, fleet score change or default promotion", "First workspace creation contains process/context startup and is reported separately"],
        "helper_sha256": sha(helper_path), "npz_parser_source_sha256": sha(parser_path),
    }
    (OUT / "results-findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    (OUT / "results-readback-index.json").write_text(json.dumps(raw_index, indent=2) + "\n")
    print(json.dumps({"passed": True, "calls": 80, "candidates": seen, "max_formula_error": grand_error, "max_paired_error": paired_error, "ratios": {r["id"]: r["ordinary_over_compact_warm_ratio"] for r in rows}}))


if __name__ == "__main__":
    main()
