"""Independent saved-output review of combined integration smoke; no CUDA."""
from pathlib import Path
import ast
import hashlib
import json
import math
import struct
import types
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "build/performance/combined-smoke-v622a"
PRIOR = ROOT / "build/performance/completion-model-gpu-v622d"
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


def arrays(path):
    with zipfile.ZipFile(path) as archive:
        return {Path(n).stem: npy(archive.read(n)) for n in archive.namelist()}


def main():
    report = read(DATA / "report.json")
    assert sha(DATA / "report.json") == "aae7e90de93a7a13887712dea85167c2e5f9d8c5ef93cf22f3642b9f47e73a44"
    assert report["complete"] and report["passed"] and not report["failures"]
    assert report["compute_before"] == report["compute_after"] == ""
    assert report["runner_sha256"] == sha(DATA / "run.py") == read(OUT / "smoke-preparation-findings.json")["runner_sha256"]
    assert report["manifest_sha256"] == sha(DATA / "build-manifest.json") == read(OUT / "build-findings.json")["manifest_sha256"]
    assert report["host_report_sha256"] == sha(DATA / "host-report.json")
    assert report["host_full_source_sha256"] == sha(DATA / "host-source-sha256.json")
    assert report["core_sha256"] == read(DATA / "build-manifest.json")["library"]["sha256"]
    assert [c["name"] for c in report["cases"]] == ["mass", "legacy", "compact"]
    for case in report["cases"]:
        assert case["exit_code"] == 0 and case["wall_seconds"] < 90
        assert case["log_sha256"] == sha(DATA / (case["name"] + ".log"))
    assert sha(DATA / "mass.log") == sha(ROOT / "build/performance/mass-tiny-v622/tiny.log") == "a3b32d6b6df42b7473e1cef264df95397ff9d77e082d8bfdc229146f5195664e"
    raw_mass = [[line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1])] for line in (DATA / "mass.log").read_text().splitlines() if line.startswith("MASS_")]
    assert raw_mass == report["cases"][0]["records"]
    assert raw_mass[-1] == ["MASS_TEST_SUMMARY", {"passed": True, "solve_API_calls": 7, "iteration_caps": 9, "actual_updates": 5}]
    assert report["cases"][1]["records"] == [{"phase": "native", "kernel_launches": 2, "candidates": 262, "passed": True}]
    suites = list(ET.parse(DATA / "compact.xml").getroot().iter("testsuite"))
    junit = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    assert junit == report["cases"][2]["junit"] == {"tests": 7, "failures": 0, "errors": 0, "skipped": 0}
    prior_report = read(PRIOR / "report.json")
    assert set(report["readbacks"]) == set(prior_report["readbacks"])
    files = {}
    count = 0
    for name, identity in report["readbacks"].items():
        path = DATA / "readbacks" / name
        assert sha(path) == identity["sha256"] and path.stat().st_size == identity["bytes"]
        prior_path = PRIOR / "readbacks" / name
        assert sha(prior_path) == prior_report["readbacks"][name]["sha256"]
        current, prior = arrays(path), arrays(prior_path)
        assert set(current) == set(prior)
        cross = 0.0
        for key in current:
            if key in ("actual_3", "expected_3"):
                for field in ("candidates", "deploy_slots", "leg_slots"):
                    assert current[key][0][field] == prior[key][0][field]
            else:
                cross = max(cross, compare(current[key], prior[key], strict=key in ("actual_2", "expected_2")))
        result_error = max(compare(current["actual_0"], current["expected_0"]), compare(current["actual_1"], current["expected_1"]))
        compare(current["actual_2"], current["expected_2"], strict=True)
        metadata_error = compare(current["expanded"], current["full_3"], atol=1e-12)
        formula_error = 0.0
        for index, candidate in enumerate(current["full_1"]):
            d0, nd, l0, nl = (candidate[k] for k in ("deploy_begin", "deploy_count", "leg_begin", "leg_count"))
            case = {"partial_mass": candidate["partial_mass"], "deploys": current["full_2"][d0:d0 + nd], "legs": current["full_3"][l0:l0 + nl]}
            answer = helper.replay(case, current["full_0"][0])
            formula_error = max(formula_error, compare(current["actual_0"][index], answer["result"]), compare(current["actual_1"][l0:l0 + nl], answer["leg_results"]))
            compare(current["actual_2"][d0:d0 + nd], answer["collected_by_deploy"], strict=True)
            assert current["actual_0"][index]["collected"] == answer["result"]["collected"]
            for a, b in zip(current["actual_1"][l0:l0 + nl], answer["leg_results"], strict=True):
                assert a["gained"] == b["gained"]
        count += len(current["full_1"])
        files[name] = {**identity, "candidate_pairs": len(current["full_1"]), "maximum_combined_vs_standalone_difference_excluding_times": cross, "maximum_result_difference": result_error, "maximum_expanded_metadata_difference": metadata_error, "maximum_independent_formula_difference": formula_error}
    assert count == 786 and len(files) == 6
    findings = {
        "scope": "Read-only saved logs/NPZ/source-identity audit; no GPU or native calls by reviewer",
        "report_sha256": sha(DATA / "report.json"), "core_sha256": report["core_sha256"],
        "manifest_sha256": report["manifest_sha256"], "runner_sha256": report["runner_sha256"],
        "mass_log_bitwise_equal_to_isolated_tiny": True,
        "mass_solve_APIs": 7, "mass_actual_updates": 5,
        "mass_scope": "Exact same frozen fixture/native assertions and raw seven outcomes; full vectors are asserted in the test but not emitted in its log",
        "legacy_nonempty_evaluations": 2, "legacy_candidates": 262,
        "legacy_scope": "Unchanged native fixture performs internal numerical/gate assertions; raw log emits summary, not all 262 vectors",
        "compact_junit": junit, "compact_valid_calls_from_test_control_flow": 18,
        "compact_candidate_evaluations_from_test_control_flow": 2358, "compact_malformed_calls_from_test_control_flow": 6,
        "saved_original_compact_pairs_independently_replayed": count,
        "compact_scope": "Six NPZs cover 786 paired ordinary/compact candidates; production capture checks are test assertions rather than additional NPZ exports",
        "files": files, "decision": "Combined linkage smoke passes; no new cold-convergence, performance, fleet-score or default-promotion claim",
        "reviewer_GPU_calls": 0,
    }
    (OUT / "smoke-results-findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps({"passed": True, "saved_pairs": count, "cross_core_max": max(x["maximum_combined_vs_standalone_difference_excluding_times"] for x in files.values()), "formula_max": max(x["maximum_independent_formula_difference"] for x in files.values()), "findings_sha256": sha(OUT / "smoke-results-findings.json")}))


if __name__ == "__main__":
    main()
