"""Recheck the saved nine-call test without loading or executing CUDA."""
from pathlib import Path
import hashlib
import json
import math


def main():
    root = Path("build/performance/l1-tiny-v615")
    build = Path("build/performance/l1-v615c")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads((root / "report.json").read_bytes())
    assert sha(root / "report.json") == "3f2b85397508ced11536d72800a093e07abdff154d415532583db6924da07483"
    assert report["complete"] and report["status"] == "passed"
    assert report["gpu_calls_started"] == len(report["cases"]) == 1
    entry = report["cases"][0]
    assert entry["returncode"] == 0 and sha(root / "tiny.log") == entry["log_sha256"]
    assert sha(root / "run_l1_tiny_v615.py") == report["runner_sha256"] == "4d98ade975f17d05e72a7bbf7f50df413e360cd63e12ebd72bc79b606220aad9"
    parsed = [{"prefix": line.split(" ", 1)[0], "record": json.loads(line.split(" ", 1)[1])}
              for line in (root / "tiny.log").read_text().splitlines()]
    assert parsed == entry["records"]
    manifest = json.loads((build / "manifest.json").read_bytes())
    assert report["manifest_sha256"] == sha(build / "manifest.json")
    assert report["core_sha256"] == manifest["library_sha256"]
    assert report["test_sha256"] == manifest["persistent_l1_test_sha256"]
    cases = {x["record"]["case"]: x["record"] for x in parsed if x["prefix"] == "L1_TEST"}
    assert len(cases) == 9 and sum(x["iterations"] for x in cases.values()) == 10
    for name, result in cases.items():
        if name.endswith("_prox"):
            assert result["termination"] == 2 and result["iterations"] == result["updates"] == result["completions"] == 2
            assert result["finite"] and result["common_valid"] and not result["common_passes"]
            b = 1 / (1 + math.sqrt(2))
            o = 1 / (1 + math.sqrt(17 if "zero" in name else 81))
            eta = .9 / math.sqrt(3)
            for key, expected in (("B", b), ("O", o), ("eta", eta), ("threshold", 4 * eta * o / b)):
                assert math.isclose(result[key], expected, rel_tol=3e-15), (name, key)
        elif name in ("disabled_original_scaling_refresh", "fresh_original_control"):
            assert result["termination"] == 2 and result["iterations"] == 1
            assert not result["l1_enabled"] and not result["l1_valid"]
        elif name == "original_weak_seed_zero_step":
            assert result["termination"] == 1 and result["iterations"] == result["updates"] == result["completions"] == 0
            assert result["common_passes"] and result["finite"] and result["gap"] == 4e-20
        elif name == "cancel_before_initial":
            assert result["termination"] == 3 and result["iterations"] == 0 and not result["common_valid"]
        elif name == "nonfinite_initial":
            assert result["termination"] == 4 and result["iterations"] == 0
            assert not result["finite"] and not result["common_passes"] and result["gap"] is None
        else:
            raise AssertionError(name)
    disabled = cases["disabled_original_scaling_refresh"].copy()
    control = cases["fresh_original_control"].copy()
    disabled.pop("case")
    control.pop("case")
    assert disabled == control
    assert not (root / "compute-processes-before.txt").read_text().strip()
    result = {"scope": "Independent saved-record, identity and analytic scaling check; zero additional GPU calls",
              "script_sha256": sha(Path(__file__)), "report_sha256": sha(root / "report.json"),
              "raw_log_sha256": sha(root / "tiny.log"), "manifest_sha256": report["manifest_sha256"],
              "core_sha256": report["core_sha256"], "test_sha256": report["test_sha256"],
              "executions": 1, "solve_api_calls": 9, "actual_updates": 10,
              "all_expected_outcomes": True, "analytic_B_O_eta_thresholds_match": True,
              "mode_off_and_fresh_control_records_match": True,
              "vector_evidence_scope": "Raw logs contain no vectors. The pinned test's exit0 certifies its in-process GPU-download versus independent CPU prox/SOC oracle checks and exact weak-seed preservation assertion. This script does not independently replay those unrecorded vectors.",
              "interpretation": "Expected iteration limits, cancellation and nonfinite rejection are successful test outcomes, not newly qualified optimizer solutions. No cold convergence or speedup claim."}
    output = Path(__file__).with_name("tiny-review.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output)}))


if __name__ == "__main__":
    main()
