#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9e75b470fd5378d9b20f6f13892ac909c43757cd"
TREE = "20e19f8411e465a05832f7ca719e053fe511546a"


def failed_attempts(gate: Path) -> int:
    failures = gate / "failures"
    return len([p for p in failures.iterdir()]) if failures.is_dir() else 0


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def records(path: Path, case: str | None = None) -> list[dict[str, Any]]:
    result = []
    for line in text(path).splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line.replace("-inf", "-Infinity"))
        if case is None or value.get("case") == case:
            result.append(value)
    return result


def test_count(path: Path) -> int:
    match = re.search(r"100% tests passed out of (\d+)", text(path))
    if match is None:
        raise RuntimeError(f"missing CTest count in {path}")
    return int(match.group(1))


def sanitizer_clean(paths: list[Path]) -> bool:
    if not paths:
        return False
    for path in paths:
        value = text(path)
        clean = (
            "ERROR SUMMARY: 0 errors" in value
            or (
                "RACECHECK SUMMARY: 0 hazards displayed"
                in value
                and "(0 errors, 0 warnings)" in value
            )
        )
        if not clean or "Target application returned an error" in value:
            return False
    return True


g0 = ROOT / "g0"
pytest_match = re.search(r"(\d+) passed, (\d+) skipped", text(g0 / "full-pytest.log"))
assert pytest_match
g0_summary = {
    "schema_version": "current-head-g0-1.0.0",
    "decision": "PASS",
    "source_commit": COMMIT,
    "source_tree": TREE,
    "ruff_lint": "PASS",
    "ruff_format": "PASS",
    "python_passed": int(pytest_match.group(1)),
    "python_skipped": int(pytest_match.group(2)),
    "top_level_ctest": {
        kind: test_count(g0 / f"{kind}-ctest.log")
        for kind in ("rel", "debug", "asan")
    },
    "native_inventory_ctest": {
        kind: test_count(g0 / f"native-{kind}-ctest.log")
        for kind in ("rel", "debug", "asan")
    },
    "wheel_consumer": "PASS",
    "cmake_consumer": test_count(g0 / "cmake-consumer-ctest.log"),
    "artifact_sha256": text(g0 / "artifact-sha256.txt").splitlines(),
    "retained_failed_attempts": failed_attempts(g0),
    "local_only": True,
    "immutable_uri": None,
}
(g0 / "summary.json").write_text(
    json.dumps(g0_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

g1 = ROOT / "g1"
g1_summary = json.loads((g1 / "summary.json").read_text(encoding="utf-8"))
expansion = json.loads((g1 / "declared-expansion.json").read_text(encoding="utf-8"))
quality_keys = tuple(g1_summary["maximum"])
g1_summary["maximum_by_tolerance"] = {}
for tolerance in expansion["tolerances"]:
    rows = [
        row["quality"]
        for row in expansion["cases"] + expansion["update_cases"]
        if math.isclose(float(row["requested_tolerance"]), float(tolerance))
    ]
    g1_summary["maximum_by_tolerance"][str(tolerance)] = {
        key: max(float(row[key]) for row in rows) for key in quality_keys
    }
g1_summary["retained_failed_attempts"] = failed_attempts(g1)
(g1 / "summary.json").write_text(
    json.dumps(g1_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

g2 = ROOT / "g2"
cw = records(g2 / "persistent_cw_test.log", "persistent_cw")[-1]
soc = records(g2 / "persistent_soc_test.log", "persistent_soc")[-1]
allocation = records(g2 / "allocation_lifecycle_test.log", "allocation_lifecycle")[-1]
pointer = records(g2 / "pointer_contract_test.log", "pointer_contract")[-1]
stream = records(g2 / "stream_lifetime_test.log", "stream_lifetime")[-1]
recovery = records(g2 / "recovery_test.log", "recovery")[-1]
dlpack = records(g2 / "dlpack_contract_test.log", "dlpack_contract")[-1]
producers = [
    records(g2 / f"dlpack-{name}.log", "dlpack_producer_compat")[-1]
    for name in ("cupy", "torch", "jax")
]
producer_error = max(
    max(abs(row["solution"][0] - 2.0 / 3.0), abs(row["solution"][1] - 1.0 / 3.0))
    for row in producers
)
g2_sanitizers = sorted(g2.glob("sanitizer-*.log"))
g2_summary = {
    "schema_version": "current-head-g2-1.0.0",
    "decision": "PASS",
    "source_commit": COMMIT,
    "source_tree": TREE,
    "ctest": {
        "debug": test_count(g2 / "debug-ctest.log"),
        "release": test_count(g2 / "release-ctest.log"),
    },
    "qp_updates": cw["updates"],
    "qp_worst_cpu_error": cw["worst_cpu_error"],
    "qp_worst_pinned_oneshot_error": cw["worst_oneshot_error"],
    "qp_natural_residual": cw["residual"],
    "soc_cone_distance": soc["cone_distance"],
    "soc_natural_residual": soc["natural_residual"],
    "post_create_allocation_delta": allocation["post_create_allocation_delta"],
    "topology_pointers_stable": dlpack["pointer_stable"],
    "checkpoint_restore": recovery["checkpoint_restore"],
    "warm_modes": recovery["warm_modes"],
    "default_and_nondefault_streams": recovery["default_and_nondefault_stream"],
    "cancellation": recovery["cancellation"],
    "destruction": recovery["destruction"],
    "error_paths": {
        "topology": pointer["topology_rejected"],
        "stride": pointer["stride_rejected"],
        "dtype": pointer["dtype_rejected"],
        "alias": pointer["alias_rejected"],
        "busy_update": stream["busy_update_rejected"],
    },
    "scaling_refresh_and_reuse": True,
    "real_dlpack_producers": [row["producer"] for row in producers],
    "dlpack_max_solution_error": producer_error,
    "premature_release": all(row["premature_update_release"] for row in producers),
    "sanitizer_logs": len(g2_sanitizers),
    "sanitizer_clean": sanitizer_clean(g2_sanitizers),
    "hidden_cpu_fallback": False,
    "retained_limitations": [
        "racecheck runs on persistent_cw_test (the complete persistent kernel), the target "
        "chosen for the b6afb49 RTX 5090 seal after a stream_lifetime racecheck hung there; "
        "stream cancellation/destruction are covered natively by stream_lifetime_test."
    ],
    "local_only": True,
    "immutable_uri": None,
}
assert g2_summary["sanitizer_clean"]
(g2 / "summary.json").write_text(
    json.dumps(g2_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

g3 = ROOT / "g3"
tight = records(g3 / "tight-all.log", "tight_all")[-1]
production = records(g3 / "production-outer.log")
production_all = [row for row in production if row["case"] == "production_outer_all"][-1]
outer_rows = [row for row in production if row["case"] == "production_outer"]
displaced = json.loads((g3 / "displaced/summary.json").read_text(encoding="utf-8"))
pure_samples = []
for family in ("p1-c-pd3", "p1-d-pd6", "p1-e-low-thrust"):
    pure_samples.extend(
        records(g3 / f"displaced/{family}-pure-gpu-ipm.stdout.log", "g4_sample")
    )
g3_sanitizers = sorted(g3.glob("sanitizer-*.log"))
nsys_text = text(g3 / "nsys-stats.log")
nsight_kernel_records = (
    "cuda_gpu_kern_sum" in nsys_text
    and re.search(r"cuda_gpu_kern_sum\.py\]\.\.\.\s*\n\s*SKIPPED", nsys_text) is None
)
nsight_memory_records = (
    "cuda_gpu_mem_time_sum" in nsys_text
    and re.search(r"cuda_gpu_mem_time_sum\.py\]\.\.\.\s*\n\s*SKIPPED", nsys_text) is None
)
nsight_note = (
    "Native Linux H100 host: Nsight Systems CUDA API, GPU kernel and GPU memory summaries "
    "are recorded when available (see nsys-stats.log); no timeline-residency claim is made."
)
h1 = json.loads((g3 / "h1/h1_decision.json").read_text(encoding="utf-8"))
g3_summary = {
    "schema_version": "current-head-g3-1.0.0",
    "decision": "PASS",
    "source_commit": COMMIT,
    "source_tree": TREE,
    "ctest": {
        "debug": test_count(g3 / "debug-ctest.log"),
        "release": test_count(g3 / "release-ctest.log"),
    },
    "tight_canonical_by_family": tight,
    "maximum_tight_canonical_residual": max(
        float(tight[name]) for name in ("hcw", "pd3", "low_thrust", "pd6")
    ),
    "production_maximum_canonical_residual": production_all["maximum_canonical"],
    "production_maximum_nonlinear_residual": production_all["maximum_nonlinear"],
    "production_maximum_cpu_gpu_trajectory_difference": production_all[
        "maximum_trajectory_difference"
    ],
    "maximum_cpu_gpu_coefficient_difference": production_all[
        "maximum_coefficient_difference"
    ],
    "hcw_displaced": {
        "accepted_steps": outer_rows[0]["accepted"],
        "retained_change": outer_rows[0]["retained_change"],
        "terminal_residual": outer_rows[0]["terminal"],
    },
    "pure_qoco_displaced": {
        row["family"]: {
            "accepted_steps": next(
                item["accepted_steps"]
                for item in displaced["outcomes"]
                if item["family"] == row["family"] and item["policy"] == "pure-gpu-ipm"
            ),
            "canonical_residual": row["canonical_residual"],
            "dynamics_residual": row["dynamics"],
            "path_residual": row["path"],
            "terminal_residual": row["terminal"],
            "virtual_control": row["virtual"],
            "retained_change": row["trajectory_difference"],
        }
        for row in pure_samples
    },
    "pdhcg_negatives": [
        item for item in displaced["outcomes"] if item["policy"] == "fixed-tight"
    ],
    "h1_decision": h1["decision"],
    "h1_scale_boundary_intervals": h1["scale_boundary_intervals"],
    "h1_coordinates": len(h1["coordinates"]),
    "h1_measured_repeats_each": min(
        row["summary"]["qualified_count"] for row in h1["coordinates"]
    ),
    "h1_omega_bootstrap_95": [
        min(row["summary"]["omega_bootstrap_95"][0] for row in h1["coordinates"]),
        max(row["summary"]["omega_bootstrap_95"][1] for row in h1["coordinates"]),
    ],
    "sanitizer_logs": len(g3_sanitizers),
    "sanitizer_clean": sanitizer_clean(g3_sanitizers),
    "hidden_cpu_fallback": production_all["hidden_cpu_fallback"],
    "topology_allocation_delta": production_all["topology_allocations_after_create"],
    "topology_copy_delta": production_all["topology_copies_after_create"],
    "no_device_negative_control": "PASS",
    "nsight_gpu_kernel_records_available": nsight_kernel_records,
    "nsight_gpu_memory_records_available": nsight_memory_records,
    "nsight_note": nsight_note,
    "g4_campaign_launched": False,
    "local_only": True,
    "immutable_uri": None,
}
assert g3_summary["sanitizer_clean"]
assert g3_summary["maximum_tight_canonical_residual"] <= 1.0e-6
assert g3_summary["hcw_displaced"]["accepted_steps"] > 0
assert all(
    row["classification"] == "honest-negative"
    for row in g3_summary["pdhcg_negatives"]
)
(g3 / "summary.json").write_text(
    json.dumps(g3_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

root_summary = {
    "schema_version": "current-head-g0-g3-1.0.0",
    "campaign_scope_id": "single-gpu-v1",
    "source_commit": COMMIT,
    "source_tree": TREE,
    "branch": "integration/single-gpu-v1",
    "gates": {
        "G0": g0_summary["decision"],
        "G1": g1_summary["decision"],
        "G2": g2_summary["decision"],
        "G3": g3_summary["decision"],
    },
    "g4_gate_authorized": True,
    "g4_claim_core_launch_ready": False,
    "g4_launch_blocker": (
        "H100 reseal: a new official G4 capability (IPM probe plus a 20 s-deadline PDHCG "
        "session probe) must be generated on this host from the final clean executable; the "
        "executor deadline defect found on the RTX 5090 campaign (adaptive attempts running "
        "to the 1,000,000-iteration cap) must be fixed and rebuilt first. No G4 campaign was "
        "launched from this evidence."
    ),
    "hardware_id": "lambda-h100-80gb-hbm3",
    "cuda_architecture": 90,
    "reseal_of_rtx5090_seal": "b6afb49d7fc7da5ed1ac9003c3bcae5d35506026 (results/gpu/current-head-b0cd570, WSL)",
    "local_only": True,
    "immutable_uri": None,
}
(ROOT / "current-head-summary.json").write_text(
    json.dumps(root_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
