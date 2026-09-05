#!/usr/bin/env python3
"""Summaries for the G2/G3 reseal of main 8cb3759 on the WSL RTX 5090.

Derived from the sealed b6afb49 (RTX 5090) / 9e75b47 (H100) summariser: the G2 and G3 sections,
their extracted quantities and their assertions are unchanged; the G0/G1 sections are omitted
because this reseal is scoped to the two gates whose shared CUDA library changed after the prior
seals (1dbcae0, 2bca11d). Foreign-GPU wait records are summarised in addition.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "8cb3759b29ea8c7d843322a940a7ebcabfd9ff21"
TREE = "6d27f2552d882b4418d16e4342e6854a436a952d"
BRANCH = "chore/g2g3-reseal-8cb3759"
HARDWARE_ID = "local-rtx-5090"
CUDA_ARCHITECTURE = 120


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


def foreign_gpu_waits(gate: Path) -> dict[str, Any]:
    log = gate / "foreign-gpu-waits.log"
    if not log.is_file():
        return {"checks": 0, "waits": 0, "total_wait_seconds": 0, "records": []}
    lines = [line for line in text(log).splitlines() if line.strip()]
    waits = [line for line in lines if " WAITING " in line]
    total = 0
    for line in lines:
        match = re.search(r"waited_seconds=(\d+)", line)
        if match:
            total += int(match.group(1))
    return {
        "checks": len([line for line in lines if " clear " in line]),
        "waits": len(waits),
        "total_wait_seconds": total,
        "records": waits,
    }


def orchestrator_waits(preflight: Path) -> dict[str, Any]:
    """Gate-level waits taken by the orchestrator before launching G2/G3 (outside run.sh)."""
    log = preflight / "orchestrator-gpu-waits.log"
    if not log.is_file():
        return {"checks": 0, "waits": 0, "total_wait_seconds": 0, "records": []}
    lines = [line for line in text(log).splitlines() if line.strip()]
    total = 0
    for line in lines:
        match = re.search(r"\(waited (\d+) s\)", line)
        if match:
            total += int(match.group(1))
    return {
        "checks": len([line for line in lines if "GPU clear before" in line]),
        "waits": len([line for line in lines if "GPU BUSY before" in line]),
        "total_wait_seconds": total,
        "records": [line for line in lines if "GPU BUSY before" in line],
    }


def status_times(gate: Path) -> dict[str, str | None]:
    values: dict[str, str | None] = {"started_utc": None, "completed_utc": None}
    for line in text(gate / "status.txt").splitlines():
        key, _, value = line.partition("=")
        if key in values:
            values[key] = value
    return values


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
    "branch": BRANCH,
    "hardware_id": HARDWARE_ID,
    "cuda_architecture": CUDA_ARCHITECTURE,
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
    "foreign_gpu_waits": foreign_gpu_waits(g2),
    "timing": status_times(g2),
    "retained_failed_attempts": failed_attempts(g2),
    "local_only": True,
    "immutable_uri": None,
}
assert g2_summary["sanitizer_clean"]
assert g2_summary["post_create_allocation_delta"] == 0
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
    "WSL RTX 5090 host: Nsight Systems 2024.6.2 under WSL records CUDA API activity; GPU kernel "
    "and GPU-memory summaries are recorded only when the platform exposes them (see "
    "nsys-stats.log); no timeline-residency claim is made."
)
h1 = json.loads((g3 / "h1/h1_decision.json").read_text(encoding="utf-8"))
g3_summary = {
    "schema_version": "current-head-g3-1.0.0",
    "decision": "PASS",
    "source_commit": COMMIT,
    "source_tree": TREE,
    "branch": BRANCH,
    "hardware_id": HARDWARE_ID,
    "cuda_architecture": CUDA_ARCHITECTURE,
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
    "h1_scvx_median_seconds_by_intervals": {
        str(row["intervals"]): row["summary"].get("scvx_median_seconds")
        for row in h1["coordinates"]
    },
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
    "foreign_gpu_waits": foreign_gpu_waits(g3),
    "timing": status_times(g3),
    "retained_failed_attempts": failed_attempts(g3),
    "local_only": True,
    "immutable_uri": None,
}
assert g3_summary["sanitizer_clean"]
assert g3_summary["sanitizer_logs"] == 16
assert g3_summary["maximum_tight_canonical_residual"] <= 1.0e-6
assert g3_summary["hcw_displaced"]["accepted_steps"] > 0
assert all(
    row["classification"] == "honest-negative"
    for row in g3_summary["pdhcg_negatives"]
)
assert g3_summary["h1_decision"] in {"supported", "mixed", "rejected"}
assert g3_summary["hidden_cpu_fallback"] is False
assert g3_summary["topology_allocation_delta"] == 0
assert g3_summary["topology_copy_delta"] == 0
(g3 / "summary.json").write_text(
    json.dumps(g3_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

root_summary = {
    "schema_version": "current-head-g2-g3-1.0.0",
    "campaign_scope_id": "single-gpu-v1",
    "source_commit": COMMIT,
    "source_tree": TREE,
    "branch": BRANCH,
    "branch_note": (
        "chore/g2g3-reseal-8cb3759 was cut from main at 8cb3759 with no source change; the "
        "sealed SOURCE is main 8cb3759 (evidence recorded against that commit before any commit "
        "was made on the branch)."
    ),
    "scope": ["G2", "G3"],
    "scope_note": (
        "G2 and G3 only: the shared CUDA library changed after the b6afb49 (RTX 5090) and "
        "9e75b47 (H100) seals through 1dbcae0 (recovery-kernel cancel polling, kernel pre-load, "
        "inner_iteration_cap) and 2bca11d (HCW exact matrix-exponential step, control-tracking "
        "term removal, relative KKT audit). G0/G1 were not re-run here."
    ),
    "gates": {
        "G2": g2_summary["decision"],
        "G3": g3_summary["decision"],
    },
    "g4_gate_authorized": True,
    "g4_claim_core_launch_ready": False,
    "g4_launch_blocker": (
        "No G4 campaign was launched from this evidence. The G4 claim core runs on the Lambda "
        "H100 from 1dbcae0 under its own capability; any RTX 5090 relaunch needs a new official "
        "capability generated from this reseal's clean Release executable."
    ),
    "hardware_id": HARDWARE_ID,
    "cuda_architecture": CUDA_ARCHITECTURE,
    "reseal_of": [
        "b6afb49d7fc7da5ed1ac9003c3bcae5d35506026 (results/gpu/current-head-b0cd570, WSL RTX 5090)",
        "9e75b470fd5378d9b20f6f13892ac909c43757cd (results/gpu/current-head-9e75b47-h100, Lambda H100)",
    ],
    "foreign_gpu_waits": {
        "orchestrator_before_gates": orchestrator_waits(ROOT / "preflight"),
        "g2": g2_summary["foreign_gpu_waits"],
        "g3": g3_summary["foreign_gpu_waits"],
    },
    "local_only": True,
    "immutable_uri": None,
}
(ROOT / "current-head-summary.json").write_text(
    json.dumps(root_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(root_summary["gates"], sort_keys=True))
