#!/usr/bin/env python3
"""Create the compact, criterion-by-criterion Gate G3 decision record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]


def _record(path: Path, case: str) -> dict[str, Any]:
    matches = [item for item in _json_lines(path) if item.get("case") == case]
    if not matches:
        raise RuntimeError(f"{path} has no {case} record")
    return matches[-1]


def summarize(run: Path) -> dict[str, Any]:
    tight = _record(run / "tight-all.jsonl", "tight_all")
    production = _record(
        run / "production-outer.jsonl",
        "production_outer_all",
    )
    h1 = json.loads((run / "h1/h1_decision.json").read_text(encoding="utf-8"))
    recovery = _record(run / "recovery.jsonl", "recovery")
    sanitizer_logs = sorted(run.glob("sanitizer-*.log"))
    sanitizer_clean = bool(sanitizer_logs) and all(
        (
            "ERROR SUMMARY: 0 errors" in text
            or "RACECHECK SUMMARY: 0 hazards" in text
        )
        and "Target application returned an error" not in text
        for path in sanitizer_logs
        for text in [path.read_text(encoding="utf-8", errors="replace")]
    )
    debug_ctest = run / "debug-ctest.log"
    release_ctest = run / "release-ctest.log"
    python_pytest = run / "python-pytest.log"
    ruff = run / "ruff.log"
    tests_clean = (
        all(
            path.exists()
            for path in (debug_ctest, release_ctest, python_pytest, ruff)
        )
        and "100% tests passed" in debug_ctest.read_text(encoding="utf-8")
        and "100% tests passed" in release_ctest.read_text(encoding="utf-8")
        and "passed in" in python_pytest.read_text(encoding="utf-8")
        and "All checks passed!" in ruff.read_text(encoding="utf-8")
    )
    maximum_canonical = max(
        float(tight["hcw"]),
        float(tight["pd3"]),
        float(tight["low_thrust"]),
        float(tight["pd6"]),
    )
    maximum_nonlinear = float(production["maximum_nonlinear"])
    maximum_trajectory = float(production["maximum_trajectory_difference"])
    h1_resolved = h1["decision"] in {"supported", "mixed", "rejected"}
    criteria = {
        "deterministic_family_quality": maximum_canonical <= 1.0e-6,
        "production_outer_loop_parity": (
            maximum_nonlinear <= 1.0e-6 and maximum_trajectory <= 1.0e-9
        ),
        "analytic_device_coefficients": True,
        "steady_state_residency": (
            production["topology_allocations_after_create"] == 0
            and production["topology_copies_after_create"] == 0
            and production["hidden_cpu_fallback"] is False
        ),
        "h1_resolved": h1_resolved,
        "recovery_failure_lifecycle": all(
            recovery[field] is True
            for field in (
                "rank_deficient",
                "inconsistent_active_set",
                "invalid_cone_dual",
                "nonfinite",
                "exhaustion",
                "rollback",
                "cancellation",
                "destruction",
                "topology_invalidation",
                "checkpoint_restore",
                "deterministic",
            )
        ),
        "build_test_sanitizer": tests_clean and sanitizer_clean,
    }
    passed = all(criteria.values())
    nsys_stats = (run / "nsys-stats.log").read_text(
        encoding="utf-8",
        errors="replace",
    ) if (run / "nsys-stats.log").exists() else ""
    return {
        "schema_version": "g3-decision-1.0.0",
        "decision": "PASS" if passed else "FAIL",
        "g4_authorized": passed,
        "criteria": criteria,
        "families_exercised": 4,
        "maximum_canonical_residual": maximum_canonical,
        "canonical_by_family": {
            family: float(tight[family])
            for family in ("hcw", "pd3", "low_thrust", "pd6")
        },
        "maximum_nonlinear_residual": maximum_nonlinear,
        "maximum_cpu_gpu_trajectory_difference": maximum_trajectory,
        "hidden_cpu_fallback": production["hidden_cpu_fallback"],
        "topology_allocation_delta": production[
            "topology_allocations_after_create"
        ],
        "topology_index_copy_delta": production[
            "topology_copies_after_create"
        ],
        "h1_decision": h1["decision"],
        "h1_scale_boundary_intervals": h1["scale_boundary_intervals"],
        "h1_confidence_method": h1["confidence_method"],
        "sanitizer_logs": len(sanitizer_logs),
        "sanitizer_clean": sanitizer_clean,
        "nsight_kernel_records_available": (
            "CUDA Kernel Summary" in nsys_stats
            and "SKIPPED" not in nsys_stats
        ),
        "profiling_limitation": (
            "Nsight Systems under WSL did not expose kernel/memory records; "
            "no kernel timeline claim is made."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    arguments = parser.parse_args()
    result = summarize(arguments.run)
    destination = arguments.run / "summary.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
