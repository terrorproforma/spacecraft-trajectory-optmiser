"""Reproduce terminal component timing summaries using saved evidence only."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / "completion-pack-benchmark-v622"
OUTPUT = KIT / "output"
READY_SHA = "64d36a429381edeae90f3300520516f5310337f8d154ed292d5c89642143ba9d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def median(rows, field):
    return statistics.median(row[field] for row in rows)


def main():
    assert sha(KIT / "ready.json") == READY_SHA
    ready = read(KIT / "ready.json")
    for name, digest in ready["files"].items():
        assert sha(KIT / name) == digest, name
    plan = read(KIT / "plan.json")
    report = read(OUTPUT / "benchmark-report.json")
    launch = read(OUTPUT / "launch-report.json")
    assert report["complete"] and report["passed"] and not report["failures"]
    assert launch["complete"] and launch["passed"] and launch["exit_code"] == 0
    assert launch["ready_sha256"] == READY_SHA
    assert report["native_calls"] == report["attempted_method_calls"] == 80
    assert report["candidates"] == {"ordinary": 13520, "compact": 13520}
    assert report["fresh_Lambert_solves"] == report["fresh_SCvx_refinements"] == 0
    assert report["fresh_CPU_finish_calls"] == 0 and report["captures_enabled"] is False
    events = [json.loads(line) for line in (OUTPUT / "events.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "requested" for row in events) == 80
    assert sum(row["event"] == "completed" for row in events) == 80
    groups = []
    comparisons = 0
    largest_error = 0.0
    evaluated = {"ordinary": 0, "compact": 0}
    accepted = {"ordinary": 0, "compact": 0}
    for number, group in enumerate(report["groups"]):
        metadata = group["group"]
        assert metadata == plan["groups"][number]
        rows = group["measurements"]
        sequence = ["ordinary", "compact"] if number % 2 == 0 else ["compact", "ordinary"]
        sequence += plan["warm_order"]
        assert [row["backend"] for row in rows] == sequence
        summary = {
            "id": metadata["id"],
            "model": metadata["model"],
            "batch_size": metadata["size"],
            "unique_controls": metadata["unique_control_count"],
            "expected_accepted_each_call": metadata["expected_accepted"],
            "request_sha256": group["request_sha256"],
            "construction_seconds": group["construction_seconds"],
            "workspace_create_seconds": group["workspace_create_seconds"],
            "workspace_close_seconds": group["workspace_close_seconds"],
            "arms": {},
        }
        for row in rows:
            arm = row["backend"]
            label = f"{row['trial']:02}-{arm}"
            prefix = OUTPUT / metadata["id"]
            call = read(prefix / f"{label}-call.json")
            check = read(prefix / f"{label}-comparison.json")
            assert call["run_returned"] and call["raw_readback_saved"]
            assert call["native_evaluation_count"] == 1
            assert call["method_seconds"] == row["method_seconds"]
            assert (prefix / f"{label}.npz").is_file()
            assert check["passed"] and not check["failures"]
            assert row["validation"]["passed"]
            assert row["validation"]["accepted"] == metadata["expected_accepted"]
            assert row["telemetry_delta"]["completion_batches"] == 1
            assert row["telemetry_delta"]["completion_candidates"] == metadata["size"]
            if arm == "compact":
                assert row["native_model_rebuilds"] == 1
                assert row["ordinary_geometry_cache_pairs"] == 0
            comparisons += check["field_comparisons"]
            largest_error = max(
                largest_error, check["maximum_absolute_finite_difference"]["absolute"]
            )
            evaluated[arm] += metadata["size"]
            accepted[arm] += row["validation"]["accepted"]
        for arm in ("ordinary", "compact"):
            selected = [row for row in rows if row["backend"] == arm]
            first = [row for row in selected if row["first_call"]]
            warm = [row for row in selected if not row["first_call"]]
            assert len(first) == 1 and len(warm) == 4
            assert len({row["numeric_readback_sha256"] for row in selected}) == 1
            time_value = median(warm, "method_seconds")
            assert time_value == group[arm + "_warm_median_seconds"]
            timings = [row["telemetry_delta"] for row in warm]
            summary["arms"][arm] = {
                "first_call_seconds": first[0]["method_seconds"],
                "warm_sample_count": 4,
                "warm_method_samples_seconds": [row["method_seconds"] for row in warm],
                "warm_median_seconds": time_value,
                "warm_min_seconds": min(row["method_seconds"] for row in warm),
                "warm_max_seconds": max(row["method_seconds"] for row in warm),
                "warm_proxy_evaluations_per_second": metadata["size"] / time_value,
                "warm_pack_median_seconds": median(timings, "completion_pack_seconds"),
                "warm_paired_pack_share_median": statistics.median(
                    row["telemetry_delta"]["completion_pack_seconds"] / row["method_seconds"]
                    for row in warm
                ),
                "warm_native_call_median_seconds": median(
                    timings, "completion_native_call_seconds"
                ),
                "warm_kernel_median_seconds": median(timings, "completion_kernel_seconds"),
                "numeric_readback_sha256": selected[0]["numeric_readback_sha256"],
                "geometry_cache_pairs": selected[-1]["ordinary_geometry_cache_pairs"],
                "model_rebuilds": selected[-1].get("native_model_rebuilds", 0),
            }
        a, b = (summary["arms"][arm] for arm in ("ordinary", "compact"))
        ratio = a["warm_median_seconds"] / b["warm_median_seconds"]
        assert ratio == group["ordinary_over_compact_warm_ratio"]
        summary["ordinary_over_compact_warm_ratio"] = ratio
        summary["compact_method_time_reduction_fraction"] = 1.0 - 1.0 / ratio
        summary["first_call_ordinary_over_compact_ratio"] = (
            a["first_call_seconds"] / b["first_call_seconds"]
        )
        summary["cross_arm_numeric_payload_identical"] = (
            a["numeric_readback_sha256"] == b["numeric_readback_sha256"]
        )
        groups.append(summary)
    assert len(groups) == 8 and evaluated == report["candidates"]
    output_hashes = {
        path.relative_to(OUTPUT).as_posix(): sha(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file()
    }
    assert len(list(OUTPUT.rglob("*.npz"))) == 80
    write(HERE / "raw-output-sha256.json", output_hashes)
    result = {
        "scope": "Saved component timing arithmetic; no new model or GPU evaluation",
        "device": launch["gpu_inventory"],
        "ready_sha256": READY_SHA,
        "library_sha256": launch["core_sha256"],
        "report_sha256": sha(OUTPUT / "benchmark-report.json"),
        "launch_sha256": sha(OUTPUT / "launch-report.json"),
        "worker_seconds_including_setup_evidence_validation": report["total_worker_seconds"],
        "catalogue_load_seconds": report["catalogue_load_seconds"],
        "native_evaluate_calls": 80,
        "kernel_launches_from_frozen_source": 160,
        "candidate_evaluations_each_arm": evaluated,
        "accepted_proxy_evaluations_each_arm": accepted,
        "historical_unique_controls": 20,
        "reported_frozen_oracle_field_comparisons": comparisons,
        "maximum_reported_absolute_finite_difference": largest_error,
        "correctness_scope": (
            "All saved classifications/cargo checks passed and reports were cross-checked; "
            "this summary does not redo the independent numerical/formula audit"
        ),
        "new_mission_solves": 0,
        "score_changes": 0,
        "raw_output_files": len(output_hashes),
        "groups": groups,
    }
    assert math.isfinite(largest_error)
    write(HERE / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}))
    for row in groups:
        print(row["id"], row["ordinary_over_compact_warm_ratio"])


if __name__ == "__main__":
    main()
