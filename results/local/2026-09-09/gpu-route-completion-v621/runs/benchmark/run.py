"""Finite matched production CPU/GPU completion benchmark; supervisor only."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import time
import traceback

from common import (
    KIT,
    construct,
    np,
    read,
    sha,
    signature,
    validate_native,
    validate_plans,
    write,
)

from spacepdhcg.gtoc12.lambert import using_lambert_backend


def main():
    if not __debug__:
        raise RuntimeError("Assertions are required for evidence qualification")
    assert os.environ.get("SPACEPDHCG_COMPLETION_BENCHMARK_SUPERVISED") == str(os.getppid())
    output = KIT / "output"
    plan = read(KIT / "plan.json")
    report = {
        "complete": False,
        "groups": [],
        "CPU_candidate_evaluations": 0,
        "GPU_candidate_evaluations": 0,
        "GPU_evaluate_calls": 0,
        "new_geometry_calls": 0,
        "new_refinements": 0,
        "fleet_promotions": 0,
        "throughput_unit": plan["throughput_unit"],
        "failures": [],
    }
    began = time.perf_counter()
    gc_enabled = gc.isenabled()
    # Keep normal production GC behavior; its setting is recorded, not suppressed.
    report["gc_enabled"] = gc_enabled

    def event(value):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    try:
        for number, group in enumerate(plan["groups"]):
            assert time.perf_counter() - began < plan["maximum_worker_seconds"]
            folder = output / group["id"]
            folder.mkdir(exist_ok=False)
            search, requests, cases, construction_seconds = construct(group)
            before = signature(requests)
            summary = {
                "group": group,
                "construction_seconds": construction_seconds,
                "input_signature": before,
                "measurements": [],
            }
            outcome_hashes = {}
            report["groups"].append(summary)
            write(
                folder / "inputs.json",
                {"fixture_indices": group["fixture_indices"], "input_signature": before},
            )
            opened = time.perf_counter()
            with using_lambert_backend("cuda", maximum_batch_size=7) as gpu:
                summary["backend_setup_seconds"] = time.perf_counter() - opened
                workspace_identity = None
                sequence = (["cpu", "gpu"] if number % 2 == 0 else ["gpu", "cpu"]) + plan[
                    "warm_order"
                ]
                for trial, backend in enumerate(sequence):
                    if backend == "gpu":
                        assert report["GPU_evaluate_calls"] < plan["gpu_evaluate_calls"]
                    name = f"{trial:02}-{backend}"
                    event(
                        {
                            "event": "method_requested",
                            "group": group["id"],
                            "trial": trial,
                            "backend": backend,
                            "size": len(requests),
                            "GPU_evaluate_calls_before": report["GPU_evaluate_calls"],
                        }
                    )
                    old_telemetry = dict(gpu.telemetry)
                    started = time.perf_counter()
                    if backend == "cpu":
                        rows = []
                        for partial, deploy, collect, forward, use_table in requests:
                            route = search._finish_cpu(
                                partial, deploy, collect, forward, use_collect_table=use_table
                            )
                            rows.append((route, search.last_failure if route is None else ""))
                    else:
                        rows = search._finish_many(requests)
                    elapsed = time.perf_counter() - started
                    post_started = time.perf_counter()
                    measurement = {
                        "trial": trial,
                        "backend": backend,
                        "first_call": trial < 2,
                        "method_seconds": elapsed,
                        "proxy_evaluations_per_second": len(requests) / elapsed,
                    }
                    if backend == "gpu":
                        report["GPU_evaluate_calls"] += 1
                        report["GPU_candidate_evaluations"] += len(requests)
                        delta = {
                            key: gpu.telemetry.get(key, 0) - old_telemetry.get(key, 0)
                            for key in gpu.telemetry
                            if key.startswith("completion_")
                        }
                        assert delta["completion_batches"] == 1 and delta[
                            "completion_candidates"
                        ] == len(requests)
                        measurement["telemetry_delta"] = delta
                        identity = (
                            id(gpu.completion_workspace),
                            *(a.ctypes.data for a in gpu.completion_workspace.inputs),
                        )
                        if workspace_identity is None:
                            workspace_identity = identity
                            summary["workspace_capacities"] = list(
                                gpu.completion_workspace.capacities
                            )
                        else:
                            assert workspace_identity == identity
                        native_check_started = time.perf_counter()
                        arrays, parity = validate_native(gpu.completion_workspace, requests, cases)
                        measurement["native_readback_and_validation_seconds"] = (
                            time.perf_counter() - native_check_started
                        )
                        capture_started = time.perf_counter()
                        np.savez_compressed(folder / f"{name}-readback.npz", **arrays)
                        write(folder / f"{name}-parity.json", parity)
                        measurement["raw_capture_seconds"] = time.perf_counter() - capture_started
                        assert parity["passed"], parity["failures"]
                    else:
                        report["CPU_candidate_evaluations"] += len(requests)
                        assert gpu.telemetry == old_telemetry
                    validation_start = time.perf_counter()
                    measurement["plan_validation"] = validate_plans(rows, requests, cases)
                    measurement["plan_validation_seconds"] = time.perf_counter() - validation_start
                    outcomes = [
                        {"reason": reason}
                        if route is None
                        else {
                            "reason": reason,
                            "mass": route.final_mass_proxy_kg,
                            "fuel": route.propellant_proxy_kg,
                            "cargo_in_order": list(route.collected_mass.items()),
                            "inflation": [leg.inflation for leg in route.legs],
                        }
                        for route, reason in rows
                    ]
                    outcome_digest = hashlib.sha256(
                        json.dumps(outcomes, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest()
                    measurement["raw_outcome_sha256"] = outcome_digest
                    if backend in outcome_hashes:
                        measurement["raw_outcome_identical_to_first"] = (
                            outcome_hashes[backend] == outcome_digest
                        )
                    else:
                        outcome_hashes[backend] = outcome_digest
                        write(folder / f"first-{backend}-outcomes.json", outcomes)
                    assert measurement["plan_validation"]["accepted"] == group["expected_accepted"]
                    assert search.collect_table.lambert_evaluations == 0
                    assert not search.collect_table.return_sweeps
                    assert gpu.evaluations == 0 and gpu.batches == 0
                    assert gpu.telemetry["completed_branch_requests"] == 0
                    measurement["all_post_method_check_capture_seconds"] = (
                        time.perf_counter() - post_started
                    )
                    summary["measurements"].append(measurement)
                    event({"event": "method_completed", "group": group["id"], **measurement})
                assert signature(requests) == before
                summary["telemetry"] = dict(gpu.telemetry)
                closing = time.perf_counter()
            summary["backend_close_seconds"] = time.perf_counter() - closing
            for backend in ("cpu", "gpu"):
                samples = [
                    m["method_seconds"]
                    for m in summary["measurements"]
                    if m["backend"] == backend and not m["first_call"]
                ]
                assert len(samples) == 4
                median = statistics.median(samples)
                summary[backend + "_warm"] = {
                    "samples_seconds": samples,
                    "median_seconds": median,
                    "proxy_evaluations_per_second": group["size"] / median,
                }
            summary["warm_host_inclusive_speed_ratio_cpu_over_gpu"] = (
                summary["cpu_warm"]["median_seconds"] / summary["gpu_warm"]["median_seconds"]
            )
            write(folder / "summary.json", summary)
        assert report["GPU_evaluate_calls"] == plan["gpu_evaluate_calls"] == 60
        assert report["GPU_candidate_evaluations"] == plan["gpu_candidate_evaluations"] == 14200
        assert report["CPU_candidate_evaluations"] == plan["cpu_candidate_evaluations"] == 14200
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["failures"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        report.update(
            complete=True,
            total_worker_seconds=time.perf_counter() - began,
            run_sha256=sha(__file__),
            plan_sha256=sha(KIT / "plan.json"),
        )
        write(output / "benchmark-report.json", report)
        print(
            json.dumps(
                {
                    "passed": report.get("passed", False),
                    "GPU_evaluate_calls": report["GPU_evaluate_calls"],
                    "CPU_candidate_evaluations": report["CPU_candidate_evaluations"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
