"""Finite ordinary/compact production component benchmark; supervisor required."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import statistics
import threading
import time
import traceback

from common import (
    KIT,
    construct,
    np,
    raw_workspace,
    read,
    sha,
    signature,
    validate_native,
    validate_plans,
    write,
)

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_completion import GpuCompletion


class Owner:
    """Thread-owned real component library, without unused Lambert symbols."""

    def __init__(self, library):
        self.library = library
        self.device_id = 0
        self.telemetry = {}
        self.thread = threading.get_ident()

    def _owned(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError("component workspace used from a different thread")


def measured_call(owner, search, inputs, folder, label):
    """Time only run; retain available buffers before any caller validation."""
    outcome = {"run_returned": False, "native_evaluation_count": "unknown_until_return"}
    started = time.perf_counter()
    try:
        rows = owner.run(search, inputs)
    except BaseException as error:
        elapsed = time.perf_counter() - started
        outcome["run_error"] = repr(error)
        raise
    else:
        elapsed = time.perf_counter() - started
        outcome.update(run_returned=True, native_evaluation_count=1)
    finally:
        post = time.perf_counter()
        outcome.update(
            method_seconds=elapsed,
            buffer_status=(
                "available buffers only; not yet validated; may be stale if run did not return"
            ),
        )
        try:
            arrays, _ = raw_workspace(owner, inputs)
            np.savez_compressed(folder / f"{label}.npz", **arrays)
            outcome["raw_readback_saved"] = True
        except BaseException as error:
            outcome["raw_readback_saved"] = False
            outcome["capture_error"] = repr(error)
            raise
        finally:
            write(folder / f"{label}-call.json", outcome)
    return rows, elapsed, post


def main():
    if not __debug__:
        raise RuntimeError("Assertions are required")
    assert os.environ.get("SPACEPDHCG_PACK_BENCHMARK_SUPERVISED") == str(os.getppid())
    plan, profile = read(KIT / "plan.json"), read(KIT / "profile.json")
    output = KIT / "output"
    report = {
        "complete": False,
        "passed": False,
        "groups": [],
        "native_calls": 0,
        "attempted_method_calls": 0,
        "native_call_count_status": "exact for returned methods; unknown if a method raises",
        "candidates": {"ordinary": 0, "compact": 0},
        "failures": [],
        "fresh_Lambert_solves": 0,
        "fresh_SCvx_refinements": 0,
        "fresh_CPU_finish_calls": 0,
        "captures_enabled": False,
        "throughput_unit": plan["throughput_unit"],
    }
    began = time.perf_counter()

    def journal(value):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    try:
        loaded = time.perf_counter()
        catalogue = load_catalogue()
        assert catalogue.source_sha256 == plan["catalogue_sha256"]
        report["catalogue_load_seconds"] = time.perf_counter() - loaded
        library = ctypes.CDLL(profile["library"]["path"])
        for number, group in enumerate(plan["groups"]):
            assert time.perf_counter() - began < plan["worker_seconds"]
            folder = output / group["id"]
            folder.mkdir(exist_ok=False)
            summary = {
                "group": group,
                "construction_seconds": {},
                "workspace_create_seconds": {},
                "measurements": [],
            }
            report["groups"].append(summary)
            searches, requests, cases, owners, fingerprints = {}, {}, {}, {}, {}
            try:
                for backend in ("ordinary", "compact"):
                    search, inputs, reference, seconds = construct(group, catalogue)
                    assert not getattr(search, "completion_capture", None)
                    assert not search.collect_table._geometry
                    searches[backend], requests[backend], cases[backend] = search, inputs, reference
                    fingerprints[backend] = signature(inputs)
                    summary["construction_seconds"][backend] = seconds
                    started = time.perf_counter()
                    owners[backend] = GpuCompletion(
                        Owner(library),
                        len(inputs),
                        sum(len(x[1]) for x in inputs),
                        sum(len(x[3]) for x in inputs),
                    )
                    summary["workspace_create_seconds"][backend] = time.perf_counter() - started
                assert fingerprints["ordinary"] == fingerprints["compact"]
                summary["request_sha256"] = fingerprints["ordinary"]
                sequence = ["ordinary", "compact"] if number % 2 == 0 else ["compact", "ordinary"]
                sequence += plan["warm_order"]
                counts, first_hash = {"ordinary": 0, "compact": 0}, {}
                for trial, backend in enumerate(sequence):
                    assert report["native_calls"] < plan["native_evaluate_calls"]
                    assert time.perf_counter() - began < plan["worker_seconds"]
                    os.environ["SPACEPDHCG_TEST_GTOC12_COMPLETION_NATIVE_MODEL"] = (
                        "1" if backend == "compact" else "0"
                    )
                    owner, search, inputs = owners[backend], searches[backend], requests[backend]
                    before = dict(owner.gpu.telemetry)
                    journal(
                        {
                            "event": "requested",
                            "group": group["id"],
                            "trial": trial,
                            "backend": backend,
                            "size": len(inputs),
                        }
                    )
                    report["attempted_method_calls"] += 1
                    label = f"{trial:02}-{backend}"
                    rows, elapsed, post = measured_call(owner, search, inputs, folder, label)
                    # Raw readbacks are now saved before every post-run assertion.
                    report["native_calls"] += 1
                    report["candidates"][backend] += len(inputs)
                    delta = {
                        k: owner.gpu.telemetry[k] - before.get(k, 0)
                        for k in owner.gpu.telemetry
                        if k.startswith("completion_")
                    }
                    assert delta["completion_batches"] == 1
                    assert delta["completion_candidates"] == len(inputs)
                    arrays, comparison = validate_native(owner, inputs, cases[backend])
                    write(folder / f"{label}-comparison.json", comparison)
                    assert comparison["passed"], comparison
                    checks = validate_plans(rows, inputs, cases[backend])
                    assert checks["accepted"] == group["expected_accepted"]
                    numeric_hash = hashlib.sha256(
                        b"".join(
                            array.tobytes()
                            for key, array in sorted(arrays.items())
                            if key != "stats"
                        )
                    ).hexdigest()
                    if backend in first_hash:
                        assert first_hash[backend] == numeric_hash
                    else:
                        first_hash[backend] = numeric_hash
                    measurement = {
                        "trial": trial,
                        "backend": backend,
                        "first_call": counts[backend] == 0,
                        "method_seconds": elapsed,
                        "proxy_completions_per_second": len(inputs) / elapsed,
                        "telemetry_delta": delta,
                        "numeric_readback_sha256": numeric_hash,
                        "ordinary_geometry_cache_pairs": len(search.collect_table._geometry),
                        "validation": checks,
                    }
                    counts[backend] += 1
                    if backend == "compact":
                        assert owner.native_model.rebuilds == 1
                        assert not search.collect_table._geometry
                        measurement["native_model_signature"] = owner.native_model.signature
                        measurement["native_model_rebuilds"] = owner.native_model.rebuilds
                    assert search.collect_table.lambert_evaluations == 0
                    assert not search.collect_table.return_sweeps
                    measurement["post_method_validation_capture_seconds"] = (
                        time.perf_counter() - post
                    )
                    summary["measurements"].append(measurement)
                    journal({"event": "completed", "group": group["id"], **measurement})
                assert counts == {"ordinary": 5, "compact": 5}
                for backend in ("ordinary", "compact"):
                    assert signature(requests[backend]) == fingerprints[backend]
                    samples = [
                        x["method_seconds"]
                        for x in summary["measurements"]
                        if x["backend"] == backend and not x["first_call"]
                    ]
                    assert len(samples) == 4
                    summary[backend + "_warm_median_seconds"] = statistics.median(samples)
                summary["ordinary_over_compact_warm_ratio"] = (
                    summary["ordinary_warm_median_seconds"] / summary["compact_warm_median_seconds"]
                )
            finally:
                closed = time.perf_counter()
                for owner in owners.values():
                    owner.close()
                summary["workspace_close_seconds"] = time.perf_counter() - closed
                write(folder / "summary.json", summary)
        assert report["native_calls"] == 80
        assert report["candidates"] == {"ordinary": 13520, "compact": 13520}
        report["passed"] = True
    except BaseException as error:
        report["failures"].append({"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report.update(
            complete=True,
            total_worker_seconds=time.perf_counter() - began,
            worker_sha256=sha(KIT / "run.py"),
            plan_sha256=sha(KIT / "plan.json"),
        )
        write(output / "benchmark-report.json", report)
        print(
            json.dumps({"passed": report["passed"], "native_calls": report["native_calls"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
