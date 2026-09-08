"""One ABI-v1 GPU evaluation of the 42 already-frozen completion fixtures.

Run only as an owned child of gpu_runner.py. No CPU model is evaluated here.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

KIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"float64"}:
            return float(value["float64"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"float64": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def write(path, value):
    with path.open("x") as stream:
        json.dump(encode(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


I, D = ct.c_int32, ct.c_double


class Policy(ct.Structure):
    _fields_ = [(name, I) for name in ("abi_version", "sum_mode", "reserved0", "reserved1")] + [
        (name, D) for name in ("initial_mass", "dry_mass", "miner_mass", "thrust", "exhaust", "mining_rate", "year_days", "minimum_stay")]


class Candidate(ct.Structure):
    _fields_ = [(name, I) for name in ("deploy_begin", "deploy_count", "leg_begin", "leg_count")] + [("partial_mass", D)]


class Deploy(ct.Structure):
    _fields_ = [("deploy_epoch", D), ("collect_epoch", D), ("has_collect", I), ("reserved", I)]


class Leg(ct.Structure):
    _fields_ = [(name, I) for name in ("role", "model", "source_deploy", "reserved")] + [
        (name, D) for name in ("departure", "arrival", "dv", "flat", "floor", "slope")] + [
        ("fit", D * 5)] + [(name, D) for name in ("delta_a_au", "delta_longitude_rad", "authority_ratio")]


class Result(ct.Structure):
    _fields_ = [(name, I) for name in ("failure", "failed_leg", "failed_deploy", "processed_legs")] + [
        (name, D) for name in ("propellant", "final_mass", "collected", "margin")]


class LegResult(ct.Structure):
    _fields_ = [("stage", I), ("pickup", I)] + [(name, D) for name in (
        "mass_before", "gained", "departure_mass", "mass_after", "authority", "inflation", "propellant", "tof")]


class Stats(ct.Structure):
    _fields_ = [(name, D) for name in ("upload_ms", "kernel_ms", "download_ms")] + [
        (name, ct.c_uint64) for name in ("candidates", "deploy_slots", "leg_slots")]


def structure(cls, values):
    assert set(values) == {name for name, _ in cls._fields_}
    out = cls()
    for name, value in values.items():
        if name == "fit":
            out.fit[:] = value
        else:
            setattr(out, name, value)
    return out


def values(row):
    return {name: getattr(row, name) for name, _ in row._fields_}


def compare(cases, raw, policy):
    """Classification exact; tolerances never relax a rejection or boundary."""
    failures, count = [], 0
    worst = {"absolute": 0.0, "path": None}
    atol, rtol = policy["absolute_tolerance"], policy["relative_tolerance"]

    def check(path, observed, expected, exact=False):
        nonlocal count
        count += 1
        if isinstance(expected, int) or exact:
            ok = observed == expected
        elif math.isnan(expected):
            ok = math.isnan(observed)
        elif math.isinf(expected):
            ok = observed == expected
        else:
            difference = abs(observed - expected)
            if difference > worst["absolute"]:
                worst.update(absolute=difference, path=path)
            ok = math.isfinite(observed) and math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol)
        if not ok:
            failures.append({"path": path, "observed": observed, "expected": expected, "exact": exact})

    li = di = 0
    for i, case in enumerate(cases):
        expected = case["expected"]
        observed = raw["results"][i]
        strict_boundary = case["id"] in policy["exact_result_cases"]
        for field, value in expected["result"].items():
            check(f"{case['id']}.result.{field}", observed[field], value,
                  exact=field == "collected" or strict_boundary)
        kind = "RoutePlan" if observed["failure"] == 0 else "ValueError" if observed["failure"] == 6 else "None"
        if kind != expected["source_return_kind"]:
            failures.append({"path": f"{case['id']}.return_kind", "observed": kind,
                             "expected": expected["source_return_kind"]})
        insertion, seen = [], set()
        for j, detail in enumerate(expected["leg_results"]):
            got = raw["leg_results"][li+j]
            for field, value in detail.items():
                check(f"{case['id']}.leg{j}.{field}", got[field], value, exact=field == "gained")
            if got["pickup"] and got["stage"] != 5:
                source = case["legs"][j]["source_deploy"]
                if source not in seen:
                    insertion.append(source)
                    seen.add(source)
        if insertion != expected["pickup_insertion_order"]:
            failures.append({"path": f"{case['id']}.pickup_insertion_order", "observed": insertion,
                             "expected": expected["pickup_insertion_order"]})
        for j, value in enumerate(expected["collected_by_deploy"]):
            check(f"{case['id']}.collected_by_deploy{j}", raw["collected_by_deploy"][di+j], value, exact=True)
        li += len(case["legs"])
        di += len(case["deploys"])
    return {"passed": not failures, "field_comparisons": count, "failures": failures,
            "maximum_absolute_finite_difference": worst, "comparison_policy": policy}


def main():
    output = Path(sys.argv[1]).resolve()
    assert output == KIT / "gpu-output"
    assert os.environ.get("SPACEPDHCG_COMPLETION_SUPERVISED") == str(os.getppid())
    pins = json.loads((KIT / "gpu-profile.json").read_text())
    assert sha(KIT / "fixtures.json") == pins["fixtures_sha256"]
    assert sha(pins["library"]["path"]) == pins["library"]["sha256"]
    document = decode(json.loads((KIT / "fixtures.json").read_text()))
    assert document["historical_batch"]["policy"] == document["synthetic_batch"]["policy"]
    cases = document["historical"] + document["synthetic"]
    assert len(cases) == 42
    candidates, deployments, flights = [], [], []
    for case in cases:
        candidates.append(structure(Candidate, {"deploy_begin": len(deployments), "deploy_count": len(case["deploys"]),
                          "leg_begin": len(flights), "leg_count": len(case["legs"]), "partial_mass": case["partial_mass"]}))
        deployments.extend(structure(Deploy, row) for row in case["deploys"])
        flights.extend(structure(Leg, row) for row in case["legs"])
    assert (len(candidates), len(deployments), len(flights)) == (42, 196, 260)
    sizes = {cls.__name__: ct.sizeof(cls) for cls in (Policy, Candidate, Deploy, Leg, Result, LegResult, Stats)}
    assert sizes == {"Policy": 80, "Candidate": 24, "Deploy": 24, "Leg": 128,
                     "Result": 48, "LegResult": 72, "Stats": 48}
    p = structure(Policy, document["historical_batch"]["policy"])
    candidate_array = (Candidate * len(candidates))(*candidates)
    deploy_array = (Deploy * len(deployments))(*deployments)
    leg_array = (Leg * len(flights))(*flights)
    result_array, detail_array = (Result * len(candidates))(), (LegResult * len(flights))()
    collected_array, stats = (D * len(deployments))(), Stats()
    for array in (result_array, detail_array, collected_array):
        ct.memset(ct.addressof(array), 0xA5, ct.sizeof(array))
    library = ct.CDLL(pins["library"]["path"])
    create = library.spacepdhcg_gtoc12_completion_create
    create.argtypes, create.restype = [I, I, I, I, ct.POINTER(ct.c_void_p)], ct.c_int
    evaluate = library.spacepdhcg_gtoc12_completion_evaluate_host
    evaluate.argtypes = [ct.c_void_p, I, I, I, ct.POINTER(Policy), ct.POINTER(Candidate),
                         ct.POINTER(Deploy), ct.POINTER(Leg), ct.POINTER(Result),
                         ct.POINTER(LegResult), ct.POINTER(D), ct.POINTER(Stats)]
    evaluate.restype = ct.c_int
    destroy = library.spacepdhcg_gtoc12_completion_destroy
    destroy.argtypes, destroy.restype = [ct.POINTER(ct.c_void_p)], ct.c_int
    workspace = ct.c_void_p()
    journal = output / "batch-events.jsonl"

    def event(kind, **details):
        with journal.open("a") as stream:
            stream.write(json.dumps({"event": kind, "time": time.time(), **details}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    event("create_requested", capacities=[42, 196, 260], pid=os.getpid())
    status = create(0, 42, 196, 260, ct.byref(workspace))
    event("create_returned", status=status)
    assert status == 0 and workspace.value
    try:
        event("evaluate_requested", api_call=1, candidate_count=42, deploy_slots=196, leg_slots=260)
        start = time.perf_counter()
        status = evaluate(workspace, 42, 196, 260, ct.byref(p), candidate_array, deploy_array,
                          leg_array, result_array, detail_array, collected_array, ct.byref(stats))
        elapsed = time.perf_counter() - start
        event("evaluate_returned", status=status, wall_seconds=elapsed)
        raw = {"api_status": status, "outputs_valid": status == 0, "evaluate_calls": 1,
               "candidate_ids": [c["id"] for c in cases], "structure_sizes": sizes,
               "fixtures_sha256": pins["fixtures_sha256"], "library_sha256": pins["library"]["sha256"],
               "wall_seconds_including_API": elapsed, "stats": values(stats),
               "results": [values(row) for row in result_array],
               "leg_results": [values(row) for row in detail_array],
               "collected_by_deploy": list(collected_array)}
        write(output / "batch-readback.json", raw)
        assert status == 0, "API error: readback retained but is not valid evidence"
        assert (stats.candidates, stats.deploy_slots, stats.leg_slots) == (42, 196, 260)
        report = compare(cases, raw, pins["comparison"])
        report.update(api_evaluate_calls=1, kernel_launches=1, candidates=42, deploy_slots=196,
                      leg_slots=260, fresh_physics_certifications=0, fleet_promotions=0)
        write(output / "batch-comparison.json", report)
        assert report["passed"], "Completion parity failed; all mismatches retained"
        print(json.dumps({"passed": True, "evaluate_calls": 1, "kernel_launches": 1,
                          "candidates": 42, "field_comparisons": report["field_comparisons"]}), flush=True)
    finally:
        event("destroy_requested")
        status = destroy(ct.byref(workspace))
        event("destroy_returned", status=status, cleared=workspace.value is None)
        assert status == 0 and workspace.value is None


if __name__ == "__main__":
    main()
