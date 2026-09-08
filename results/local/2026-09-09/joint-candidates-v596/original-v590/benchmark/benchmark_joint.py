"""Time a complete incumbent mesh neighbourhood with scalar and batched CUDA.

This measures the joint search surrogate and its Python controller. It does not
run SCvx, the independent verifier, or certify a changed trajectory.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12.bundles import (
    ClusterPricingSettings,
    cluster_retime_settings,
    cluster_search_settings,
)
from spacepdhcg.gtoc12.data import (
    load_bonus_table,
    load_catalogue,
    pins_path,
    rules_path,
    verified_path,
)
from spacepdhcg.gtoc12.gpu_joint import evaluate_joint
from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.retiming import Retimer, visits_of

JOINT_FLAG = "SPACEPDHCG_TEST_GTOC12_JOINT_BATCH"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = ROOT / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
ORDER = ("A", "B", "B", "A")
FIELDS = ("objective", "weighted_kg", "collected_kg", "spare_kg", "propellant_kg")


def plain(value):
    """Strict JSON, including explicit strings for infeasible infinities."""
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [plain(v) for v in sorted(value)]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unrecordable configuration value: {type(value).__name__}")


def digest_json(value):
    raw = json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def cache_digest(cache):
    # Hexadecimal binary64 values make the retained cost/key identity explicit.
    return digest_json(
        [[a, b, float(t0).hex(), float(tf).hex(), float(v).hex()]
         for (a, b, t0, tf), v in sorted(cache.items())]
    )


def epoch_digest(arrivals, departures):
    return digest_json({"arrivals": arrivals, "departures": departures})


@contextmanager
def mode(label):
    previous = os.environ.get(JOINT_FLAG)
    os.environ[JOINT_FLAG] = "0" if label == "A" else "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(JOINT_FLAG, None)
        else:
            os.environ[JOINT_FLAG] = previous


def clone(template, cache):
    # No mutable learned state or Lambert dict is shared between A/B runs.
    # The frozen catalogue's arrays are read-only inputs and do not need copies.
    joint = deepcopy(template, {id(template.catalogue): template.catalogue})
    joint._lambert = dict(cache)
    joint.evaluations = joint.lambert_evaluations = 0
    joint.retimer.lambert_evaluations = 0
    return joint


def trial_rows(visits, arrivals, departures, delta):
    # Both timed paths call this generator themselves, including moves(), copies
    # and epoch updates. A consumes it lazily; B materializes the entire matrix.
    for shift in JointItinerary.moves(len(visits), delta):
        arr, dep = arrivals.copy(), departures.copy()
        for j, (da, dd) in shift.items():
            arr[j] += da
            dep[j] += dd
        yield arr, dep


def matrices(visits, arrivals, departures, delta):
    rows = list(trial_rows(visits, arrivals, departures, delta))
    return np.asarray([a for a, _ in rows]), np.asarray([d for _, d in rows])


def scalar_select(joint, visits, rows, minimum):
    winner, best, selected_epochs = None, None, None
    for index, (arr, dep) in enumerate(rows):
        value = joint.evaluate(visits, arr, dep)
        if not value.feasible or value.objective <= minimum + 1e-9:
            continue
        if best is None or value.objective > best.objective:
            winner, best, selected_epochs = index, value, (arr, dep)
    return winner, best, selected_epochs


def select_values(values, minimum):
    winner = None
    for index, value in enumerate(values):
        if value.feasible and value.objective > minimum + 1e-9:
            if winner is None or value.objective > values[winner].objective:
                winner = index
    return winner


def counters(gpu):
    result = {k: v for k, v in gpu.telemetry.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    result["geometry_branch_requests"] = gpu.evaluations
    result["geometry_batches"] = gpu.batches
    return result


def counter_delta(before, gpu):
    after = counters(gpu)
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(before.keys() | after.keys())}


class Checks:
    def __init__(self, atol, rtol):
        self.atol, self.rtol = atol, rtol
        self.errors = []
        self.count = 0
        self.max_absolute_error = {}

    def equal(self, label, actual, expected):
        self.count += 1
        if actual != expected and len(self.errors) < 64:
            self.errors.append(f"{label}: {actual!r} != {expected!r}")

    def numbers(self, label, actual, expected):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.count += 1
        if a.shape != e.shape:
            self.errors.append(f"{label}: shapes {a.shape} != {e.shape}")
            return
        finite = np.isfinite(a) & np.isfinite(e)
        error = float(np.max(np.abs(a[finite] - e[finite]), initial=0.0))
        self.max_absolute_error[label] = max(self.max_absolute_error.get(label, 0.0), error)
        if not np.allclose(a, e, atol=self.atol, rtol=self.rtol, equal_nan=False):
            if len(self.errors) < 64:
                self.errors.append(f"{label}: values differ, maximum finite absolute error {error}")

    def evaluation(self, actual, expected, label):
        if actual is None or expected is None:
            self.equal(label + ".exists", actual is not None, expected is not None)
            return
        for field in ("feasible", "failure", "measured_legs"):
            self.equal(label + "." + field, getattr(actual, field), getattr(expected, field))
        for field in FIELDS:
            self.numbers(field, getattr(actual, field), getattr(expected, field))
        self.numbers("masses", actual.masses, expected.masses)
        if actual.plan is None or expected.plan is None:
            return
        a, e = actual.plan, expected.plan
        for field in ("deploy_epochs", "collect_epochs", "foreign_deploy_epochs"):
            self.equal(label + ".plan." + field, getattr(a, field), getattr(e, field))
        self.equal(label + ".collected_bodies", list(a.collected_mass), list(e.collected_mass))
        self.numbers("collected_mass", list(a.collected_mass.values()), list(e.collected_mass.values()))
        self.numbers("plan_mass", [a.propellant_proxy_kg, a.final_mass_proxy_kg],
                     [e.propellant_proxy_kg, e.final_mass_proxy_kg])
        self.equal(label + ".leg_count", len(a.legs), len(e.legs))
        for j, (al, el) in enumerate(zip(a.legs, e.legs)):
            identity = lambda leg: (leg.from_id, leg.to_id, leg.role,
                                    leg.departure_epoch, leg.arrival_epoch)
            self.equal(f"{label}.leg[{j}]", identity(al), identity(el))
            self.numbers("leg_proxy_inflation", [al.delta_v_proxy_km_s, al.inflation],
                         [el.delta_v_proxy_km_s, el.inflation])

    def record(self):
        return {"passed": not self.errors, "checks": self.count, "errors": self.errors,
                "atol": self.atol, "rtol": self.rtol,
                "max_absolute_error": self.max_absolute_error}


def prepare_fixture(path, catalogue, bonus, protect_earth_leg):
    route = route_from_summary(json.loads(path.read_text(encoding="utf-8")))
    weights = {body: bonus.for_asteroid(body) for body in route.collected_mass}
    pricing = ClusterPricingSettings(collect_dp_inflation_fit="")
    retimer = Retimer(catalogue, cluster_search_settings(pricing, 40),
                      cluster_retime_settings(pricing, last=True), weights)
    joint = JointItinerary(catalogue, retimer, weights=weights)
    learned = joint.learn(route)
    if not route.certified or learned != len(route.legs) or not joint.measured:
        raise ValueError(f"fixture does not supply every certified measured leg: {path}")
    if protect_earth_leg:
        retimer.protect_earth_leg(route.plan)
    visits, arrivals, departures = visits_of(route.plan)
    visits = [dataclasses.replace(v, foreign_deploy_epoch=route.plan.foreign_deploy_epochs.get(v.body))
              if v.collect and v.body not in route.plan.deploy_epochs else v for v in visits]
    arrivals, departures = (np.asarray(x, dtype=np.float64) for x in (arrivals, departures))
    arrivals.setflags(write=False)
    departures.setflags(write=False)
    config = {"pricing": pricing, "search": retimer.search_settings, "retime": retimer.settings,
              "joint": joint.settings, "bonus_weights": "loaded from verified pinned bonus table; values omitted",
              "visits": visits,
              "measured": [[list(k), v] for k, v in sorted(joint.measured.items())],
              "pair_inflations": [[list(k), v] for k, v in sorted(retimer.inflations.items())],
              "pair_bans": [[list(k), v] for k, v in sorted(retimer.bans.items())],
              "protect_earth_leg": protect_earth_leg,
              "earth_out_tof_floor": retimer.earth_out_tof_floor,
              "earth_out_inflation": joint.earth_out_inflation,
              "free_earth_leg": joint.free_earth_leg, "screen_earth_out": joint._screen_earth_out}
    return joint, visits, arrivals, departures, {
        "archive": file_record(path), "certified_archive_input": route.certified,
        "archive_collected_kg": route.total_collected_kg, "learned_measured_legs": learned,
        "config": plain(config), "config_sha256": digest_json(config),
        "base_arrivals": arrivals.tolist(), "base_departures": departures.tolist(),
    }


def warm_cache(gpu, joint, visits, arrivals, departures):
    """One common, complete cost dictionary; it is never mutated by a run."""
    cache, pending = {}, {}
    for arr, dep in zip(arrivals, departures, strict=True):
        for j, (visit, nxt) in enumerate(pairwise(visits)):
            t0, tf = float(dep[j]), float(arr[j + 1])
            key = joint.key(visit.body, nxt.body, t0, tf)
            if tf <= t0:
                cache[key] = math.inf
            else:
                pending.setdefault((visit.body, nxt.body), {}).setdefault(key, (t0, tf - t0))
    for (source, target), queries in pending.items():
        times = np.asarray(list(queries.values()), dtype=np.float64)
        hops = gpu.paired_hops(joint.catalogue, source, target, times[:, 0], times[:, 1])
        for key, value, feasible in zip(queries, hops.total_delta_v, hops.feasible, strict=True):
            cache[key] = float(value) if feasible and math.isfinite(value) else math.inf
    return cache


def audit(gpu, template, visits, arrivals, departures, cache, minimum, checks):
    values, joints, records = {}, {}, {}
    for label in ("A", "B"):
        joint = joints[label] = clone(template, cache)
        before = counters(gpu)
        with mode(label):
            if label == "A":
                result = [joint.evaluate(visits, a, d)
                          for a, d in zip(arrivals, departures, strict=True)]
            else:
                result = evaluate_joint(joint, visits, arrivals, departures)
        if result is None:
            raise RuntimeError("batch audit silently fell back from CUDA")
        values[label] = result
        checks.equal(label + ".audit_count", len(result), len(arrivals))
        checks.equal(label + ".evaluation_count", joint.evaluations, len(arrivals))
        records[label] = {
            "failure_counts": dict(Counter(v.failure or "feasible" for v in result)),
            "failures_by_row": [v.failure for v in result],
            "objectives_by_row": [v.objective for v in result],
            "measured_legs_by_row": [v.measured_legs for v in result],
            "evaluations": joint.evaluations, "lambert_branch_requests": joint.lambert_evaluations,
            "telemetry_delta": counter_delta(before, gpu),
            "winner_index": select_values(result, minimum),
        }
    for index, (actual, expected) in enumerate(zip(values["B"], values["A"], strict=True)):
        checks.evaluation(actual, expected, f"audit.row[{index}]")
    checks.equal("audit.winner_index", records["B"]["winner_index"], records["A"]["winner_index"])
    common = sorted(joints["A"]._lambert.keys() & joints["B"]._lambert.keys())
    checks.numbers("common_lambert_costs", [joints["B"]._lambert[k] for k in common],
                   [joints["A"]._lambert[k] for k in common])
    records["common_lambert_keys"] = len(common)
    return values, records


def execute(label, joint, visits, arr, dep, delta, minimum):
    if label == "A":
        return scalar_select(joint, visits, trial_rows(visits, arr, dep, delta), minimum)
    arrivals, departures = matrices(visits, arr, dep, delta)
    result = evaluate_joint(joint, visits, arrivals, departures, minimum_objective=minimum)
    if result is None:
        raise RuntimeError("winner-only evaluator silently fell back from CUDA")
    index, value = result
    return index, value, None if index is None else (arrivals[index], departures[index])


def run_case(gpu, template, visits, arr, dep, delta, condition, cache, args):
    arrivals, departures = matrices(visits, arr, dep, delta)
    arrivals.setflags(write=False)
    departures.setflags(write=False)
    rows_hash = epoch_digest(arrivals, departures)
    initial_cache_hash = cache_digest(cache)
    checks = Checks(args.cold_atol if condition == "cold" else args.warm_atol,
                    args.cold_rtol if condition == "cold" else args.warm_rtol)
    with mode("A"):
        baseline = clone(template, cache).evaluate(visits, arr, dep)
    if not baseline.feasible:
        raise ValueError(f"incumbent baseline failed: {baseline.failure}")
    values, audit_record = audit(gpu, template, visits, arrivals, departures, cache,
                                 baseline.objective, checks)
    case = {"mesh_days": delta, "cache_condition": condition,
            "candidate_count": len(arrivals), "epoch_rows_sha256": rows_hash,
            "initial_cache_entries": len(cache), "initial_cache_sha256": initial_cache_hash,
            "baseline": plain(baseline), "untimed_full_candidate_audit": plain(audit_record),
            "runs": [], "correctness": checks.record()}
    if condition == "warm":
        for label in ("A", "B"):
            checks.equal(label + ".warm_audit_geometry", audit_record[label]["lambert_branch_requests"], 0)
    if checks.errors:
        case["correctness"] = checks.record()
        return case
    expected_index = audit_record["A"]["winner_index"]
    expected = None if expected_index is None else values["A"][expected_index]
    # Full audit warms the library, scan grids, kernels and correctly-sized joint
    # workspace. Winner-only Python/materialization also gets explicit warmups.
    for _ in range(args.warmups):
        for label in ("A", "B"):
            with mode(label):
                execute(label, clone(template, cache), visits, arr, dep, delta, baseline.objective)
    for block in range(args.repeats):
        for position, label in enumerate(ORDER):
            joint = clone(template, cache)
            before = counters(gpu)
            with mode(label):
                started = time.perf_counter_ns()
                index, value, epochs = execute(label, joint, visits, arr, dep, delta, baseline.objective)
                seconds = (time.perf_counter_ns() - started) * 1e-9
            telemetry = counter_delta(before, gpu)
            checks.equal("timed.winner_index", index, expected_index)
            checks.evaluation(value, expected, f"timed.{block}.{position}.{label}")
            checks.equal("timed.evaluation_count", joint.evaluations, len(arrivals))
            checks.equal("timed.joint_native_evaluations",
                         telemetry.get("completed_joint_evaluations", 0),
                         len(arrivals) if label == "B" else 0)
            if epochs is not None and index is not None:
                checks.equal("timed.winner_arrivals", epochs[0].tolist(), arrivals[index].tolist())
                checks.equal("timed.winner_departures", epochs[1].tolist(), departures[index].tolist())
            if condition == "warm":
                checks.equal("timed.warm_geometry", joint.lambert_evaluations, 0)
                checks.equal("timed.warm_gpu_branches", telemetry["geometry_branch_requests"], 0)
            case["runs"].append({
                "block": block, "position": position, "path": label, "seconds": seconds,
                "winner_index": index, "winner": plain(value),
                "selected_objective": baseline.objective if value is None else value.objective,
                "evaluations": joint.evaluations, "lambert_branch_requests": joint.lambert_evaluations,
                "telemetry_delta": telemetry, "initial_cache_sha256": initial_cache_hash,
                "final_cache_entries": len(joint._lambert),
            })
    checks.equal("shared_cache_unchanged", cache_digest(cache), initial_cache_hash)
    checks.equal("canonical_epochs_unchanged", epoch_digest(arrivals, departures), rows_hash)
    case["correctness"] = checks.record()
    summaries = {}
    for label in ("A", "B"):
        times = [r["seconds"] for r in case["runs"] if r["path"] == label]
        summaries[label] = {"seconds": times, "median_seconds": statistics.median(times),
                            "minimum_seconds": min(times), "maximum_seconds": max(times)}
    case["timings"] = summaries
    if not checks.errors:
        case["speedup_A_over_B"] = summaries["A"]["median_seconds"] / summaries["B"]["median_seconds"]
        case["block_speedups_A_over_B"] = [
            statistics.median(r["seconds"] for r in case["runs"] if r["block"] == block and r["path"] == "A")
            / statistics.median(r["seconds"] for r in case["runs"] if r["block"] == block and r["path"] == "B")
            for block in range(args.repeats)
        ]
    return case


def provenance(args, catalogue):
    package = importlib.import_module("spacepdhcg")
    package_root = Path(package.__file__).resolve().parent
    try:
        package_version = importlib.metadata.version("spacepdhcg")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source import; no installed distribution metadata"
    modules = {str(p.relative_to(package_root)): file_record(p)
               for p in sorted(package_root.rglob("*.py"))}
    native = {}
    for name in (
        "cpp/CMakeLists.txt", "cpp/cuda/src/gtoc12_joint.cu",
        "cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h",
        "cpp/cuda/src/orbitweaver_gpu.cu", "cpp/include/spacepdhcg/orbitweaver/lambert.hpp",
    ):
        path = args.source_root / name
        if path.is_file():
            native[name] = file_record(path)
    return {
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "package_root": str(package_root), "package_version": package_version,
        "loaded_package_python_files": modules, "loaded_python_manifest_sha256": digest_json(modules),
        "native_source_files": native, "native_source_manifest_sha256": digest_json(native),
        "benchmark_script": file_record(__file__), "core": file_record(args.core),
        "pins": file_record(pins_path()), "rules": file_record(rules_path()),
        "catalogue": file_record(verified_path("GTOC12_Asteroids_Data.txt")),
        "catalogue_source_sha256": catalogue.source_sha256,
        "bonus_input": "bonus_coefficients.txt verified against the recorded pins manifest; values and content hash omitted",
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith("SPACEPDHCG_TEST_") or k in {
                            "SPACEPDHCG_GTOC12_CUDA_LIBRARY", "SPACEPDHCG_GTOC12_DATA",
                            "SPACEPDHCG_GTOC12_JOINT_ARCHIVE", "SPACEPDHCG_ASSET_ROOT",
                            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "PYTHONHASHSEED", "PYTHONPATH"}},
        "gc_enabled": gc.isenabled(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new JSON report path")
    parser.add_argument("--archive-dir", type=Path,
                        default=Path(os.environ.get("SPACEPDHCG_GTOC12_JOINT_ARCHIVE", DEFAULT_ARCHIVE)))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--core", type=Path, default=os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY"))
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--ships", nargs="+", default=["ship-01.json", "ship-02.json"])
    parser.add_argument("--mesh-days", type=float, nargs="+", default=[3.0])
    parser.add_argument("--repeats", type=int, default=3, help="number of complete A/B/B/A blocks")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-batch-size", type=int, default=16384)
    parser.add_argument("--no-protect-earth-leg", action="store_true")
    parser.add_argument("--cold-atol", type=float, default=5e-5)
    parser.add_argument("--cold-rtol", type=float, default=2e-8)
    parser.add_argument("--warm-atol", type=float, default=5e-9)
    parser.add_argument("--warm-rtol", type=float, default=5e-12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    if args.core is None:
        parser.error("set SPACEPDHCG_GTOC12_CUDA_LIBRARY or pass --core")
    if args.repeats < 1 or args.warmups < 1 or args.maximum_batch_size < 1:
        parser.error("repeats, warmups and maximum-batch-size must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in args.mesh_days):
        parser.error("mesh-days must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in
           (args.cold_atol, args.cold_rtol, args.warm_atol, args.warm_rtol)):
        parser.error("tolerances must be finite and nonnegative")
    args.core = args.core.expanduser().resolve(strict=True)
    args.source_root = args.source_root.expanduser().resolve(strict=True)
    args.archive_dir = args.archive_dir.expanduser().resolve(strict=True)
    os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"] = str(args.core)
    if args.data_dir is not None:
        os.environ["SPACEPDHCG_GTOC12_DATA"] = str(args.data_dir.expanduser().resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "joint-controller-benchmark-v593", "complete": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": plain(vars(args)),
        "claim": "search-surrogate performance and parity only; no trajectory certification",
        "paths": {"A": "scalar CUDA Lambert, sequential JointItinerary.evaluate and strict winner selection",
                  "B": "batched CUDA elements and joint arithmetic, winner-only evaluate_joint"},
        "execution_order_per_block": list(ORDER),
        "timing_scope": {
            "included": ["JointItinerary.moves generation", "epoch copies and updates",
                         "A lazy iteration or B matrix construction", "Lambert keys and cache lookups",
                         "uncached geometry", "C API packing and transfers", "joint arithmetic",
                         "plan materialization", "ordered winner selection"],
            "excluded": ["data loading and checksum verification", "baseline evaluation",
                         "fixture and common warm-cache construction", "state cloning",
                         "CUDA context/workspace/kernel warmup", "full candidate parity audit",
                         "telemetry snapshots and JSON reporting"],
            "clock": "time.perf_counter_ns; blocking host C APIs complete before returning",
            "cold_means": "empty per-run joint Lambert cache; CUDA context and workspaces are warm",
            "warm_means": "identical complete paired-GPU Lambert costs cloned into every run",
        }, "ships": [],
    }
    status = 1
    try:
        catalogue, bonus = load_catalogue(), load_bonus_table()
        report["provenance"] = provenance(args, catalogue)
        with using_lambert_backend("cuda", maximum_batch_size=args.maximum_batch_size) as gpu:
            report["cuda_device_id"] = gpu.device_id
            report["loaded_core"] = file_record(gpu.library._name)
            for ship in args.ships:
                template, visits, arr, dep, fixture = prepare_fixture(
                    args.archive_dir / ship, catalogue, bonus, not args.no_protect_earth_leg)
                ship_record = {"ship": ship, "fixture": fixture, "cases": []}
                report["ships"].append(ship_record)
                for delta in args.mesh_days:
                    rows_a, rows_d = matrices(visits, arr, dep, delta)
                    cache = warm_cache(gpu, template, visits,
                                       np.vstack((arr, rows_a)), np.vstack((dep, rows_d)))
                    for condition, initial in (("cold", {}), ("warm", cache)):
                        case = run_case(gpu, template, visits, arr, dep, delta, condition, initial, args)
                        ship_record["cases"].append(case)
                        args.output.write_text(json.dumps(plain(report), indent=2, allow_nan=False) + "\n")
                        print(json.dumps({"ship": ship, "mesh_days": delta, "cache": condition,
                                          "passed": case["correctness"]["passed"],
                                          "speedup": case.get("speedup_A_over_B")}), file=sys.stderr)
        report["complete"] = True
        report["correctness_passed"] = all(
            case["correctness"]["passed"] for ship in report["ships"] for case in ship["cases"])
        status = 0 if report["correctness_passed"] else 2
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        print(f"benchmark failed: {error}", file=sys.stderr)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(plain(report), indent=2, allow_nan=False) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
