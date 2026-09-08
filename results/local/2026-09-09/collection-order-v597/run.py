"""Bounded new-incumbent collection-order experiment; explicit invocation only.

The default preparation imports no solver. A full run requires caller-selected,
validated native libraries and the shared GPU lock. Only a fully verified fleet
with greater weighted score and nondecreasing physical haul can replace the best.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

INCUMBENT_SHA256 = "33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8"
SHIP = 15


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cases(summaries):
    """Change only the seven interior pickups of the promoted nine-collect route."""
    summary = summaries[SHIP]
    plan = summary["plan"]
    deploy = {int(a): float(t) for a, t in plan["deploy_epochs"].items()}
    collect = {int(a): float(t) for a, t in plan["collect_epochs"].items()}
    if not summary.get("certified") or plan.get("foreign_deploy_epochs") or plan.get("orphaned"):
        raise ValueError("expected the certified, self-contained v595 ship 15")
    if len(deploy) != 9 or set(deploy) != set(collect):
        raise ValueError("expected nine deployed and nine collected own miners")
    for other, value in summaries.items():
        if other != SHIP:
            occupied = {int(a) for a in value["plan"]["deploy_epochs"]}
            occupied.update(int(a) for a in value["plan"]["collect_epochs"])
            if set(deploy) & occupied:
                raise ValueError(f"ship {other} shares the selected asteroid footprint")
    deploy_order = sorted(deploy, key=deploy.get)
    collect_order = sorted(collect, key=collect.get)
    if collect_order != [19102, 6757, 51831, 49856, 3299, 5162, 23153, 35473, 13077]:
        raise ValueError("input is not the promoted v595 collection order")
    if deploy_order[-1] != collect_order[0]:
        raise ValueError("the existing deployment/collection camp must be retained")
    interior = collect_order[1:-1]
    variants = {}

    def add(order, move):
        sequence = (collect_order[0], *order, collect_order[-1])
        if sequence != tuple(collect_order):
            variants.setdefault(sequence, []).append(move)

    # Local swaps first; the later wider moves are still recorded if the deadline
    # ends the scan early, so an unfinished scan is never called full coverage.
    for separation in range(1, len(interior)):
        for i in range(len(interior) - separation):
            j = i + separation
            order = list(interior)
            order[i], order[j] = order[j], order[i]
            add(order, {"kind": "swap", "interior_positions": [i, j]})
    for length in range(2, len(interior) + 1):
        for i in range(len(interior) - length + 1):
            order = list(interior)
            order[i:i + length] = reversed(order[i:i + length])
            add(order, {"kind": "reverse", "interior_start": i, "length": length})
    for i in range(len(interior)):
        for j in range(len(interior)):
            if i != j:
                order = list(interior)
                order.insert(j, order.pop(i))
                add(order, {"kind": "relocate", "interior_from": i, "interior_to": j})
    cases = []
    for index, (order, moves) in enumerate(variants.items()):
        assert len(order) == len(set(order)) == 9
        assert set(order) == set(deploy_order)
        cases.append({"case": f"ship_15_collect_order_{index:02d}", "ship": SHIP,
                      "deploy_order": deploy_order, "collect_order": list(order),
                      "moves": moves, "same_asteroid_footprint": True,
                      "camp_asteroid": 19102, "last_collector": 13077})
    return cases


@contextmanager
def native_progress(out, report):
    from spacepdhcg.gtoc12 import gpu_scvx

    original = gpu_scvx.solve_native
    count = 0

    def measured(*args, **kwargs):
        nonlocal count
        count += 1
        began = time.perf_counter()
        boundary = args[0]
        row = {"solve": count, "departure_epoch": boundary.departure_epoch,
               "arrival_epoch": boundary.arrival_epoch, "initial_mass_kg": boundary.initial_mass}
        try:
            result = original(*args, **kwargs)
            row.update(status=result.status, iterations=result.iterations,
                       accepted_iterations=result.accepted_iterations, nodes=len(result.node_epochs_mjd))
            return result
        except Exception as error:
            row["error"] = repr(error)
            raise
        finally:
            row["seconds"] = time.perf_counter() - began
            report["native_solves_completed"] = count
            with (out / "native-solves.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    gpu_scvx.solve_native = measured
    try:
        yield
    finally:
        gpu_scvx.solve_native = original


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--proxy-seconds", type=float, default=120)
    p.add_argument("--retime-fraction", type=float, default=0.7)
    p.add_argument("--joint-seconds", type=float, default=6)
    p.add_argument("--joint-candidates", type=int, default=6)
    p.add_argument("--max-certifications", type=int, default=4)
    p.add_argument("--wall-seconds", type=float, default=1800)
    p.add_argument("--minimum-proxy-weighted-gain", type=float, default=0.5)
    p.add_argument("--minimum-proxy-raw-gain", type=float, default=0.5)
    p.add_argument("--gpu-execution", choices=("graph", "dispatch"), default="graph")
    p.add_argument("--lock", type=Path, default=Path.home() / ".spacepdhcg-gpu.lock")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--proxy-only", action="store_true")
    args = p.parse_args()
    args.repo, args.inputs = args.repo.resolve(), args.inputs.resolve()
    if not 0 < args.proxy_seconds <= 120 or not 0 < args.wall_seconds <= 3600:
        p.error("proxy budget must be in (0,120], total soft wall budget in (0,3600]")
    if not 0 < args.retime_fraction <= 1 or not 0 <= args.joint_seconds <= 30:
        p.error("retime fraction must be in (0,1], per-seed joint budget in [0,30]")
    if not 0 <= args.joint_candidates <= 12 or not 1 <= args.max_certifications <= 4:
        p.error("joint candidates must be 0..12 and whole-route certifications 1..4")
    if args.minimum_proxy_weighted_gain < 0 or args.minimum_proxy_raw_gain < 0:
        p.error("proxy gain thresholds cannot be negative")
    return args


def run(args):
    paths = {ship: args.inputs / f"ship-{ship:02d}.json" for ship in range(1, 24)}
    old_path = args.inputs / "Result.txt"
    if sha256(old_path) != INCUMBENT_SHA256:
        raise ValueError("baseline must be the exact promoted v595 Result.txt")
    manifest = json.loads((args.inputs / "sha256.json").read_text())
    for name, expected in manifest.items():
        if sha256(args.inputs / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    summaries = {ship: json.loads(path.read_text()) for ship, path in paths.items()}
    cases = make_cases(summaries)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args._output_created = True
    write_json(out / "orders.json", cases)
    report = {"pid": os.getpid(), "complete": False, "status": "prepared", "stage": "prepared",
              "orders": len(cases), "configuration": {k: str(v) if isinstance(v, Path) else v
                                                        for k, v in vars(args).items()},
              "driver_sha256": sha256(Path(__file__)), "input_sha256": manifest,
              "rows": [], "joint_rows": [], "refinements": [], "promotions": [], "best": None,
              "native_solves_completed": 0, "physics_tolerances_changed": False}
    save = lambda: write_json(out / "report.json", report)
    save()
    if args.plan_only:
        report.update(complete=True, status="plan_only_no_gpu_execution")
        save()
        print(json.dumps({"status": report["status"], "orders": len(cases), "output": str(out)}))
        return 0
    for name in ("SPACEPDHCG_GTOC12_CUDA_LIBRARY", "SPACEPDHCG_QOCO_LIBRARY"):
        library = Path(os.environ.get(name, ""))
        if not library.is_file():
            raise ValueError(f"{name} must name a validated native library")
        report.setdefault("native_sha256", {})[name] = sha256(library)
    if args.joint_seconds > 0 and args.joint_candidates > 0 and os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_BATCH") != "1":
        raise ValueError("native joint polishing requires SPACEPDHCG_TEST_GTOC12_JOINT_BATCH=1")
    report["feature_flags"] = {k: v for k, v in os.environ.items() if k.startswith("SPACEPDHCG_TEST_")}
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    sys.path.insert(0, str(args.repo / "src"))
    import fcntl
    import numpy as np
    from spacepdhcg.gtoc12 import constants as C
    from spacepdhcg.gtoc12.bundles import (ClusterPricingSettings, cluster_retime_settings,
                                         cluster_search_settings, profile_for_orders)
    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.jointopt import JointItinerary, JointSettings, route_from_summary, weighted_mass
    from spacepdhcg.gtoc12.lambert import using_lambert_backend
    from spacepdhcg.gtoc12.low_thrust import ScvxSettings
    from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
    from spacepdhcg.gtoc12.pipeline import emit_solution, refine_route, write_route_artifacts
    from spacepdhcg.gtoc12.retiming import Retimer, calibrate_from_route, orders_of, visits_of, weighted_collected
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
    from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset

    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    if not official_verifier_available():
        raise RuntimeError("both final checkers are required")
    report["source_sha256"] = {str(p.relative_to(args.repo)): sha256(p)
                               for p in sorted((args.repo / "src/spacepdhcg/gtoc12").glob("*.py"))}
    report.update(catalogue_sha256=catalogue.source_sha256, bonus_sha256=bonus.source_sha256,
                  tolerances={"position_km": C.TOLERANCE_POSITION_KM,
                              "velocity_km_s": C.TOLERANCE_VELOCITY_KM_S,
                              "mass_kg": C.TOLERANCE_MASS_KG})
    scvx = ScvxSettings(max_iterations=40, node_days=2.0, discretisation_backend="cuda",
                        assembly_backend="cuda", convex_solver_backend="qoco",
                        qoco_ruiz_iterations=0, outer_loop_backend="cuda", seed_backend="cuda")
    report["scvx_settings"] = dataclasses.asdict(scvx)
    execution = argparse.Namespace(gpu_execution=args.gpu_execution, outer_loop_backend="cuda", workers=1)
    current = Solution.read(old_path)
    if current.ship_count != 23 or [s.ship_id for s in current.ships] != list(range(1, 24)):
        raise ValueError("baseline must contain ships 1..23 in order")
    selected = next(s for s in current.ships if s.ship_id == SHIP)
    actual_visits = Counter((e.event_id, e.before.epoch) for e in selected.asteroid_visits())
    expected_visits = Counter((int(a), float(t)) for phase in ("deploy_epochs", "collect_epochs")
                              for a, t in summaries[SHIP]["plan"][phase].items())
    if actual_visits != expected_visits:
        raise ValueError("selected route summary disagrees with the actual baseline event inventory")
    archived = route_from_summary(summaries[SHIP])
    before_weighted = weighted_mass(archived.collected_mass, weights)
    started = time.perf_counter()
    deadline = started + args.wall_seconds

    def verify(path):
        began = time.perf_counter()
        histories = {}
        independent = Gtoc12Verifier(catalogue, bonus=bonus, history=histories).verify_file(path)
        official = run_official_verifier(path)
        summary = independent.summary()
        score = summary.get("weighted_score_fixed_bonus_kg")
        ok = bool(independent.ok and official.ok and score is not None and math.isfinite(score))
        return {"ok": ok, "independent": summary, "official": official.summary(),
                "score_kg": score if ok else None, "total_mass_kg": independent.total_mass_kg,
                "verification_seconds": time.perf_counter() - began}, histories

    def check_footprint(plan, case):
        deploy_order, collect_order = orders_of(plan)
        if (deploy_order != case["deploy_order"] or collect_order != case["collect_order"]
                or plan.foreign_deploy_epochs or plan.orphaned
                or set(plan.collected_mass) != set(archived.plan.collect_epochs)):
            raise ValueError("candidate changed the authorized order or exact asteroid footprint")

    proxy_candidates = []
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["stage"] = "baseline_fleet_verification"
        save()
        baseline, _ = verify(old_path)
        if not baseline["ok"]:
            raise RuntimeError("baseline failed fresh verification; no candidate work started")
        report["baseline"] = baseline
        report["best"] = baseline | {"solution": str(old_path), "source": "retained_v595_incumbent"}
        save()
        with using_gpu_execution(execution), using_lambert_backend("cuda") as gpu, native_progress(out, report):
            proxy_start = time.perf_counter()
            proxy_deadline = min(deadline, proxy_start + args.proxy_seconds)
            retime_deadline = min(proxy_deadline, proxy_start + args.proxy_seconds * args.retime_fraction)
            policy = ClusterPricingSettings()
            retimer = Retimer(catalogue, cluster_search_settings(policy, len(catalogue.ids)),
                              dataclasses.replace(cluster_retime_settings(policy, last=True), step_days=15),
                              weights=weights)
            retimer.protect_earth_leg(archived.plan)
            report["calibrated_legs"] = calibrate_from_route(retimer, archived)
            report["stage"] = "gpu_new_order_scan"
            save()
            for case in cases:
                if time.perf_counter() >= retime_deadline:
                    break
                began = time.perf_counter()
                profile = profile_for_orders(archived.plan, retimer, case["deploy_order"], case["collect_order"], None)
                result = retimer.retime_order(case["deploy_order"], case["collect_order"], profile,
                                              before=before_weighted, original=archived.plan)
                row = {"case": case["case"], "result": result.summary(), "seconds": time.perf_counter() - began}
                if result.plan is not None and result.plan.feasible:
                    candidate = result.plan
                    check_footprint(candidate, case)
                    gain = weighted_collected(candidate, weights) - before_weighted
                    raw_gain = candidate.total_collected_kg - archived.total_collected_kg
                    location = out / "proxies" / f"{case['case']}_retimed.json"
                    location.parent.mkdir(exist_ok=True)
                    write_json(location, candidate.summary())
                    row.update(raw_gain_kg=raw_gain, weighted_gain_kg=gain, plan=str(location))
                    proxy_candidates.append((gain, raw_gain, candidate, case, "retimed"))
                report["rows"].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k != "result"}), flush=True)
            report["all_orders_retimed"] = len(report["rows"]) == len(cases)
            seeds = sorted(proxy_candidates, key=lambda x: (-x[0], -x[1], x[3]["case"]))[:args.joint_candidates]
            report["stage"] = "gpu_joint_polish"
            save()
            for _, _, candidate, case, _ in seeds:
                if args.joint_seconds <= 0 or time.perf_counter() >= proxy_deadline:
                    break
                began = time.perf_counter()
                joint = JointItinerary(catalogue, retimer, weights=weights,
                                       settings=JointSettings(time_budget_seconds=args.joint_seconds))
                joint.learn(archived)
                visits, arrivals, departures = visits_of(candidate)
                _, _, evaluation, moves = joint.optimise_epochs(
                    visits, np.asarray(arrivals), np.asarray(departures), mesh=(20.0, 8.0, 3.0, 1.0),
                    deadline=min(proxy_deadline, time.perf_counter() + args.joint_seconds))
                row = {"case": case["case"], "moves": moves, "evaluations": joint.evaluations,
                       "failure": evaluation.failure, "seconds": time.perf_counter() - began}
                if evaluation.feasible and evaluation.plan is not None:
                    polished = evaluation.plan
                    check_footprint(polished, case)
                    gain = weighted_collected(polished, weights) - before_weighted
                    raw_gain = polished.total_collected_kg - archived.total_collected_kg
                    location = out / "proxies" / f"{case['case']}_joint.json"
                    write_json(location, polished.summary())
                    row.update(raw_gain_kg=raw_gain, weighted_gain_kg=gain, plan=str(location))
                    proxy_candidates.append((gain, raw_gain, polished, case, "joint"))
                report["joint_rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
            report.update(proxy_seconds=time.perf_counter() - proxy_start,
                          proxy_soft_overrun_seconds=max(0, time.perf_counter() - proxy_deadline))
            eligible = [item for item in proxy_candidates if item[0] > args.minimum_proxy_weighted_gain
                        and item[1] > args.minimum_proxy_raw_gain]
            report["eligible_proxy_candidates"] = len(eligible)
            report["screening_telemetry"] = {} if gpu is None else dict(gpu.telemetry)
            save()
            seen = set()
            for _, _, candidate, case, origin in sorted(eligible, key=lambda x: (-x[0], -x[1], x[3]["case"])):
                if args.proxy_only or len(report["refinements"]) >= args.max_certifications or time.perf_counter() >= deadline:
                    break
                signature = json.dumps(candidate.summary(), sort_keys=True)
                if signature in seen:
                    continue
                seen.add(signature)
                attempt = len(report["refinements"]) + 1
                directory = out / "candidates" / f"ship_15_attempt_{attempt:02d}"
                directory.mkdir(parents=True)
                record = {"ship": SHIP, "case": case["case"], "origin": origin, "attempt": attempt}
                report["refinements"].append(record)
                report.update(stage="gpu_route_refinement", active_candidate=record.copy())
                save()
                refined = refine_route(candidate, catalogue, scvx=scvx)
                write_json(directory / "refinement.json", refined.summary())
                record.update(certified=refined.certified, raw_kg=refined.total_collected_kg,
                              failures=refined.failures, seconds=refined.wall_seconds)
                if not refined.certified:
                    record["status"] = "failed_refinement"
                    save()
                    continue
                check_footprint(refined.plan, case)
                if (weighted_mass(refined.collected_mass, weights) <= before_weighted + 1e-8
                        or refined.total_collected_kg < archived.total_collected_kg - 1e-8):
                    record["status"] = "certified_without_joint_raw_and_weighted_improvement"
                    save()
                    continue
                write_route_artifacts(refined, catalogue, directory / "route")
                replacement = emit_solution(refined, catalogue, ship_id=SHIP).ships[0]
                trial = Solution([replacement if old.ship_id == SHIP else old for old in current.ships])
                trial_path = directory / "fleet-Result.txt"
                trial.write(trial_path)
                report["stage"] = "candidate_full_fleet_verification"
                save()
                checked, histories = verify(trial_path)
                record["verification"] = checked
                write_json(directory / "verification.json", checked)
                best = report["best"]
                if (checked["ok"] and checked["score_kg"] > best["score_kg"] + 1e-8
                        and checked["total_mass_kg"] >= best["total_mass_kg"] - 1e-8):
                    current = trial
                    best_dir = out / "best"
                    best_dir.mkdir(exist_ok=True)
                    shutil.copy2(trial_path, best_dir / "Result.txt")
                    viewer = write_viewer_dataset(best_dir / "viewer", trial, histories, catalogue,
                                                  run_id="collect_order_v597", commit="experimental",
                                                  verification=checked["independent"], solution_path=best_dir / "Result.txt")
                    report["best"] = checked | {"solution": str(best_dir / "Result.txt"),
                                               "viewer_manifest": str(best_dir / "viewer/manifest.json"),
                                               "viewer": viewer, "source": "verified_candidate"}
                    report["promotions"].append({"ship": SHIP, "case": case["case"], **checked})
                    record["status"] = "promoted_verified_fleet"
                else:
                    record["status"] = "retained_verified_incumbent"
                save()
            report["screening_telemetry"] = {} if gpu is None else dict(gpu.telemetry)
    if sha256(old_path) != INCUMBENT_SHA256:
        raise RuntimeError("baseline input changed during the experiment")
    report.update(complete=True, stage="complete", status="improved" if report["promotions"] else "incumbent_retained",
                  seconds=time.perf_counter() - started, soft_budget_overrun_seconds=max(0, time.perf_counter() - deadline))
    save()
    print(json.dumps({"status": report["status"], "best": report["best"], "output": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    args = arguments()
    try:
        raise SystemExit(run(args))
    except Exception as error:
        path = args.output.resolve() / "report.json"
        if getattr(args, "_output_created", False) and path.exists():
            report = json.loads(path.read_text())
            report.update(complete=True, status="failed_after_verified_promotion" if report.get("promotions")
                          else "failed_incumbent_unchanged", error=repr(error))
            write_json(path, report)
        raise
