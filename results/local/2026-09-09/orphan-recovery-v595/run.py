"""Bounded own-orphan collection experiment; no solver runs unless invoked explicitly.

Run --plan-only to inspect all new visit orders without importing the GPU stack.
The full run requires caller-supplied validated CUDA libraries and pinned GTOC12 data.
It leaves the original fleet untouched and promotes only a doubly verified improvement.
"""
from __future__ import annotations

import argparse
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


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def native_progress(out, report):
    """Record actual native calls; do not infer solve counts from requested layouts."""
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
            value = original(*args, **kwargs)
            row.update(status=value.status, iterations=value.iterations,
                       accepted_iterations=value.accepted_iterations, nodes=len(value.node_epochs_mjd))
            return value
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


def make_cases(summaries, selected):
    """New collect-order layouts, with exact own-deployer and fleet exclusivity checks."""
    cases = []
    for ship in selected:
        summary = summaries[ship]
        plan = summary["plan"]
        deploy = {int(a): float(t) for a, t in plan["deploy_epochs"].items()}
        collect = {int(a): float(t) for a, t in plan["collect_epochs"].items()}
        if plan.get("foreign_deploy_epochs"):
            raise ValueError(f"ship {ship}: expected no foreign dependencies")
        if not summary.get("certified"):
            raise ValueError(f"ship {ship}: archived input is not certified")
        orphans = set(deploy) - set(collect)
        if orphans != set(plan.get("orphaned", [])) or len(orphans) != 1:
            raise ValueError(f"ship {ship}: expected exactly one own orphan")
        orphan = next(iter(orphans))
        for other, value in summaries.items():
            if other == ship:
                continue
            q = value["plan"]
            if str(orphan) in q["collect_epochs"] or str(orphan) in q["deploy_epochs"]:
                raise ValueError(f"asteroid {orphan} is already used by incumbent ship {other}")
        deploy_order = sorted(deploy, key=deploy.get)
        collect_order = sorted(collect, key=collect.get)
        if deploy_order[-1] != orphan:
            raise ValueError("this experiment targets the last deployed asteroid")
        for position in range(len(collect_order) + 1):
            new_collect = [*collect_order[:position], orphan, *collect_order[position:]]
            assert len(set(new_collect)) == len(new_collect)
            assert set(new_collect) == set(deploy_order)
            cases.append({
                "case": f"ship_{ship:02d}_position_{position:02d}",
                "ship": ship, "orphan": orphan, "position": position,
                "deploy_order": deploy_order, "collect_order": new_collect,
                "merges_last_deploy_and_first_collect": position == 0,
                "additional_low_thrust_legs": 0 if position == 0 else 1,
                "same_asteroid_footprint": True,
                "source_raw_kg": summary["total_collected_kg"],
            })
    return cases


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ships", default="15,3")
    p.add_argument("--grids", default="15,5")
    p.add_argument("--wall-seconds", type=float, default=1800)
    p.add_argument("--max-certifications", type=int, default=4)
    p.add_argument("--per-ship-certifications", type=int, default=2)
    p.add_argument("--minimum-proxy-weighted-gain", type=float, default=0.5)
    p.add_argument("--minimum-proxy-raw-gain", type=float, default=0.5)
    p.add_argument("--joint-seconds", type=float, default=0)
    p.add_argument("--screening-backend", choices=("cuda", "numpy"), default="cuda")
    p.add_argument("--gpu-execution", choices=("graph", "dispatch"), default="graph")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--proxy-only", action="store_true")
    p.add_argument("--lock", type=Path, default=Path.home() / ".spacepdhcg-gpu.lock")
    args = p.parse_args()
    args.repo = args.repo.resolve()
    args.ships = [int(x) for x in args.ships.split(",")]
    args.grids = [float(x) for x in args.grids.split(",")]
    if not args.ships or len(set(args.ships)) != len(args.ships) or not set(args.ships) <= {3, 15}:
        p.error("--ships must select ship 3, ship 15, or both, without duplicates")
    if not args.grids or len(set(args.grids)) != len(args.grids) or not set(args.grids) <= {5.0, 15.0}:
        p.error("--grids must select 15, 5, or both")
    if not 0 < args.wall_seconds <= 7200 or not 0 <= args.joint_seconds <= 120:
        p.error("wall budget must be in (0,7200], joint budget in [0,120]")
    if not 1 <= args.max_certifications <= 8 or not 1 <= args.per_ship_certifications <= 4:
        p.error("certification caps must be bounded: total 1..8, per-ship 1..4")
    if args.minimum_proxy_weighted_gain < 0 or args.minimum_proxy_raw_gain < 0:
        p.error("proxy gain thresholds cannot be negative")
    return args


def run(args):
    source = args.repo / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
    paths = {ship: source / f"ship-{ship:02d}.json" for ship in range(1, 24)}
    summaries = {ship: json.loads(path.read_text()) for ship, path in paths.items()}
    cases = make_cases(summaries, args.ships)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args._output_created = True
    write_json(out / "orders.json", cases)
    report = {
        "pid": os.getpid(), "complete": False, "status": "prepared", "stage": "prepared", "orders": len(cases),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "input_sha256": {str(path): sha256(path) for path in paths.values()},
        "driver_sha256": sha256(Path(__file__)), "rows": [], "refinements": [],
        "best": None, "promotions": [], "physics_tolerances_changed": False,
    }
    save = lambda: write_json(out / "report.json", report)
    save()
    if args.plan_only:
        report.update(complete=True, status="plan_only_no_gpu_execution")
        save()
        print(json.dumps({"status": report["status"], "orders": len(cases), "output": str(out)}))
        return 0

    # Caller chooses the core/QOCO library and diagnostic flags. The experimental
    # driver changes no numerical tolerances or acceptance thresholds in the solver.
    for name in ("SPACEPDHCG_GTOC12_CUDA_LIBRARY", "SPACEPDHCG_QOCO_LIBRARY"):
        library = Path(os.environ.get(name, ""))
        if not library.is_file():
            raise ValueError(f"{name} must name a validated native library")
        report.setdefault("native_sha256", {})[name] = sha256(library)
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    sys.path.insert(0, str(args.repo / "src"))
    import fcntl
    import numpy as np
    from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, cluster_retime_settings, cluster_search_settings
    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.jointopt import JointItinerary, JointSettings, route_from_summary, weighted_mass
    from spacepdhcg.gtoc12.lambert import using_lambert_backend
    from spacepdhcg.gtoc12.low_thrust import ScvxSettings
    from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
    from spacepdhcg.gtoc12.pipeline import emit_solution, refine_route, write_route_artifacts
    from spacepdhcg.gtoc12.retiming import Retimer, build_visits, calibrate_from_route, visits_of, weighted_collected
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
    from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset

    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    if not official_verifier_available():
        raise RuntimeError("both final checkers are required; official verifier is unavailable")
    report["source_sha256"] = {
        str(p.relative_to(args.repo)): sha256(p)
        for p in sorted((args.repo / "src/spacepdhcg/gtoc12").glob("*.py"))
    }
    report["bonus_sha256"] = bonus.source_sha256
    report["joint_batch_requested"] = os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_BATCH", "0")
    scvx = ScvxSettings(max_iterations=40, node_days=2.0, discretisation_backend="cuda",
                        assembly_backend="cuda", convex_solver_backend="qoco",
                        qoco_ruiz_iterations=0, outer_loop_backend="cuda", seed_backend="cuda")
    execution = argparse.Namespace(gpu_execution=args.gpu_execution, outer_loop_backend="cuda", workers=1)
    report["scvx_settings"] = dataclasses.asdict(scvx)
    old_path = args.repo / "results/lambda/2026-09-06/fleet_master_v11/fleet/Result.txt"
    report["incumbent_solution_sha256"] = sha256(old_path)
    current = Solution.read(old_path)
    if current.ship_count != 23 or [s.ship_id for s in current.ships] != list(range(1, 24)):
        raise ValueError("the retained fleet must contain ships 1..23 in order")
    for ship in current.ships:
        for case in (c for c in cases if c["position"] == 0):
            count = sum(e.event_id == case["orphan"] for e in ship.asteroid_visits())
            expected = 1 if ship.ship_id == case["ship"] else 0
            if count != expected:
                raise ValueError("actual incumbent solution does not match the own-orphan inventory")
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    lock = args.lock.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def verify(path):
        histories = {}
        independent = Gtoc12Verifier(catalogue, bonus=bonus, history=histories).verify_file(path)
        official = run_official_verifier(path)
        summary = independent.summary()
        score = summary.get("weighted_score_fixed_bonus_kg")
        ok = bool(independent.ok and official.ok and score is not None and math.isfinite(score))
        return {"ok": ok, "independent": summary, "official": official.summary(),
                "score_kg": score if ok else None, "total_mass_kg": independent.total_mass_kg}, histories

    report["stage"] = "baseline_fleet_verification"
    save()
    baseline, _ = verify(old_path)
    if not baseline["ok"]:
        raise RuntimeError("retained fleet failed fresh verification; no candidate work started")
    report["baseline"] = baseline
    report["best"] = baseline | {"solution": str(old_path), "source": "retained_incumbent"}
    save()
    base_routes = {ship: route_from_summary(summaries[ship]) for ship in args.ships}
    proxy_candidates = []
    report["stage"] = "gpu_order_scan"
    save()
    with using_gpu_execution(execution), using_lambert_backend(args.screening_backend) as gpu, native_progress(out, report):
        # Enumerate only the new structural orders. Retiming controls are unchanged;
        # calibration is taken from the actual flown incumbent legs, including margin.
        for ship in args.ships:
            archived = base_routes[ship]
            before_weighted = weighted_mass(archived.collected_mass, weights)
            for step in args.grids:
                if time.perf_counter() >= deadline:
                    break
                policy = ClusterPricingSettings()
                retimer = Retimer(catalogue, cluster_search_settings(policy, len(catalogue.ids)),
                                  dataclasses.replace(cluster_retime_settings(policy, last=True), step_days=step),
                                  weights=weights)
                retimer.protect_earth_leg(archived.plan)
                calibrated = calibrate_from_route(retimer, archived)
                for case in (c for c in cases if c["ship"] == ship):
                    if time.perf_counter() >= deadline:
                        break
                    profile = list(retimer._plan_masses(archived.plan))
                    needed = len(build_visits(case["deploy_order"], case["collect_order"])) - 1
                    camp = len(case["deploy_order"])
                    while len(profile) < needed:
                        profile.insert(camp, profile[min(camp, len(profile) - 1)])
                    while len(profile) > needed:
                        profile.pop(camp)
                    result = retimer.retime_order(case["deploy_order"], case["collect_order"], profile,
                                                  before=before_weighted, original=archived.plan)
                    row = dict(case=case["case"], ship=ship, grid_days=step, calibrated_legs=calibrated,
                               result=result.summary(), elapsed_seconds=time.perf_counter() - started)
                    if result.plan is not None and result.plan.feasible:
                        candidate = result.plan
                        # Optional native-joint polishing requires a caller-selected library
                        # and feature flag. It never uses speculative credit as verified ore.
                        if args.joint_seconds > 0 and time.perf_counter() < deadline:
                            joint = JointItinerary(catalogue, retimer, weights=weights,
                                                  settings=JointSettings(time_budget_seconds=args.joint_seconds))
                            joint.learn(archived)
                            visits, arrivals, departures = visits_of(candidate)
                            _, _, evaluation, moves = joint.optimise_epochs(
                                visits, np.asarray(arrivals), np.asarray(departures),
                                mesh=(20.0, 8.0, 3.0, 1.0),
                                deadline=min(deadline, time.perf_counter() + args.joint_seconds))
                            row["joint"] = {"moves": moves, "evaluations": joint.evaluations,
                                            "failure": evaluation.failure}
                            if evaluation.feasible and evaluation.plan is not None and (
                                weighted_collected(evaluation.plan, weights) > weighted_collected(candidate, weights)
                                and evaluation.plan.total_collected_kg >= candidate.total_collected_kg
                            ):
                                candidate = evaluation.plan
                        if (set(candidate.deploy_epochs) != set(case["deploy_order"])
                                or set(candidate.collect_epochs) != set(case["collect_order"])
                                or candidate.foreign_deploy_epochs):
                            raise ValueError("proxy changed the authorized collection footprint")
                        raw_gain = candidate.total_collected_kg - archived.total_collected_kg
                        weighted_gain = weighted_collected(candidate, weights) - before_weighted
                        row.update(raw_gain_kg=raw_gain, weighted_gain_kg=weighted_gain)
                        location = out / "proxies" / f"{case['case']}_grid_{step:g}.json"
                        location.parent.mkdir(exist_ok=True)
                        write_json(location, candidate.summary())
                        row["plan"] = str(location)
                        if raw_gain > args.minimum_proxy_raw_gain and weighted_gain > args.minimum_proxy_weighted_gain:
                            proxy_candidates.append((weighted_gain, raw_gain, ship, candidate, row))
                    report["rows"].append(row)
                    save()
                    print(json.dumps({k: v for k, v in row.items() if k != "result"}), flush=True)

        report["eligible_proxy_candidates"] = len(proxy_candidates)
        tried = dict.fromkeys(args.ships, 0)
        seen = set()
        for _, _, ship, candidate, row in sorted(proxy_candidates, key=lambda x: (-x[0], -x[1], x[2])):
            if args.proxy_only or time.perf_counter() >= deadline or sum(tried.values()) >= args.max_certifications:
                break
            if tried[ship] >= args.per_ship_certifications:
                continue
            signature = json.dumps(candidate.summary(), sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            tried[ship] += 1
            directory = out / "candidates" / f"ship_{ship:02d}_attempt_{tried[ship]:02d}"
            directory.mkdir(parents=True)
            record = {"ship": ship, "case": row["case"], "grid_days": row["grid_days"],
                      "started_seconds": time.perf_counter() - started}
            report["refinements"].append(record)
            report["stage"] = "gpu_route_refinement"
            report["active_candidate"] = {"ship": ship, "case": row["case"], "attempt": tried[ship]}
            save()
            refined = refine_route(candidate, catalogue, scvx=scvx)
            write_json(directory / "refinement.json", refined.summary())
            record.update(certified=refined.certified, raw_kg=refined.total_collected_kg,
                          failures=refined.failures, seconds=refined.wall_seconds)
            if not refined.certified:
                record["status"] = "failed_refinement"
                save()
                continue
            actual_weighted = weighted_mass(refined.collected_mass, weights)
            original = base_routes[ship]
            if (actual_weighted <= weighted_mass(original.collected_mass, weights) + 1e-8
                    or refined.total_collected_kg < original.total_collected_kg - 1e-8):
                record["status"] = "certified_without_joint_raw_and_weighted_improvement"
                save()
                continue
            write_route_artifacts(refined, catalogue, directory / "route")
            trial = Solution([emit_solution(refined, catalogue, ship_id=ship).ships[0]
                              if old.ship_id == ship else old for old in current.ships])
            trial_path = directory / "fleet-Result.txt"
            trial.write(trial_path)
            report["stage"] = "candidate_fleet_verification"
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
                                              run_id="orphan_recovery_v595", commit="experimental",
                                              verification=checked["independent"], solution_path=best_dir / "Result.txt")
                report["best"] = checked | {"solution": str(best_dir / "Result.txt"),
                                           "viewer_manifest": str(best_dir / "viewer/manifest.json"),
                                           "viewer": viewer, "source": "verified_candidate"}
                report["promotions"].append({"ship": ship, "case": row["case"], **checked})
                record["status"] = "promoted_verified_fleet"
            else:
                record["status"] = "retained_verified_incumbent"
            save()
        report["screening_telemetry"] = {} if gpu is None else dict(gpu.telemetry)

    report.update(complete=True, stage="complete", status="improved" if report["promotions"] else "incumbent_retained",
                  seconds=time.perf_counter() - started, soft_budget_overrun_seconds=max(0, time.perf_counter() - deadline))
    save()
    lock.close()
    print(json.dumps({"status": report["status"], "best": report["best"], "output": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    args = arguments()
    try:
        raise SystemExit(run(args))
    except Exception as error:
        report_path = args.output.resolve() / "report.json"
        if getattr(args, "_output_created", False) and report_path.exists():
            error_report = json.loads(report_path.read_text())
            error_report.update(complete=True, status="failed_after_verified_promotion" if error_report.get("promotions") else "failed_incumbent_unchanged", error=repr(error))
            write_json(report_path, error_report)
        raise
