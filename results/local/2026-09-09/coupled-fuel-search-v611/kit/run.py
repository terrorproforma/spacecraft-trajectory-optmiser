"""Finite native whole-itinerary fuel/objective sweep; explicit execution only."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import common
import numpy as np


@contextmanager
def count_native(report, directory):
    from spacepdhcg.gtoc12 import gpu_scvx

    original = gpu_scvx.solve_native

    def call(boundary, *args, **kwargs):
        if report["native_legs_started"] >= 68:
            raise RuntimeError("Hard68-native-leg budget exhausted")
        report["native_legs_started"] += 1
        record = {
            "number": report["native_legs_started"],
            "case": report["active_case"],
            "departure": boundary.departure_epoch,
            "arrival": boundary.arrival_epoch,
            "initial_mass_kg": boundary.initial_mass,
            "minimum_final_mass_kg": boundary.minimum_final_mass,
        }
        started = time.perf_counter()
        try:
            value = original(boundary, *args, **kwargs)
            record.update(
                status=value.status,
                iterations=value.iterations,
                accepted_iterations=value.accepted_iterations,
            )
            return value
        except Exception as error:
            record["error"] = repr(error)
            raise
        finally:
            report["native_legs_finished"] += 1
            record["wrapper_seconds"] = time.perf_counter() - started
            with (directory / "native-legs.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    gpu_scvx.solve_native = call
    try:
        yield
    finally:
        gpu_scvx.solve_native = original


def main(args):
    import fcntl

    common.activate()
    ready = common.read(common.ROOT / "ready-manifest.json")
    for name, expected in ready["files"].items():
        if common.sha(common.ROOT / name) != expected:
            raise ValueError("Prepared kit changed: " + name)
    profile = common.runtime_check()
    if args.lock.resolve() != Path(profile["lock"]).resolve():
        raise ValueError("Exact shared GPU lock required")
    from domain import bounds, incumbent_joint, search_order, shortlist
    from fixed_refine import refine_fixed, validate_prescription
    from freeze_incumbents import ship_bytes

    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.gpu_joint import evaluate_joint
    from spacepdhcg.gtoc12.jointopt import weighted_mass
    from spacepdhcg.gtoc12.lambert import using_lambert_backend
    from spacepdhcg.gtoc12.low_thrust import ScvxSettings
    from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
    from spacepdhcg.gtoc12.pipeline import emit_solution, plan_from_route_summary
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier

    catalogue, bonus = load_catalogue(), load_bonus_table()
    if (
        catalogue.source_sha256
        != "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675"
        or bonus.source_sha256 != "e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21"
    ):
        raise ValueError("Catalogue/bonus pin changed")
    if not official_verifier_available():
        raise ValueError("Both fullfleet checkers are required")
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    baseline_path = common.ROOT / "inputs/Result.txt"
    baseline_fleet = Solution.read(baseline_path)
    controls = common.read(common.ROOT / "incumbent-controls.json")
    if controls["fresh_control_refinement_required"] or controls["critical_source_differences"]:
        raise ValueError("This reviewed kit requires equivalent archived controls")
    for ship in common.SHIPS:
        if ship_bytes(baseline_path, ship) != ship_bytes(
            common.ROOT / "inputs/archived-control-fleet.txt", ship
        ):
            raise ValueError("Archived control no longer matches exact incumbent ship")
    if args.lock_fd is None:
        handle = args.lock.open("a+")
    else:
        passed, actual = os.fstat(args.lock_fd), args.lock.stat()
        if (passed.st_dev, passed.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("Wrong supervisor lock descriptor")
        handle = os.fdopen(os.dup(args.lock_fd), "a+")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.created_output = True
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    scvx = ScvxSettings(
        max_iterations=40,
        node_days=2.0,
        discretisation_backend="cuda",
        assembly_backend="cuda",
        convex_solver_backend="qoco",
        qoco_ruiz_iterations=0,
        outer_loop_backend="cuda",
        seed_backend="cuda",
    )
    report = {
        "complete": False,
        "status": "running",
        "pid": os.getpid(),
        "source_commit": common.read(common.ROOT / "preparation.json")["source_commit"],
        "ready_sha256": common.sha(common.ROOT / "ready-manifest.json"),
        "native_libraries": profile["native_libraries"],
        "scvx_settings": dataclasses.asdict(scvx),
        "bounds": bounds(),
        "searches": [],
        "refinements": [],
        "promotions": [],
        "native_legs_started": 0,
        "native_legs_finished": 0,
        "full_refinements_started": 0,
        "fresh_control_refinements": 0,
        "control_kind": "equivalent_archived_ship_bytes_and_fresh_baseline_fleet_checks",
        "proxy_cargo_sizing": (
            "native estimated cargo; exact selected quantities frozen before refinement"
        ),
        "score_kind": "fixed_bonus_weighted_kg",
        "cargo_shrinking_during_refinement": False,
    }

    def save():
        common.write(out / "report.json", report)

    def verify(path):
        before = time.perf_counter()
        independent = Gtoc12Verifier(catalogue, bonus=bonus).verify_file(path)
        official = run_official_verifier(path)
        summary = independent.summary()
        score = summary.get("weighted_score_fixed_bonus_kg")
        return {
            "ok": bool(
                independent.ok and official.ok and score is not None and math.isfinite(score)
            ),
            "independent": summary,
            "official": official.summary(),
            "score_kg": score,
            "total_mass_kg": independent.total_mass_kg,
            "seconds": time.perf_counter() - before,
        }

    qualified = []
    with handle as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["stage"] = "baseline_and_archived_control_gate"
        save()
        basecheck = verify(baseline_path)
        if not basecheck["ok"]:
            raise ValueError("Baseline failed fresh both-checker gate")
        baseline = {"raw_kg": basecheck["total_mass_kg"], "weighted_kg": basecheck["score_kg"]}
        report.update(
            baseline=basecheck,
            archived_controls_accepted=True,
            best=basecheck | {"solution": str(baseline_path), "sha256": common.sha(baseline_path)},
        )
        originals = {}
        execution = argparse.Namespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
        with (
            using_gpu_execution(execution),
            using_lambert_backend("cuda") as gpu,
            count_native(report, out),
        ):
            if gpu is None:
                raise RuntimeError("CUDA backend unavailable; no numerical fallback")
            for number, (seed, price) in enumerate(search_order(), 1):
                ship = common.SHIPS[seed]
                if number > 12:
                    raise AssertionError("Twelve device searches maximum")
                if time.perf_counter() >= deadline:
                    report.setdefault("not_started_searches", []).append(
                        {"ship": ship, "price": price}
                    )
                    continue
                joint, original, visits, a, d = incumbent_joint(catalogue, weights, ship, price)
                originals[ship] = {
                    "raw_kg": original.total_collected_kg,
                    "weighted_kg": weighted_mass(original.collected_mass, weights),
                }
                row = {
                    "id": f"search-{number:02d}",
                    "ship": ship,
                    "margin_price": price,
                    "scope": "uncertified whole-itinerary proxy",
                    "started": True,
                    "initial_arrivals": a.tolist(),
                    "initial_departures": d.tolist(),
                }
                report["searches"].append(row)
                report["stage"] = "native_whole_itinerary_search"
                save()
                before = time.perf_counter()
                try:
                    selected, arrivals, departures, stats = evaluate_joint(
                        joint,
                        visits,
                        a[None],
                        d[None],
                        minimum_objective=-math.inf,
                        _mesh_delta=1.0,
                        _search_config=(
                            np.asarray(common.MESH),
                            common.MAX_MOVES,
                            min(deadline, before + 30),
                        ),
                    )
                    ev = selected[1]
                    work = {k: int(stats[k]) for k in stats.dtype.names}
                    if work["evaluations"] > bounds()["max_rows_per_search"]:
                        raise AssertionError("Per-search row budget exceeded")
                    row.update(
                        work=work,
                        search_seconds=time.perf_counter() - before,
                        feasible=ev.feasible,
                        failure=ev.failure,
                        objective=ev.objective,
                        weighted_kg=ev.weighted_kg,
                        raw_kg=ev.collected_kg,
                        spare_kg=ev.spare_kg,
                        propellant_estimate_kg=ev.propellant_kg,
                        arrivals=arrivals[0].tolist(),
                        departures=departures[0].tolist(),
                        weighted_gain_kg=ev.weighted_kg - originals[ship]["weighted_kg"],
                        objective_eligible=False,
                    )
                    if ev.plan is not None:
                        validate_prescription(ev.plan, ev.plan.collected_mass)
                        if set(ev.plan.deploy_epochs) != set(original.plan.deploy_epochs) or set(
                            ev.plan.collect_epochs
                        ) != set(original.plan.collect_epochs):
                            raise AssertionError("Fixed asteroid inventory changed")
                        row["plan"] = ev.plan.summary()
                        row["plan_sha256"] = hashlib.sha256(
                            json.dumps([ship, row["plan"]], sort_keys=True).encode()
                        ).hexdigest()
                        row["objective_eligible"] = common.eligible(
                            ev.collected_kg, ev.weighted_kg, baseline, originals[ship]
                        )
                except Exception as error:
                    row.update(
                        error=repr(error), objective_eligible=False, status="search_exception"
                    )
                common.write(out / "searches" / (row["id"] + ".json"), row)
                save()
            report["screening_telemetry"] = dict(gpu.telemetry)
            candidates = shortlist(report["searches"])
            report["selected"] = [row["id"] for row in candidates]
            save()
            for number, row in enumerate(candidates, 1):
                if time.perf_counter() >= deadline:
                    report.setdefault("not_started_refinements", []).append(row["id"])
                    continue
                if number > 3 or report["full_refinements_started"] >= 4:
                    raise AssertionError("Full-route refinement budget exceeded")
                ship = row["ship"]
                case = out / "refinements" / f"candidate-{number:02d}"
                case.mkdir(parents=True, exist_ok=False)
                plan = plan_from_route_summary({"plan": row["plan"]})
                cargo = dict(plan.collected_mass)
                validate_prescription(plan, cargo)
                report["full_refinements_started"] += 1
                report["active_case"] = case.name
                report["stage"] = "native_fixed_cargo_fullroute_refinement"
                detail = {
                    "id": case.name,
                    "search": row["id"],
                    "ship": ship,
                    "cargo_kg": cargo,
                    "plan": plan.summary(),
                    "status": "refining",
                    "expected_legs": 17,
                }
                report["refinements"].append(detail)
                common.write(case / "prescription.json", detail)
                save()

                def record_leg(index, item, saved, _case=case):
                    if item.solution is not None:
                        path = _case / f"leg-{index:02d}.npz"
                        np.savez_compressed(
                            path,
                            node_epochs=item.solution.node_epochs_mjd,
                            thrust_n=item.solution.thrust_n,
                            states_scaled=item.solution.states_scaled,
                            departure_vinf=item.solution.departure_vinf_km_s,
                            arrival_vinf=item.solution.arrival_vinf_km_s,
                        )
                        saved["arrays"] = {"path": path.name, "sha256": common.sha(path)}
                    common.write(_case / f"leg-{index:02d}.json", saved)

                try:
                    native_before = report["native_legs_started"]
                    refined = refine_fixed(
                        plan, catalogue, cargo, settings=scvx, on_leg=record_leg, deadline=deadline
                    )
                    common.write(case / "refinement.json", refined.summary())
                    detail.update(
                        status="route_certified" if refined.certified else "route_failed",
                        certified=refined.certified,
                        failures=refined.failures,
                        seconds=refined.wall_seconds,
                        actual_legs=len(refined.legs),
                        native_leg_calls=report["native_legs_started"] - native_before,
                    )
                    if detail["native_leg_calls"] != len(refined.legs):
                        raise AssertionError("Every retained leg requires exactly one native solve")
                    if not refined.certified:
                        save()
                        continue
                    if refined.collected_mass != cargo or refined.plan.summary() != plan.summary():
                        raise AssertionError("Refinement changed selected cargo or schedule")
                    replacement = emit_solution(refined, catalogue, ship_id=ship).ships[0]
                    fleet = Solution(
                        [replacement if s.ship_id == ship else s for s in baseline_fleet.ships]
                    )
                    path = case / "fleet-Result.txt"
                    fleet.write(path)
                    checked = verify(path)
                    expected = {
                        "raw_kg": baseline["raw_kg"]
                        - originals[ship]["raw_kg"]
                        + refined.total_collected_kg,
                        "weighted_kg": baseline["weighted_kg"]
                        - originals[ship]["weighted_kg"]
                        + weighted_mass(cargo, weights),
                    }
                    common.write(case / "verification.json", checked)
                    detail["verification"] = checked
                    if common.fleet_accept(checked, baseline, expected):
                        detail["status"] = "individually_both_checked_gain"
                        qualified.append(
                            {
                                "ship": ship,
                                "replacement": replacement,
                                "raw_gain": expected["raw_kg"] - baseline["raw_kg"],
                                "weighted_gain": expected["weighted_kg"] - baseline["weighted_kg"],
                                "path": path,
                                "checked": checked,
                            }
                        )
                except Exception as error:
                    detail.update(status="refinement_exception", error=repr(error))
                save()
        # At most three independently qualified variants: enumerate at most eight subsets.
        best_subset, best_score = common.best_combination(qualified, baseline)
        if best_subset:
            if len(best_subset) == 1:
                item = best_subset[0]
                path, checked = item["path"], item["checked"]
            else:
                replacements = {item["ship"]: item["replacement"] for item in best_subset}
                fleet = Solution([replacements.get(s.ship_id, s) for s in baseline_fleet.ships])
                path = out / "combined-Result.txt"
                fleet.write(path)
                checked = verify(path)
                expected = {
                    "raw_kg": baseline["raw_kg"] + sum(item["raw_gain"] for item in best_subset),
                    "weighted_kg": best_score,
                }
                common.write(out / "combined-verification.json", checked)
                if not common.fleet_accept(checked, baseline, expected):
                    item = max(qualified, key=lambda x: x["weighted_gain"])
                    path, checked = item["path"], item["checked"]
            report["best"] = checked | {"solution": str(path), "sha256": common.sha(path)}
            report["promotions"].append(dict(report["best"]))
    common.activate()
    execution_errors = sum("error" in row for row in report["searches"] + report["refinements"])
    report.update(
        complete=True,
        status="improved"
        if report["promotions"]
        else ("completed_with_execution_errors" if execution_errors else "incumbent_retained"),
        execution_errors=execution_errors,
        stage="complete",
        seconds=time.perf_counter() - started,
    )
    if (
        report["native_legs_started"] != report["native_legs_finished"]
        or report["native_legs_started"] > 68
    ):
        raise AssertionError("Native solve accounting mismatch")
    save()
    print(
        json.dumps(
            {
                "status": report["status"],
                "searches": len(report["searches"]),
                "full_refinements": report["full_refinements_started"],
                "native_legs": report["native_legs_finished"],
            }
        )
    )
    return int(execution_errors != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1800:
        parser.error("Wall budget must be positive and at most1800seconds")
    try:
        raise SystemExit(main(args))
    except Exception as error:
        if getattr(args, "created_output", False):
            path = args.output / "report.json"
            saved = common.read(path) if path.exists() else {}
            saved.update(complete=True, status="failed_incumbent_preserved", error=repr(error))
            common.write(path, saved)
        raise
