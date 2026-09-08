"""Four-route maximum native low-thrust truth set; no cargo shrinking or retiming."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import common


@contextmanager
def native_progress(out, report):
    from spacepdhcg.gtoc12 import gpu_scvx

    original = gpu_scvx.solve_native

    def observed(*args, **kwargs):
        report["native_solves_started"] += 1
        row = {
            "number": report["native_solves_started"],
            "case": report["active_case"],
            "departure": args[0].departure_epoch,
            "arrival": args[0].arrival_epoch,
            "initial_mass_kg": args[0].initial_mass,
            "minimum_final_mass_kg": args[0].minimum_final_mass,
        }
        began = time.perf_counter()
        try:
            result = original(*args, **kwargs)
            row.update(
                status=result.status,
                iterations=result.iterations,
                accepted_iterations=result.accepted_iterations,
            )
            return result
        except Exception as error:
            row["error"] = repr(error)
            raise
        finally:
            row["seconds"] = time.perf_counter() - began
            report["native_solves_completed"] += 1
            with (out / "native-solves.jsonl").open("a") as stream:
                import json

                stream.write(json.dumps(row) + "\n")

    gpu_scvx.solve_native = observed
    try:
        yield
    finally:
        gpu_scvx.solve_native = original


def should_run_case(case, controls):
    return case["kind"] == "control" or controls.get(case["requires_control"], False)


def main(args):
    import fcntl

    import numpy as np

    common.activate_source()
    from fixed_refine import refine_fixed
    from plan import make_plan

    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.low_thrust import ScvxSettings
    from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
    from spacepdhcg.gtoc12.pipeline import emit_solution, write_route_artifacts
    from spacepdhcg.gtoc12.search import RoutePlan
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
    from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset

    preparation = common.read(common.ROOT / "preparation.json")
    common.validate_ready()
    profile, runtime_libraries = common.validate_profile(args.profile)
    if args.lock.resolve() != Path(profile["lock"]).resolve():
        raise ValueError("the selected host's shared GPU lock is mandatory")
    if not official_verifier_available():
        raise RuntimeError("official and independent full-fleet checkers are required")
    summaries, input_audit, baseline_solution = common.load_inputs()
    catalogue, bonus = load_catalogue(), load_bonus_table()
    if catalogue.source_sha256 != common.DATA_SHA or bonus.source_sha256 != common.BONUS_SHA:
        raise ValueError("catalogue or fixed bonus table changed")
    weights = {int(body): float(bonus.for_asteroid(int(body))) for body in catalogue.ids}
    plan = common.read(common.ROOT / "plan/plan.json")
    prepared_plan = common.read(common.ROOT / "plan/report.json")
    if common.sha(common.ROOT / "plan/plan.json") != prepared_plan["plan_sha256"]:
        raise ValueError("reviewed four-route plan changed")
    import json

    if json.loads(json.dumps(make_plan(summaries, input_audit, weights))) != plan:
        raise ValueError("prescribed schedules are not reproducible")
    if len(plan["cases"]) != 4 or plan["maximum_whole_route_refinements"] != 4:
        raise ValueError("four-route global budget required")
    settings = ScvxSettings(**preparation["scvx_settings"])
    if dataclasses.asdict(settings) != preparation["scvx_settings"]:
        raise ValueError("validated native SCvx settings changed")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.created_output = True
    report = {
        "pid": os.getpid(),
        "complete": False,
        "status": "running",
        "stage": "prepared",
        "source_commit": preparation["source_commit"],
        "source_manifest_sha256": common.sha(common.ROOT / "source-sha256.json"),
        "driver_sha256": common.sha(__file__),
        "fixed_wrapper_sha256": common.sha(common.ROOT / "fixed_refine.py"),
        "ready_manifest_sha256": common.sha(common.ROOT / "ready-manifest.json"),
        "plan_sha256": prepared_plan["plan_sha256"],
        "scvx_settings": dataclasses.asdict(settings),
        "profile": args.profile,
        "profiles_sha256": common.sha(common.ROOT / "profiles.json"),
        "native_libraries": runtime_libraries,
        "runtime_python": __import__("sys").version,
        "input_sha256": common.INCUMBENT_SHA,
        "cargo_resizing_allowed": False,
        "retiming_allowed": False,
        "physics_tolerances_changed": False,
        "score_kind": "fixed_bonus_weighted_kg",
        "control_start": (
            "actual solver with CUDA cold seed; archived trajectory is not replayed as control"
        ),
        "maximum_whole_route_refinements": 4,
        "maximum_leg_calls": plan["maximum_leg_calls"],
        "whole_route_refinements_started": 0,
        "native_solves_started": 0,
        "native_solves_completed": 0,
        "cases": [],
        "promotions": [],
        "best": None,
        "feature_flags": {
            key: value for key, value in os.environ.items() if key.startswith("SPACEPDHCG_TEST_")
        },
    }

    def save():
        common.write(out / "report.json", report)

    def verify(path):
        began = time.perf_counter()
        history = {}
        independent = Gtoc12Verifier(catalogue, bonus=bonus, history=history).verify_file(path)
        official = run_official_verifier(path)
        summary = independent.summary()
        score = summary.get("weighted_score_fixed_bonus_kg")
        checked = {
            "ok": bool(
                independent.ok and official.ok and score is not None and math.isfinite(score)
            ),
            "independent": summary,
            "official": official.summary(),
            "score_kg": score,
            "score_kind": "fixed_bonus_weighted_kg",
            "total_mass_kg": independent.total_mass_kg,
            "verification_seconds": time.perf_counter() - began,
        }
        return checked, history

    started = time.perf_counter()
    deadline = started + args.wall_seconds
    controls = {}
    save()
    if args.lock_fd is not None:
        passed_lock = os.fstat(args.lock_fd)
        shared_lock = args.lock.stat()
        if (passed_lock.st_dev, passed_lock.st_ino) != (shared_lock.st_dev, shared_lock.st_ino):
            raise ValueError("supervisor did not pass the selected shared GPU lock")
        lock_handle = os.fdopen(os.dup(args.lock_fd), "a+")
    else:
        lock_handle = args.lock.open("a+")
    with lock_handle as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report.update(stage="baseline_full_fleet_verification", gpu_lock_acquired=True)
        save()
        baseline, _ = verify(common.ROOT / "inputs/Result.txt")
        if not common.checker_pass(baseline):
            raise ValueError("retained baseline failed fresh full-fleet verification")
        report["baseline"] = baseline
        report["best"] = baseline | {
            "source": "retained_v595",
            "solution": str(common.ROOT / "inputs/Result.txt"),
            "solution_sha256": common.INCUMBENT_SHA,
        }
        execution = argparse.Namespace(
            gpu_execution=preparation["gpu_execution"], outer_loop_backend="cuda", workers=1
        )
        with using_gpu_execution(execution), native_progress(out, report):
            for case in plan["cases"]:
                record = {
                    "id": case["id"],
                    "kind": case["kind"],
                    "ship": case["ship"],
                    "prescription": case,
                    "legs": [],
                    "status": "pending",
                }
                report["cases"].append(record)
                if not should_run_case(case, controls):
                    record["status"] = "skipped_corresponding_control_failed"
                    record["proxy_interpretation_allowed"] = False
                    save()
                    continue
                if time.perf_counter() >= deadline:
                    record["status"] = "not_run_campaign_deadline"
                    save()
                    continue
                directory = out / "cases" / case["id"]
                directory.mkdir(parents=True)
                route_plan = RoutePlan.from_summary(case["plan"])
                cargo = {
                    int(body): float(mass) for body, mass in case["prescribed_cargo_kg"].items()
                }
                common.write(directory / "prescription.json", case)
                report.update(stage="fixed_cargo_native_refinement", active_case=case["id"])
                record["status"] = "refining"
                report["whole_route_refinements_started"] += 1
                save()
                if report["whole_route_refinements_started"] > 4:
                    raise AssertionError("four-route budget exceeded before solver call")

                def on_leg(index, leg, details, directory=directory, record=record):
                    path = directory / f"leg-{index:02d}.json"
                    if leg.solution is not None:
                        array_path = directory / f"leg-{index:02d}-solution.npz"
                        np.savez_compressed(
                            array_path,
                            node_epochs=leg.solution.node_epochs_mjd,
                            states_scaled=leg.solution.states_scaled,
                            thrust_n=leg.solution.thrust_n,
                            departure_vinf=leg.solution.departure_vinf_km_s,
                            arrival_vinf=leg.solution.arrival_vinf_km_s,
                        )
                        details["solution_arrays"] = {
                            "path": str(array_path),
                            "sha256": common.sha(array_path),
                        }
                    common.write(path, details)
                    record["legs"].append(
                        {
                            "leg": index,
                            "certified": leg.certified,
                            "details": str(path),
                            "status": details["status"],
                            "initial_mass_kg": leg.mass_before,
                            "certified_mass_after_kg": leg.mass_after_leg,
                        }
                    )
                    save()

                refined = refine_fixed(
                    route_plan,
                    catalogue,
                    cargo,
                    settings=settings,
                    on_leg=on_leg,
                    deadline=deadline,
                )
                common.write(directory / "refinement.json", refined.summary())
                if (
                    json.loads(json.dumps(refined.plan.summary())) != case["plan"]
                    or refined.collected_mass != cargo
                ):
                    raise AssertionError("prescribed epochs/cargo changed")
                record.update(
                    certified=refined.certified,
                    failures=refined.failures,
                    refinement_seconds=refined.wall_seconds,
                    completed_legs=len(refined.legs),
                    requested_legs=case["maximum_leg_calls"],
                    cargo_scope="prescribed_only_until_both_full_fleet_checks_pass",
                )
                attempted_last = max(
                    [len(refined.legs) - 1]
                    + [
                        int(item["leg"])
                        for item in refined.failures
                        if "leg" in item and item["status"] != "campaign_deadline_before_leg"
                    ]
                )
                record["unattempted_leg_indices"] = list(
                    range(attempted_last + 1, case["maximum_leg_calls"])
                )
                if not refined.certified:
                    record["status"] = "fixed_refinement_failed"
                    record["proxy_interpretation_allowed"] = case[
                        "kind"
                    ] == "probe" and controls.get(case["requires_control"], False)
                    if case["kind"] == "control":
                        controls[case["id"]] = False
                    save()
                    continue
                measured_propellant = sum(
                    leg.mass_before - leg.mass_after_leg for leg in refined.legs
                )
                record.update(
                    certified_propellant_kg=measured_propellant,
                    certified_final_dry_mass_kg=refined.final_mass_kg,
                    solver_reported_propellant_kg=sum(
                        leg.solution.propellant_kg for leg in refined.legs
                    ),
                )
                if case["proxy_prediction"] is not None:
                    predicted = case["proxy_prediction"]["mass_projection"]["propellant_kg"]
                    record.update(
                        projected_propellant_kg=predicted,
                        projected_minus_certified_propellant_kg=predicted - measured_propellant,
                    )
                write_route_artifacts(refined, catalogue, directory / "route")
                replacement = emit_solution(refined, catalogue, ship_id=case["ship"]).ships[0]
                trial = Solution(
                    [
                        replacement if ship.ship_id == case["ship"] else ship
                        for ship in baseline_solution.ships
                    ]
                )
                path = directory / "fleet-Result.txt"
                trial.write(path)
                report["stage"] = "full_fleet_verification"
                save()
                checked, history = verify(path)
                record["verification"] = checked
                common.write(directory / "verification.json", checked)
                expected_raw = baseline["total_mass_kg"] + case["raw_gain_kg"]
                expected_score = baseline["score_kg"] + case["weighted_gain_kg"]
                prescribed_verified = (
                    common.checker_pass(checked)
                    and abs(checked["total_mass_kg"] - expected_raw) <= 1e-7
                    and abs(checked["score_kg"] - expected_score) <= 1e-7
                )
                record["prescribed_cargo_verified"] = prescribed_verified
                if case["kind"] == "control":
                    controls[case["id"]] = prescribed_verified
                    record["status"] = (
                        "control_passed"
                        if prescribed_verified
                        else "control_full_fleet_check_failed"
                    )
                    record["proxy_interpretation_allowed"] = False
                else:
                    record["proxy_interpretation_allowed"] = controls.get(
                        case["requires_control"], False
                    )
                    record["status"] = (
                        "probe_certified"
                        if prescribed_verified
                        else "probe_full_fleet_check_failed"
                    )
                    if prescribed_verified and common.can_promote(
                        checked,
                        report["best"]["score_kg"],
                        controls.get(case["requires_control"], False),
                    ):
                        best_dir = out / "best" / f"promotion-{len(report['promotions']) + 1:02d}"
                        best_dir.mkdir(parents=True)
                        destination = best_dir / "Result.txt"
                        shutil.copy2(path, destination)
                        report["best"] = checked | {
                            "source": case["id"],
                            "solution": str(destination),
                            "solution_sha256": common.sha(destination),
                        }
                        report["promotions"].append(dict(report["best"]))
                        record["status"] = "promoted_verified_fixed_cargo_probe"
                        save()
                        try:
                            report["best"]["viewer"] = write_viewer_dataset(
                                best_dir / "viewer",
                                trial,
                                history,
                                catalogue,
                                run_id="truth_set_v606",
                                commit=preparation["source_commit"],
                                verification=checked["independent"],
                                solution_path=destination,
                            )
                        except Exception as error:
                            report["best"]["viewer_export_error"] = repr(error)
                save()
    if common.sha(common.ROOT / "inputs/Result.txt") != common.INCUMBENT_SHA:
        raise AssertionError("retained fleet mutated")
    if (
        report["native_solves_started"] > plan["maximum_leg_calls"]
        or report["whole_route_refinements_started"] > 4
    ):
        raise AssertionError("native leg-solve budget exceeded")
    control_failed = any(
        not controls.get(case["id"], False) for case in plan["cases"] if case["kind"] == "control"
    )
    status = (
        "improved"
        if report["promotions"]
        else ("control_failure_no_proxy_conclusion" if control_failed else "incumbent_retained")
    )
    report.update(
        complete=True,
        stage="complete",
        status=status,
        seconds=time.perf_counter() - started,
        control_outcomes=controls,
        soft_wall_budget_overrun_seconds=max(0.0, time.perf_counter() - deadline),
    )
    save()
    print(
        {
            "status": report["status"],
            "controls": controls,
            "native_calls": report["native_solves_completed"],
            "output": str(out),
        }
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("local", "h100"), required=True)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 3600:
        parser.error("bounded wall budget of 0..3600 seconds required")
    try:
        raise SystemExit(main(args))
    except Exception as error:
        if getattr(args, "created_output", False):
            path = args.output.resolve() / "report.json"
            report = common.read(path) if path.exists() else {}
            report.update(
                complete=True,
                status="failed_after_verified_promotion"
                if report.get("promotions")
                else "failed_incumbent_preserved",
                error=repr(error),
            )
            common.write(path, report)
        raise
