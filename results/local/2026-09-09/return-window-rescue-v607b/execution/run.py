"""One original-prefix control and at most four return-only candidate solves."""

import argparse
import dataclasses
import math
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import support


@contextmanager
def observe_native(out, report):
    from spacepdhcg.gtoc12 import gpu_scvx

    original = gpu_scvx.solve_native

    def observed(boundary, *args, **kwargs):
        if report["native_returns_started"] >= 5:
            raise RuntimeError("five native return solves are the global hard limit")
        report["native_returns_started"] += 1
        row = {
            "number": report["native_returns_started"],
            "case": report["active_case"],
            "departure": boundary.departure_epoch,
            "arrival": boundary.arrival_epoch,
            "initial_mass_kg": boundary.initial_mass,
            "minimum_final_mass_kg": boundary.minimum_final_mass,
        }
        started = time.perf_counter()
        try:
            result = original(boundary, *args, **kwargs)
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
            import json

            row["wrapper_wall_seconds"] = time.perf_counter() - started
            report["native_returns_completed"] += 1
            with (out / "native-returns.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    gpu_scvx.solve_native = observed
    try:
        yield
    finally:
        gpu_scvx.solve_native = original


def main(args):
    import fcntl

    support.activate()
    support.validate_ready()
    profile = support.validate_runtime()
    if args.lock.resolve() != Path(profile["lock"]).resolve():
        raise ValueError("selected shared GPU lock required")
    from prefix import assert_prefix_emission, load_prefix, p, solve_return
    from screen import screen_windows

    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
    from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset

    catalogue, bonus = load_catalogue(), load_bonus_table()
    if (
        catalogue.source_sha256
        != "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675"
        or bonus.source_sha256 != "e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21"
    ):
        raise ValueError("catalogue or fixed-bonus hash changed")
    if not official_verifier_available():
        raise ValueError("both full-fleet verifiers are mandatory")
    preparation = support.read(support.ROOT / "preparation.json")
    settings = p.ScvxSettings(**preparation["scvx_settings"])
    plan = support.read(support.ROOT / "inputs/v607-plan.json")
    baseline_solution = Solution.read(support.ROOT / "inputs/Result.txt")
    if baseline_solution.ship_count != 23:
        raise ValueError("exact 23-ship baseline required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.created_output = True
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    report = {
        "pid": os.getpid(),
        "complete": False,
        "status": "running",
        "stage": "prepared",
        "source_commit": preparation["source_commit"],
        "source_manifest_sha256": support.sha(support.ROOT / "source-sha256.json"),
        "ready_manifest_sha256": support.sha(support.ROOT / "ready-manifest.json"),
        "driver_sha256": support.sha(__file__),
        "prefix_importer_sha256": support.sha(support.ROOT / "prefix.py"),
        "native_libraries": profile["native_libraries"],
        "scvx_settings": dataclasses.asdict(settings),
        "GPU_execution": "graph",
        "profile": "local",
        "whole_route_reruns": 0,
        "native_returns_started": 0,
        "native_returns_completed": 0,
        "maximum_native_returns": 5,
        "maximum_candidate_returns": 4,
        "cases": [],
        "promotions": [],
        "best": None,
        "cargo_scaling_allowed": False,
        "prefix_retiming_allowed": False,
        "score_kind": "fixed_bonus_weighted_kg",
        "control_passed": False,
    }

    def save():
        support.write(out / "report.json", report)

    def verify(path):
        before = time.perf_counter()
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
            "total_mass_kg": independent.total_mass_kg,
            "score_kind": "fixed_bonus_weighted_kg",
            "verification_seconds": time.perf_counter() - before,
        }
        return checked, history

    def attempt(prefix, candidate, kind, number):
        identifier = "control_return" if kind == "control" else f"candidate-{number:02d}"
        directory = out / "cases" / identifier
        directory.mkdir(parents=True, exist_ok=False)
        row = {
            "id": identifier,
            "kind": kind,
            "window": candidate,
            "cargo_kg": dict(prefix.cargo),
            "prefix_fingerprint": prefix.fingerprint(),
            "reused_prefix_legs": 16,
            "status": "refining",
        }
        report["cases"].append(row)
        report.update(stage="native_return_refinement", active_case=identifier)
        support.write(directory / "prescription.json", row)
        before = time.perf_counter()
        calls_before = report["native_returns_started"]
        save()

        def on_result(item, detail):
            if item.solution is not None:
                arrays = directory / "return-solution.npz"
                np.savez_compressed(
                    arrays,
                    node_epochs=item.solution.node_epochs_mjd,
                    thrust_n=item.solution.thrust_n,
                    states_scaled=item.solution.states_scaled,
                    departure_vinf=item.solution.departure_vinf_km_s,
                    arrival_vinf=item.solution.arrival_vinf_km_s,
                )
                detail["solution_arrays"] = {"path": str(arrays), "sha256": support.sha(arrays)}
            support.write(directory / "return.json", detail)

        route, detail = solve_return(
            prefix,
            catalogue,
            candidate["departure"],
            candidate["arrival"],
            settings,
            delta_v=candidate.get("lambert_delta_v_km_s", 0),
            on_result=on_result,
        )
        support.write(directory / "outcome.json", detail)
        row.update(
            status=detail["status"],
            refinement_seconds=time.perf_counter() - before,
            native_return_calls=report["native_returns_started"] - calls_before,
            outcome_sha256=support.sha(directory / "outcome.json"),
        )
        if route is None:
            save()
            return False
        if row["native_return_calls"] != 1:
            raise AssertionError("qualified return requires exactly one fresh native solve")
        solution = p.emit_solution(route, catalogue, ship_id=7)
        row["prefix_emission_check"] = assert_prefix_emission(prefix, solution)
        support.write(directory / "route-summary.json", route.summary())
        solution.write(directory / "ship7-Result.txt")
        replacement = solution.ships[0]
        fleet = Solution(
            [replacement if ship.ship_id == 7 else ship for ship in baseline_solution.ships]
        )
        path = directory / "fleet-Result.txt"
        fleet.write(path)
        report["stage"] = "full_fleet_verification"
        save()
        checked, history = verify(path)
        expected = (
            report["baseline"]
            if kind == "control"
            else {
                "total_mass_kg": plan["expected_certified_fleet"]["raw_kg"],
                "score_kg": plan["expected_certified_fleet"]["weighted_fixed_bonus_kg"],
            }
        )
        qualified = support.qualifies(checked, expected["total_mass_kg"], expected["score_kg"])
        checked["prescribed_cargo_verified"] = qualified
        support.write(directory / "verification.json", checked)
        row.update(
            verification=checked,
            final_dry_mass_kg=route.final_mass_kg,
            status="control_passed"
            if kind == "control" and qualified
            else ("candidate_fleet_verified" if qualified else "full_fleet_check_failed"),
        )
        if kind == "control":
            report["control_passed"] = qualified
        elif qualified and support.better(
            checked, report["best"], route.final_mass_kg, report["control_passed"]
        ):
            destination = out / "best" / f"promotion-{len(report['promotions']) + 1:02d}"
            destination.mkdir(parents=True, exist_ok=False)
            retained = destination / "Result.txt"
            shutil.copy2(path, retained)
            report["best"] = checked | {
                "source": identifier,
                "solution": str(retained),
                "solution_sha256": support.sha(retained),
                "ship7_final_dry_mass_kg": route.final_mass_kg,
            }
            report["promotions"].append(dict(report["best"]))
            row["status"] = "promoted_verified_fixed_cargo_return"
            save()
            try:
                report["best"]["viewer"] = write_viewer_dataset(
                    destination / "viewer",
                    fleet,
                    history,
                    catalogue,
                    run_id="return_window_rescue_v607b",
                    commit=preparation["source_commit"],
                    verification=checked["independent"],
                    solution_path=retained,
                )
            except Exception as error:
                report["best"]["viewer_export_error"] = repr(error)
        save()
        return qualified

    save()
    if args.lock_fd is None:
        handle = args.lock.open("a+")
    else:
        passed, actual = os.fstat(args.lock_fd), args.lock.stat()
        if (passed.st_dev, passed.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("incorrect supervisor lock descriptor")
        handle = os.fdopen(os.dup(args.lock_fd), "a+")
    with handle as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report.update(stage="baseline_full_fleet_verification", gpu_lock_acquired=True)
        save()
        checked, _ = verify(support.ROOT / "inputs/Result.txt")
        old = support.read(support.ROOT / "inputs/v606-report.json")["baseline"]
        if not support.qualifies(checked, old["total_mass_kg"], old["score_kg"]):
            raise ValueError("retained baseline failed fresh verification")
        report["baseline"] = checked
        report["best"] = checked | {
            "source": "retained_v595",
            "solution": str(support.ROOT / "inputs/Result.txt"),
            "solution_sha256": support.sha(support.ROOT / "inputs/Result.txt"),
        }
        report["stage"] = "fresh_CPU_prefix_certification"
        save()
        prefixes = {name: load_prefix(name, catalogue) for name in ("control", "probe")}
        for name, prefix in prefixes.items():
            support.write(
                out / "prefix" / (name + ".json"),
                prefix.audit | {"fingerprint": prefix.fingerprint()},
            )
        execution = argparse.Namespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
        with using_gpu_execution(execution), observe_native(out, report):
            if time.perf_counter() >= deadline:
                report["stop_reason"] = "deadline_before_control"
            elif attempt(
                prefixes["control"], {"departure": 69218.0, "arrival": 69728.0}, "control", 0
            ):
                if time.perf_counter() < deadline:
                    report["stage"] = "GPU_return_window_screening"
                    save()
                    screening = screen_windows(prefixes["probe"], catalogue, out / "screening")
                    report["screening"] = screening
                    save()
                    for number, candidate in enumerate(screening["selected"], 1):
                        if number > 4:
                            raise AssertionError("candidate refinement budget exceeded")
                        if time.perf_counter() >= deadline:
                            report.setdefault("not_attempted_selected", []).append(candidate)
                            continue
                        attempt(prefixes["probe"], candidate, "probe", number)
                else:
                    report["stop_reason"] = "deadline_after_control"
            else:
                report["stop_reason"] = "control_failed_candidates_not_screened_or_refined"
    support.activate()
    report.update(
        complete=True,
        stage="complete",
        seconds=time.perf_counter() - started,
        status="improved"
        if report["promotions"]
        else (
            "incumbent_retained"
            if report["control_passed"]
            else "control_failed_no_probe_conclusion"
        ),
        soft_wall_budget_overrun_seconds=max(0.0, time.perf_counter() - deadline),
    )
    if (
        report["native_returns_started"] > 5
        or report["native_returns_started"] != report["native_returns_completed"]
    ):
        raise AssertionError("native return accounting mismatch")
    save()
    print(
        {
            "status": report["status"],
            "native_returns": report["native_returns_completed"],
            "output": str(out),
        }
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1800:
        parser.error("wall budget must be positive and at most 1800 seconds")
    try:
        raise SystemExit(main(args))
    except Exception as error:
        if getattr(args, "created_output", False):
            path = args.output.resolve() / "report.json"
            result = support.read(path) if path.exists() else {}
            result.update(
                complete=True,
                error=repr(error),
                status="failed_after_verified_promotion"
                if result.get("promotions")
                else "failed_incumbent_preserved",
            )
            support.write(path, result)
        raise
