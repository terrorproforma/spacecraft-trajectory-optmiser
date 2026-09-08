"""Bounded asteroid substitutions in the retained v595 fleet; preparation is CPU-only.

Run with --plan-only to freeze a deterministic search inventory. CUDA requires
--execute, pinned libraries and the shared lock. Only complete fleets accepted
by both checkers with a strictly greater fixed-bonus score can be promoted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import incident

ROOT = Path(__file__).resolve().parent
COMMIT = "3091c716714c8bdec364d54c5e7357f2b5d85730"
INCUMBENT_SHA = "33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8"
CORE_SHA = "86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671"
QOCO_SHA = "0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315"
DATA_SHA = "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675"
BONUS_SHA = "e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21"
SHIP_COUNT = 23
MIN_FLEET_RAW_KG = SHIP_COUNT * math.log(SHIP_COUNT / 2.0) / 0.004


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def finite(item):
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(finite(value), indent=2, default=float, allow_nan=False) + "\n")
    temp.replace(path)


def activate_source(source=ROOT / "source"):
    for name, expected in json.loads((ROOT / "source-sha256.json").read_text()).items():
        if sha256(source / name) != expected:
            raise ValueError(f"published source mismatch: {name}")
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(source / "src"))


def load_inputs(inputs=ROOT / "inputs"):
    manifest = json.loads((inputs / "sha256.json").read_text())
    for name, expected in manifest.items():
        if sha256(inputs / name) != expected:
            raise ValueError(f"input mismatch: {name}")
    if sha256(inputs / "Result.txt") != INCUMBENT_SHA:
        raise ValueError("exact retained v595 input required")
    audit = json.loads((inputs / "fresh-verification.json").read_text())
    if (
        audit["solution_sha256"] != INCUMBENT_SHA
        or not audit["independent"]["ok"]
        or not audit["official"]["ok"]
        or audit["bonus_sha256"] != BONUS_SHA
    ):
        raise ValueError("input is not accompanied by its accepted independent scoring audit")
    summaries = {i: json.loads((inputs / f"ship-{i:02d}.json").read_text()) for i in range(1, 24)}
    return summaries, audit, manifest


def inventory(summaries, audit, weights, solution):
    """Attribute the independent verifier's material to actual ship event inventories."""
    if len(summaries) != SHIP_COUNT or solution.ship_count != SHIP_COUNT:
        raise ValueError("the retained fleet must contain exactly 23 ships")
    masses = {int(a): float(v) for a, v in audit["scored_masses"].items()}
    footprints, rows, seen = {}, [], set()
    for ship in solution.ships:
        summary = summaries[ship.ship_id]
        if not summary["certified"]:
            raise ValueError("uncertified incumbent summary")
        plan = summary["plan"]
        deploy = {int(a): float(t) for a, t in plan["deploy_epochs"].items()}
        collect = {int(a): float(t) for a, t in plan["collect_epochs"].items()}
        expected = Counter((a, t) for phase in (deploy, collect) for a, t in phase.items())
        actual = Counter((e.event_id, e.before.epoch) for e in ship.asteroid_visits())
        if actual != expected:
            raise ValueError(f"ship {ship.ship_id} summary does not describe Result.txt")
        footprint = set(deploy) | set(collect)
        if footprint & seen:
            raise ValueError("this pilot requires independent, disjoint incumbent ship footprints")
        seen.update(footprint)
        footprints[ship.ship_id] = footprint
        if (
            not 0 < len(deploy) <= 20
            or plan.get("foreign_deploy_epochs")
            or not set(collect) <= set(deploy)
        ):
            raise ValueError("unsupported miner count or cooperative incumbent")
        if any(collect[a] - deploy[a] < 365.25 - 1e-6 for a in collect):
            raise ValueError("incumbent mining stay is below one year")
        own = {int(a): float(m) for a, m in summary["collected_mass_kg"].items()}
        if set(own) != set(collect) or any(abs(own[a] - masses[a]) > 1e-7 for a in own):
            raise ValueError("route material disagrees with independent scored masses")
        if any(a not in weights or not math.isfinite(weights[a]) or weights[a] <= 0 for a in own):
            raise ValueError("fixed-bonus coefficients are mandatory and finite")
        raw = sum(masses[a] for a in collect)
        weighted = sum(masses[a] * weights[a] for a in collect)
        first, last = min(deploy, key=deploy.get), max(collect, key=collect.get)
        rows.append(
            {
                "ship": ship.ship_id,
                "verified_raw_kg": raw,
                "verified_weighted_kg": weighted,
                "weighted_to_raw_ratio": weighted / raw,
                "deploy_count": len(deploy),
                "collect_count": len(collect),
                "fixed_earth_departure_target": first,
                "fixed_earth_return_source": last,
                "replaceable": sorted((set(deploy) & set(collect)) - {first, last}),
            }
        )
    if abs(sum(r["verified_raw_kg"] for r in rows) - audit["independent"]["total_mass_kg"]) > 1e-7:
        raise ValueError("attributed raw total mismatch")
    if (
        abs(
            sum(r["verified_weighted_kg"] for r in rows)
            - audit["independent"]["weighted_score_fixed_bonus_kg"]
        )
        > 1e-7
    ):
        raise ValueError("attributed weighted total mismatch")
    return rows, footprints


def replacement_visits(visits, old, new, excluded):
    """Replace both visits, or one combined camp, without transferring flight costs."""
    bodies = {v.body for v in visits}
    matching = [v for v in visits if v.body == old]
    if (
        old == new
        or new <= 0
        or new in excluded
        or new in bodies
        or not matching
        or old in (visits[1].body, visits[-2].body)
    ):
        raise ValueError("replacement violates footprint or fixed Earth endpoints")
    if sum(v.deploy for v in matching) != 1 or sum(v.collect for v in matching) != 1:
        raise ValueError("replacement needs exactly one own deployment and collection")
    if any(v.foreign_deploy_epoch is not None or v.pinned_arrival is not None for v in visits):
        raise ValueError("cooperative or pinned visit is outside this pilot")
    result = [dataclasses.replace(v, body=new) if v.body == old else v for v in visits]
    if Counter((v.deploy, v.collect, v.role_out) for v in result) != Counter(
        (v.deploy, v.collect, v.role_out) for v in visits
    ):
        raise ValueError("replacement changed mining actions")
    return result


epoch_seeds = incident.epoch_seeds


def objective_gate(raw, weighted, before_raw, before_weighted, fleet_raw, *, minimum_gain=0.5):
    values = (raw, weighted, before_raw, before_weighted, fleet_raw, minimum_gain)
    return bool(
        all(math.isfinite(v) for v in values)
        and minimum_gain >= 0
        and weighted > before_weighted + minimum_gain
        and fleet_raw - before_raw + raw >= MIN_FLEET_RAW_KG + 1e-6
    )


def verified_promotion(checked, incumbent_score):
    """A physical proxy or failed checker can never become a fleet score."""
    score = checked.get("score_kg")
    raw = checked.get("total_mass_kg")
    return bool(
        checked.get("ok")
        and checked.get("independent", {}).get("ok")
        and checked.get("official", {}).get("ok")
        and score is not None
        and raw is not None
        and math.isfinite(score)
        and math.isfinite(raw)
        and score > incumbent_score + 1e-8
        and raw >= MIN_FLEET_RAW_KG - 1e-8
    )


def publish_verified_candidate(out, solution_path, checked, case, report, export_viewer):
    """Keep immutable promotion artifacts; a viewer error cannot corrupt the score checkpoint."""
    if not verified_promotion(checked, report["best"]["score_kg"]):
        return False
    destination = out / "best" / f"promotion_{len(report['promotions']) + 1:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    result_path = destination / "Result.txt"
    shutil.copy2(solution_path, result_path)
    report["best"] = checked | {
        "solution": str(result_path),
        "solution_sha256": sha256(result_path),
        "source": "verified_single_substitution",
        "case": case,
    }
    report["promotions"].append({"case": case, "solution": str(result_path), **checked})
    write_json(out / "report.json", report)
    try:
        report["best"]["viewer"] = export_viewer(destination)
    except Exception as error:
        report["best"]["viewer_export_error"] = repr(error)
    write_json(out / "report.json", report)
    return True


def make_plan(summaries, audit, weights, catalogue, solution):
    from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
    from spacepdhcg.gtoc12.retiming import visits_of

    inventory(summaries, audit, weights, solution)
    reference = ROOT / "reference/v599-plan.json"
    if sha256(reference) != "c58c60f081b389d37a95bdec79cf406ec98a363197d1ba1d3de2cc459675dff2":
        raise ValueError("exact 496-case v599 inventory required")
    history = ROOT / "reference/v599-report.json"
    if sha256(history) != "0f8e3d2078ad001c446ca80491cfd506bbcb261d7261a21e47e583370cee29d2":
        raise ValueError("historical control changed")
    plan = json.loads(reference.read_text())
    if catalogue.source_sha256 != DATA_SHA:
        raise ValueError("catalogue pin mismatch")
    cases = plan["replacement_cases"]
    occupied = set(plan["occupied_asteroids"])
    count, structures = 0, []
    max_polish = 0
    for selected in plan["selected_ships"]:
        route = route_from_summary(summaries[selected["ship"]])
        visits, arr, dep = visits_of(route.plan)
        max_polish = max(max_polish, 1 + 4 * len(list(JointItinerary.moves(len(visits), 8))))
        for case in cases:
            if case["ship"] != selected["ship"]:
                continue
            new_visits = replacement_visits(visits, case["old"], case["new"], occupied)
            seeds = epoch_seeds(new_visits, arr, dep, case["new"])
            indices = incident.incident_indices(new_visits, case["new"])
            count += len(seeds)
            structures.append({"case": case["case"], "seeds": len(seeds), "incident_legs": indices})
    if len(cases) != 496 or count > 12400 or 4 * max_polish > incident.MAX_POLISH_ROWS:
        raise ValueError("prepared search exceeds approved candidate budget")
    controls = [r["case"] for r in json.loads(history.read_text())["retimings"]]
    incident.choose_arms(cases, {}, controls)
    plan.update(
        kind="CPU_prepared_controlled_incident_window_search_not_physics_certificates",
        reference_case_plan_sha256=sha256(reference),
        reference_run_report_sha256=sha256(history),
        epoch_shift_days=list(incident.SHIFTS_DAYS),
        construction=structures,
        control_case_ids=controls,
        maximum_initial_epoch_candidates=count,
        maximum_total_joint_candidates=count + 4 * max_polish,
        maximum_retime_calls=24,
        maximum_retime_calls_per_arm=12,
        maximum_polish_seeds=4,
        maximum_full_refinements=4,
        comparison_scope=(
            "Shared expanded grid; twelve historical-priority versus twelve "
            "complete-incidence choices, fresh equivalent retimer states. "
            "Grid expansion versus v599 is a separate changed budget."
        ),
    )
    return plan


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--proxy-seconds", type=float, default=180)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--max-refinements", type=int, default=4)
    parser.add_argument("--lock", type=Path, default=Path.home() / ".spacepdhcg-gpu.lock")
    args = parser.parse_args()
    if (
        not 0 < args.proxy_seconds <= 180
        or not 0 < args.wall_seconds <= 3600
        or not 0 <= args.max_refinements <= 4
    ):
        parser.error("proxy budget 0..180s, total budget 0..3600s, refinements 0..4 required")
    return args


def main(args):
    activate_source()
    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
    from spacepdhcg.gtoc12.solution import Solution

    summaries, audit, manifest = load_inputs()
    catalogue, bonus = load_catalogue(), load_bonus_table()
    if catalogue.source_sha256 != DATA_SHA or bonus.source_sha256 != BONUS_SHA:
        raise ValueError("catalogue and bonus pins changed")
    weights = {int(a): float(bonus.for_asteroid(int(a))) for a in catalogue.ids}
    baseline = Solution.read(ROOT / "inputs/Result.txt")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    if args.plan_only:
        plan = make_plan(summaries, audit, weights, catalogue, baseline)
        write_json(out / "plan.json", plan)
        write_json(
            out / "report.json",
            {
                "status": "plan_only_no_GPU",
                "complete": True,
                "driver_sha256": sha256(__file__),
                "input_sha256": manifest,
                "plan_sha256": sha256(out / "plan.json"),
                "gpu_calls": 0,
            },
        )
        print(
            json.dumps(
                {
                    "status": "plan_only_no_GPU",
                    "cases": len(plan["replacement_cases"]),
                    "selected_ships": [r["ship"] for r in plan["selected_ships"]],
                    "output": str(out),
                }
            )
        )
        return 0
    return execute(args, out, summaries, audit, manifest, weights, catalogue, bonus, baseline)


def assert_candidate_plan(plan, original, case, occupied):
    from spacepdhcg.gtoc12.retiming import orders_of

    old, new = case["old"], case["new"]
    expected = tuple([new if a == old else a for a in order] for order in orders_of(original))
    if orders_of(plan) != expected or plan.foreign_deploy_epochs or new in occupied:
        raise ValueError("candidate changed a protected visit or overlaps another ship")
    if (
        len(plan.deploy_epochs) != len(original.deploy_epochs)
        or len(plan.collect_epochs) != len(original.collect_epochs)
        or set(plan.orphaned) != set(original.orphaned)
    ):
        raise ValueError("candidate changed miner inventory")
    if any(
        plan.collect_epochs[a] - plan.deploy_epochs[a] < 365.25 - 1e-6 for a in plan.collect_epochs
    ):
        raise ValueError("candidate mining interval is too short")
    if plan.legs[0].departure_epoch < 64328 or plan.legs[-1].arrival_epoch > 69807:
        raise ValueError("candidate is outside the mission window")


@contextmanager
def native_progress(out, report):
    from spacepdhcg.gtoc12 import gpu_scvx

    original = gpu_scvx.solve_native

    def measured(*args, **kwargs):
        number = report["native_solves_started"] + 1
        report["native_solves_started"] = number
        began = time.perf_counter()
        row = {
            "solve": number,
            "departure": args[0].departure_epoch,
            "arrival": args[0].arrival_epoch,
            "initial_mass_kg": args[0].initial_mass,
        }
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
                stream.write(json.dumps(row) + "\n")

    gpu_scvx.solve_native = measured
    try:
        yield
    finally:
        gpu_scvx.solve_native = original


def execute(args, out, summaries, audit, manifest, weights, catalogue, bonus, baseline_solution):
    import fcntl

    import numpy as np

    from spacepdhcg.gtoc12 import constants as C
    from spacepdhcg.gtoc12.bundles import (
        ClusterPricingSettings,
        cluster_retime_settings,
        cluster_search_settings,
        profile_for_orders,
    )
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.gpu_joint import evaluate_joint
    from spacepdhcg.gtoc12.jointopt import (
        JointItinerary,
        JointSettings,
        route_from_summary,
        weighted_mass,
    )
    from spacepdhcg.gtoc12.lambert import using_lambert_backend
    from spacepdhcg.gtoc12.low_thrust import ScvxSettings
    from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
    from spacepdhcg.gtoc12.pipeline import emit_solution, refine_route, write_route_artifacts
    from spacepdhcg.gtoc12.retiming import Retimer, orders_of, visits_of, weighted_collected
    from spacepdhcg.gtoc12.search import SearchSettings
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
    from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset

    libraries = {}
    for name, expected in (
        ("SPACEPDHCG_GTOC12_CUDA_LIBRARY", CORE_SHA),
        ("SPACEPDHCG_QOCO_LIBRARY", QOCO_SHA),
    ):
        library = Path(os.environ.get(name, ""))
        if not library.is_file() or sha256(library) != expected:
            raise ValueError(f"{name} is not the explicitly validated v596/core or QOCO540 binary")
        libraries[name] = {"path": str(library), "sha256": expected}
    for name in (
        "SPACEPDHCG_TEST_GTOC12_JOINT_BATCH",
        "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION",
    ):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name}=1 required for this recorded GPU candidate configuration")
    if not official_verifier_available():
        raise ValueError("the independent and official full-fleet checkers are both required")
    plan_path = ROOT / "plan/plan.json"
    prepared = json.loads((ROOT / "plan/report.json").read_text())
    if sha256(plan_path) != prepared["plan_sha256"]:
        raise ValueError("prepared candidate inventory changed")
    plan = json.loads(plan_path.read_text())
    if plan["source_commit"] != COMMIT or plan["incumbent_sha256"] != INCUMBENT_SHA:
        raise ValueError("candidate plan provenance mismatch")
    inventory(summaries, audit, weights, baseline_solution)
    occupied = set(plan["occupied_asteroids"])
    pool = np.asarray(sorted(set(map(int, catalogue.ids)) - occupied), dtype=np.int64)
    pool.setflags(write=False)
    settings = SearchSettings(**plan["neighbor_settings"])
    cases = plan["replacement_cases"]
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
        "pid": os.getpid(),
        "complete": False,
        "stage": "prepared",
        "status": "running",
        "source_commit": COMMIT,
        "source_manifest_sha256": sha256(ROOT / "source-sha256.json"),
        "driver_sha256": sha256(__file__),
        "input_sha256": manifest,
        "candidate_plan_sha256": sha256(plan_path),
        "native_libraries": libraries,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "feature_flags": {k: v for k, v in os.environ.items() if k.startswith("SPACEPDHCG_TEST_")},
        "catalogue_sha256": DATA_SHA,
        "bonus_sha256": BONUS_SHA,
        "scvx_settings": dataclasses.asdict(scvx),
        "physics_tolerances_changed": False,
        "tolerances": {
            "position_km": C.TOLERANCE_POSITION_KM,
            "velocity_km_s": C.TOLERANCE_VELOCITY_KM_S,
            "mass_kg": C.TOLERANCE_MASS_KG,
        },
        "maximum_full_refinements": args.max_refinements,
        "maximum_joint_candidates": plan["maximum_total_joint_candidates"],
        "native_solves_started": 0,
        "native_solves_completed": 0,
        "screened_cases": [],
        "retimings": [],
        "polishings": [],
        "refinements": [],
        "promotions": [],
        "best": None,
        "gpu_neighbor_queries": [],
        "controlled_comparison": {
            "shared_grid_rows_planned": plan["maximum_initial_epoch_candidates"],
            "retimings_per_arm": 12,
            "maximum_refinements_total": 4,
            "arms": {},
            "scope": plan["comparison_scope"],
        },
        "ranking_module_sha256": sha256(ROOT / "incident.py"),
        "joint_screen_call_seconds": 0.0,
        "CPU_ranking_diagnostic_seconds": 0.0,
        "candidate_scope": (
            "Each trial changes one ship and one asteroid relative to v595; "
            "trials are not combined."
        ),
    }

    def save():
        write_json(out / "report.json", report)

    save()
    started = time.perf_counter()
    deadline = started + args.wall_seconds

    def verify(path):
        began = time.perf_counter()
        history = {}
        independent = Gtoc12Verifier(catalogue, bonus=bonus, history=history).verify_file(path)
        official = run_official_verifier(path)
        summary = independent.summary()
        value = summary.get("weighted_score_fixed_bonus_kg")
        ok = bool(independent.ok and official.ok and value is not None and math.isfinite(value))
        return {
            "ok": ok,
            "independent": summary,
            "official": official.summary(),
            "score_kg": value if ok else None,
            "total_mass_kg": independent.total_mass_kg,
            "verification_seconds": time.perf_counter() - began,
        }, history

    ship_rows = {r["ship"]: r for r in plan["all_ships"]}
    accepted_proxies = []
    screened_best = {}
    diagnostics = {}

    def remember(candidate, case, origin):
        if candidate is None or not candidate.feasible:
            return None
        state = states[case["ship"]]
        assert_candidate_plan(candidate, state["route"].plan, case, occupied)
        row = ship_rows[case["ship"]]
        raw, weighted = candidate.total_collected_kg, weighted_collected(candidate, weights)
        if not all(math.isfinite(v) for v in (raw, weighted)):
            raise ValueError("nonfinite proxy material")
        key = f"{case['case']}_{origin}"
        location = out / "proxies" / f"{key}.json"
        write_json(location, candidate.summary())
        item = {
            "case": case,
            "plan": candidate,
            "origin": origin,
            "path": str(location),
            "predicted_raw_kg": raw,
            "predicted_weighted_kg": weighted,
            "predicted_raw_gain_kg": raw - row["verified_raw_kg"],
            "predicted_weighted_gain_kg": weighted - row["verified_weighted_kg"],
            "passes_unchanged_native_forward": True,
        }
        eligible = objective_gate(
            raw,
            weighted,
            row["verified_raw_kg"],
            row["verified_weighted_kg"],
            report["baseline"]["total_mass_kg"],
        )
        item["eligible_for_refinement"] = eligible
        if eligible:
            accepted_proxies.append(item)
        previous = screened_best.get(case["case"])
        if (
            previous is None
            or item["predicted_weighted_gain_kg"] > previous["predicted_weighted_gain_kg"]
        ):
            screened_best[case["case"]] = item
        return {k: v for k, v in item.items() if k not in ("plan", "case")}

    def candidate_rank(item):
        return (
            -item["predicted_weighted_gain_kg"],
            -item["predicted_raw_gain_kg"],
            item["case"]["case"],
            item["origin"],
        )

    with args.lock.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["gpu_lock_acquired"] = True
        report["stage"] = "baseline_full_fleet_verification"
        save()
        baseline, _ = verify(ROOT / "inputs/Result.txt")
        if not baseline["ok"]:
            raise ValueError(
                "retained baseline failed fresh verification; candidate work not started"
            )
        if (
            abs(baseline["score_kg"] - plan["baseline_weighted_kg"]) > 1e-7
            or abs(baseline["total_mass_kg"] - plan["baseline_raw_kg"]) > 1e-7
        ):
            raise ValueError("fresh baseline score differs from the prepared scoring fixture")
        report["baseline"] = baseline
        report["best"] = baseline | {
            "solution": str(ROOT / "inputs/Result.txt"),
            "source": "retained_v595",
        }
        save()
        execution = argparse.Namespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
        with (
            using_gpu_execution(execution),
            using_lambert_backend("cuda") as gpu,
            native_progress(out, report),
        ):
            proxy_started = time.perf_counter()
            proxy_deadline = min(deadline, proxy_started + args.proxy_seconds)
            fixed_deadline = min(proxy_deadline, proxy_started + 0.65 * args.proxy_seconds)
            states = {}
            policy = ClusterPricingSettings(
                collect_dp_inflation_fit=str(ROOT / "source/results/gtoc12/hop_inflation_fit.json")
            )
            for row in plan["selected_ships"]:
                ship = row["ship"]
                route = route_from_summary(summaries[ship])
                retimer = Retimer(
                    catalogue,
                    cluster_search_settings(policy, len(catalogue.ids)),
                    dataclasses.replace(cluster_retime_settings(policy, last=True), step_days=15),
                    weights=weights,
                )
                retimer.protect_earth_leg(route.plan)
                joint = JointItinerary(
                    catalogue,
                    retimer,
                    weights=weights,
                    settings=JointSettings(
                        insert=False, max_moves_per_mesh=2, time_budget_seconds=15
                    ),
                )
                learned = joint.learn(route)
                states[ship] = {
                    "route": route,
                    "retimer": retimer,
                    "joint": joint,
                    "visits": visits_of(route.plan),
                    "learned_legs": learned,
                }

            def fresh_state(ship):
                route = states[ship]["route"]
                retimer = Retimer(
                    catalogue,
                    cluster_search_settings(policy, len(catalogue.ids)),
                    dataclasses.replace(cluster_retime_settings(policy, last=True), step_days=15),
                    weights=weights,
                )
                retimer.protect_earth_leg(route.plan)
                joint = JointItinerary(
                    catalogue,
                    retimer,
                    weights=weights,
                    settings=JointSettings(
                        insert=False, max_moves_per_mesh=2, time_budget_seconds=15
                    ),
                )
                joint.learn(route)
                return retimer, joint

            neighbor_cache = {}
            report["stage"] = "gpu_substitution_screen"
            save()
            for case in cases:
                if time.perf_counter() >= fixed_deadline:
                    break
                state = states[case["ship"]]
                query_key = (case["ship"], case["old"])
                if query_key not in neighbor_cache:
                    matches = [
                        list(
                            map(int, gpu.neighbours(catalogue, pool, case["old"], epoch, settings))
                        )
                        for epoch in case["neighbor_query_epochs"]
                    ]
                    neighbor_cache[query_key] = set(matches[0]) | set(matches[1])
                    report["gpu_neighbor_queries"].append(
                        {
                            "ship": case["ship"],
                            "old": case["old"],
                            "epochs": case["neighbor_query_epochs"],
                            "ranked_ids": matches,
                        }
                    )
                record = {
                    "case": case["case"],
                    "ship": case["ship"],
                    "old": case["old"],
                    "new": case["new"],
                    "samples": [],
                }
                if case["new"] not in neighbor_cache[query_key]:
                    record["status"] = "CPU_prepared_candidate_outside_GPU_neighbor_union"
                else:
                    visits, arr, dep = state["visits"]
                    visits = replacement_visits(visits, case["old"], case["new"], occupied)
                    seeds = epoch_seeds(visits, arr, dep, case["new"])
                    native_began = time.perf_counter()
                    values = evaluate_joint(
                        state["joint"],
                        visits,
                        np.asarray([s[1] for s in seeds]),
                        np.asarray([s[2] for s in seeds]),
                    )
                    report["joint_screen_call_seconds"] += time.perf_counter() - native_began
                    if values is None or len(values) != len(seeds):
                        raise RuntimeError("native joint screening was unavailable or incomplete")
                    for (mode, seed_arr, seed_dep), value in zip(seeds, values, strict=True):
                        sample = {
                            "mode": mode,
                            "feasible_surrogate": value.feasible,
                            "failure": value.failure,
                        }
                        diagnostic_began = time.perf_counter()
                        diagnostic = incident.ranking_diagnostic(
                            state["joint"],
                            state["route"],
                            visits,
                            seed_arr,
                            seed_dep,
                            case,
                            weights,
                            ship_rows[case["ship"]],
                            baseline["total_mass_kg"],
                            MIN_FLEET_RAW_KG,
                        )
                        report["CPU_ranking_diagnostic_seconds"] += (
                            time.perf_counter() - diagnostic_began
                        )
                        sample["ranking_estimate"] = diagnostic
                        previous = diagnostics.get(case["case"])
                        if previous is None or incident.diagnostic_key(
                            diagnostic, case["case"], mode
                        ) < incident.diagnostic_key(previous, case["case"], previous["mode"]):
                            diagnostics[case["case"]] = diagnostic | {"mode": mode}
                        if value.feasible:
                            sample["proxy"] = remember(value.plan, case, mode)
                        record["samples"].append(sample)
                    record["status"] = "screened"
                report["screened_cases"].append(record)
                if len(report["screened_cases"]) % 25 == 0:
                    save()
            report["shared_grid"] = incident.screen_progress(
                report["screened_cases"], cases, plan["maximum_initial_epoch_candidates"]
            )
            report["all_prepared_cases_screened"] = report["shared_grid"]["complete"]
            report["fixed_screen_seconds"] = time.perf_counter() - proxy_started
            save()
            # Shared screening evidence, independently initialized arm state, and
            # equal counts. No time-to-first-failure ratio is a complete-incidence score.
            arms = incident.choose_arms(cases, diagnostics, plan["control_case_ids"])
            arm_states = {name: {ship: fresh_state(ship) for ship in states} for name in arms}
            report["controlled_comparison"]["shared_grid_complete"] = report[
                "all_prepared_cases_screened"
            ]
            for name, selected in arms.items():
                report["controlled_comparison"]["arms"][name] = {
                    "selected_cases": [c["case"] for c in selected],
                    "retimings_planned": len(selected),
                    "retimings_completed": 0,
                    "feasible_native_forward_plans": 0,
                    "objective_eligible_native_forward_plans": 0,
                    "retiming_seconds": 0.0,
                    "telemetry_delta": {},
                }
            report["stage"] = "gpu_substitution_retiming"
            save()
            # Alternate which arm goes first per slot; counts and complete state
            # remain visible if the shared soft time budget is reached.
            for slot in range(12):
                names = list(arms) if slot % 2 == 0 else list(reversed(arms))
                for name in names:
                    if time.perf_counter() >= proxy_deadline:
                        break
                    case = arms[name][slot]
                    began = time.perf_counter()
                    telemetry_before = dict(gpu.telemetry)
                    original = states[case["ship"]]["route"].plan
                    retimer, _arm_joint = arm_states[name][case["ship"]]
                    deploy, collect = (
                        [case["new"] if a == case["old"] else a for a in order]
                        for order in orders_of(original)
                    )
                    profile = profile_for_orders(original, retimer, deploy, collect, None)
                    result = retimer.retime_order(
                        deploy,
                        collect,
                        profile,
                        before=ship_rows[case["ship"]]["verified_weighted_kg"],
                        original=original,
                    )
                    record = {
                        "arm": name,
                        "slot": slot,
                        "case": case["case"],
                        "result": result.summary(),
                        "seconds": time.perf_counter() - began,
                        "ranking_estimate": diagnostics.get(case["case"]),
                        "telemetry_delta": {
                            k: v - telemetry_before.get(k, 0)
                            for k, v in gpu.telemetry.items()
                            if isinstance(v, (int, float))
                            and isinstance(telemetry_before.get(k, 0), (int, float))
                        },
                    }
                    arm = report["controlled_comparison"]["arms"][name]
                    arm["retimings_completed"] += 1
                    arm["retiming_seconds"] += record["seconds"]
                    for key, value in record["telemetry_delta"].items():
                        arm["telemetry_delta"][key] = arm["telemetry_delta"].get(key, 0) + value
                    if result.plan is not None:
                        record["proxy"] = remember(result.plan, case, f"{name}_retimed")
                        arm["feasible_native_forward_plans"] += 1
                        arm["objective_eligible_native_forward_plans"] += int(
                            record["proxy"]["eligible_for_refinement"]
                        )
                    report["retimings"].append(record)
                    save()
                    retimer.release_caches()
            compared = report["controlled_comparison"]["arms"]
            report["controlled_comparison"]["equal_full_budgets_completed"] = report[
                "all_prepared_cases_screened"
            ] and all(a["retimings_completed"] == 12 for a in compared.values())
            report["stage"] = "gpu_substitution_joint_polish"
            save()
            polish_candidates = sorted(screened_best.values(), key=candidate_rank)
            seen_polish = set()
            for item in polish_candidates:
                key = item["case"]["case"]
                if key in seen_polish:
                    continue
                if len(seen_polish) >= 4 or time.perf_counter() >= proxy_deadline:
                    break
                case = item["case"]
                joint = states[case["ship"]]["joint"]
                visits, arr, dep = visits_of(item["plan"])
                maximum_calls = 1 + 4 * len(list(joint.moves(len(visits), 8.0)))
                if (
                    gpu.telemetry.get("completed_joint_evaluations", 0) + maximum_calls
                    > plan["maximum_total_joint_candidates"]
                ):
                    break
                seen_polish.add(key)
                began = time.perf_counter()
                _, _, value, moves = joint.optimise_epochs(
                    visits,
                    np.asarray(arr),
                    np.asarray(dep),
                    mesh=(8.0, 3.0),
                    max_moves=2,
                    deadline=min(proxy_deadline, began + 15),
                )
                record = {
                    "case": key,
                    "moves": moves,
                    "failure": value.failure,
                    "seconds": time.perf_counter() - began,
                }
                if value.feasible:
                    record["proxy"] = remember(value.plan, case, "joint")
                report["polishings"].append(record)
                save()
            report["proxy_seconds"] = time.perf_counter() - proxy_started
            report["screening_telemetry"] = dict(gpu.telemetry)
            if (
                gpu.telemetry.get("completed_joint_evaluations", 0)
                > plan["maximum_total_joint_candidates"]
            ):
                raise RuntimeError("joint candidate work exceeded the hard preparation budget")
            report["eligible_proxy_plans"] = len(accepted_proxies)
            save()
            refined_keys = set()
            for item in sorted(accepted_proxies, key=candidate_rank):
                if (
                    args.proxy_only
                    or len(report["refinements"]) >= args.max_refinements
                    or time.perf_counter() >= deadline
                ):
                    break
                case = item["case"]
                # One certification per substitution leaves the four-call budget
                # available to different asteroid choices rather than near-duplicate seeds.
                if case["case"] in refined_keys:
                    continue
                refined_keys.add(case["case"])
                attempt = len(report["refinements"]) + 1
                directory = out / "candidates" / f"attempt_{attempt:02d}_{case['case']}"
                directory.mkdir(parents=True)
                record = {k: v for k, v in item.items() if k != "plan"}
                report["refinements"].append(record)
                report.update(stage="gpu_whole_route_refinement", active_candidate=case["case"])
                save()
                refined = refine_route(item["plan"], catalogue, scvx=scvx)
                write_json(directory / "refinement.json", refined.summary())
                record.update(
                    certified=refined.certified,
                    seconds=refined.wall_seconds,
                    failures=refined.failures,
                )
                if not refined.certified:
                    record["status"] = "failed_refinement"
                    save()
                    continue
                state, ship_row = states[case["ship"]], ship_rows[case["ship"]]
                assert_candidate_plan(refined.plan, state["route"].plan, case, occupied)
                raw, weighted = (
                    refined.total_collected_kg,
                    weighted_mass(refined.collected_mass, weights),
                )
                record.update(refined_raw_kg=raw, refined_weighted_kg=weighted)
                write_route_artifacts(refined, catalogue, directory / "route")
                if not objective_gate(
                    raw,
                    weighted,
                    ship_row["verified_raw_kg"],
                    ship_row["verified_weighted_kg"],
                    baseline["total_mass_kg"],
                    minimum_gain=0,
                ):
                    record["status"] = "certified_without_weighted_gain_or_fleet_raw_feasibility"
                    save()
                    continue
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
                write_json(directory / "verification.json", checked)

                def export_viewer(best_dir, trial=trial, history=history, checked=checked):
                    return write_viewer_dataset(
                        best_dir / "viewer",
                        trial,
                        history,
                        catalogue,
                        run_id="asteroid_substitution_v604",
                        commit=COMMIT,
                        verification=checked["independent"],
                        solution_path=best_dir / "Result.txt",
                    )

                if publish_verified_candidate(out, path, checked, case, report, export_viewer):
                    record["status"] = "promoted_verified_fleet"
                else:
                    record["status"] = "retained_verified_incumbent"
                save()
            report["screening_telemetry"] = dict(gpu.telemetry)
    if sha256(ROOT / "inputs/Result.txt") != INCUMBENT_SHA:
        raise ValueError("immutable baseline changed during the run")
    report.update(
        complete=True,
        stage="complete",
        status="improved" if report["promotions"] else "incumbent_retained",
        seconds=time.perf_counter() - started,
        soft_wall_budget_overrun_seconds=max(0, time.perf_counter() - deadline),
    )
    save()
    print(json.dumps({"status": report["status"], "best": report["best"], "output": str(out)}))
    return 0


if __name__ == "__main__":
    arguments_ = arguments()
    try:
        raise SystemExit(main(arguments_))
    except Exception as error:
        if getattr(arguments_, "_created_output", False):
            path = arguments_.output.resolve() / "report.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            report.update(
                complete=True,
                status=(
                    "failed_after_verified_promotion"
                    if report.get("promotions")
                    else "failed_incumbent_preserved"
                ),
                error=repr(error),
            )
            write_json(path, report)
        raise
