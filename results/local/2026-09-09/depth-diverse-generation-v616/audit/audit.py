"""Independent, saved-output-only audit of the finite v616 candidate generation."""

import ctypes
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / "depth-diverse-generation-v616"
EXPECTED_READY = "a5336aa90bfef484e8affa0d1770a25a22f3a6611da392c9e4a375e6cbb4f267"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def close(actual, expected, tolerance=1e-8):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0), (actual, expected)


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit():
    launch = read(KIT / "launch.json")
    assert launch.get("finished_utc") and "returncode" in launch, "Await root-confirmed completion"
    assert sha(KIT / "ready-manifest.json") == EXPECTED_READY
    ready = read(KIT / "ready-manifest.json")
    for name, digest in ready["files"].items():
        assert sha(KIT / name) == digest, name
    profile = read(KIT / "profile.json")
    for item in profile["native_libraries"].values():
        assert sha(item["path"]) == item["sha256"], item["path"]
    out = KIT / "output"
    before_hashes = {
        str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
    }
    if (out / "output-sha256.json").exists():
        for name, digest in read(out / "output-sha256.json").items():
            assert before_hashes[name] == digest, name
    raw = read(out / "report.json")
    spec = read(KIT / "plan.json")
    seed_input = read(KIT / "generation-input.json")
    assert raw["ready_sha256"] == EXPECTED_READY
    assert raw["driver_sha256"] == sha(KIT / "run.py")
    assert launch["driver_sha256"] == sha(KIT / "run.py")
    assert launch["supervisor_sha256"] == sha(KIT / "launch.py")
    assert raw["pid"] == launch["pid"]
    assert raw["settings"] == spec["settings"]
    for key in (
        "native_low_thrust_solves",
        "full_route_refinements",
        "fullfleet_verifications",
        "promotions",
    ):
        assert raw[key] == 0, key
    assert raw["incumbent_retained"] is True
    assert not list(out.rglob("Result*.txt")) and not list(out.rglob("*.npz"))

    events = lines(out / "generation-events.jsonl")
    counts = Counter()
    active = []
    last_time = -1.0
    completion_depths = Counter()
    completed_depths = Counter()
    failure_depths = Counter()
    failed_completions = []
    for entry in events:
        event, stage = entry["event"], entry["stage"]
        assert event in ("started", "finished")
        assert entry["elapsed_seconds"] >= last_time
        last_time = entry["elapsed_seconds"]
        key = stage + "_" + event
        counts[key] += 1
        assert entry["number"] == counts[key]
        assert counts[key] <= spec["limits"][stage]
        pair = (stage, entry["number"])
        if event == "started":
            active.append(pair)
            if stage == "completion_attempts":
                completion_depths[entry["detail"]["depth"]] += 1
        else:
            assert active.pop() == pair, "Non-nested work journal"
            if stage == "completion_attempts":
                detail = entry["detail"]
                if detail["feasible"]:
                    completed_depths[detail["depth"]] += 1
                else:
                    failure_depths[detail["depth"]] += 1
                    failed_completions.append(detail)
                    assert detail["failure"] is not None
    if raw.get("counts") is not None:
        assert dict(counts) == raw["counts"]
    if raw.get("complete"):
        assert not active
        assert counts["generation_calls_started"] == counts["generation_calls_finished"] == 1
        assert counts["expansions_finished"] == raw["result"]["expansions"]
    telemetry = raw.get("screening_telemetry", {})
    if telemetry:
        assert telemetry["backend"] == "cuda"
        assert (
            telemetry.get("completed_collection_dp_passes", 0)
            == counts["collection_dp_passes_finished"]
        )
        if raw.get("result"):
            assert telemetry["completed_branch_requests"] == raw["result"]["lambert_evaluations"]

    # Frozen numerical classes only; no driver/domain score or shortlist helpers imported.
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(KIT / "source/src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = profile["data"]
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("Audit must remain CPU-only")):
        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.bundles import ClusterPricingSettings, refine_candidates
        from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
        from spacepdhcg.gtoc12.search import RoutePlan
        from spacepdhcg.gtoc12.solution import Solution

        catalogue, bonus = load_catalogue(), load_bonus_table()
        assert catalogue.source_sha256 == seed_input["catalogue_sha256"]
        assert bonus.source_sha256 == seed_input["bonus_sha256"]
        weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
        fleet = Solution.read(KIT / "inputs/Result.txt")
        mass_rows = []
        for ship in fleet.ships:
            masses = [
                (event.event_id, event.after.mass - event.before.mass)
                for event in ship.asteroid_visits()
                if event.after.mass > event.before.mass
            ]
            mass_rows.append(
                {
                    "ship": ship.ship_id,
                    "raw_kg": sum(m for _, m in masses),
                    "weighted_kg": sum(weights[a] * m for a, m in masses),
                }
            )
        baseline_raw = sum(row["raw_kg"] for row in mass_rows)
        baseline_weighted = sum(row["weighted_kg"] for row in mass_rows)
        replaced = next(row for row in mass_rows if row["ship"] == 23)
        floor = len(fleet.ships) * math.log(len(fleet.ships) / 2) / 0.004
        close(baseline_raw, raw["baseline"]["raw_kg"])
        close(baseline_weighted, raw["baseline"]["weighted_kg"])
        close(floor, raw["baseline"]["raw_floor_kg"])
        assert raw["baseline"]["replaced_ship"] == replaced

        entries = (
            read(out / "candidate-pool.json") if (out / "candidate-pool.json").exists() else []
        )
        journal_summaries = lines(out / "completed-plans.jsonl")
        plans = [RoutePlan.from_summary(row["plan"]) for row in entries]
        # JSON stringifies map keys. Restoring RoutePlan first reproduces the intended
        # canonical plan identity; topology uniqueness is separately reported below.
        hashes = [canonical(p.summary()) for p in plans]
        journal_hashes = [canonical(RoutePlan.from_summary(p).summary()) for p in journal_summaries]
        if entries or raw.get("generated_candidates") == 0:
            assert Counter(hashes) == Counter(journal_hashes)
            assert len(plans) == raw["generated_candidates"]
        assert len(journal_summaries) == sum(completed_depths.values())
        expected_seed = (
            seed_input["seed"]["target"],
            seed_input["seed"]["launch_epoch"],
            seed_input["seed"]["tof_days"],
        )
        allowed, excluded = (
            set(seed_input["allowed_ids"]),
            set(seed_input["excluded_other_fleet_ids"]),
        )
        depth_counts, closed_depths, topologies, groups = (
            Counter(),
            Counter(),
            Counter(),
            defaultdict(list),
        )
        inventory, eligible = [], []
        for i, (entry, plan) in enumerate(zip(entries, plans, strict=True)):
            assert entry["index"] == i and entry["plan_sha256"] == hashes[i]
            first = plan.legs[0]
            earth = (first.to_id, first.departure_epoch, first.tof_days)
            assert earth == expected_seed == tuple(entry["earth_key"])
            touched = set(plan.deploy_epochs) | set(plan.collect_epochs)
            assert touched <= allowed and not touched & excluded
            assert not plan.foreign_deploy_epochs
            depth = len(plan.deploy_epochs)
            closed = set(plan.deploy_epochs) == set(plan.collect_epochs)
            assert depth <= 10 and entry["deploy_depth"] == depth
            assert entry["collect_depth"] == len(plan.collect_epochs)
            assert entry["plan"]["self_cleaning"] == closed
            depth_counts[depth] += 1
            if closed:
                closed_depths[depth] += 1
            topologies[
                (
                    tuple(sorted(plan.deploy_epochs, key=lambda a: (plan.deploy_epochs[a], a))),
                    tuple(sorted(plan.collect_epochs, key=lambda a: (plan.collect_epochs[a], a))),
                )
            ] += 1
            groups[earth].append(i)
            cargo, weighted = (
                sum(plan.collected_mass.values()),
                sum(weights[a] * m for a, m in plan.collected_mass.items()),
            )
            for a, mass in plan.collected_mass.items():
                # No cargo resizing here: the generator's retained epochs prescribe mining.
                close(
                    mass, C.maximum_collected_mass(plan.collect_epochs[a] - plan.deploy_epochs[a])
                )
            close(cargo, entry["raw_proxy_kg"])
            close(weighted, entry["weighted_proxy_kg"])
            score = weighted - spec["settings"]["propellant_weight"] * plan.propellant_proxy_kg
            close(score, entry["beam_score_including_propellant_penalty"])
            next_raw, next_weighted = (
                baseline_raw - replaced["raw_kg"] + cargo,
                baseline_weighted - replaced["weighted_kg"] + weighted,
            )
            close(next_raw, entry["candidate_fleet_raw_proxy_kg"])
            close(next_weighted, entry["candidate_fleet_weighted_proxy_kg"])
            raw_ok, weight_ok = next_raw >= floor, next_weighted > baseline_weighted + 1e-9
            assert raw_ok == entry["proxy_raw_floor_eligible"]
            assert weight_ok == entry["proxy_weighted_improves"]
            assert not entry["certified"] and not entry["scored_fleet"] and not entry["promoted"]
            if closed and raw_ok and weight_ok:
                eligible.append(i)
            inventory.append(
                {
                    "index": i,
                    "depth": depth,
                    "collect_depth": len(plan.collect_epochs),
                    "closed": closed,
                    "raw_proxy_kg": cargo,
                    "weighted_proxy_kg": weighted,
                    "raw_floor_margin_kg": next_raw - floor,
                    "weighted_gain_proxy_kg": next_weighted - baseline_weighted,
                    "beam_score": score,
                }
            )
        ordered = sorted(
            range(len(plans)),
            key=lambda i: (
                -inventory[i]["beam_score"],
                plans[i].propellant_proxy_kg,
                plans[i].asteroids,
            ),
        )
        assert ordered == list(range(len(plans))), "Final pool order changed"
        selected = refine_candidates(plans, {expected_seed}, ClusterPricingSettings(refine_top=2))
        selected_indices = [
            next(i for i, plan in enumerate(plans) if plan is candidate) for candidate in selected
        ]
        if (out / "shortlist-attrition.json").exists():
            attrition = read(out / "shortlist-attrition.json")
            assert attrition["actual_selected_indices"] == selected_indices
            assert attrition["immutable_pool_sha256"] == sha(out / "candidate-pool.json")
            assert attrition["closed_raw_and_weighted_proxy_eligible_indices"] == eligible
            assert {int(k): v for k, v in attrition["generated_depth_counts"].items()} == dict(
                depth_counts
            )
            assert {
                int(k): v for k, v in attrition["generated_closed_depth_counts"].items()
            } == dict(closed_depths)
            firsts = {indices[0] for indices in groups.values()}
            assert attrition["earth_dedup_discarded_indices"] == [
                i for i in range(len(plans)) if i not in firsts
            ]
            assert attrition["refine_top_discarded_indices"] == sorted(
                firsts - set(selected_indices)
            )
        report = {
            "audit_passed": True,
            "native_library_loads": 0,
            "GPU_calls": 0,
            "prepared_files_verified": len(ready["files"]),
            "ready_sha256": EXPECTED_READY,
            "raw_output_hashes": before_hashes,
            "launch_sha256": sha(KIT / "launch.json"),
            "supervisor_returncode": launch["returncode"],
            "driver_status": raw["status"],
            "complete": raw["complete"],
            "counts_complete": raw["counts_complete"],
            "generated_candidates": len(plans),
            "completion_journal_candidates": len(journal_summaries),
            "unique_exact_plans": len(set(hashes)),
            "unique_ordered_topologies": len(topologies),
            "duplicate_exact_plans": len(hashes) - len(set(hashes)),
            "depth_counts": dict(depth_counts),
            "closed_depth_counts": dict(closed_depths),
            "journal_counts": dict(counts),
            "unfinished_journal_stages": active,
            "attempted_completion_depths": dict(completion_depths),
            "completed_completion_depths": dict(completed_depths),
            "failed_completion_depths": dict(failure_depths),
            "failed_completions": failed_completions,
            "frozen_default_selected_indices": selected_indices,
            "closed_raw_and_weighted_proxy_eligible_indices": eligible,
            "eligible_but_not_default_selected_indices": sorted(
                set(eligible) - set(selected_indices)
            ),
            "same_earth_depth_contrast": any(
                len({len(plans[i].deploy_epochs) for i in group}) > 1 for group in groups.values()
            ),
            "candidate_inventory": inventory,
            "baseline_raw_kg": baseline_raw,
            "baseline_weighted_kg": baseline_weighted,
            "ship_23_raw_kg": replaced["raw_kg"],
            "ship_23_weighted_kg": replaced["weighted_kg"],
            "required_candidate_raw_kg": floor - baseline_raw + replaced["raw_kg"],
            "actual_lambert_branch_requests": telemetry.get("completed_branch_requests", 0),
            "actual_cuda_collection_dp_passes": telemetry.get("completed_collection_dp_passes", 0),
            "raw_runtime_seconds": raw.get("wall_seconds"),
            "low_thrust_solutions": 0,
            "fullfleet_verifications": 0,
            "promotions": 0,
            "scope": (
                "Source/output accounting and independent proxy arithmetic; "
                "no new physical certification"
            ),
            "limitations": [
                "A generated proxy is not proof of low-thrust feasibility "
                "or leaderboard improvement",
                "Full pool means returned completion candidates, not all internal pruned paths",
                "A partial run cannot establish depth-10 absence for the whole family",
            ],
        }
    after_hashes = {
        str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
    }
    assert before_hashes == after_hashes, "Audit must not alter outputs"
    for name, digest in ready["files"].items():
        assert sha(KIT / name) == digest, name
    return report


if __name__ == "__main__":
    result = audit()
    target = ROOT / "report-v2.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "audit_passed",
                    "generated_candidates",
                    "depth_counts",
                    "closed_depth_counts",
                    "frozen_default_selected_indices",
                    "closed_raw_and_weighted_proxy_eligible_indices",
                    "journal_counts",
                )
            },
            indent=2,
        )
    )
