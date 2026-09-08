"""Independent artifact/telemetry checks, using only the standard library."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit"
RUN = ROOT / "output-compatible"
SOURCE = ROOT / "source"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(path):
    grouped = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            parts = line.split()
            grouped[int(parts[0])].append(parts)
    return dict(grouped)


def event_inventory(rows):
    state = [r for r in rows if int(r[1]) != -1]
    deployed, collected, unloaded = {}, {}, 0.0
    for first, second in zip(state[::2], state[1::2], strict=True):
        assert first[:3] == second[:3]
        asteroid = int(first[1])
        change = float(second[-1]) - float(first[-1])
        if asteroid > 0:
            if change < -1e-6:
                assert abs(change + 40) < 1e-7
                assert asteroid not in deployed
                deployed[asteroid] = float(first[2])
            else:
                assert asteroid not in collected
                collected[asteroid] = change
        elif asteroid == -3:
            unloaded -= change
    return deployed, collected, unloaded


report = read(RUN / "report.json")
fresh = read(OUT / "fresh-verification.json")
launch = read(ROOT / "compatible-launch.json")
old = SOURCE / "results/lambda/2026-09-06/fleet_master_v11/fleet/Result.txt"
best = RUN / "best/Result.txt"
attempt1 = RUN / "candidates/ship_15_attempt_01/fleet-Result.txt"
attempt2 = RUN / "candidates/ship_15_attempt_02/fleet-Result.txt"
manifest = read(RUN / "best/viewer/manifest.json")
source_checks = {}
for relative, expected in report["source_sha256"].items():
    source_checks[relative] = sha(SOURCE / relative) == expected
assert all(source_checks.values())
assert sha(SOURCE / "run.py") == report["driver_sha256"]
assert sha(old) == report["incumbent_solution_sha256"]
assert sha(best) == sha(attempt1) == fresh["solution_sha256"] == manifest["source"]["sha256"]
assert sha(best) != sha(attempt2)
assert sha(RUN / "best/viewer/trajectories.json") == manifest["files"]["trajectories.json"]["sha256"]
assert best.stat().st_size == manifest["source"]["bytes"]
assert fresh["independent"] == report["best"]["independent"]
assert fresh["independent"]["ok"] and fresh["official"]["ok"]
prior, promoted = sections(old), sections(best)
assert sorted(prior) == sorted(promoted) == list(range(1, 24))
changed = [i for i in prior if prior[i] != promoted[i]]
assert changed == [15], changed
all_inventory = {i: event_inventory(rows) for i, rows in promoted.items()}
old_inventory = {i: event_inventory(rows) for i, rows in prior.items()}
dep = {a for d, _, _ in all_inventory.values() for a in d}
col = {a for _, c, _ in all_inventory.values() for a in c}
old_col = {a for _, c, _ in old_inventory.values() for a in c}
assert col - old_col == {19102} and old_col - col == set()
assert set(all_inventory[15][0]) == set(old_inventory[15][0])
assert all(a not in c for i, (_, c, _) in old_inventory.items() for a in [19102] if i != 15)
unloaded = sum(v[2] for v in all_inventory.values())
assert abs(unloaded - fresh["independent"]["total_mass_kg"]) < 1e-7
route = read(RUN / "candidates/ship_15_attempt_01/route/route_summary.json")
old_route = read(SOURCE / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources/ship-15.json")
native = [json.loads(line) for line in (RUN / "native-solves.jsonl").read_text().splitlines()]
assert len(native) == report["native_solves_completed"] == 36
assert [r["solve"] for r in native] == list(range(1, 37))
assert all(r.get("status") == "converged" and "error" not in r for r in native)
assert len(report["rows"]) == report["screening_telemetry"]["completed_retime_driver_calls"] == 36
assert len(report["refinements"]) == report["eligible_proxy_candidates"] == 2
assert len(report["promotions"]) == 1
assert report["refinements"][0]["status"] == "promoted_verified_fleet"
assert report["refinements"][1]["status"] == "retained_verified_incumbent"
new_score = fresh["independent"]["weighted_score_fixed_bonus_kg"]
old_score = report["baseline"]["score_kg"]
new_raw = fresh["independent"]["total_mass_kg"]
old_raw = report["baseline"]["total_mass_kg"]
telemetry = report["screening_telemetry"]
result = {
    "status": "passed",
    "fresh_cpu_verifiers": {"independent_ok": fresh["independent"]["ok"], "official_ok": fresh["official"]["ok"], "independent_seconds": fresh["independent_seconds"]},
    "solution_sha256": sha(best),
    "artifact_checks": {"promoted_equals_first_candidate": True, "promoted_is_not_second_candidate": True,
                         "viewer_source_and_data_hashes_match": True, "source_modules_match_run_hashes": len(source_checks),
                         "driver_matches_run_hash": True, "original_solution_matches_recorded_hash": True,
                         "unchanged_ship_sections_token_identical": [i for i in prior if i not in changed]},
    "changed_ships": changed, "newly_collected_asteroids": sorted(col - old_col),
    "deployed_asteroids": len(dep), "mined_asteroids_before": len(old_col), "mined_asteroids_after": len(col),
    "score": {"raw_kg_before": old_raw, "raw_kg_after": new_raw, "raw_gain_kg": new_raw - old_raw,
              "weighted_kg_before": old_score, "weighted_kg_after": new_score, "weighted_gain_kg": new_score - old_score,
              "raw_kg_per_ship_before": old_raw / 23, "raw_kg_per_ship_after": new_raw / 23,
              "weighted_kg_per_ship_after": new_score / 23, "physical_ore_recomputed_from_earth_unload_events_kg": unloaded},
    "ship_15": {"raw_kg_before": old_route["total_collected_kg"], "raw_kg_after": route["total_collected_kg"],
                "new_asteroid": 19102, "new_asteroid_ore_kg": route["collected_mass_kg"]["19102"],
                "deploy_mjd": route["plan"]["deploy_epochs"]["19102"], "collect_mjd": route["plan"]["collect_epochs"]["19102"],
                "other_asteroid_net_raw_change_kg": sum(v - old_route["collected_mass_kg"].get(k, 0) for k, v in route["collected_mass_kg"].items() if k != "19102"),
                "final_mass_after_unload_kg": route["final_mass_kg"]},
    "work": {"distinct_new_orders": report["orders"], "grid_days": report["configuration"]["grids"],
             "retime_driver_calls": len(report["rows"]), "proxy_feasible_cases": sum("joint" in r for r in report["rows"]),
             "proxy_failures": dict(Counter(r["result"]["failure"] for r in report["rows"])),
             "eligible_proxies": report["eligible_proxy_candidates"], "route_refinements": len(report["refinements"]),
             "native_leg_solves": len(native), "native_solve_statuses": dict(Counter(r["status"] for r in native)),
             "native_iterations": sum(r["iterations"] for r in native),
             "native_accepted_iterations": sum(r["accepted_iterations"] for r in native),
             "native_solve_seconds_sum": sum(r["seconds"] for r in native),
             "route_refinement_seconds_sum": sum(r["seconds"] for r in report["refinements"]),
             "joint_batches": telemetry["completed_joint_batches"], "joint_candidate_evaluations": telemetry["completed_joint_evaluations"],
             "retime_internal_dp_calls": telemetry["completed_retime_dp_calls"],
             "retime_internal_forward_calls": telemetry["completed_retime_forward_calls"],
             "lambert_logical_branch_requests": telemetry["completed_branch_requests"],
             "lambert_element_hops": telemetry["completed_element_hops"],
             "resident_table_cells": telemetry["retime_resident_cells"],
             "verified_promotions": len(report["promotions"])},
    "timing": {"campaign_timer_seconds": report["seconds"],
               "scope": "Begins after imports, data/native hashes and original fleet parsing; includes fresh baseline verification, GPU context and searches, both refinements and both full-fleet candidate verifications, viewer export, checkpoint IO; ends before final report serialization and process exit.",
               "not_pure_gpu_time": True, "not_a_repeated_speed_benchmark": True,
               "fresh_audit_seconds_excluded": fresh["independent_seconds"]},
    "execution_qualification": {"gpu_joint_evaluation": report["joint_batch_requested"] == "1",
                                "joint_device_selection": launch["device_selection"],
                                "winner_selection": "CPU NumPy argmax (compatible setting)",
                                "native_refinement_backends": report["scvx_settings"]},
    "verification_limits": {"fresh_run_default_tolerances": fresh["tolerances"],
                            "max_position_error_km_unchanged_from_baseline": report["baseline"]["independent"]["max_position_error_km"] == fresh["independent"]["max_position_error_km"],
                            "max_velocity_error_unchanged_from_baseline": report["baseline"]["independent"]["max_velocity_error_km_s"] == fresh["independent"]["max_velocity_error_km_s"]},
    "publication_issues": [],
    "publication_caveats": ["Publish output-compatible, not the failed preceding output directory.",
                            "195 is the mined-asteroid count; the asteroid footprint remains196 deployed asteroids.",
                            "Lambert requests/joint evaluations are proxy work, not that many certified trajectories.",
                            "The 78.115-second campaign timer is not pure GPU time or complete process startup time.",
                            "Reproduction needs SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=0 with the recorded library; the failed preceding run requested an unavailable device-winner entry point.",
                            "The driver and independent verification retain CPU orchestration; do not claim the entire program is GPU-native."]
}
(OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("status", "score", "ship_15", "work", "artifact_checks")}, indent=2))
