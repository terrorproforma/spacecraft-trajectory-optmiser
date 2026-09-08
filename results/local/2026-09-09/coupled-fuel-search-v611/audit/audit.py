"""Read-only independent accounting/objective audit of the completed finite v611 run."""

import ctypes
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / "coupled-fuel-search-v611"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tolerance=1e-7):
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, (a, b)


def main():
    target = ROOT / "report.json"
    if target.exists():
        raise FileExistsError("Preserve existing audit")
    ready = read(KIT / "ready-manifest.json")
    assert sha(KIT / "ready-manifest.json") == (
        "1d9428be654cf30f49367d5ba98f4a2674d810fc4e2ee728906b4518a1cf8986"
    )
    for name, digest in ready["files"].items():
        assert sha(KIT / name) == digest, name
    output = KIT / "output"
    report, launch = read(output / "report.json"), read(KIT / "launch.json")
    assert report["complete"] and report["status"] == "incumbent_retained"
    assert launch["returncode"] == 0 and not launch.get("timed_out")
    assert launch["outer_timeout_seconds"] == 630 and launch["termination_grace_seconds"] == 10
    assert launch["driver_sha256"] == sha(KIT / "run.py")
    assert launch["supervisor_sha256"] == sha(KIT / "launch.py")
    assert report["execution_errors"] == 0 and not report["promotions"]
    base = report["baseline"]
    assert base["ok"] and base["independent"]["ok"] and base["official"]["ok"]
    close(base["score_kg"], 12810.135953048577)
    close(base["total_mass_kg"], 14051.854893908598)
    assert report["best"]["sha256"] == sha(KIT / "inputs/Result.txt")
    assert report["best"]["score_kg"] == base["score_kg"]
    assert report["fresh_control_refinements"] == 0 and report["archived_controls_accepted"]
    # Load only the frozen data parser, with native loading explicitly forbidden.
    sys.path.insert(0, str(KIT / "source/src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "profile.json")["data"]
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("No GPU audit")):
        from spacepdhcg.gtoc12.data import load_bonus_table

        bonus = load_bonus_table()
    assert bonus.source_sha256 == "e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21"
    searches = report["searches"]
    assert len(searches) == 12
    assert Counter(row["margin_price"] for row in searches) == {0.05: 4, 0.25: 4, 1.0: 4}
    rows, identities = [], set()
    originals = {}
    for ship in (20, 7, 10, 11):
        original = read(KIT / "inputs" / f"ship-{ship:02d}.json")
        originals[ship] = original
    for row in searches:
        assert read(output / "searches" / (row["id"] + ".json")) == row
        assert row["feasible"] and not row["failure"] and "error" not in row
        assert row["work"]["evaluations"] <= 7969 and row["work"]["moves"] <= 48
        assert row["work"]["invalid_stay"] == 0
        plan = row["plan"]
        original = originals[row["ship"]]
        assert set(plan["deploy_epochs"]) == set(original["plan"]["deploy_epochs"])
        assert set(plan["collect_epochs"]) == set(original["plan"]["collect_epochs"])
        assert not plan["foreign_deploy_epochs"] and not plan["orphaned"]
        cargo = plan["collected_mass_kg"]
        raw = sum(cargo.values())
        weighted = sum(mass * float(bonus.coefficient[int(body) - 1]) for body, mass in cargo.items())
        old_weighted = sum(
            mass * float(bonus.coefficient[int(body) - 1])
            for body, mass in original["collected_mass_kg"].items()
        )
        expected_raw = base["total_mass_kg"] - sum(original["collected_mass_kg"].values()) + raw
        close(raw, row["raw_kg"])
        close(weighted, row["weighted_kg"])
        close(weighted - old_weighted, row["weighted_gain_kg"])
        close(weighted + row["margin_price"] * row["spare_kg"], row["objective"])
        eligible = weighted > old_weighted + 1e-8 and expected_raw >= 23 * math.log(23 / 2) / .004 - 1e-8
        assert eligible == row["objective_eligible"]
        # The producer hashed integer asteroid keys before JSON converted them to strings.
        hash_plan = dict(plan)
        for field in ("deploy_epochs", "collect_epochs", "foreign_deploy_epochs", "collected_mass_kg"):
            hash_plan[field] = {int(key): value for key, value in plan[field].items()}
        digest = hashlib.sha256(json.dumps([row["ship"], hash_plan], sort_keys=True).encode()).hexdigest()
        assert digest == row["plan_sha256"]
        identities.add(digest)
        rows.append({key: row[key] for key in (
            "id", "ship", "margin_price", "raw_kg", "weighted_kg", "weighted_gain_kg",
            "spare_kg", "objective_eligible", "search_seconds", "work", "plan_sha256",
        )})
    eligible_rows = [row for row in rows if row["objective_eligible"]]
    assert len(eligible_rows) == 4 and all(row["margin_price"] == .05 for row in eligible_rows)
    winner = max(eligible_rows, key=lambda row: row["weighted_gain_kg"])
    assert report["selected"] == [winner["id"]] == ["search-01"]
    unrefined = [row for row in eligible_rows if row["id"] not in report["selected"]]
    evaluations = sum(row["work"]["evaluations"] for row in rows)
    moves = sum(row["work"]["moves"] for row in rows)
    telemetry = report["screening_telemetry"]
    assert evaluations == telemetry["completed_joint_evaluations"] == 39852
    assert moves == telemetry["joint_search_accepted_moves"] == 202
    assert telemetry["completed_joint_searches"] == 12 and telemetry["gpu_used"]
    assert telemetry["completed_branch_requests"] == 2 * telemetry["joint_geometry_computed_hops"]
    assert evaluations * 17 == (
        telemetry["joint_geometry_computed_hops"] + telemetry["joint_geometry_rejected_hops"]
    )
    case = output / "refinements/candidate-01"
    detail = report["refinements"][0]
    refined = read(case / "refinement.json")
    prescribed = read(case / "prescription.json")
    assert len(report["refinements"]) == report["full_refinements_started"] == 1
    assert refined["plan"] == prescribed["plan"] == searches[0]["plan"]
    assert refined["collected_mass_kg"] == prescribed["cargo_kg"] == searches[0]["plan"]["collected_mass_kg"]
    assert not refined["certified"] and not refined["master_certified"]
    assert detail["native_leg_calls"] == detail["actual_legs"] == 17
    assert not list(case.glob("*Result.txt")) and not (case / "verification.json").exists()
    calls = [json.loads(line) for line in (output / "native-legs.jsonl").read_text().splitlines()]
    assert len(calls) == report["native_legs_started"] == report["native_legs_finished"] == 17
    assert [call["number"] for call in calls] == list(range(1, 18))
    flown = [leg for leg in prescribed["plan"]["legs"] if leg["role"] != "camp"]
    assert len(flown) == 17
    records, certificate_burns, summaries = [], [], []
    cargo, deploy = prescribed["cargo_kg"], prescribed["plan"]["deploy_epochs"]
    collect = prescribed["plan"]["collect_epochs"]
    for i, (flight, call) in enumerate(zip(flown, calls)):
        leg = read(case / f"leg-{i:02d}.json")
        assert sha(case / leg["arrays"]["path"]) == leg["arrays"]["sha256"]
        assert leg["from"] == flight["from"] and leg["to"] == flight["to"]
        assert leg["departure"] == flight["t0"] and leg["arrival"] == flight["tf"]
        assert call["departure"] == flight["t0"] and call["arrival"] == flight["tf"]
        carried = sum(cargo[a] for a, epoch in collect.items() if epoch <= flight["t0"])
        deployed = sum(epoch <= flight["t0"] for epoch in deploy.values())
        expected_initial = 3000 - sum(certificate_burns) - 40 * deployed + carried
        close(leg["initial_mass_kg"], expected_initial)
        close(leg["initial_mass_kg"], call["initial_mass_kg"])
        close(leg["minimum_final_mass_kg"], 500 + carried)
        solution = leg["solution"]
        assert call["iterations"] == solution["iterations"]
        assert call["accepted_iterations"] == solution["accepted_iterations"]
        assert len(solution["history"]) == solution["iterations"]
        if i < 16:
            assert leg["certified"] and leg["certificate"]
            certificate = leg["certificate"]
            assert certificate["position_error_km"] <= 1000
            assert certificate["velocity_error_km_s"] <= .001
            assert certificate["minimum_sun_distance_au"] >= .3
            assert certificate["maximum_thrust_n"] <= .6 + 1e-8
            assert certificate["final_mass_kg"] >= leg["minimum_final_mass_kg"] - 1e-3
            certificate_burns.append(leg["initial_mass_kg"] - certificate["final_mass_kg"])
        else:
            assert not leg["certified"] and leg["certificate"] is None
            assert solution["max_defect"] > report["scvx_settings"]["defect_tolerance"]
            close(solution["propellant_kg"], leg["initial_mass_kg"] - leg["minimum_final_mass_kg"])
        summaries.append({
            "index": i, "from": leg["from"], "to": leg["to"],
            "certified": leg["certified"], "iterations": solution["iterations"],
            "accepted_iterations": solution["accepted_iterations"],
            "native_wrapper_seconds": call["wrapper_seconds"],
            "solver_reports": len(solution["solver_reports"]),
            "qualified_solver_reports": sum(bool(x["qualified"]) for x in solution["solver_reports"]),
            "inner_iterations": sum(x["iterations"] for x in solution["solver_reports"]),
        })
        records.append(leg)
    failure = records[-1]
    search_seconds = sum(row["search_seconds"] for row in rows)
    native_seconds = sum(call["wrapper_seconds"] for call in calls)
    audit = {
        "status": "audited_incumbent_retained_no_new_GPU_work", "GPU_calls": 0,
        "ready_sha256": sha(KIT / "ready-manifest.json"), "indexed_prepared_files_verified": 247,
        "run_report_sha256": sha(output / "report.json"), "launch_sha256": sha(KIT / "launch.json"),
        "source_commit": ready["source_commit"], "score_kind": "fixed_bonus_weighted_kg",
        "baseline_raw_kg": base["total_mass_kg"], "baseline_weighted_kg": base["score_kg"],
        "score_improvement_kg": 0, "baseline_both_fleet_checks_pass": True,
        "new_candidate_fullfleet_verifications": 0, "promotions": 0,
        "searches": rows, "unrefined_eligible_candidates": unrefined,
        "unrefined_reason": "Frozen one-candidate-per-margin-price rule; no automatic backfill",
        "selected_candidate": winner["id"], "itinerary_evaluations": evaluations,
        "unique_final_plans": len(identities), "accepted_device_moves": moves,
        "actual_lambert_direction_requests": telemetry["completed_branch_requests"],
        "full_refinements": 1, "native_leg_calls": 17, "certified_prefix_legs": 16,
        "failed_leg": {key: failure[key] for key in (
            "leg", "from", "to", "departure", "arrival", "initial_mass_kg", "minimum_final_mass_kg",
            "status", "diagnostic", "certified", "certificate",
        )},
        "failed_leg_max_defect": failure["solution"]["max_defect"],
        "fixed_cargo_kg": sum(cargo.values()), "exact_cargo_epochs_preserved": True,
        "certified_prefix_propellant_from_mass_balance_kg": sum(certificate_burns),
        "available_return_propellant_kg": failure["initial_mass_kg"] - failure["minimum_final_mass_kg"],
        "failure_interpretation": (
            "Return exhausted its fixed-cargo mass allowance with substantial remaining dynamics error. "
            "No accepted trajectory or physical infeasibility certificate resulted."
        ),
        "leg_accounting": summaries,
        "seconds": {"whole_run": report["seconds"], "search_wrappers": search_seconds,
                    "native_leg_wrappers": native_seconds, "baseline_both_checkers": base["seconds"]},
        "itinerary_evaluations_per_search_wrapper_second": evaluations / search_seconds,
        "rate_scope": "Proxy search only, includes first-call overhead; not certified solutions/s",
        "supervisor_timeout_seconds": 630, "timeout_triggered": False,
        "certificate_audit_scope": "Recorded independent certificates and mass/event consistency; no repeated integration",
        "raw_output_files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    assert len(audit["raw_output_files"]) == 50
    target.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "status", "itinerary_evaluations", "actual_lambert_direction_requests", "seconds",
        "available_return_propellant_kg", "fixed_cargo_kg",
    )}, indent=2))


if __name__ == "__main__":
    main()
