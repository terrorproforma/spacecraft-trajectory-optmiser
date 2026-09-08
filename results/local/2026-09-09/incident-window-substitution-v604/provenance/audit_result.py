"""Audit completed v604 accounting, saved proxies and independently verified fleet score."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import incident

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ready = read(ROOT / "ready-manifest.json")
for name, expected in ready["files"].items():
    assert digest(ROOT / name) == expected, name
report = read(ROOT / "output/report.json")
supervisor = read(ROOT / "supervisor.json")
launch = read(ROOT / "launch.json")
plan = read(ROOT / "plan/plan.json")
assert report["complete"] and supervisor["returncode"] == 0
assert supervisor["child_pid"] == launch["pid"] == report["pid"]
assert digest(ROOT / "foreground_supervisor.py") == supervisor["supervisor_sha256"]
assert report["driver_sha256"] == ready["files"]["run.py"] == launch["driver_sha256"]
assert report["ranking_module_sha256"] == ready["files"]["incident.py"]
assert report["candidate_plan_sha256"] == digest(ROOT / "plan/plan.json")
assert report["source_commit"] == ready["source_commit"] == plan["source_commit"]
assert report["physics_tolerances_changed"] is False
assert len(report["refinements"]) <= 4
assert len(report["retimings"]) <= 24
assert len(report["polishings"]) <= 4
assert report["screening_telemetry"]["completed_joint_evaluations"] <= 15220
assert report["screening_telemetry"]["gpu_used"]

cases = {case["case"]: case for case in plan["replacement_cases"]}
shared = incident.screen_progress(report["screened_cases"], list(cases.values()), 12400)
assert shared == report["shared_grid"]
assert report["all_prepared_cases_screened"] == shared["complete"]
structures = {row["case"]: row for row in plan["construction"]}
best_diagnostics = {}
for record in report["screened_cases"]:
    for sample in record["samples"]:
        diagnostic = sample["ranking_estimate"]
        assert diagnostic["scope"] == "uncertified_ranking_estimates_only"
        assert [leg["leg"] for leg in diagnostic["incident_legs"]] == structures[record["case"]][
            "incident_legs"
        ]
        assert len(diagnostic["incident_legs"]) == 4
        if diagnostic["complete"]:
            assert all(
                math.isfinite(leg["authority_normalized_estimate"])
                for leg in diagnostic["incident_legs"]
            )
            previous = best_diagnostics.get(record["case"])
            if previous is None or incident.diagnostic_key(
                diagnostic, record["case"], sample["mode"]
            ) < incident.diagnostic_key(previous, record["case"], previous["mode"]):
                best_diagnostics[record["case"]] = diagnostic | {"mode": sample["mode"]}
arms = incident.choose_arms(list(cases.values()), best_diagnostics, plan["control_case_ids"])
comparison = report["controlled_comparison"]
for name, selected in arms.items():
    stored = comparison["arms"][name]
    assert stored["selected_cases"] == [case["case"] for case in selected]
    rows = [row for row in report["retimings"] if row["arm"] == name]
    assert len(rows) == stored["retimings_completed"] <= 12
    assert len({row["case"] for row in rows}) == len(rows)
    assert [row["case"] for row in rows] == stored["selected_cases"][: len(rows)]
    assert stored["feasible_native_forward_plans"] == sum("proxy" in row for row in rows)
    assert stored["objective_eligible_native_forward_plans"] == sum(
        row.get("proxy", {}).get("eligible_for_refinement", False) for row in rows
    )
assert comparison["equal_full_budgets_completed"] == (
    shared["complete"] and len(report["retimings"]) == 24
)

baseline, best = report["baseline"], report["best"]
threshold = 23 * math.log(23 / 2) / 0.004
for checked in (baseline, best):
    assert checked["ok"] and checked["independent"]["ok"] and checked["official"]["ok"]
    assert checked["official"]["return_code"] == 0
    assert checked["total_mass_kg"] >= threshold
    assert checked["score_kg"] == checked["independent"]["weighted_score_fixed_bonus_kg"]
assert baseline["total_mass_kg"] == 14051.854893908598
assert baseline["score_kg"] == 12810.135953048577
assert digest(ROOT / "inputs/Result.txt") == plan["incumbent_sha256"]
if report["promotions"]:
    assert best["score_kg"] > baseline["score_kg"]
    assert digest(Path(best["solution"])) == best["solution_sha256"]
else:
    assert best["score_kg"] == baseline["score_kg"]
    assert best["total_mass_kg"] == baseline["total_mass_kg"]

bonus = Path(
    "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data/bonus_coefficients.txt"
)
assert digest(bonus) == report["bonus_sha256"]
coefficients = [float(line.split()[0]) for line in bonus.read_text().splitlines() if line.strip()]
ship_rows = {row["ship"]: row for row in plan["all_ships"]}
occupied = set(plan["occupied_asteroids"])
proxy_records = []
for record in report["screened_cases"]:
    for sample in record["samples"]:
        if "proxy" in sample:
            proxy_records.append((record["case"], sample["proxy"]))
for record in [*report["retimings"], *report["polishings"]]:
    if "proxy" in record:
        proxy_records.append((record["case"], record["proxy"]))
proxy_checks = []
for key, proxy in proxy_records:
    case = cases[key]
    original = read(ROOT / f"inputs/ship-{case['ship']:02d}.json")["plan"]
    path = ROOT / "output/proxies" / Path(proxy["path"]).name
    candidate = read(path)
    deploy = {int(a): float(t) for a, t in candidate["deploy_epochs"].items()}
    collect = {int(a): float(t) for a, t in candidate["collect_epochs"].items()}
    for phase, values in (("deploy_epochs", deploy), ("collect_epochs", collect)):
        assert set(values) == {
            case["new"] if int(a) == case["old"] else int(a) for a in original[phase]
        }
    assert case["new"] not in occupied and not candidate["foreign_deploy_epochs"]
    masses = {int(a): float(m) for a, m in candidate["collected_mass_kg"].items()}
    assert set(masses) == set(collect)
    assert all(collect[a] - deploy[a] >= 365.25 - 1e-6 for a in collect)
    assert all(0 <= m <= 10 * (collect[a] - deploy[a]) / 365.25 + 1e-7 for a, m in masses.items())
    raw = sum(masses.values())
    weighted = sum(m * coefficients[a - 1] for a, m in masses.items())
    before = ship_rows[case["ship"]]
    raw_gain, weighted_gain = (
        raw - before["verified_raw_kg"],
        weighted - before["verified_weighted_kg"],
    )
    for actual, field in (
        (raw, "predicted_raw_kg"),
        (weighted, "predicted_weighted_kg"),
        (raw_gain, "predicted_raw_gain_kg"),
        (weighted_gain, "predicted_weighted_gain_kg"),
    ):
        assert abs(actual - proxy[field]) < 1e-8
    eligible = weighted_gain > 0.5 and baseline["total_mass_kg"] + raw_gain >= threshold + 1e-6
    assert eligible == proxy["eligible_for_refinement"]
    assert proxy["passes_unchanged_native_forward"] is True
    proxy_checks.append(
        {
            "case": key,
            "origin": proxy["origin"],
            "raw_kg": raw,
            "weighted_kg": weighted,
            "raw_gain_kg": raw_gain,
            "weighted_gain_kg": weighted_gain,
            "eligible": eligible,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path),
        }
    )
native_path = ROOT / "output/native-solves.jsonl"
native_rows = (
    [json.loads(line) for line in native_path.read_text().splitlines()]
    if native_path.exists()
    else []
)
assert len(native_rows) == report["native_solves_completed"]
result = {
    "passed": True,
    "additional_GPU_calls": 0,
    "report_sha256": digest(ROOT / "output/report.json"),
    "ready_manifest_sha256": digest(ROOT / "ready-manifest.json"),
    "supervisor_sha256": digest(ROOT / "foreground_supervisor.py"),
    "source_commit": report["source_commit"],
    "driver_sha256": report["driver_sha256"],
    "status": report["status"],
    "shared_grid": shared,
    "controlled_comparison": comparison,
    "raw_kg": best["total_mass_kg"],
    "weighted_kg": best["score_kg"],
    "weighted_gain_kg": best["score_kg"] - baseline["score_kg"],
    "full_refinements": len(report["refinements"]),
    "promotions": len(report["promotions"]),
    "both_stored_checkers_pass": True,
    "proxy_checks": proxy_checks,
    "native_solves": len(native_rows),
    "native_exceptions": sum("error" in row for row in native_rows),
    "retime_failures": dict(
        Counter(row["result"].get("failure", "") for row in report["retimings"])
    ),
    "telemetry": report["screening_telemetry"],
    "timing": {
        "campaign_seconds": report["seconds"],
        "baseline_verification_seconds": baseline["verification_seconds"],
        "shared_screen_seconds": report["fixed_screen_seconds"],
        "proxy_seconds": report["proxy_seconds"],
        "joint_screen_call_seconds": report["joint_screen_call_seconds"],
        "CPU_ranking_diagnostic_seconds": report["CPU_ranking_diagnostic_seconds"],
    },
}
audit_path = ROOT / "audit/result-audit.json"
audit_path.parent.mkdir(parents=True, exist_ok=True)
with audit_path.open("x") as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            key: value
            for key, value in result.items()
            if key not in ("proxy_checks", "controlled_comparison", "telemetry")
        }
    )
)
