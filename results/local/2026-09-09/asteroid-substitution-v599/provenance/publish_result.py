"""Publish the completed v599 negative result and exact compact provenance locally."""

import hashlib
import json
import math
import shutil
import tarfile
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[2]
target = repo / "results/local/2026-09-09/asteroid-substitution-v599"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), destination
    shutil.copy2(source, destination)
    assert digest(source) == digest(destination)


ready = read(root / "ready-manifest.json")
for name, expected in ready["files"].items():
    assert digest(root / name) == expected, name
report = read(root / "output/report.json")
plan = read(root / "plan/plan.json")
launch = read(root / "launch.json")
authority = read(root / "audit/authority-replay.json")
assert report["complete"] and report["status"] == "incumbent_retained"
assert report["driver_sha256"] == ready["files"]["run.py"] == launch["driver_sha256"]
assert report["source_commit"] == ready["source_commit"] == plan["source_commit"]
assert report["pid"] == launch["pid"] == 399
assert report["candidate_plan_sha256"] == digest(root / "plan/plan.json")
assert report["source_manifest_sha256"] == digest(root / "source-sha256.json")
assert not report["refinements"] and not report["promotions"]
assert report["native_solves_started"] == report["native_solves_completed"] == 0
assert not (root / "output/native-solves.jsonl").exists()
baseline, best = report["baseline"], report["best"]
assert baseline["ok"] and baseline["independent"]["ok"] and baseline["official"]["ok"]
assert baseline["official"]["return_code"] == 0
assert baseline["score_kg"] == best["score_kg"] == 12810.135953048577
assert baseline["total_mass_kg"] == best["total_mass_kg"] == 14051.854893908598
assert best["source"] == "retained_v595"
assert digest(root / "inputs/Result.txt") == plan["incumbent_sha256"]
assert baseline["independent"]["ships"] == baseline["official"]["ships"] == 23
threshold = 23 * math.log(23 / 2) / 0.004
assert baseline["total_mass_kg"] >= threshold
cases = {case["case"]: case for case in plan["replacement_cases"]}
assert len(cases) == len(report["screened_cases"]) == 496
assert {row["case"] for row in report["screened_cases"]} == set(cases)
assert all(row["status"] == "screened" for row in report["screened_cases"])
samples = [sample for row in report["screened_cases"] for sample in row["samples"]]
assert len(samples) == 1488
assert all(not sample["feasible_surrogate"] for sample in samples)
assert len(report["retimings"]) == 12 and len(report["polishings"]) == 2
assert authority["GPU_report_sha256"] == digest(root / "output/report.json")
assert authority["all_GPU_CPU_outcomes_agree"] and authority["GPU_calls"] == 0
assert not authority["native_library_load_attempts"]
assert all(row["feasible"] for row in authority["originals"])
assert all(row["measured_legs"] == row["flight_legs"] for row in authority["originals"])

bonus_path = Path("/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data/bonus_coefficients.txt")
assert digest(bonus_path) == report["bonus_sha256"]
coefficients = [float(line.split()[0]) for line in bonus_path.read_text().splitlines() if line.strip()]
ship_rows = {row["ship"]: row for row in plan["all_ships"]}
occupied = set(plan["occupied_asteroids"])
proxy_checks = []
expected_joint_rows = len(samples)
for row in [*report["retimings"], *report["polishings"]]:
    proxy = row.get("proxy")
    if not proxy:
        continue
    case = cases[row["case"]]
    original = read(root / f"inputs/ship-{case['ship']:02d}.json")["plan"]
    path = root / "output/proxies" / Path(proxy["path"]).name
    candidate = read(path)
    deploy = {int(a): float(t) for a, t in candidate["deploy_epochs"].items()}
    collect = {int(a): float(t) for a, t in candidate["collect_epochs"].items()}
    expected_deploy = {case["new"] if int(a) == case["old"] else int(a) for a in original["deploy_epochs"]}
    expected_collect = {case["new"] if int(a) == case["old"] else int(a) for a in original["collect_epochs"]}
    assert set(deploy) == expected_deploy and set(collect) == expected_collect
    assert case["new"] not in occupied and not candidate["foreign_deploy_epochs"]
    masses = {int(a): float(m) for a, m in candidate["collected_mass_kg"].items()}
    assert set(masses) == set(collect)
    assert all(collect[a] - deploy[a] >= 365.25 - 1e-6 for a in collect)
    assert all(m >= 0 and m <= 10 * (collect[a] - deploy[a]) / 365.25 + 1e-7 for a, m in masses.items())
    raw = sum(masses.values())
    weighted = sum(m * coefficients[a - 1] for a, m in masses.items())
    before = ship_rows[case["ship"]]
    raw_gain, weighted_gain = raw - before["verified_raw_kg"], weighted - before["verified_weighted_kg"]
    assert abs(raw - proxy["predicted_raw_kg"]) < 1e-8
    assert abs(weighted - proxy["predicted_weighted_kg"]) < 1e-8
    assert abs(raw_gain - proxy["predicted_raw_gain_kg"]) < 1e-8
    assert abs(weighted_gain - proxy["predicted_weighted_gain_kg"]) < 1e-8
    eligible = weighted_gain > 0.5 and baseline["total_mass_kg"] + raw_gain >= threshold + 1e-6
    assert eligible == proxy["eligible_for_refinement"] is False
    if proxy["origin"] == "joint":
        # Both meshes took their full two accepted moves, hence exactly one
        # initial row and four complete move batches for each polished plan.
        assert row["moves"] == 4
        visits = sum(leg["role"] != "camp" for leg in candidate["legs"]) + 1
        expected_joint_rows += 1 + 4 * 2 * (3 + 5 * (visits - 2))
    proxy_checks.append({"case": row["case"], "origin": proxy["origin"], "raw_kg": raw,
                         "weighted_kg": weighted, "raw_gain_kg": raw_gain,
                         "weighted_gain_kg": weighted_gain, "eligible": eligible,
                         "path": path.relative_to(root).as_posix(), "sha256": digest(path)})
telemetry = report["screening_telemetry"]
assert telemetry["gpu_used"] and telemetry["backend"] == "cuda"
assert telemetry["completed_joint_evaluations"] == expected_joint_rows == 2778
assert telemetry["completed_joint_batches"] == 496 + 2 * 5 == 506
assert telemetry["completed_retime_driver_calls"] == 12
assert telemetry["completed_neighbour_queries"] == 24
assert report["eligible_proxy_plans"] == 0
assert len(proxy_checks) == 4
authority_rows = [row for row in authority["samples"] if row["CPU_failure"] == "leg_authority"]
assert len(authority_rows) == 1487
assert all(row["failure_on_changed_incident_leg"] for row in authority_rows)
assert all(not row["first_failure"]["measured_key_exists"] for row in authority_rows)
ratios = sorted(row["first_failure"]["authority_ratio"] for row in authority_rows)
audit = {
    "kind": "saved_result_inventory_score_counts_and_CPU_replay_audit",
    "passed": True, "additional_GPU_calls": 0, "additional_physics_refinements": 0,
    "driver_sha256": report["driver_sha256"], "source_commit": report["source_commit"],
    "report_sha256": digest(root / "output/report.json"),
    "ready_manifest_sha256": digest(root / "ready-manifest.json"),
    "scored_objective": "fixed_bonus_weighted_returned_mass_kg",
    "retained_raw_kg": best["total_mass_kg"], "retained_weighted_kg": best["score_kg"],
    "both_stored_checkers_pass": True, "promotions": 0, "full_refinements": 0,
    "cases": len(cases), "initial_samples": len(samples),
    "initial_failure_counts": dict(Counter(sample["failure"] for sample in samples)),
    "retime_calls": 12, "feasible_retime_plans": 2, "polished_plans": 2,
    "joint_rows_reconciled": expected_joint_rows, "joint_batches_reconciled": 506,
    "proxy_checks": proxy_checks,
    "authority_replay": {
        "originals_pass_identical_proxy_settings": True,
        "all_CPU_GPU_sample_outcomes_agree": True,
        "changed_incident_leg_failures": 1487, "unchanged_leg_failures": 0,
        "empirical_ratio_limit": 0.55,
        "first_failure_ratio_min_median_max": [ratios[0], ratios[len(ratios)//2], ratios[-1]],
        "first_failure_ratios_at_most_one": sum(r <= 1 for r in ratios),
        "physical_feasibility_of_rejected_samples_established": False,
    },
    "telemetry": telemetry,
    "timing": {"campaign_seconds": report["seconds"],
               "baseline_verification_seconds": baseline["verification_seconds"],
               "fixed_screen_seconds": report["fixed_screen_seconds"],
               "search_seconds": report["proxy_seconds"]},
}

target.mkdir(parents=True, exist_ok=False)
packed = {name: expected for name, expected in ready["files"].items()
          if name.startswith(("source/", "inputs/"))}
with tarfile.open(target / "source.tar.gz", "w:gz") as archive:
    for name in sorted(packed):
        archive.add(root / name, arcname="execution/" + name, recursive=False)
with tarfile.open(target / "source.tar.gz", "r:gz") as archive:
    assert len(archive.getmembers()) == len(packed)
    for member in archive.getmembers():
        name = member.name.removeprefix("execution/")
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == packed[name]
for name in ready["files"]:
    if name not in packed:
        copy(root / name, target / "execution" / name)
copy(root / "ready-manifest.json", target / "execution/ready-manifest.json")
for name in ("launch.json", "run.log"):
    copy(root / name, target / name)
copy(root / "output/report.json", target / "report.json")
copy(root / "audit/authority-replay.json", target / "audit/authority-replay.json")
copy(root / "diagnose_authority.py", target / "audit/diagnose_authority.py")
copy(repo / "build/performance/run_substitution_v599_foreground.py", target / "provenance/run_substitution_v599_foreground.py")
copy(Path(__file__), target / "provenance/publish_result.py")
raw_files = {p.relative_to(root).as_posix(): digest(p) for p in sorted((root / "output").rglob("*")) if p.is_file()}
with tarfile.open(target / "raw.tar.gz", "w:gz") as archive:
    for name in sorted(raw_files):
        archive.add(root / name, arcname=name, recursive=False)
with tarfile.open(target / "raw.tar.gz", "r:gz") as archive:
    assert len(archive.getmembers()) == len(raw_files)
    for member in archive.getmembers():
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == raw_files[member.name]
write(target / "raw-files.json", raw_files)
write(target / "audit/result-audit.json", audit)
(target / ".gitattributes").write_bytes(b"* -text whitespace=cr-at-eol\n*.log -whitespace\n")

readme = """# GPU asteroid-substitution search — v599

The bounded local RTX 5090 run completed and retained the verified v595 fleet:
**12,810.135953 weighted kg / 14,051.854894 physical kg**, 23 ships. Both freshly
run full-fleet checkers pass. No candidate qualified for physical refinement;
this experiment produced no score improvement.

| Measured work | Result |
| --- | ---: |
| Ships explored | 2, 21, 7 |
| Distinct paired asteroid substitutions | 496 |
| Initial whole-itinerary epoch samples | 1,488 |
| GPU neighbor queries | 24 |
| GPU retiming driver calls | 12 |
| Internal DP / forward calls | 93 / 93 |
| Feasible retimed surrogate plans / polished plans | 2 / 2 |
| Joint candidate rows / batches | 2,778 / 506 |
| Logical Lambert branch requests | 6,430,118 |
| Search stage | 2.044482 s |
| Initial substitution screening | 1.402124 s |
| Baseline independent and official checks | 21.187483 s |
| Campaign timer | 23.409579 s |
| Complete physical route refinements / promotions | 0 / 0 |

These are intermediate search counts, not certified trajectories per second.
Timing starts after initial imports, source/input validation, loading and parsing;
the search stage begins after entering the GPU context. The campaign includes
baseline CPU verification. Logical branch requests can repeat cached geometry.
There is no baseline comparison or whole-campaign speedup claim.

## What failed and what the diagnosis establishes

All 496 prepared cases were screened, with no CPU/GPU neighbor-union mismatch.
Of 1,488 initial epoch samples, 1,487 failed the empirical thrust-authority screen
and one failed the final surrogate mass budget. Ten of twelve retimings failed
their mass budget. The two feasible retimings improved slightly through eight
joint timing moves, but even the better alternative loses **54.476775 weighted
kg and 72.991102 raw kg** relative to its original ship. None meets the weighted
gain and fleet raw-mass gate, so no expensive SCvx refinement was attempted.

A separate **CPU-only** replay checked the exact frozen evaluator, without
changing arithmetic or thresholds or loading native libraries. All three original
certified routes pass the identical proxy settings, reusing all 16/18/17 measured
flight legs. All 1,488 CPU sample outcomes match their recorded GPU reasons.
Every authority failure is on a changed incident leg: 1,421 deployment hops and
66 collection hops. None arises on an unchanged leg losing its measured-mass
allowance. No measured cost was accidentally attached to a replacement endpoint.

The empirical authority limit is 0.55; first-failure ratios range from 0.5524 to
11.899, with median 1.5553. However, **419 lie at or below a unit authority ratio**.
These are conservative surrogate rejections, not physical infeasibility proofs.
No rejected replacement received SCvx/independent physical certification here,
so the experiment cannot establish the screen's false-negative rate.

The next distinct hypothesis is to select and seed replacements by their four
incident-leg windows, bottleneck authority and downstream mass budget, then
retime the least-excess candidates. This directly tests reachability around the
existing chain rather than ranking primarily by nearby orbital elements and
bonus gain. It does not require changing acceptance gates or rerunning this
same sampled neighborhood unchanged.

## Objective and retained evidence

Each trial substitutes an asteroid at both deployment and collection, preserves
the Earth endpoints/miner inventory/ship count, and excludes every other ship's
asteroid footprint. The fleet has 8.359441 raw kg of ship-count slack. A weighted
gain could use that slack; the driver does not incorrectly require raw mass to
increase. The four saved surrogate plans have independently recomputed raw and
weighted cargo, unchanged inventories and no eligible improvement.

The unchanged baseline Result is preserved in the source archive's inputs and
already published in [v595](../orphan-recovery-v595/Result.txt). No new viewer
dataset was added. `report.json` contains the complete run and both checker
summaries; `raw.tar.gz` retains every output file, including all four proxy plans
and the complete per-sample failures. `audit/result-audit.json` reconciles the
work counts and score arithmetic. `audit/authority-replay.json` retains every
CPU replay and inspected first-failure frame.

## Exact source and reproduction

The run uses 190 frozen published files at
`3091c716714c8bdec364d54c5e7357f2b5d85730`, the v596 CUDA core
`86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`
and QOCO540
`0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`.
The exact tested driver SHA-256 is
`9e668824fc4542da103777313c23694bae49a628cd5c5f266805cb1f537101a0`.
Joint batch/device selection were enabled; numerical and physics tolerances
were unchanged. Python orchestration and CPU verification remain.

`execution/` contains the exact driver, preparation, 61-test CPU validation,
candidate plan and original 237-file ready manifest. `source.tar.gz` stores the
frozen source and inputs once, with paths rooted at `execution/`; extracting it
reconstructs that complete preparation. Source and archive member hashes were
verified during packaging. The top-level `sha256.json` indexes publication bytes.
The original foreground wrapper is retained under `provenance/`, alongside the
publisher audit. `launch.json` records PID 399, flags and native library paths.

For a deliberate reproduction, extract `source.tar.gz` into this directory,
provide the pinned catalogue/bonus data and recorded compatible native libraries,
then inspect `execution/launch.py` without `--execute`. The saved launcher can
start a reviewed reproduction with `--execute --output /absolute/fresh/output`;
it requires the shared GPU lock and refuses existing launch/output files.
Do not treat a repeated identical run as a new search hypothesis.
"""
(target / "README.md").write_text(readme)
index = {p.relative_to(target).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
         for p in sorted(target.rglob("*")) if p.is_file()}
write(target / "sha256.json", index)
for name, info in index.items():
    assert digest(target / name) == info["sha256"]
    assert (target / name).stat().st_size == info["bytes"]
print(json.dumps({"published": str(target), "indexed_files": len(index),
                  "source_archive_files": len(packed), "raw_archive_files": len(raw_files),
                  "sha256_index": digest(target / "sha256.json"),
                  "result_audit_sha256": digest(target / "audit/result-audit.json")}))
