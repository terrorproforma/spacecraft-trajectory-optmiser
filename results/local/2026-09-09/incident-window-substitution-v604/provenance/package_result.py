"""Package all v604 evidence once, retaining raw/source hashes without duplicate trees."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parents[2] / "results/local/2026-09-09/incident-window-substitution-v604"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), destination
    shutil.copy2(source, destination)
    assert digest(source) == digest(destination)


def archive(path, files, *, prefix=""):
    with tarfile.open(path, "w:gz") as stream:
        for name in sorted(files):
            stream.add(ROOT / name, arcname=prefix + name, recursive=False)
    with tarfile.open(path, "r:gz") as stream:
        assert len(stream.getmembers()) == len(files)
        for member in stream.getmembers():
            name = member.name.removeprefix(prefix)
            assert member.isfile()
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == files[name]


ready = read(ROOT / "ready-manifest.json")
audit = read(ROOT / "audit/result-audit.json")
report = read(ROOT / "output/report.json")
assert audit["passed"] and audit["additional_GPU_calls"] == 0
assert digest(ROOT / "output/report.json") == audit["report_sha256"]
assert report["complete"] and report["status"] == "incumbent_retained"
assert report["controlled_comparison"]["equal_full_budgets_completed"]
assert (
    report["eligible_proxy_plans"] == 0 and not report["refinements"] and not report["promotions"]
)
for name, expected in ready["files"].items():
    assert digest(ROOT / name) == expected, name

TARGET.mkdir(parents=True, exist_ok=False)
packed = {
    name: expected
    for name, expected in ready["files"].items()
    if name.startswith(("source/", "inputs/"))
}
archive(TARGET / "source.tar.gz", packed, prefix="execution/")
for name in ready["files"]:
    if name not in packed:
        copy(ROOT / name, TARGET / "execution" / name)
copy(ROOT / "ready-manifest.json", TARGET / "execution/ready-manifest.json")
for name in ("launch.json", "run.log", "supervisor.json", "supervisor-attempt-01.json"):
    copy(ROOT / name, TARGET / name)
for name in ("foreground_supervisor.py", "audit_result.py", "package_result.py"):
    copy(ROOT / name, TARGET / "provenance" / name)
copy(ROOT / "audit/result-audit.json", TARGET / "audit/result-audit.json")
raw = {
    path.relative_to(ROOT).as_posix(): digest(path)
    for path in sorted((ROOT / "output").rglob("*"))
    if path.is_file()
}
archive(TARGET / "raw.tar.gz", raw)
write(TARGET / "raw-files.json", raw)

probes = []
for key, mode in (
    ("ship_02_replace_31302_with_2181", "deploy_-30_collect_-30"),
    ("ship_07_replace_15206_with_3150", "deploy_-30_collect_+15"),
):
    case = next(row for row in report["screened_cases"] if row["case"] == key)
    sample = next(row for row in case["samples"] if row["mode"] == mode)
    estimate = sample["ranking_estimate"]
    assert sample["failure"] == "mass_below_dry_plus_collected"
    assert estimate["complete"] and estimate["fleet_raw_shortfall_kg_estimate"] == 0
    assert estimate["mining_weighted_gain_kg_estimate"] > 0.5
    probes.append(
        {
            "case": key,
            "mode": mode,
            "failure": sample["failure"],
            "scope": "unexecuted_truth_set_recommendation_not_certified",
            "ranking_estimate": estimate,
        }
    )
write(
    TARGET / "audit/next-hypothesis.json",
    {
        "hypothesis": (
            "The calibrated propellant proxy may reject cargo-preserving fixed-epoch "
            "replacements that low-thrust refinement can certify."
        ),
        "status": "recommended_only_not_GPU_executed",
        "maximum_whole_route_refinements": 4,
        "positive_controls": ["original_v595_ship_02", "original_v595_ship_07"],
        "candidate_probes": probes,
        "protocol": [
            "Reproduce each original route through the same pinned native SCvx settings first; "
            "a control failure invalidates inference about the proxy.",
            "Probe the two saved schedules at their score-preserving mined quantities, "
            "retaining every native solve and failure; do not retime them into lower-cargo "
            "routes first.",
            "Compare certified propellant/cargo against the recorded projection; a failed "
            "refinement is an unresolved numerical/feasibility result, "
            "not a proof of impossibility.",
            "Only a fully certified route installed in the original fleet and accepted by both "
            "full-fleet checkers can be promoted; weighted improvement and raw ship-count "
            "feasibility remain mandatory.",
            "This two-probe set can detect a false negative or support a local calibration "
            "diagnosis. It cannot establish the population false-negative rate.",
        ],
    },
)

readme = """# GPU incident-window substitution search — v604

The controlled local RTX 5090 search completed and **did not improve the fleet**.
The retained v595 result remains **14,051.854894 raw kg / 12,810.135953 fixed-bonus
weighted kg**, with 23 ships. Fresh baseline official and independent checks passed.
No replacement became a certified fleet, and no new viewer dataset was added.

## Controlled result

Both policies used exactly the same 496 replacement cases, three ships and
12,400-row two-sided deployment/collection epoch grid. Both received twelve
retiming calls with equivalent independently initialized state and alternating
arm order. The historical arm retained v599's exact twelve choices; the new arm
ranked complete four-incident windows and downstream mass estimates.

| Recorded work / outcome | Historical priority | Complete incidence |
|---|---:|---:|
| Retiming calls | 12 | 12 |
| Logical Lambert branch requests in retiming | 6,421,104 | 6,421,104 |
| Resident retiming cells | 3,210,552 | 3,210,552 |
| Native DP / forward calls | 93 / 93 | 86 / 86 |
| Forward-feasible retimed plans | 2 | 3 |
| Objective-eligible plans | 0 | 0 |
| Recorded retiming-call seconds | 0.393263 | 0.258130 |

The new ranking produced one additional forward-feasible plan in this small
controlled neighborhood. This is not a score gain, a general performance
multiplier or proof of better global solutions. The shared grid was larger
than v599's 1,488 samples; grid-budget expansion is distinct from the equal-budget
shortlist-policy comparison. Timings are descriptive single-run call timings,
not isolated kernel benchmarks.

All 12,400 fixed-grid rows failed complete native forward: 12,369 authority
failures, ten mass failures and 21 infeasible Lambert legs. There were 10,620
complete ranking estimates and eleven near-authority-threshold estimates.
An estimate is not a feasible itinerary or a physics certificate.

Four forward-feasible plans were polished. The best new-policy polished case,
ship 2 replacing 41045 with 25996, returned a surrogate 589.404517 raw kg and
484.449193 weighted kg: **42.847365 raw kg / 32.684285 weighted kg below that
incumbent ship**. The fleet has only 8.359441 raw kg of ship-count slack. The
objective/raw-mass gates correctly admitted no full refinements: zero SCvx calls,
zero full-route refinement attempts and zero promotions. All nine saved
surrogate plans have independently recomputed cargo, bonus objective and miner
inventory in the saved-result audit.

## Work and timing scope

The run performed **14,940 total joint itinerary rows**, 516 joint batches,
12,863,886 logical Lambert branch requests, 6,431,943 element hops, 179 native
retiming DP/forward calls and 24 retiming driver calls. These are different
work units; Lambert requests and DP cells are not complete trajectory solutions.
The approved caps were 15,220 total joint rows, 24 retimings and four full
refinements. All shared-grid and per-arm budgets completed.

The campaign took **56.255700 seconds**, including **20.897167 seconds** of
baseline verification. Screening/retiming/polishing and its controller/reporting
took 34.487976 seconds. Shared-grid screening took 9.916554 seconds, of which
1.235435 seconds was inside joint-evaluation calls and 1.050799 seconds inside
the explicitly CPU ranking diagnostic. The remaining controller time was not
profiled into exclusive categories; frequent serialization of the 38.5 MB raw
report is included. Do not attribute all wall time to GPU kernels or call the
ranking/controller and CPU independent verification fully GPU-native.

## Next distinct hypothesis: a small low-thrust truth set

The next useful test is whether the calibrated propellant proxy rejects
cargo-preserving schedules that native low-thrust refinement can certify.
Re-running a larger neighbor search would not answer that question.

Use **four whole-route refinements maximum**: the original v595 ships 2 and 7
as positive controls, then these two retained schedules:

| Unexecuted probe | Epoch shifts | Weighted mining potential | Propellant shortfall estimate |
|---|---:|---:|---:|
| Ship 2: 31302 → 2181 | -30 / -30 days | +6.236203 kg | 148.881649 kg |
| Ship 7: 15206 → 3150 | -30 / +15 days | +9.908765 kg | 153.112966 kg |

Their mining quantities preserve the fleet raw-mass threshold, but the proxy
predicts 6.83% and 7.02% excess propellant use respectively. Those are meaningful
mass deficits, not merely rounding errors. Both were rejected for mass by
native forward. The truth set should retain their proposed cargo and epochs
while measuring low-thrust feasibility, instead of first retiming away the
potential score gain. Exact saved diagnostics and the proposed protocol are in
`audit/next-hypothesis.json`; **no such probe has been executed here**.

Positive-control failure must be resolved before drawing conclusions about proxy
false negatives. A failed candidate solve is not proof of physical impossibility.
Only unchanged physical tolerances, a certified route and both full-fleet
checkers can establish a promotable weighted improvement. Two probes can detect
a false negative or inform local calibration; they cannot estimate a population
false-negative rate reliably. No production acceptance gate was weakened.

## Source, process and raw evidence

The exact 190-file Python snapshot is published commit
`3091c716714c8bdec364d54c5e7357f2b5d85730`. The CUDA core is validated v596
`86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`,
and QOCO540 is
`0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`.
The tested driver hash is
`05be6b3638e3e1209228f74272fcb29815a5aef5a0fadd45ef54362a603bbe66`.
The ready manifest indexed 241 files; 78 CPU tests passed with zero skipped,
plus the construction audit and Ruff. All pinned hashes were rechecked.

The child launched once as Linux PID 382 under foreground supervisor PID 305
(tool session 22812) and exited zero. Its preflight acquired the actual shared
GPU lock and observed no compute processes. Launch, supervisor source/hash,
preflight record and complete log are retained. The supervisor remained in the
foreground until exit, and the GPU slot was released before another agent's
tests resumed. No remote operations or production source changes were made.

`source.tar.gz` stores the frozen source and incumbent inputs once, rooted at
`execution/`; extracting it reconstructs the original preparation alongside
the exact files in `execution/`. `raw.tar.gz` retains every output byte,
including the full report and all nine proxy plans, with archive paths rooted
at `output/`. `raw-files.json` indexes those uncompressed bytes. The raw report
is compressed rather than duplicated as another 38.5 MB file. The independent
saved-result audit is directly readable at `audit/result-audit.json`.
`sha256.json` indexes every publication file; each index and archive member was
verified when packaging. The unchanged flown Result is also published in
[v595](../orphan-recovery-v595/Result.txt).

For an intentional reproduction, extract the source archive into this directory,
provide the recorded native libraries and pinned catalogue/bonus table, and
inspect `execution/launch.py` without `--execute`. A reviewed reproduction must
use fresh output/launch files and the shared GPU lock. The production search
source may have evolved after this frozen run; it is not substituted silently.
"""
(TARGET / "README.md").write_text(readme)
(TARGET / ".gitattributes").write_bytes(b"* -text whitespace=cr-at-eol\n*.log -whitespace\n")
index = {
    path.relative_to(TARGET).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
    for path in sorted(TARGET.rglob("*"))
    if path.is_file()
}
write(TARGET / "sha256.json", index)
for name, info in index.items():
    assert (TARGET / name).stat().st_size == info["bytes"]
    assert digest(TARGET / name) == info["sha256"]
print(
    json.dumps(
        {
            "published": str(TARGET),
            "indexed_files": len(index),
            "source_archive_files": len(packed),
            "raw_archive_files": len(raw),
            "sha256_index": digest(TARGET / "sha256.json"),
            "result_audit_sha256": digest(TARGET / "audit/result-audit.json"),
        }
    )
)
