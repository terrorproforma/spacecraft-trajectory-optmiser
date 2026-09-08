# GPU incident-window substitution search — v604

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
