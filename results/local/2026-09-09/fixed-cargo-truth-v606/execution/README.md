# Fixed-cargo low-thrust truth set v606

Prepared experiment, **no GPU execution yet**. This tests whether the calibrated propellant proxy rejects two fixed schedules that the actual low-thrust solver can certify. It does not widen a previous search or change acceptance tolerances.

The retained v595 fleet has 23 ships, **14,051.854893908598 raw kg** and **12,810.135953048577 fixed-bonus weighted kg**, with both archived full-fleet checker certificates. A fresh baseline check is mandatory at execution.

| Case | Fixed change | Prescribed raw kg | Potential weighted fleet gain | Max leg solves |
| --- | --- | ---: | ---: | ---: |
| Ship 2 control | Original epochs and verified cargo | 632.251882 | 0 | 16 |
| Ship 2 probe | 31302 → 2181; deployment −30 days, collection −30 days | 632.251882 | +6.236203 kg | 16 |
| Ship 7 control | Original epochs and verified cargo | 614.428474 | 0 | 17 |
| Ship 7 probe | 15206 → 3150; deployment −30 days, collection +15 days | 615.660507 | +9.908765 kg | 17 |

The table rounds for display. `plan/plan.json` preserves the exact original route-artifact cargo and each prescribed replacement cargo; original values agree with the independently scored serialized Result to within the unchanged 1e-7 kg comparison threshold.

The hard limit is **four whole-route refinements, at most 66 native leg solves on one host**. H100 is preferred; the local profile is an alternative, not another budget. Each probe is skipped if its matching actual solver control fails. A route stops at its first failed leg because subsequent initial masses cannot then be certified. Controls use the same CUDA cold seed as probes; archived trajectories are not substituted for solver work. The original ship 2 has only about 2.68 kg of final dry-mass margin, so cold reproduction is a meaningful test.

`fixed_refine.py` preserves exact prescribed cargo and epochs in a single pass. It reuses the frozen stock scheduler, G3 adapter, SCvx driver, thrust clamp, independent DOP853 leg certificate and route master. The stock `refine_route` is intentionally not called because it replaces prescribed cargo with maximum mining production and can shrink cargo after a mass deficit. The isolated wrapper keeps the same event/mass ordering and requires final dry mass ≥500 kg with the existing numerical slack. No cargo scaling or epoch search is permitted on failure.

Every candidate must pass both full-fleet checkers with the prescribed cargo, retain the fleet's raw-mass ship-count admissibility, and improve the verified weighted objective before promotion. Failed native calls, solver reports/history, independent leg certificates, available solution arrays, unattempted legs and full-fleet certificates are retained. Each probe replaces one ship in the unchanged baseline; combining probes is outside this run. A saved improvement survives later failures or viewer-export errors.

The saved proxy deficits are 148.88 kg and 153.11 kg of propellant capacity, about 6.83% and 7.02%. These are selected calibration probes, not rounding errors. A certified probe would demonstrate one proxy false negative; a failed SCvx solve does not establish physical infeasibility. Two probes cannot estimate an overall false-negative rate. Complete measured propellant comparisons require a fully certified route.

## Frozen implementation and profiles

Python is pinned to `b08b1f5aa464659e558926713d29bcd364a3778f`. All 179 unchanged source files were checked against the actual successful v595 archive after normalizing CRLF to LF only; both raw hashes are retained in `reference/v595-python-compatibility.json`. The five central pipeline/refinement/verification modules also retain their original v595 bytes. The three changed joint/Lambert workspace modules are unused by fixed-route refinement. `source-sha256.json` indexes the complete 190-file Python/assets snapshot.

`preparation.json` preserves the initial local preparation record. `profiles.json` and `reference/h100-native-compatibility-report.json` add the later H100 compatibility evidence; the earlier `H100_exact_equivalent_proven: false` field is historical, not a failed H100 test.

Local uses v590 core `ffbae813…` and QOCO540 `0cc27a1d…`, exactly those used in successful v595 full-route validation. Exact v590 native source archive, manifest and build identity are in `reference/`. H100 uses existing v632 core `29a6b10b…` and QOCO545 `b50d6190…`, previously qualified by full-fleet runs. The root's H100 comparison found 352/355 native files identical; the two joint-search files are unused, and the remaining orbitweaver difference is an opt-in parallel-direction kernel forced off here. This is source-path evidence, not a new H100 measurement. Both binary hashes are checked freshly at launch and again in the child.

Both profiles use the exact v595 SCvx settings: CUDA seed, assembly, discretization and outer loop, GPU QOCO, graph execution, 40 SCvx iterations, 2-day nodes, ZOH, 8 substeps and the existing polish/certification settings. The complete settings are in `preparation.json`; no tolerance is changed. Python orchestration and independent CPU verification remain in this diagnostic harness.

## Review and launch

Inspect `fixed_refine.py`, `run.py`, `plan/plan.json`, `cpu-audit.json`, `validation/report.json` and `ready-manifest.json`. `launch.py` is itself the foreground supervisor; there is no detached-launch wrapper. Print-only mode performs no GPU work. Its execution mode takes the shared nonblocking GPU lock, checks actual compute processes, inherits the held lock into the child, and waits for completion. Busy/timeout observations are saved and never retried automatically. A single `launch.json` marker covers both profiles in this kit. The coordinator must retain the same global four-route limit across copies/hosts.

After root approval and transfer of **only indexed files** to the selected H100 working directory:

```bash
# Print and inspect the exact H100 recipe; this does not launch GPU work.
/home/ubuntu/spacepdhcg/v1/.venv/bin/python -B /path/to/truth-set-v606/launch.py --profile h100

# Authorized execution, once, in a durable foreground session.
/home/ubuntu/spacepdhcg/v1/.venv/bin/python -B /path/to/truth-set-v606/launch.py --profile h100 --execute
```

The local alternative uses `/home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B` with `--profile local`; do not run both. No launcher uploads files, compiles libraries, changes Git or mutates the visualizer. The default wall budget is 1800 seconds, checked before each route and leg; an in-flight native call retains the validated 900-second solver limit and may overrun the soft campaign budget. Output is written to a fresh `output/`, with `launch.json` and `run.log` beside this README. Every outcome, including no gain or failed controls, must be retained.

CPU tests mock native solve/rollout boundaries and block library loading. They verify construction, fixed cargo/mass events, failure paths, actual adapter plumbing, objective/control gates, source identity and supervisor exclusion. They do not certify either probe's low-thrust physics.
