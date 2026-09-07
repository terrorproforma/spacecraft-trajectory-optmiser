# Certified objective and incumbent fleet diagnosis — 8 September 2026

The retiming certification loop optimised a weighted objective but chose its
preferred certified variant by raw collected kilograms. For example, it preferred
150 kg at weight 0.5 over 100 kg at weight 1.0, despite the latter scoring 100
against 75. It now ranks actual refined payloads with `plan_value`, preserving
the configured bonus weights and speculative orphan credit. Orphan credit remains
a planning estimate, not material already collected or a verified fleet score.

Refinement can reduce payload to close the mass budget. The comparison therefore
uses `RefinedRoute.collected_mass`, not the original proxy payload. All certified
variants remain available to the fleet master. Equal objectives keep the first
variant. This change does not alter dynamics, solver tolerances, physics gates,
or the policy used to generate the next candidate.

Five controller regression cases cover raw/weighted disagreement, no weights,
payload sizing, equal scores and orphan credit. They deliberately inject candidate
and refinement outcomes; they test selection semantics, not physical feasibility.
Together with twelve existing GPU DP parity cases, **17 tests pass locally in
1.32 s and on H100 in 0.86 s**. The H100 copy's source hashes match the published
Python change. No CUDA library changes are required.

## Fleet diagnosis

The eight previously missing exact-mass incumbent summaries were recovered from
Lambda's `joint_itinerary_v11` archive, with byte hashes retained. All 23 selected
ship summaries are included in the evidence directory. These are recovered
historical inputs, not newly certified trajectories.

The previous v222 scan combined `cluster_retime_settings` with generic
`SearchSettings`. That missed the cluster-specific Earth-leg propellant model.
The archived ship 1 report and the isolated CPU/GPU comparison both report a
forward mass-budget failure after eight price rounds; they do not establish a DP
implementation disagreement. The new scan uses `cluster_search_settings` as well.
The production cluster planner already supplies this configuration; this corrects
the exploratory experiment rather than changing the production search defaults.

| Experiment on RTX 5090 | Evaluations | Elapsed | Outcome |
|---|---:|---:|---|
| v226: 23 ships, uncalibrated and certified-leg-calibrated models | 25,473,600 Lambert branches; 201 DP calls | 14.328 s | All 46 proxy schedules close; none improves weighted score |
| v227: ships 18 and 20, 5-day grid | 9,204,208 Lambert branches; 9 DP calls | 2.590 s | Ship 20 predicts +0.424800 weighted kg and +0.602327 raw kg |

These are single scans, including table construction, lookup, bookkeeping and
incremental report writing. They are not repeated comparative speed benchmarks.
Both calibration passes retain the existing 3% calibration margin. A feasible
proxy schedule still requires every low-thrust leg to be flown and independently
verified.

H100 v228 attempts the promising ship-20 schedule using GPU seed, discretisation,
assembly, QOCO and outer-loop backends, at the existing 2-day node spacing and
40-iteration SCvx limit. It stops on the first Earth-to-4524 leg in **22.435 s**,
reporting residual virtual control **7.817e-3** and no certified route. No solution
is sent to the physics checkers because refinement did not certify it. This is a
failed optimisation attempt, not proof that the schedule is physically impossible.

No candidate replaces an incumbent. The verified fleet remains **23 ships,
14,047.803 raw kg and 12,805.194 weighted kg**. Route-order search, better seeding
and continuously refined departure timings remain open avenues. The GPU pipeline
also still has the host table round trip and CPU orchestration described in
[GPU_RETIMING_DP.md](GPU_RETIMING_DP.md).

## Evidence and reproduction

All inputs, scans, proposed schedules, the unsuccessful H100 refinement and test
logs are in [fleet-objective-v229](../results/lambda/2026-09-08/fleet-objective-v229/).
`evidence-sha256.json` covers 102 evidence artifacts. `incumbent-sources` contains
the 23 exact input summaries, and `recovered-archives` retains the eight-file
retrieval manifest and available original solution text. `local-v226/report.json`
records the source path and hash for each ship and both calibration policies.

The archived `fleet_retime_v226.py` and `fleet_retime_v227.py` are the exact scan
drivers. They use the paths from the original WSL environment: the pinned data at
`/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data` and the frozen CUDA
library at `/home/angus/build-spacepdhcg-retime-dp-v219/final/libspacepdhcg_cuda.so`.
To replay elsewhere, restore the 23 summaries to their report-recorded repository
paths, configure those two runtime paths, and choose unused output directories in
the scripts. Run from the repository root with the project's Python dependencies.
The H100 `mission-v228/run.py` embeds the proposed plan and records solver settings;
its report retains the source hashes and failure diagnostics.

For the regression checks, set `SPACEPDHCG_GTOC12_CUDA_LIBRARY` to a CUDA library
built from the current source, set `SPACEPDHCG_GTOC12_GPU_TESTS=1`, and run with
`PYTHONPATH=src` under the per-device GPU lock:

```sh
python -m pytest tests/test_gtoc12_certified_objective.py tests/test_gtoc12_gpu_retime.py -q
```

The saved `remote-validation.py` copies the existing v220 source into an isolated
H100 directory, overlays the two changed Python files, and runs these checks.
