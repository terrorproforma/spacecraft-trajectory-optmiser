# Conditioning helps some failures but loses otherwise valid trajectories

**No production setting changed.** A six-mode diagnostic of state translation
and GPU Ruiz equilibration looked promising on selected difficult legs, but the
full trajectory and campaign tests rejected enabling two Ruiz passes globally.
The zero-Ruiz default and all physics/objective tolerances remain unchanged.

Ruiz equilibration rescales the numerical system to balance coefficient
magnitudes. It leaves the intended optimization problem unchanged, but finite
precision, regularization and stopping behavior can differ. These measurements
use the same pinned native and QOCO binaries in every mode on each GPU.

## Pilot versus complete workloads

The eight-case pilot contains three previously certified legs and five rejected
legs. All modes use fresh workspaces and a uniform 30-second diagnostic budget
per leg. State origin off/on and zero/two/five Ruiz passes retain the same three
certified legs. With origin off, zero versus two passes gives:

| Hardware | Eight-case solve time, zero passes | Two passes |
|---|---:|---:|
| RTX 5090 | 28.1535 s | 6.2600 s |
| Lambda H100 | 25.5137 s | 7.2513 s |

The full 225-leg replay uses the original fixture's budgets and acceptance
limits. Pooling remains enabled where eligible: zero-Ruiz workspaces reuse
storage, whereas scaled workspaces rebuild. The full results reverse the pilot:

| Hardware | Solver time, zero passes | Two passes | Certified legs, zero / two |
|---|---:|---:|---:|
| RTX 5090 | 131.9902 s | 167.8111 s | 205 / 191 |
| Lambda H100 | 137.0648 s | 204.7752 s | 205 / 192 |

The scaled runs lose 14 and 13 previously certified legs respectively. All lost
legs end at the iteration limit. They are not counted as valid trajectories,
even when an intermediate state has small physical defects: objective
convergence and the remaining qualification requirements still apply.

Complete one-ship campaigns alternate baseline/candidate/candidate/baseline,
with two runs per mode. They preserve identical initial plans and logical search
counts and pass both mission checkers at 548.254620 weighted kg. Nevertheless,
median complete-process time rises from **28.2461 to 72.3277 seconds locally**
and **29.9808 to 82.4540 seconds on H100**. Workspace creations rise from 17 to
47, but repeated inner solves also contribute substantially. The scaled local
campaigns converge on only 41/40 of 47 attempted legs, versus 46 in both baseline
runs. The final fleet score alone would conceal that regression.

## Isolated numerical failure

Case 68 normally converges in four outer iterations. With scaling, its initial
inner solve can reach 200 iterations and miss the unchanged 1e-9 duality-gap
gate. Repeated qualification failures eventually shrink the trust region; the
outer solve then exhausts its budget before recovering the original objective.

The v526 capture freezes that case's first convex subproblem with zero and two
Ruiz passes. Dimensions, sparse topology and numerical input arrays are
identical; only the requested scaling count differs. The capture intentionally
permits one outer iteration and no polishing to isolate this QP, not to report a
completed trajectory.

The standalone GPU replay uses independent long-double sparse products to audit
primal/dual residuals, duality gap and cone membership. Its existing thresholds
remain 1e-9 for normalized residuals/gap and 1e-8 for cone violations. At the
default 1e-8 static regularization, the unscaled form passes **3/3** repeats and
the scaled form **0/3**, with 200 inner iterations in each scaled repeat.

Changing static regularization to 1e-10, 1e-12 or 1e-6 does not restore reliable
scaled qualification: only one of three repeats passes at 1e-10, and none at the
other two values. Tightening iterative-refinement tolerance from 1e-12 to 1e-14
or 1e-16, including increasing its maximum from 20 to 80, also fails all sampled
scaled repeats. Those experimental settings were not promoted. Translating the
state origin likewise fails to recover all three sampled lost trajectory cases.

## Objective normalization isolates the regression on both GPUs

Follow-up v529-v532 tests use the same captured QP on RTX 5090 and H100. Each
hardware runs three fresh processes per setup/graph combination. Zero Ruiz passes
qualify all 12 trials; two passes qualify none of 12. This holds with CUDA Graphs
enabled or disabled and with direct setup or device numerical updates. The graph
and numerical-update paths are therefore not necessary to reproduce this failure.

An instrumented diagnostic downloads the scaled matrices before solving. They
match the intended transformations `k D P D`, `E A D`, `F G D`, `k D c`, `E b`
and `F h`, including the configured P regularization. Maximum relative matrix
errors are 3.34e-16 on both GPUs. This checks the assembled problem, not each
subsequent factorization or Newton direction. CPU Clarabel 0.11.1 solves the
original QP and passes the same independent original-equation audit on both hosts.

The observed objective scale is **k = 0.0001**. The variable scaling is close to
one, and cone-row scaling ranges from 1 to approximately 1.297. Separating those
transformations outside the solver, with native Ruiz disabled, gives:

| Equivalent input transformation | RTX qualified / repeats | H100 qualified / repeats |
|---|---:|---:|
| None | 3 / 3 | 3 / 3 |
| Variable and constraint scalings only | 3 / 3 | 3 / 3 |
| Objective multiplied by 0.0001 only | 0 / 3 | 0 / 3 |
| Both transformations | 0 / 3 | 0 / 3 |

All unscaled-objective trials take 27 inner iterations. The external-transformation
diagnostic maps primal/dual/slack values back to the original problem before
qualification. It tightens native stopping tolerances in proportion to the
objective multiplier; the independent 1e-9 residual/gap and 1e-8 cone gates remain
unchanged. No accuracy improvement is claimed merely from evaluating smaller
scaled residuals. The instrumented traces are diagnostic, not throughput measurements.

This isolates objective normalization as sufficient to trigger this numerical
regression. It does not yet identify a faulty kernel or establish a general cure.
The [follow-up opt-in policy](GPU_OBJECTIVE_PRESERVING_RUIZ.md) now preserves
objective magnitude during equilibration and retains all 205 certified legs on
both GPUs. It does not establish an overall speedup; zero Ruiz remains the default.
The earlier static-regularization and iterative-refinement sweeps did not restore
reliable qualification. All production solver settings remain unchanged.

## Evidence

- [Local pilots, full replay, campaigns and isolated QP diagnostics](../results/lambda/2026-09-08/gpu-conditioning-local-v528/summary.json)
- [H100 pilot](../results/lambda/2026-09-08/gpu-conditioning-v520/analysis.json)
- [H100 full trajectory comparison](../results/lambda/2026-09-08/gpu-conditioning-legs-v522/analysis.json)
- [H100 complete campaigns](../results/lambda/2026-09-08/gpu-conditioning-campaign-v524/analysis.json)
- [Local setup-path, matrix and objective-scaling isolation](../results/lambda/2026-09-08/gpu-conditioning-cost-local-v531/summary.json)
- [H100 replication and retrieved logs](../results/lambda/2026-09-08/gpu-conditioning-paths-v532/summary.json)

Archives retain source/input snapshots, exact commands, numerical outputs and
library hashes. SHA-256 manifests cover published files and archive members.
The incumbent remains 12,805.194 weighted kg; these diagnostics create no new
fleet record or official leaderboard placement.
