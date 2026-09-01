# OrbitWeaver G7 implementation devlog

## 2026-09-01

- Created `feat/orbitweaver-gpu` at
  `a33e950e06b0a302815fb079dc95f356c13db5fd` in the isolated WSL worktree.
- Mapped existing Lambert, low-thrust, route-master, column-generation, dynamic-grid,
  robust-risk, certification and public G3 APIs.
- Chose fixed Lambert output slots rather than atomic compaction to preserve ordering and
  all failure classifications.
- Added CPU C ABI parity and CUDA persistent workspace paths.
- Added a solver-neutral persistent backend seam and rank/device ownership policy.
- Added bounded scheduler/backpressure telemetry, stable top-K, scenario risk,
  deterministic restart and independent certification.
- Added Python orchestration/records, CLI validation and frozen Paper 2 matrix hash.
- Configured unique native Debug/Release and CUDA Debug/Release directories.
- Built native Werror/sanitizer-capable and CUDA 12.8 `sm_120` targets.
- Passed bounded CPU, Python and actual one-GPU Lambert parity tests.
- Deliberately did not run timing, energy, full route, or physical multi-GPU experiments.

## Corrections made during the loop

- Corrected new-file destination paths to the isolated WSL worktree before compiling.
- Used the worktree-local Python environment to avoid the canonical editable-install hook.
- Replaced release-disabled test assertions where Werror exposed unused variables.
- Pinned the benchmark loader to the byte-exact worktree matrix hash.

## Unified-roadmap serialized validation

- Rebuilt the unified Release, Debug, and sanitizer-capable CUDA targets for `sm_120`.
- Passed the Release one-GPU OrbitWeaver test on the RTX 5090 after native-QOCO integration.
- Exercised the concrete persistent G3 backend callback and route-result propagation seam.
- Retained the boundary: this is one-GPU correctness only. No physical multi-GPU scaling,
  complete route campaign, energy claim, G7 acceptance, or Paper 2 claim is made.

## 2026-09-02 schema audit

- Confirmed Python manifests had drifted from their JSON schema by omitting the required
  Paper 2 matrix hash.
- Moved config, manifest, checkpoint and result schemas into one in-package authority and
  added deterministic schema generation/check mode.
- Added strict record read/write paths with atomic output, finite JSON enforcement,
  nested unknown-field rejection and manifest pin/repeat cross-checks.
- Expanded terminal status semantics for failed, censored, unsupported, OOM, timeout,
  infeasible and cancelled records while retaining partial bounds where valid.
- Added repository/config/matrix, toolchain, hardware, seed and repeat capture.
- Added round-trip, adversarial and seeded differential schema tests.
- Built and installed the wheel, ran its CLI, generated a pinned manifest and validated it
  independently with Draft 2020-12 `jsonschema`.
- Ran no GPU timing, energy, throughput or multi-GPU experiment.

## 2026-09-02 G3/G5 concrete adapters

- Created the isolated `feat/orbitweaver-g3-g5-adapter` worktree from unified commit
  `e95b902d718ceaf05523e469cbe21945013c2f41`.
- Cherry-picked only schema-parity commit
  `bf9d10af541c995f1bdcd10b031486cff6b4351e`; retained the integrated original G7 code.
- Added a bounded Python G3 adapter with persistent topology/rank/device workspaces,
  in-place numerical updates, opaque compatible warm states, separate canonical/replay/path/
  terminal diagnostics and explicit failure/censor classifications.
- Added the C++ `G3PersistentTrajectoryAdapter` over the public device-SCvx C API.
- Added deterministic G5 route/arc/scenario partitioning, rank-local ownership/backend
  adapters, checkpoint compatibility, status propagation and collective telemetry surfaces.
- Connected scenario risk results to route columns using real returned costs/lower bounds;
  only independently certified route combinations may become incumbents.
- Added deterministic fixture tests for the full coarse/refined/scenario/master/certification
  path and failure modes. Fixtures are labelled non-evidence.
- Added logical-rank ownership coverage and a CUDA compile/link contract test.
- Validation:
  - native Debug ASan/UBSan/Werror: 43/43 CPU tests passed;
  - native Release Werror: 43/43 CPU tests passed;
  - CUDA 12.8 + G5 Debug/Release compiled for `sm_120`;
  - G5 logical-rank contract passed in Debug and Release;
  - adapter/schema Python selection: 40 passed;
  - Ruff and generated-schema checks passed.
- Kept all GPU executables, energy collection and physical multi-GPU runs disabled while
  shared validation remained active.

## 2026-09-02 single-GPU scope

- Added schema-v2 G7 manifests with `campaign_scope_id`; schema-v1 historical records remain
  readable.
- Made `single-gpu-v1` require one-device ownership and reject physical-multi-GPU evidence labels.
- Defined `complete-in-scope` as the full coarse/refined/scenario/pricing-master/certification/
  visualisation flow with independently certified results.
- Kept physical route-by-scenario scaling, throughput, energy, memory crossover, and
  tractability-frontier claims in the preserved deferred backlog.
