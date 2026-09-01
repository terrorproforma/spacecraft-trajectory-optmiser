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
