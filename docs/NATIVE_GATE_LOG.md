# Full native gate log

## 31 August 2026 — gate expansion

The historical root CMake target compiled only an interface-header smoke test. The pre-GPU
completion branch changed the root build so every `cpp/tests/*_smoke.cpp` target is compiled
with warnings as errors and also exercised under ASan/UBSan.

This immediately exposed previously hidden integration gaps rather than regressions in a
formerly complete native build:

1. the higher-order transcription test existed, but its Euler/RK4 selector was not connected to
   the 3-DoF, 6-DoF or low-thrust transcriptions;
2. 6-DoF path parameters and an Euler step required by the native truth tests were absent;
3. the native 3-DoF SCvx driver referenced a decision decoder that had never been implemented;
4. the driver used stale forcing-policy and trust-action names.

The branch implements those interfaces instead of excluding the failing targets.

## 31 August 2026 — complete native suite reaches runtime

After the interface repairs, all 23 host-native smoke targets compiled with warnings treated as
errors. Twenty-one tests passed immediately. The two runtime failures identified valid domain
boundary defects:

- low-thrust discrete-flow finite differences perturbed a zero thrust-magnitude epigraph into
  the negative half-line;
- 6-DoF rollout rejected a finite near-unit initial quaternion before it could be normalised.

The finite-difference lineariser now uses central differences in the interior and a valid
one-sided derivative when one perturbation crosses a physical-domain boundary. The 6-DoF
rollout now normalises the initial attitude at the public rollout boundary and validates the
normalised state before integration.

## 31 August 2026 — unsanitized native truth gate passes

With those boundary fixes applied, all 23 host-native smoke tests pass. This includes the
selectable higher-order transcriptions, nonlinear 3-DoF SCvx driver, 6-DoF and low-thrust
models, robust scenario CQPs, expected/worst/CVaR risk augmentations, dynamic time-grid
discovery, route-column master, multi-fidelity trajectory oracle, Lambert screening and
continuous inter-node checks.

One-shot source-migration scripts and workflows were removed after their asserted edits were
applied. The Python reference-layer rename was completed without suppressing lint rules. The
ASan/UBSan, install/package-consumer and Python-reference gates are rerun from this
human-authored commit before the pre-GPU branch is declared fully green.
