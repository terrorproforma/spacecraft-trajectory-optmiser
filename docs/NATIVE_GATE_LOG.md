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

The branch now implements those interfaces instead of excluding the failing targets. This file
is an evidence log, not a green-build claim: each newly exposed compile, sanitizer or runtime
failure is cleared before the pre-GPU gate is declared complete.
