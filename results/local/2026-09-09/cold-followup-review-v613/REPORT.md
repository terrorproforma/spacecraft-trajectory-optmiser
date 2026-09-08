# Independent review of automatic-cold v610 and zero-Q v613

No package, equation, dispatch or call-accounting blocker was found. All 52
indexed files and eight compressed raw-log members match their hashes. Every
raw final record matches its runner report. A fresh Decimal65 calculation from
the exact represented FP64 input and output vectors agrees with all eight
original-equation qualification verdicts. The retained mathematical auditor
source is compiled directly with `exec(compile(source_bytes))`; this review does
not execute inherited Python bytecode, run a solver, or mutate either package.

| Package | Indexed files | Indexed bytes | Saved native calls | Qualified |
|---|---:|---:|---:|---:|
| automatic-cold-v610 | 18 | 2,971,787 | 4 | 0 |
| upstream-zero-quadratic-v613 | 34 | 4,551,847 | 4 | 2 supplied seeds; 0 cold |

The v610 index is
`802f02b50d9a4923eb367dbd3cdfb7bc8f0f5a72a30ff0c5d2f89660feff6acd`.
The v613 index is
`7c642523e3748314d54d7ec066a1aa3ae43c9ecacedd0698674afac541abb3e7`.

The automatic-grid calls all reach 100,000 iterations without a deadline,
bootstrap, retry or recovery. Requested/effective block fields are null in this
executable, so no actual block count can be inferred from those fields. Final
common-versus-natural primal differences are at most about 1.95e-14 and dual
differences at most about 3.41e-11. This supports the documented unchanged
numerical path to rounding; only final vectors, not every intermediate iterate,
are retained. Both policies fail the common gate. The earlier null-grid parser
failure occurred after one successful executable return. Its exact log hash and
elapsed time are unchanged in the resumed report; source control flow skips the
existing call and launches only the other three.

The only v613 source-overlay change relative to the v608 manifest is
`cpp/cuda/tests/upstream_snapshot_replay.cu`. The archived source checks every
canonical Q coefficient with exact `value == 0.0` before CUDA and rejects a
nonzero coefficient; canonical conversion retains all original P values and
mirrors upper entries once. Only the opt-in flag changes the passed descriptor
to null. c, scalar/SOC matrices, bounds, offset, initial-point conversion and
original audit remain unchanged. The default still passes full symmetric CSC Q.
All five recorded hidden-device validation checks have their expected status.

The local pinned upstream tree is clean at commit
`167c8b72b4b96d2f94d405b8763e485514192b81`. A null Q with no low-rank term selects
NON_Q through `src/preconditioner.c:793` and `src/utils.cu:834`.
`src/solver_state.cu:307` skips BB initialization for NON_Q and
`src/pdhg_core_op.cu:727` calls the explicit LP primal update. Its line 762
increments `inner_solver->total_count` unconditionally once per outer step.
Therefore the reported 100,000 inner counter at 100,000 outer steps does not
mean 100,000 BB solves. The README states that distinction correctly.

The zero-Q run's initial harness failed its incorrect expected-zero-counter
assertion after three completed calls. All three original log hashes, commands,
return codes and wall times match the resumed report. The corrected runner
reuses them and launches only the fourth call. All four raw logs include their
normal final DONE record. The saved budget remains four total calls, with no
evidence of a repeated completed solve.

Decimal65 gaps are about 1.26e-11 and 4.00e-11 for the two zero-step seeds, which
qualify, and 0.00133143 / 0.01708122 for the two 100,000-step cold results, which
do not. The reported lower one-shot native times on the zero-Q branch are single
unqualified diagnostic observations. Both READMEs correctly avoid a qualified
throughput, convergence, fleet score or SOTA claim. The scripts copied into the
packages preserve their original execution context; they are evidence, not
portable turnkey launchers from their archive locations.

Reproduce this review from the repository root using only standard-library Python:

```powershell
& 'C:/Users/Angus/.local/bin/python3.12.exe' -B build/performance/cold-followup-review-v613/review_packages.py --output build/performance/cold-followup-review-v613/findings.json
```
