# Exact-zero quadratic reference dispatch v613

The opt-in `--omit-zero-quadratic` diagnostic passes a null quadratic descriptor
only after verifying every canonical Hessian coefficient equals exactly zero.
Original snapshot bytes, objective, constraints, cone mapping and independent KKT
thresholds remain unchanged. The default comparator still retains the original
full symmetric CSC descriptor. Nonzero quadratic fixtures and duplicate flags
are rejected with CUDA hidden; all five CPU validation checks pass.

Four native calls use the pinned upstream commit 167c8b72b4b96d2f94d405b8763e485514192b81:
two supplied qualified points accepted at zero iterations, followed by two cold
100,000-iteration/60-second runs. Both cold runs remain unqualified.

| Capture | Iterations | Reported inner counter | Native C API wall seconds | Original normalized gap | Qualified |
|---|---:|---:|---:|---:|---|
| conditioning-seeded | 0 | 0 | 1.989537312 | 1.26051763869e-11 | True |
| difficult-seeded | 0 | 0 | 0.254250876 | 3.99939651022e-11 | True |
| conditioning-cold | 100000 | 100000 | 10.315961813 | 0.00133143448985 | False |
| difficult-cold | 100000 | 100000 | 8.796659007 | 0.0170812209045 | False |

The saved [v608 calls](../upstream-identical-capture-v608/README.md) used the same
snapshots, upstream commit, accuracy settings and cold iteration/time caps, with
the structurally nonempty zero Hessian. Their conditioning/difficult C API times
were 37.332045778 / 34.817582406 seconds, versus 10.315961813 / 8.796659007 here. These are
single diagnostic samples including native setup, host transfers and synchronization;
neither path produced a qualified cold result, so this is not a qualified solver
speedup or a mission improvement.

`inner_iterations` faithfully exports upstream's counter. Its source increments
`inner_solver->total_count` once unconditionally per outer update, even on NON_Q.
Thus 100,000 is expected here despite no BB inner solve. The earlier general-Q
path reported 200,000, including the unconditional counts. The initial harness
expected zero, failed after the third completed call, and was corrected. All
three completed outputs were reused; only the fourth outstanding call was then
launched. Both harnesses, the observation failure and every raw output are retained.

The mathematical dispatch proof is in pinned upstream `src/utils.cu:834`,
`src/solver_state.cu:307` and `src/pdhg_core_op.cu:727`. No upstream source was
changed. The static reference library identity and build command are recorded in
manifest.json; no executable or library is included in this evidence package.
