# Optional common-KKT stopping on identical captured problems

The new GPU stopping option accepts all four already-qualified real-capture starts at zero optimization iterations, with bit-for-bit unchanged primal and original equality/conic dual vectors. Independent original-equation audits pass for both execution strategies. The four cold runs remain unqualified at their fixed deadlines; this experiment establishes an acceptance correction, not a cold-convergence or mission-score improvement.

| Capture | Start / rule | Kernel blocks | Termination | Iterations | Normalized gap | Native solve seconds | Independent qualification |
|---|---|---:|---|---:|---:|---:|---|
| conditioning | cold / common KKT | 0 | cancelled | 79,375 | 1.0002738 | 58.958551 | Fail |
| conditioning | cold / native natural | 0 | cancelled | 81,525 | 0.999907553 | 59.004367 | Fail |
| conditioning | qualified seed / common KKT | 0 | optimal | 0 | 1.26051763e-11 | 0.016343 | Pass |
| conditioning | qualified seed / common KKT | 2 | optimal | 0 | 1.26051763e-11 | 0.000607 | Pass |
| difficult | cold / common KKT | 0 | cancelled | 70,775 | 0.173749298 | 58.864398 | Fail |
| difficult | cold / native natural | 0 | cancelled | 72,609 | 0.204742906 | 58.883016 | Fail |
| difficult | qualified seed / common KKT | 0 | optimal | 0 | 3.99901969e-11 | 0.017231 | Pass |
| difficult | qualified seed / common KKT | 2 | optimal | 0 | 3.99901969e-11 | 0.000647 | Pass |

Block setting 0 means the single-block kernel; 2 means two cooperative blocks. The cold comparison forces setting 0 in both arms. It does not benchmark native automatic selection, which can choose a much larger grid, and does not establish general throughput or state-of-the-art performance. Both arms retain 1e-9 requested solver tolerances, a 100,000-iteration cap and the same 60-second phase deadline. Returned iteration counts can differ because both runs were bounded by wall time, not equal work.

The common rule uses normalized original-equation primal, dual, gap and maximum block-complementarity gates of 1e-9, and separate absolute primal/dual cone gates of 1e-8. The old absolute natural-residual telemetry is still computed. A passing original-equation point need not pass that distinct native natural threshold. The new policy checks before an update, preserves cancellation precedence and disables recovery explicitly. It is opt-in, restricted to unshifted generic captures with free primal variables, and assumes a full symmetric convex Hessian; these two captures have zero numerical Hessians.

A GPU common record is a certificate for the current reported iterate only when valid=true. Cancellation between scheduled checks invalidates that record; cancellation at a completed check may leave current metrics valid while termination remains cancelled. Neither case grants solver acceptance. Every returned vector set in this bundle has a fresh independent CPU audit, including deadline outcomes. GPU compensated FP64 and CPU long-double arithmetic are compared at the same fixed gate, without claiming bit-identical arithmetic at arbitrary boundaries.

The real experiment ran eight executables and 12 solve API calls, including 4 separately reported unseeded bootstrap iterations. The four seeded actual solves used zero iterations. The analytic bundle adds two executables, fourteen bounded solve calls and six actual iterations. Bootstrap, setup, solve-event, download and audit times are separate in raw records. Native solve time includes legacy natural-report work and common evaluation; common evaluation_clock_cycles measures block-zero clock cycles, not seconds. The strong block-count dependence of these diagnostics must not be presented as end-to-end trajectory throughput.

Default kernels are separate compiled instantiations; their register/shared/stack footprints match the frozen v603 baseline. The optional kernels have separate occupancy limits. The final core preserves all non-persistent v603 object files by hash. No physics tolerances, production GTOC12 backend, fleet itinerary, score or incumbent changed in this experiment.

Evidence: [analytic source/build/tiny checks](analytic/README.md), [original real runner report](real/run/report.json), [fresh independent audit and counts](audit/findings.json), and [raw-log member hashes](real/raw-log-members.json). The compressed raw logs retain every original point and cold-result vector, metadata, bootstrap and pre-step record. Inputs and exact runners/auditor are retained in real/. Earlier compile failures and the superseded CPU-only runtime-branch prototype are preserved in analytic/earlier-builds/. The parent sha256.json indexes every file except itself.
