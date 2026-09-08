# Identical assembled return QP exposes inner-solver accuracy failure

This extends the fixed-input v672 diagnostic on the pinned RTX 5090 mesh core
and QOCO540 binary. No production setting, physical tolerance or fleet changed.

The first assembled conic problem is byte-identical across all six captures:
`1b5b36ac2ae38cd25b5c3c3b8f1bffcbc68fdc998baf02306eaa472586a46c2f`.
It includes CSC topology, original values, cones and solver settings. GPU seed
arrays are not separately downloaded; the resulting first conic input is exact.

The first three calls capture two priming problems and then return unsupported
because host snapshot writes cannot run inside the outer CUDA graph. These are
instrumentation exceptions, not reproduced physics failures. The supported
ordinary native dispatch comparison then produces infeasible / infeasible /
converged. The converged trajectory passes the unchanged independent certificate.
All numerical trajectory work in that comparison still runs on CUDA.

Eight standalone replays of the identical first QP, without any trajectory loop,
all fail the independent long-double 1e-9 residual/gap audit and produce different
numerical results. Explicitly setting static factor regularization to 1e-8 fails
8/8 too. Correction: this repeats the captured baseline value, not an increase;
the original description overstated that comparison. Raw inputs/logs are unchanged.
Disabling only the factor graph while retaining the IPM graph is unsupported and
exits before producing a replay. Disabling both graphs completes eight replays;
all eight still fail the audit. Graph execution alone does not explain the issue.

The independent CPU Clarabel diagnostic reports AlmostSolved. Its original-QP
residuals and relative gap pass the numerical audit (about 1.18e-10 primal and
2.36e-12 gap), but the helper conservatively does not qualify that status. It is
an accuracy reference only, never a production fallback or promoted trajectory.

This isolates an inner-solver numerical accuracy problem for a concrete saved
input. It does not yet identify a faulty kernel, prove every mission failure has
the same cause, or provide a validated fix. Next work should use this QP to test
conditioning, factorization and refinement changes against unchanged equations,
then validate successful candidates on complete graph-mode trajectories.

`raw.tar.gz` preserves every captured QP, failed and successful trajectory record,
exact drivers, standalone executable/source, independent oracle point and replay
coordinates. `raw-manifest.json` checks every member. Frozen runtime/source are
in the published mesh-v670 local archive; runtime hashes are in the reports.
