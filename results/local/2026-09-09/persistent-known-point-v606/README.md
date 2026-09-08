# Known-qualified-point GPU replay

The imported solutions are correct; the persistent solver's stopping predicate
and subsequent numerical updates remain the issue to investigate. **All eight
initial certificates pass the common original-equation KKT gate, one final
output retains that gate, and zero outputs have native accepted termination.**
No production solver code, tolerance or fleet result changes in this experiment.

The diagnostic source adds `--initial-point PATH` with strict dimensions,
finite FP64 values, exact snapshot SHA-256 and explicit coordinate convention.
Original and coordinate-roundtrip certificates must qualify before CUDA. Actual
GPU seed buffers are checked bit for bit before and after residual-only
measurement. A full reset separates a one-iteration bootstrap from seeded work;
pre-step records have zero seeded iterations and `UNSPECIFIED` termination.
[Format, analytic cases and lifecycle details](analytic/README.md).

The two QOCO reference points are the first qualified repeat, repeat zero, in
each archived three-repeat reference. Both captures have numerically zero
Hessians. They are convex subproblems, not nonlinear trajectory certificates.
Source extraction normalized three equality-dual signed zeros per point; no
nonzero value changed. The encoded point and actual imported vectors match
exactly. [Independent source chain and full-vector audit](audit/README.md).

| Capture | Representation | Steps | Final normalized gap | Common KKT gate | Native termination |
| --- | --- | ---: | ---: | --- | --- |
| Conditioning | Generic | 1 | 6.04613e-7 | Fail | Iteration limit |
| Conditioning | Folded bounds | 1 | 1.27414e-7 | Fail | Iteration limit |
| Conditioning | Generic | 1,000 | 4.21818e-7 | Fail | Iteration limit |
| Conditioning | Folded bounds | 1,000 | 1.84037e-6 | Fail | Iteration limit |
| Difficult | Generic | 1 | 2.49764e-10 | Pass | Iteration limit |
| Difficult | Folded bounds | 1 | 1.59897e-7 | Fail | Iteration limit |
| Difficult | Generic | 1,000 | 1.23070e-7 | Fail | Iteration limit |
| Difficult | Folded bounds | 1,000 | 1.81965e-8 | Fail | Iteration limit |

The common gate retains normalized primal/dual/gap/per-block-complementarity
limits of 1e-9 and absolute cone limits of 1e-8. Native acceptance additionally
requires its absolute natural residual at 1e-9. Initial natural residuals are
3.81865e-8 and 1.40971e-8 despite qualified original certificates. Weak
complementarity explains part of this distinction. Subsequent dual-objective
drift can break gap accuracy despite tiny primal movement. This is not a
measured loss of mission cargo or proof of global divergence.
[Numerical reconstruction and next experiment](analysis/REPORT.md).

Ten GPU-hidden JSON checks and six corrected analytic GPU cases pass. The
analytic bundle preserves the original JSON serialization failure, its original
raw output, the adapter correction, and a busy attempt with no GPU calls. Real
work totals eight executable invocations, sixteen native solve calls, **4,004
seeded iterations plus eight bootstrap iterations**, with no recovery. Summed
real native solve-event time is 0.203400 seconds and executable wall time is
3.638010 seconds. These are different scopes; this one-run-per-case diagnostic
does not establish certified throughput or a speedup.

`analytic/manifest.json` pins frozen source `659f5abd7b41d16dbeac980a6e10d6ca92666314`,
executable SHA-256 `7d248cf1361cf8df83232951fee07cf9d274af06a6abf6e186823f6e9aae84d3`,
and unchanged production core SHA-256
`d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633`.
`real/report.json` is the untouched run report; `real/raw.tar.gz` retains exact
inputs, point preparation, source auditor, commands, and all full-vector logs.
The `audit/` and `analysis/` CPU scripts retain source hashes and recomputed
metrics. `objective-balance/` shows why a global 2^-13 objective rescaling changes
effective original-coordinate steps only about 2%; it is not a GPU experiment.
Archives include source and evidence, not compiled binaries.

The persistent/cooperative main iteration is explicit PDHG; recovery CGLS is
separate from upstream's quadratic/conic proximal solve. For these zero-Hessian
cases the explicit linear-objective proximal map is already applicable. Keep
this distinction when planning general quadratic comparisons.
[Implementation boundary and roadmap](../../../../docs/GPU_PERSISTENT_CAPTURE_REPLAY.md).
