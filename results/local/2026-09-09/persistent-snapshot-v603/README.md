# Captured GTOC12 conic problems through the persistent GPU core

The new diagnostic importer passes its analytic checks, but the unmodified
persistent core does not qualify either real captured GTOC12 problem within
100,000 iterations. GPU QOCO qualifies five of six repeated reference solves.
Independent Python and native original-equation audits agree for both persistent
outputs. No fleet score or nonlinear trajectory is qualified by this experiment.

| Capture | Variables | SOCs | Persistent qualified / attempted | QOCO qualified / attempted |
| --- | ---: | ---: | ---: | ---: |
| Conditioning regression | 5,261 | 210 | 0 / 1 | 3 / 3 |
| Difficult refinement | 5,839 | 234 | 0 / 1 | 2 / 3 |

Both Hessians are numerically zero despite retained structural entries. These
fixtures test conic LP convergence; they cannot establish a conjugate-gradient
advantage on a nonzero quadratic objective.

## Convergence evidence

| Persistent metric after 100,000 iterations | Conditioning | Difficult |
| --- | ---: | ---: |
| Normalized primal residual | 8.790083e-5 | 4.158997e-6 |
| Normalized dual residual | 3.483413e-4 | 2.125477e-5 |
| Relative objective gap | 0.9999243 | 0.01122705 |
| Absolute primal cone violation | 1.555828e-4 | 7.798669e-6 |
| Scaling preamble | 0.001745 s | 0.001771 s |
| Native solve stage | 5.011544 s | 5.141791 s |
| Complete invocation | 5.327859 s | 5.535290 s |

Both stop at the iteration limit; neither hits the 30-second cancellation deadline
or enters recovery. The gap is substantial, so this is not merely a marginal
qualification-threshold disagreement. The current preamble is a small fraction
of these measured solves.

QOCO's three-repeat invocations take 0.719610 and 0.867800 seconds. Workspace
setup is shared between its repeats; the persistent invocations contain one cold
solve each. These process times include startup and raw vector output and are
not matched steady-state timing distributions or a speedup ratio. The failed
QOCO repeat has relative gap 3.476514e-9 and remains rejected despite vendor
status 2. Five other QOCO outputs pass the identical external audit.

## What was verified

[Analytic validation](analytic/README.md) records six qualified GPU solves,
including shifted coordinates, cold/reused workspaces and fully retained state.
It includes CPU conversion tests, source/build hashes and a frozen source archive.
No production numerical kernel or solver default was changed by this adapter.

[The real report](real/report.json) records commands, loaded library hashes,
requested budgets, per-repeat original-equation audits and exact input hashes.
The common diagnostic gate requires normalized primal/dual residuals, objective
gap and maximum scalar/SOC block complementarity at most 1e-9, and absolute
primal/dual cone violations at most 1e-8. Native optimal termination is separately
required for qualification. The block complementarity check prevents large
opposite-sign errors from cancelling in a global sum. It is an explicit guard
in the new replay auditor; production tolerances were not modified.

The Python auditor consumes the exact FP64 input and reconstructed primal values,
then accumulates sparse products in long double. It checks emitted snapshot hashes
and coordinate declarations. The native reader additionally checks translated
coefficient/offset consistency before CUDA. No arbitrary-Hessian convexity or
nonlinear spacecraft certification is claimed.

## Reproduction and next intervention

- `analytic/source.tar.gz` contains the frozen native source, including the adapter.
- `real/source/` contains the independent auditor/tests, existing QOCO replay and
  reference build scripts. The first missing-header build failure is preserved.
- `real/inputs/` contains both byte-identical captured problems.
- `real/raw.tar.gz` contains every full-vector log, the bounded runner and report.
- `publication-audit.json` records independent archive/source reconciliation.

The current generic import represents simple bound rows as scalar inequalities.
The two captures contain 5,273 and 5,848 exact singleton rows, all with coefficient
+1 or -1. Folding these into the existing native variable-bound projection is
the next algebraically equivalent representation experiment. Its performance
and convergence are not established by this baseline. PDHCG is still not wired
into the production GTOC12 SCvx backend selector.
