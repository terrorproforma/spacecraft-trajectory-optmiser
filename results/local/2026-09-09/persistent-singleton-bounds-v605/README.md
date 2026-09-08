# Exact bound projection did not improve the captured solves

The optional replay representation passes seven analytic GPU checks, but it
does not qualify either real GTOC12 capture within 100,000 iterations. It remains
off by default. No production numerical kernel, objective or accuracy threshold
changed; both representations use the same executable and byte-identical native
solver library. This experiment produces no new fleet score.

| Capture / representation | Native scalar rows | Normalized primal residual | Normalized dual residual | Relative objective gap | Qualified |
| --- | ---: | ---: | ---: | ---: | --- |
| Conditioning / generic | 9,911 | 8.790083e-5 | 3.483413e-4 | 0.9999243 | No |
| Conditioning / folded | 4,638 | 1.239396e-4 | 1.773888e-3 | 1.0000870 | No |
| Difficult / generic | 10,992 | 4.158997e-6 | 2.125477e-5 | 0.0112271 | No |
| Difficult / folded | 5,144 | 8.213558e-5 | 4.372539e-4 | 1.0106375 | No |

All four invocations reach the iteration limit, without recovery or deadline
cancellation. The generic controls reproduce the residuals from the
[first captured-problem baseline](../persistent-snapshot-v603/README.md).
Native and independently recomputed original-equation audits agree in every case.
These two captures have numerically zero Hessians: this tests conic LP behavior,
without exercising a nonzero quadratic CG subproblem.

## What the experiment rules out

Folding moves 5,273 and 5,848 exact singleton inequalities into the core's existing
variable-bound projection. Every folded bound is satisfied exactly in the
returned vectors. Nevertheless, original equality defects worsen by approximately
1.41 and 19.75 times. The primal failure exists independently of how bound duals
are reconstructed.

There are 2,317 and 2,434 off-contact normals, with no one-ULP cases. The maximum
distance to the proposed active bound is approximately 1.08 and 2.00 in the
captured local variable units. A CPU counterfactual that assigns multipliers
despite missing contact still fails stationarity, complementarity and gap checks.
That counterfactual is diagnostic only: it supplies invalid dual support and must
not replace the guarded export or be accepted as a solution.

Native natural residuals use different operator representations and should not
be compared alone. The original-equation audit provides the common comparison.
Its gate remains normalized primal/dual residual, gap and maximum scalar/SOC
block complementarity at most 1e-9, with absolute cone violations at most 1e-8.
Native optimal termination is separately required. No nonlinear spacecraft
trajectory certification follows from these conic checks.

Observed native solve stages were 4.991/4.853 seconds for generic/folded
conditioning and 5.091/4.940 seconds for generic/folded difficult. Scaling took
1.56–1.78 milliseconds. These are single bounded runs with no qualified result;
slightly shorter iteration-budget timings do not establish a speedup or improve
qualified solutions per second.

## Evidence and reproduction

- [Analytic source, conversion tests and initial failure history](analytic/README.md)
  include seven corrected GPU passes and the preserved initial no-SOC binding
  failure. The corrected adapter source matches all four current owned files.
- [Real comparison report](real/report.json) records exact commands, input hashes,
  runtime metadata, every original-equation metric and explicit failed outcomes.
- `real/raw.tar.gz` contains the complete vectors, identical input captures,
  bounded runner, independent auditor and CPU counterfactual analysis.
- [Publication reconciliation](publication-audit.json) independently verifies
  360 source archive members and recomputes all four real audits. All 356
  non-owned source members match the original baseline archive.

Frozen adapter source: `efaabd712df62c095ef3ee782ae561ee5afa8e05`.
Executable SHA256: `d44ce03670072f2e192e47458248bd8afd1df23e206286b0a3f00041775974df`.
Unchanged core SHA256: `d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633`.

The captured runner enforces the shared GPU lock and refuses an existing output
directory. Each real invocation uses `--tolerance 1e-9 --iterations 100000
--deadline-seconds 30 --repeats 1 --mode cold`; only the folded arm adds
`--fold-singleton-bounds`. Capture order reverses between arms. The adapter is a
diagnostic executable, not a production GTOC12 backend selector.

The next useful diagnostic starts from an independently qualified primal/dual
point to separate mapping and native stopping behavior from cold-start
convergence. A fresh workspace's diagnostics currently zero residual fields
until its first solve epoch; reading those zeros must not be mistaken for a
fixed-point check. After a valid known-point test, investigate scaling, restart
and primal-dual balance with the same unchanged external gates.
