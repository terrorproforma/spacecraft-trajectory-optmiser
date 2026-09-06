# Retained cuDSS and conditioning diagnostics

These are rejected experimental settings, not production recommendations or
speedup evidence. They investigate the pre-existing strict coast equality
failures documented with [native v133](GTOC12_DEVICE_ASSEMBLY_GUARD.md).
The published numerical settings remain QOCO v128 with vendor determinism off
and the caller's existing Ruiz setting (zero for the named transfer fixture).

## Deterministic vendor solves

`scripts/gpu/prepare_qoco_deterministic.py` only patches an isolated prepared
QOCO tree. It queries the actual configuration, checks that vendor iterative
refinement is disabled before enabling determinism, and optionally traces the
active deterministic and superpanel values. QOCO's own refinement is separate.
It is not part of normal runtime preparation.

On RTX 5090, CUDA 12.8 and cuDSS 0.8.0.10:

- Baseline QOCO v128 and v129 with determinism disabled each pass 18 coast
  updates. All six fresh v129 deterministic processes fail with illegal GPU
  memory accesses; their traces confirm `active=1` and `vendor_ir=0`.
- Disabling IPM replay, factor graphs, QOCO device refinement and metric graphs
  does not eliminate the failure in the tested configurations. Memcheck with
  those paths disabled reports an invalid shared-memory read in cuDSS's
  `fwd_dtmn_ker` during solve. This locates a failure, but does not establish a
  complete root cause.
- v130's superpanels option does not rescue either tested deterministic QOCO
  path. Its nondeterministic comparison completes three updates, of which two
  pass the strict original-equation gates.

The standalone `scripts/gpu/cudss_deterministic_probe.cu` eliminates QOCO,
graphs, custom allocation and refinement. It builds small, strictly diagonally
dominant systems with the known solution `x=1`, and checks both the original
equations and coordinates against 1e-12.

The original frozen probe v130 passes ten of twelve configurations but fails
the deterministic symmetric-indefinite N=64 and N=256 cases. A subsequent v131
build with optional general-matrix and pivot configuration passes all eight
follow-up cases, **including the symmetric/default-pivot cases that failed
before**. Both selected v131 memcheck cases also pass. The outcomes therefore
remain build/execution sensitive: this is not a universal standalone failure,
nor a verified general-matrix or pivot workaround. All outcomes are retained.

## Ruiz scaling

Ninety coast updates using native v133/QOCO v128 produce these strict passes:
18/18 with zero passes, 16/18 with one, 18/18 with three, 17/18 with five, and
17/18 with ten. The four rejected cases fail the external solver audit, so the
Python wrapper returns unavailable primal vectors. This batch does not fix or
erase the earlier absolute equality failures.

The complete-transfer experiment changes the next action more decisively:
zero Ruiz passes qualify **12/12** attempts, whereas three, five and ten passes
each qualify **0/12**. All 48 attempts include warmups and retain full reports
and independent certificates. Process exit code zero is not qualification.
The cyclic four-variant ordering was not fully balanced; these results are
negative qualification evidence rather than a timing comparison.

The raw coast harness called its Ruiz-count field `deterministic` despite
actually keeping vendor determinism off. A separately labelled derivative
corrects that field and preserves the original raw file. No result was replaced
or silently reclassified.

The [v134 checkpoint](../artifacts/performance/native-refresh-v134-checkpoint.json)
embeds the diagnostic sources, original probe source, build helpers, runtime
hashes, raw failures, labelled derivative and complete transfer evidence.
