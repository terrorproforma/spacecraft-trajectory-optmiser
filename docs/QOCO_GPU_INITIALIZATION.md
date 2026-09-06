# Captured GPU initialisation

Experimental v124 extends the retained v123 graph to include initial solver
state, the initial linear solve, cone interior shifts and conditional warm-start
selection. Replays execute these operations on the GPU before the complete IPM
loop. Current warm-start and dynamic-regularisation settings arrive through the
device parameter record, so changing either does not require recapture.

The first graph construction still performs ordinary initialisation once to
prime the vendor factorisation/solve kernels. Initial topology, setup/analysis,
terminal restoration/reporting and native SCvx dispatch still involve the host.
This is progress toward GPU orchestration, not a fully GPU-controlled pipeline
or a new fleet result. Default builders remain unchanged.

## Implementation

The finite initial cone shift uses the existing parallel residual reduction,
a scalar device decision and parallel updates to LP entries and SOC leading
entries. It removes the residual download and serial scan from the ordinary
path. Exceptional non-finite inputs use a device-only serial fallback to retain
the original NaN comparison order. SOC norm summation order remains unchanged.
Scratch reservation now accounts for the combined LP and SOC reduction count.

The patch also fixes pure-SOC NT identity initialisation: the compact scaling
buffer must be cleared even when there are no LP rows. Previously, old SOC tail
values could survive the identity reset.

Warm-start selection is a conditional IF emitted into the parent graph, using
the current device setting. Each workspace retains its graph operands and
resources. Initialisation mode joins the existing graph invalidation key.

Apply after the complete [v123 preparation chain](QOCO_RETAINED_IPM.md):

```sh
python scripts/gpu/prepare_qoco_ipm_initialization.py --destination /path/to/isolated/source
```

Compile CUDA with `--default-stream per-thread` and select
`SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1` inside the existing queued scope. Presence of
`SPACEPDHCG_TEST_QOCO_IPM_INIT_DISABLE` retains host initialisation for ablation.
Presence of `SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE` selects the original cone
routine; it cannot be combined with captured initialisation. The original IPM
mode's control, count, verbose and arithmetic-audit restrictions still apply.

## Accuracy and failure evidence

- Graph-enabled and disabled modes each pass the native controller and all 51
  trajectory integration tests. Seven convergence/failure tests pass.
- PD6 N20 and N500 pass independent certificates and unchanged 1e-8 objective
  gates, with reference errors 5.399e-10 and 9.705e-10.
- The 32-case interleaved two-workspace probe changes coefficients, RHS,
  tolerances, iteration budgets, cost scaling, warm starts and dynamic
  regularisation. All printed objectives, statuses and IPM/refinement counts
  match exactly across v123, host-loop v124, host-initialisation v124 and captured
  v124. Each workspace builds twice for 16 replays, including an intentional
  static-regularisation invalidation. The diagnostic's Ruiz setup is host-side.
- The direct cone probe checks 84 finite, boundary, NaN and infinity cases,
  including 1,029 reduction partials. Every returned bit matches the original
  GPU routine, both normally and under capture; finite cases also pass an
  independent formula check. The identity probe fails on v123's retained SOC
  tails and passes on v124 for pure and mixed cones.
- Isolated cone memcheck, initcheck and synccheck report zero errors; memcheck
  reports zero leaks. Identity memcheck also passes. These sanitizer runs used
  the earlier fingerprint-reporting probe; the subsequently strengthened probe
  compares every returned bit directly and passes separately.

A new pure-SOC analytic QP probe exposed a precision floor shared by v123 and
both v124 modes. An initial coordinate tolerance of 1e-7 failed after a warm
start despite approximately 1e-11 objective error and 1e-12 residual. Demanding
1e-13 solver tolerances also fails status on that warm-start case in all three
variants. These failures and a reproducible strict mode remain in the evidence.

Under the existing 1e-8 objective and feasibility limits, all 12 coefficient
updates pass in all three variants. The normal probe derives a separate
coordinate-distance bound from strong convexity and allowed cone violation;
it does not assume objective error and coordinate error have the same units
or tolerance. Results are not bit-identical across these pure-SOC variants.
Passing the normal probe does not qualify the stricter accuracy setting.

Full solver memcheck still aborts with CUDA 999 in the one-time initial
conditional-refinement warm-up, before executing the combined graph. It reports
110 errors and 108 outstanding allocations after abort. The full solver is
**not sanitizer-qualified**.

## Complete-transfer timing

Six balanced triples run a warm-up and measured transfer per process through
the same v107 GPU seed/SCvx core. All 36 transfers pass independent physics and
the unchanged 1e-5 kg gate against 2445.3111007852112 kg final mass.

| Variant | Median measured complete transfer |
|---|---:|
| Retained IPM v123 | 333.859 ms |
| v124 with host initialisation | 363.031 ms |
| v124 with captured initialisation | 332.389 ms |

This establishes no additional end-to-end speedup over v123. The display RTX
5090 has unlocked clocks. Warm-ups and outliers are retained, and compilation
did not overlap this timing experiment. Lambda's H100 remained at 100% utilisation
on its existing campaign during the read-only check; no work was offloaded.

The [v124 checkpoint](../artifacts/performance/qoco-ipm-initialization-v124-checkpoint.json)
embeds seven sources, prepared source, 16 runtime hashes, 47 helpers and 18
evidence reports, including the failures. All 12 modified prepared files
reproduce exactly from the formatted postprocessor; Ruff passes.
Checkpoint SHA-256:
`63908cb4aeec3c3c3e81b0447500f018dd0940e63d99cdf9d6229d7608408912`.
