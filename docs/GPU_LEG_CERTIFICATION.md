# CUDA certification in native trajectory refinement

Native CUDA refinement now uses the existing CUDA DOP853 propagator for its
per-leg numerical certificate. Previously `Gtoc12ScvxDriver.solve` downloaded the
solution and called CPU `certify_leg` before deciding whether to accept it.
The driver now retains certificate buffers across solves and grows them only
when an input exceeds their capacity. Failed CUDA propagation returns numerical
failure; it does not call a CPU fallback or reuse a previous certificate.

`ScvxSettings.certification_backend` and `--certification-backend` accept
`auto`, `cpu` and `cuda`. Auto selects CUDA when the outer loop is CUDA and CPU
otherwise. Explicit CPU mode provides the comparison path. CUDA selection checks
the runtime and single-worker requirement at the command boundary. Route summaries
record the backend that actually produced each certificate; configuration reports
only record the selection until work has occurred.

Final independent CPU fleet verification, official checking and visualiser history
generation remain separate. CPU route/fleet bookkeeping, thrust clamping and
packing also remain. This change removes one recurring CPU numerical propagation
stage; it does not establish a fully GPU-controlled whole application.

## Why this stage

The v708 profiles replay the same v707 campaign and native binaries. CPU fleet
verification dominates both profiles:

| Profile | Total process | Three fleet checks | Native solves | Local retiming |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 host | 115.54 s | 98.02 s | 8.30 s | 2.48 s |
| Lambda H100 host | 210.06 s | 188.03 s | 9.59 s | See raw profile |

These are instrumented diagnostic timings, not benchmark timings: cProfile adds
substantial overhead to Python's millions of RHS/interpolation calls. The prior
unprofiled campaigns took approximately 76 s locally and 135 s on H100. CPU host
differences must not be interpreted as a GPU hardware comparison.

The profile also exposed CPU propagation inside each native leg refinement. That
is now removed in CUDA mode. Repeated final fleet audits still dominate this
bounded campaign and require further work at the search/publication boundary.

## Paired certificate-stage measurements

Two real emitted routes each contain 19 event-to-event legs, including coast/mining
segments. Both backends read identical emitted samples. Each leg is certified
sequentially, as in the production driver; CUDA reuses a session across legs.
Five alternating-order pairs per route discard the first pair as warmup, leaving
four measured samples per mode. Timings include array packing and output assembly,
but exclude trajectory solving, file parsing, original burn-sample emission and
the full mission bookkeeping/checker. This is a certificate-stage comparison.

| Route | RTX CPU / CUDA | RTX ratio | H100 CPU / CUDA | H100 ratio |
| --- | ---: | ---: | ---: | ---: |
| Attempt 1 | 846.31 / 173.45 ms | 4.88x | 1615.23 / 82.61 ms | 19.55x |
| Attempt 2 | 880.26 / 173.13 ms | 5.08x | 1596.61 / 81.55 ms | 19.58x |

Every repeated certificate has the same tolerance decision. CPU/CUDA endpoint,
velocity, mass and radius differences are checked independently of elapsed time;
the final numeric certificates and all samples are included in the evidence.
This does not measure full solved trajectories per second.

## Complete campaigns and validation

The same source and native binary per GPU run explicit CPU certificates and auto
CUDA certificates. Both use the v707 whole epoch-search graph and conditioning
retry. All four campaigns perform 36 converged native leg solves. Each produces
two 18-leg certified route records and passes both complete fleet checkers.
The auto route records all identify CUDA certification.

| GPU | CPU certificates, process total | CUDA certificates, process total |
| --- | ---: | ---: |
| RTX 5090 | 76.976 s | 76.644 s |
| H100 | 135.464 s | 133.784 s |

There is only one full-campaign pair per GPU, with CPU first. These small timing
differences do not establish a whole-campaign speedup. The retained fleet remains
23 ships, 195 collected asteroids, approximately 14,051.854894 raw kg and
**12,810.135953 weighted kg**. Sub-nanogram score differences are not a score gain.

On each GPU, 49 configuration/verifier/final-check tests and two native integration
tests pass. The integration test makes CPU certification raise if called. A
subsequent 23-test configuration run covers the final CLI reporting/preflight
changes (overlapping earlier cases). Retained-session memcheck, racecheck and
synccheck pass on both GPUs. The existing epoch-search active-loop memcheck
limitation on RTX remains as documented in [v707](GPU_DEVICE_EPOCH_SEARCH.md);
the new sanitizer runs cover verifier buffer reuse, not the whole application.

The first follow-up launcher failed before testing because its source-prefix
extraction matched a quoted string. The corrected launcher, original traceback
and final successful logs are all retained. No numerical implementation change
was needed. Both GPUs reuse the already qualified v702 native libraries; these
Python bridge/driver changes do not require a new CUDA build.

## Evidence and display

[Downloaded evidence](../results/lambda/2026-09-09/gpu-leg-certificate-v712/)
contains hashed raw archives, profiles, sources, original failure logs, tests,
paired samples, four campaigns and the H100 result. Its manifests identify the
earlier frozen fixture/library inputs; no concurrent persistent-solver changes
were included in the numerical runtime used here.

Visualiser dataset: `gtoc12-certificate-v712`, labelled **H100 CUDA certificates v712**.
Result: `results/lambda/2026-09-09/gpu-leg-certificate-v712/h100-best/Result.txt`.

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-certificate-v712&epoch=69807&preset=oblique&z=1'
```
