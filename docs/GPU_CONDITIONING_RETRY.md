# GPU conditioning for inaccurate conic retries

The captured asteroid 13077 → Earth return sometimes exhausted its refinement
budget despite identical physical inputs. Its first assembled conic problem was
also byte-identical across captures. Replaying that problem separately exposed
insufficient conic accuracy, including with IPM and factor graphs disabled.

The new optional policy uses ordinary Ruiz equilibration on the first existing
cold retry after an inaccurate conic result. Six complete return replays now
converge and pass independent physics certification on each of RTX 5090 and H100.
All H100 replays take seven outer iterations. This is a fix for the captured
failure, subject to the broader comparisons below, not proof that every difficult
trajectory or vendor numerical issue is resolved.

## Why retry with different conditioning

For the saved first problem, standard Ruiz scaling passes four of four original-
equation audits. Preserving objective magnitude during scaling passes none of
four. Smaller/larger static regularization and tighter, longer linear refinement
also fail this initial screening. Conversely, the earlier
[conditioning diagnosis](GPU_CONDITIONING_DIAGNOSIS.md) contains a problem that
fails with objective normalization and passes without it. One fixed policy is
therefore insufficient for these two concrete cases.

Always using standard scaling is not the answer for the new return either:
all six complete trajectory trials reached their iteration limit. Their physical
rollouts passed the certificate, but they did not converge and are not promoted
as successful optimisation results.

The conditional policy keeps the normal zero-Ruiz attempt. If QOCO reports an
inaccurate result that fails qualification, the existing native counter selects
five standard Ruiz passes for the first unchanged-problem retry. The second
existing retry returns to zero Ruiz. An accepted result resets the counter.
There are no extra attempts, enlarged deadlines, changed trust-region rules,
relaxed conic gates or CPU solver fallbacks.

All selection and scaling run on CUDA. The counter is borrowed from the native
SCvx state. Uniform guards enable the scaling kernels inside the existing CUDA
graph; current incoming objective coefficients are used for normalization.
Numerical updates publish the current objective scale to the IPM parameters,
and output vectors are mapped back before the existing original-equation audit.
The lease drains work and clears the borrowed pointer before retaining a solver
or freeing the outer state. An explicitly selected nonzero Ruiz count is rejected
with this experiment rather than silently overridden.

## Enable the experiment

Use a core containing this change and a prepared QOCO library extended by
`scripts/gpu/prepare_qoco_retry_conditioning.py`. The preparation step follows the
existing device numeric-update/objective-policy preparation. It checks every
expected source location before writing, reproduces the tested CUDA source
byte-for-byte, and rejects reapplication without partial mutation.

```bash
export SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY=1
# Keep the trajectory setting --qoco-ruiz-iterations 0.
```

The default remains disabled; this is an opt-in policy validated on the captured
return and the bounded comparisons below.
An explicitly enabled policy requires the new vendor capability; it does not
silently fall back to identical retries with an older library. Ordinary calls
with the experiment disabled retain compatibility with the earlier library.

Final Python report materialization exposes `conditioning_retry` and the actual
`ruiz_iterations` per attempt. It reconstructs the native counter from recorded
retry transitions after the solve; it makes no host numerical decision and adds
no device synchronization. Counts and timings include rejected attempts.

## Validation and measurements

The twelve-attempt standalone test repeatedly switches one workspace through
counter values 0 → 1 → 2. All four scaled attempts pass the independent
long-double original-QP audit; the four attempts in each unscaled mode remain
unqualified on the local GPU. This checks that scaling does not leak into the
next update. The H100 executes the same test with complete vectors retained.

Both GPUs pass five new captured-return tests covering ordinary/graph execution,
fresh/pooled solvers, pointer rebinding, complete convergence and independent
certification. Each also passes all 117 existing regression tests, including the
older-library compatibility check, workspace reuse and physics verification.

Each GPU's 225-leg comparison retains the same 205 converged, independently
certified legs with nonnegative mass margin and rejects the other 20 attempts.
Neither loses a baseline qualified leg. Both modes create 72 workspaces.

| GPU | Baseline solver time | Retry solver time | Reduction | Largest shared final-mass difference |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 121.3415 s | 112.6747 s | 7.1% | 6.55e-8 kg |
| H100 | 155.8527 s | 137.6051 s | 11.7% | 3.92e-7 kg |

These are one paired run per GPU, including failed attempts, with identical
binaries and only the policy flag changed. They are preliminary solver-time
measurements, not a statistically established whole-application speedup.

Four local full campaigns ran in baseline/candidate/candidate/baseline order.
Both candidates completed all 36 native legs; each baseline failed on leg 36
of a lower-scoring alternative. All four retained best fleets pass both complete
physics checkers. Process medians were 79.607 s baseline and 76.157 s candidate.
The candidates complete more verification work because the baselines skip the
failed alternative's full-fleet check; these are not equal-work timing samples.
The H100 candidate also completes all 36 legs and both full-fleet checks, taking
135.406 s including process startup. No matched H100 full baseline was run here.

All five best fleets reproduce **12,810.136 weighted kg / 14,051.855 raw kg**,
with 23 ships, 195 collected asteroids and 196 deployed miners. This improves
return convergence without raising the retained fleet score. The numerical
retry runs entirely on CUDA, while Python still orchestrates mesh/search control
and CPU checkers provide independent certification.

The first campaign launches lacked incumbent fixture files and failed before
GPU work. Those failures are retained alongside the hash-checked fixture
correction and successful reruns. A wrapper exit code alone was insufficient:
publication checks the campaign's `success` field and both physics reports.

The final H100 pooled conditional-graph regression also passes targeted
Compute Sanitizer memcheck with zero errors (two complete captured-return calls).

Existing full-solver sanitizer limitations are separate from ordinary regression
success. In particular, prior local instrumentation failed in vendor iterative
refinement and H100 synchronization/race checks reported cuDSS failures. No
whole-program sanitizer-clean claim follows from the results above.

## Correction to the initial replay description

The saved v680 input already uses static P/A/G regularization of 1e-8. The earlier
trial explicitly setting 1e-8 repeats that baseline; it does not increase it.
The raw input and outputs are unchanged, and the current description is corrected.
The new screening records its exact settings separately, including 1e-11,
1e-13, 1e-7 and the 100-iteration tighter-refinement trial.

## Retrieved evidence and visualiser

[Complete evidence package](../results/lambda/2026-09-09/gpu-conditioning-retry-v696/)
contains both raw archives, source and binary hashes, all failures, comparisons,
campaigns, the H100 solution and the imported viewer dataset. Native builds were
frozen from `e7da7d9e` plus the recorded changes; the final reporting overlay was
tested separately using the same binaries.

Select **H100 GPU conditioning retry v696** in the existing web visualiser.
Complete downloaded solution:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-conditioning-retry-v696\h100-best\Result.txt
```

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-retry-v696&epoch=69807&preset=oblique&z=1'
```
