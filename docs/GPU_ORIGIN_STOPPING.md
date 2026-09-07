# GPU origin and convergence diagnosis, 7 September 2026

**Experimental; numerical reliability remains unresolved.** This checkpoint fixes
the translated objective gap, false convergence on exhaustion, and invalid JSON
iteration ratios. It does not establish a fully reliable solver, a new fleet
score, or a complete application speedup. The prior failures are retained in
[the earlier diagnosis](GPU_OUTER_DIAGNOSIS.md).

## Translated objective gap

For the original QP, write the primal variables as `x = q + d`, where `q` is
the retained origin and `d` is the solver variable. The translated coefficients
are `c' = c + Pq`, `b' = b - Aq`, and `h' = h - Gq`. With

```
gamma = 0.5 q^T P q + c^T q
r = P x + c + A^T y + G^T z
```

the objective identities, including nonzero stationarity error, are

```
primal_original = primal_shifted + gamma
dual_original   = dual_shifted + gamma - q^T r
gap_original    = gap_shifted + q^T r
```

Ignoring `q^T r` can declare the shifted problem solved while the independent
original-coordinate audit rejects its objective accuracy. Ruiz scaling must
already have been removed from `r` before taking this product.

`scripts/gpu/prepare_qoco_origin_stopping.py` applies after
`prepare_qoco_stopping_accuracy.py`. The native audit retains the origin and a
device objective-offset scalar; parallel reductions update that scalar before
each translated numerical update. The vendor's stopping kernels compute the
corrected gap and original objective scale entirely on the GPU. Complementarity
remains an additional stopping requirement. Solver-coordinate objective output
slot 6 retains its existing meaning. Independent conic and physical certificates
and their tolerances are unchanged.

The optional `qoco_gpu_set_stopping_origin` extension borrows fixed device
addresses. Bind before the first metric evaluation; subsequent calls can retain
the same addresses but cannot replace them after capture. Their contents can
change in stream order. Native teardown destroys vendor graphs before freeing
the borrowed audit storage. Older libraries without this extension retain their
historical behavior; they do not acquire this correction automatically.

The 128-case original-equation test covers Ruiz 0/1/5/10, absent constraints,
zero quadratic terms, changing origins after metric capture, and nonoptimal
iterates. Both GPU metrics and the explicit host-dispatched arithmetic oracle
are checked against independently reconstructed original QPs. The oracle's
offset download is test-only and is absent from graph execution.

## Convergence and polishing

Two independent SCvx status defects were found while repeating trajectories:

* `predicted <= tolerance` treated a large negative predicted reduction as a
  convergence signal. Both native and reference paths now use its magnitude.
* Finalization changed budget/trust exhaustion to `converged` based only on
  relaxed local defect bounds. Those promotions are removed. Small local
  defects alone establish neither optimality nor accumulated trajectory error.

Polishing now confirms a preceding accepted convergence step with the finer
propagator. That same point can finish if the strict defect/virtual bounds and
the merit-change tolerance still hold. This avoids unnecessary near-identical
IPM solves without promoting an exhausted solve that lacks a convergence step.
Five native cases distinguish valid confirmation, excessive defect, excessive
merit change, missing convergence history, and excessive virtual control.
Whole-trajectory independent certification is still required.

## Reporting and remaining failures

The recovery executor printed undefined iteration reduction ratios as `-inf`,
which is invalid JSON. It now writes `null`, preserving finite values at full
precision. The recovery test parses every emitted JSON record strictly.

Early origin-correction runs passed 32/32 local and 8/8 H100 repeats. Subsequent
tests still found numerical failures, so those early results are not a
reliability guarantee. After the final polishing change, the isolated repeated
comparison qualifies **22/24 locally and 7/8 on H100**. H100 integration reports
**92 passed, 2 failed**. The rejected trajectories and full certificates remain
in the evidence, including cases with correct final mass but unacceptable
position error or a failure termination.

The final full local suite reports **333 passed, 4 failed**. Two failures were
numerical/trajectory failures; two were an old assertion that assumed graph
control traffic must always be strictly smaller. With faster four-attempt
convergence, the fixed graph exit makes the totals equal. That assertion now
checks the exact initial/priming/exit byte contract. Its focused rerun reports
**3 passed, 1 independent trajectory failure**; the numerical checks remain
unchanged. These counts are kept separate rather than presented as a green suite.

On the final H100 build, all six archived recovery problems converge and pass
their independent physics checks, taking 0.216–5.158 seconds under the separate
recovery profile. Native graph ownership, guards, deadlines and polishing tests
pass. Nested SCvx and guarded-solver memcheck pass, as does kernel initcheck with
API-memory checking disabled. The existing API-instrumentation blocking and
local WSL sanitizer limitations remain; this does not claim those are resolved.

A powered-descent fault-injection test also encountered a real numerical
failure before its intended injection. A separate fresh-process retry passed;
the failed run remains evidence of intermittent behavior. The IPM-session
capability test also produced a numerical disposition in one local run. Neither
test was weakened or bypassed, and no hidden solver retry was introduced.

The next diagnostic is the remaining ill-conditioned inner solves and their
effect on accumulated trajectory error. Correct metric identities and accurate
status reporting are necessary fixes, not proof of robust convergence.

## Reproduction

The final local core is frozen at
`/home/angus/build-spacepdhcg-gtoc12-v167/final`; the prepared vendor is at
`/home/angus/build-qoco-gpu-device-ir-v137/final`.
The matching H100 experiment is `/home/ubuntu/spacepdhcg-diagnose-v168`.
The vendor preparation starts from a fresh QOCO135 tree; do not apply it twice
or overwrite frozen libraries. Use the recorded CUDA 12.8/cuDSS build commands
and runtime checksums rather than relying on a directory name alone.

The repeated transfer criterion is convergence, independent physical
certification, and final mass `2445.3111007852112 +/- 1e-5 kg`.
The [evidence summary](../results/lambda/2026-09-07/gpu-origin-v168/summary.json)
records all stages, including failures and intermediate versions. No new fleet
trajectory or leaderboard rank is implied by these synthetic solver tests.
