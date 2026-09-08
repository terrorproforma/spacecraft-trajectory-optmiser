# Stop stationary unsuccessful SCvx attempts

The four-ship v403 profile spent 103.979 seconds on 20 unsuccessful low-thrust
legs, versus 51.787 seconds on 205 converged legs. Several difficult legs reached
a stationary penalized point with substantial virtual control, then repeatedly
shrunk the trust region without improving physical feasibility. Eventually the
small trust region made the conic solves more difficult too.

The CUDA SCvx controller now stops this local attempt after two consecutive
qualified candidates have all of the following:

- Remaining dynamics defects or virtual control outside the existing feasibility gate.
- A step within the existing step tolerance.
- Nonlinear merit change within the existing objective tolerance.
- Model objective change from the previous qualified candidate within that same tolerance.

An invalid or unqualified candidate clears the consecutive evidence. The model
comparison requires a prior qualified candidate, so termination needs at least
three qualified candidates in sequence. It returns **failed**, diagnostic 7:
`stationary penalized point with nonzero defects; no infeasibility certificate`.
The terminating candidate is not accepted or copied into the trajectory. This
does not prove global infeasibility, cache a failed boundary, or prevent another
seed or route attempt. Downstream route and return refinement reject failed
solutions. Feasible convergence, conic qualification, physical thrust limits,
polishing and independent mission verification retain their existing gates.

The first experiment compared model and nonlinear merit directly. It stopped
none of the 20 failed legs: the nonlinear penalty subtracts a per-component
defect allowance, while the model uses raw virtual control. Their absolute
difference need not approach zero at stationarity. That negative experiment is
archived alongside the corrected comparison of successive model values.

The default is enabled in both host dispatch and GPU graph execution.
`SPACEPDHCG_TEST_GTOC12_STATIONARY_FAILURE=0` disables it for controlled
comparisons. This switch changes local stopping, not solution acceptance.

## Fixed 225-leg replay

Each replay reconstructs all saved boundaries and solver settings, uses a CUDA
seed and GPU graph execution, and independently propagates every converged
output using DOP853. Each GPU uses the same frozen binary for its off/on pair.
These times include seed generation and solving, excluding certification.

| GPU | Disabled | Enabled | Less solve time | Converged outputs |
|---|---:|---:|---:|---:|
| RTX 5090 | 161.230 s | 128.494 s | 20.30% | 205 / 205 retained |
| H100 | 195.558 s | 157.501 s | 19.46% | 205 / 205 retained |

All 205 outputs in each mode pass independent propagation and the minimum-mass
comparison. The maximum change in propagated final mass is 3.95e-8 kg locally
and 5.89e-7 kg on H100. Five unsuccessful local legs and six H100 legs terminate
with the new diagnostic. Other unsuccessful legs remain unsuccessful. There is
one full replay per mode, so these are fixture results, not universal speedups.

## Complete missions

An H100 baseline/candidate/candidate/baseline comparison reduced median complete
one-ship runtime from **43.138 to 38.661 seconds**: **10.38% less time, 1.116x**.
Every run processed the same 45,188,558 logical branch requests and 2,782,091
collection options, returned 548.254620 weighted kg, and passed both final
mission checkers. There are two runs per mode.

A wider local confirmation completed in **247.183 seconds**, preserving four
ships, 29 mined asteroids and **2,088.668592 weighted kg**, with both checkers
passing. Its 169,753,864 logical branch requests and 18,250,121 collection options
match v403. This is a single confirmation, not a paired timing experiment.

Native controller tests cover qualification from both host and device packets,
the model/nonlinear offset, excluded cases, counter resets, unchanged feasible
convergence and preservation of the failed diagnostic through finalization.
Memory, synchronization and race checks pass on both GPUs, as do 41 selected
Python tests. The final default-on build has separate validation evidence.

The best retained fleet remains **12,805.194 weighted kg**. Python orchestration
and independent CPU checking remain; this change does not complete GPU migration.

## Reproducible evidence

- [Local negative experiment and fixed replay](../results/lambda/2026-09-08/gpu-stationary-local-v407/summary.json)
- [H100 replay, source overlay and checks](../results/lambda/2026-09-08/gpu-stationary-v408/summary.json)
- [Local wider mission and final default build](../results/lambda/2026-09-08/gpu-stationary-local-v413/summary.json)
- [H100 paired complete campaigns](../results/lambda/2026-09-08/gpu-stationary-fleet-v410/summary.json)
- [H100 final default-on confirmation](../results/lambda/2026-09-08/gpu-stationary-v412/summary.json)

Archives include per-member SHA-256 manifests, exact boundaries/settings, per-leg
histories, certificates, converged state/thrust samples, commands and runtime
hashes. H100 v408 uses the archived v402 source plus its included overlay; v412
uses frozen v408 plus its overlay. v410 uses the unchanged v408 source/runtime.
Local native source overlays are recorded in v404/v406/v411. Replay recipes run
from the repository root with the recorded library and data paths restored.
