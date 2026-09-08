# Faster CUDA Lambert screening

The zero-revolution hop solver now uses safeguarded interpolation inside its
existing sign bracket. This reduces serial root evaluations in both the warp-per-hop
and thread-per-hop kernels. Screening, branch selection and velocity calculation
remain CUDA C++; no new CPU numerical path is introduced.

Up to 16 Illinois-style interpolation steps retain the true endpoint residuals
for bracketing. Every fourth step is a midpoint; invalid or stagnant interpolation
also falls back. Remaining iterations use the existing bisection implementation.
Budgets below 64 use the original bisection path throughout. At the minimum 16 scan
samples, the zero-revolution bracket is at most 8π²/16 wide, leaving sufficient
bisections for the existing 1e-13 width gate. Requested time tolerances, scan
ordering, short/long-way selection and mission acceptance thresholds are unchanged.
The generic multi-revolution solver is unchanged.

The faster path is the default. Set `SPACEPDHCG_TEST_GTOC12_FAST_LAMBERT_ROOT=0`
before launching to compare with original bisection; `1` explicitly enables it.
The switch is resolved at launch/capture time, so an already captured graph retains
its selected method.

## Complete campaign measurements

Each GPU runs A/B/B/A on one frozen binary, two complete processes per mode.
All runs use the full catalogue, one ship, beam width 16, ten maximum deployments,
48 neighbours, three refined candidates, four retiming attempts, two-day nodes,
40 SCvx iterations, CUDA numerical backends, QOCO graphs and zero Ruiz passes.
The archived workers contain exact commands and environment settings.

| GPU | Baseline process seconds | Faster process seconds | Median change |
| --- | --- | --- | --- |
| RTX 5090 | 28.8168, 29.2955 | 24.3356, 25.0303 | 29.0562 → 24.6830 s; 15.05% less time |
| H100 80 GB | 31.2086, 28.4045 | 28.2544, 29.7035 | 29.8065 → 28.9789 s; 2.78% less time |

H100 ranges overlap; the sample does not establish a reliable whole-campaign
speedup there. Local ranges do not overlap. These are measured workload results,
not a general speedup guarantee or a fleet score improvement.

Every run performs 45,188,558 logical transfer-branch requests and evaluates
2,782,091 collection options. These counters include screening and repeated
queries; they do not count certified low-thrust solutions. All eight paired
campaigns pass both the official and independent mission verifiers and return
548.254620 weighted kg, differing below 1e-6 kg. Final default-build campaigns
also pass on both GPUs (26.0579 s local; 29.5564 s H100).

## Isolated screening

Identical preconstructed requests, five warmups and 30 samples in each A/B/B/A
block. Times include the native host API's transfers and synchronization, but
exclude request construction. Results are checked against the baseline.

| Batch transfers | RTX 5090 faster throughput | RTX speedup | H100 faster throughput | H100 speedup |
| --- | --- | --- | --- | --- |
| 1,024 | 2.51 million/s | 2.10× | 9.04 million/s | 1.51× |
| 16,384 | 4.75 million/s | 2.73× | 17.40 million/s | 1.51× |
| 65,536 | 5.79 million/s | 2.95× | 22.42 million/s | 1.71× |

Each transfer screens short and long zero-revolution branches. It still needs
low-thrust refinement and mission verification before it is a usable solution.

## Accuracy and evidence

Final source passes 46 tests on each GPU. Tests compare disabled, enabled and
default modes at 16/256 scan samples, 32/64/256 iteration budgets and 31/16,385
batch sizes, spanning both kernel dispatches. Independent Kepler propagation
requires position closure below 0.01 km and velocity closure below 1e-8 km/s.
Other tests cover CPU reference agreement, invalid inputs, element propagation,
fused tables and resident collection tables. Native graph replay, memcheck,
synccheck and racecheck pass for the Lambert probe on both GPUs.

These sanitizer results apply to Lambert. Previously documented full-QOCO/cuDSS
sanitizer failures remain unresolved. CPU orchestration, some symbolic setup and
mission bookkeeping remain; the complete application is not yet fully GPU controlled.

The first test fixture tried propagating an empty result at a deliberately short
iteration budget. Its failure is retained in v581; the corrected fixture preserves
baseline agreement and requires nonempty accepted results at normal budgets.
The initial v579 profile overcounted elapsed time around HiGHS callbacks and is
invalid. v580 isolates those callbacks and passes clock consistency checks;
instrumented profiles are retained as diagnostics, not performance benchmarks.

[Retrieved evidence](../results/lambda/2026-09-08/gpu-fast-lambert-v589/) includes
raw process logs, trajectories, certificates, source snapshots, runtime binaries,
commands, microbenchmark samples and SHA-256 manifests. The final source differs
from the paired experiment only in formatting/comments and default selection.

## Display the final H100 trajectory

Dataset `gtoc12-v588` contains one ship visiting eight asteroids, 510 exact
independently propagated samples and 3,010 checked context-orbit points.
It is a performance benchmark; the best fleet remains 12,805.194 weighted kg.

Viewer directory:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser`

If the server is already running, open
[the final H100 result](http://127.0.0.1:4173/?dataset=gtoc12-v588&epoch=69807&preset=oblique&z=1).
Otherwise start it with:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
```

Select **H100 faster Lambert v588** in the dataset menu. The archived raw H100
solution is `campaign-final/default/output/fleet/Result.txt` in `lambda-raw.tar.gz`.
