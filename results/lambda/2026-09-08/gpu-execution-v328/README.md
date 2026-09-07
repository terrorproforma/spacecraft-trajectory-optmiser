# Explicit GPU execution and matched H100 campaigns

Development candidate on top of `d1758141`; not merged into main. This change
selects existing native GPU loops through the CLI. It does not fix the outstanding
inner-solver repeatability failures or make the entire application GPU-native.

`--gpu-execution auto|graph|dispatch` is available on run, cluster-fleet,
fleet-master, retime-returns and joint-itinerary. Auto selects graph execution
when `--outer-loop-backend cuda` is requested. Explicit graph requires CUDA SCvx
and one worker. Dispatch disables the graph switches for controlled comparisons.
Command scopes restore inherited environment values, including on exceptions,
and serialize process-wide switch changes. Direct library calls retain their
existing switch-based API. Reports distinguish requested and selected modes;
selection alone is not evidence that refinement actually ran.

The current native graph implementation requires its pinned QOCO/cuDSS extension
and supports SM90/SM120 (tested H100/RTX 5090). Unsupported graphs fail explicitly;
there is no CPU solver fallback. Setup and cold priming remain host operations.
Python still orchestrates search/fleet work; this campaign still downloads
11,143,968 bytes of collection tables for remaining host consumers.

## Validation and complete-campaign comparison

Both GPUs pass 38 CLI/refinement tests, including all three public execution modes,
contradictory inherited flags, independent physical certificates and the unchanged
final-mass target. Runtime report rows and command-download counts distinguish
actual graph execution from dispatch. No native numerical binaries changed for
this comparison: H100 uses native v314 and QOCO v174.

| Run | Public execution option | Complete seconds | Verified kg |
|---|---|---:|---:|
| v329 | graph | 58.811 | 548.255 |
| v330 | dispatch | 61.040 | 548.255 |
| v331 | dispatch | 60.615 | 548.255 |
| v332 | graph | 58.884 | 548.255 |

Median time falls **60.828 -> 58.847 seconds (3.26% less time)**. Two runs per
mode on one fixed campaign are a limited comparison, not a universal speedup or
a reliability guarantee. All four pass the official and independent mission
checkers. Each evaluates 45,188,558 transfer branches and 2,782,091 collection
options and selects one eight-asteroid mission from three final columns. These
are overlapping search counters, not millions of certified trajectories.
The separate incumbent remains 12,805.194 weighted kg.

`summary.json` counts actual graph report rows in the initial top-three refinements;
its counts exclude retiming reports to avoid double counting repeated exports.
Complete timing and final checker results include retiming. `sha256.json` pins the
103 downloaded files; `retrieval.json` pins the remote archive. `evidence-sha256.json`
additionally pins local tests, recipes, diagnostics and viewer input/QA. Source
files are preserved in `repo/`; per-run reports pin unchanged native binaries.

## Linear-system arithmetic diagnosis

An isolated instrumented RTX build captures 138 before/after-refinement snapshots
from one replay of the exact difficult departure QP. It synchronizes and recomputes
residuals for observation; it is not a performance build, and those synchronizations
can perturb numerical scheduling. The production library is unchanged.

The audit reconstructs the symmetric KKT product with long-double arithmetic and
applies the compact Nesterov-Todd scaling twice. It independently cross-checks the
worst finite row using 80-digit decimal arithmetic. Those cross-checks differ by
at most 1.80e-12 over 106 finite snapshots. The remaining 32 snapshots contain
nonfinite iterates, recorded as null metrics; the solver ultimately restores an
iterate which passes the original external QP audit.

For snapshot 0048 after refinement, the double-precision device calculation gives
an infinity residual of **2.715e-10**, while the higher-precision product gives
**1.710e-9**. Their vectors differ by up to **1.439e-9**; the 80-digit cross-check
at the worst row agrees with long double within **2.46e-14**. The stored WtW
matrix product also differs from the compact product by up to **2.528e-9**.
This establishes meaningful arithmetic error in late linear residuals; it does
not prove the cause of every QP failure or justify loosening qualification.

`diagnostics/linear-v333` preserves the full audit, ten representative binary
snapshots, hashes for all 138 local snapshots, the instrumented backend and the
original QP/output. All original snapshots remain under WSL
`/home/angus/build-qoco-linear-snapshot-v333/snapshots`. Reproduction recipes are
in `recipes/`. The next numerical step is an independently checked GPU residual
operator using these captured inputs before integrating it into the solver.
The earlier compensated-residual prototype failed qualification and remains rejected.

## Display the new run locally

Dataset: **GPU graphs v332 (548 kg certified)**. It retains 506 exact archived
samples, with 3,010 catalogue-context samples checked during import. All viewer
tests and dataset checks pass; browser inspection confirms one ship, eight
asteroids, both checker passes and physical 1x geometry. Existing datasets retain
their original provenance.

Open <http://127.0.0.1:4173/?dataset=gtoc12-v332&epoch=69807&preset=oblique&z=1>.

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-v332&epoch=69807&preset=oblique&z=1'
# Run this only if the viewer server is stopped; keep the terminal open:
node scripts/serve.mjs --port=4173
```

Full solution:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-execution-v328\v332\output\fleet\Result.txt`
