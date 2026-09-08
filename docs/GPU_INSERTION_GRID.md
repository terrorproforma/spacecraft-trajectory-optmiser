# Native CUDA insertion timing grid

The previous insertion generator always placed the new deployment and collection
at the midpoints of their split legs. CUDA can now explore unequal splits of both
legs independently, including the existing borrowed-time seeds. The route layouts,
epochs, geometry and mass evaluation stay in the retained native workspace.

`JointSettings(insert_split_points=3)` selects fractions 1/4, 1/2 and 3/4 on each
split interval: nine split pairs, each with four ordered borrowing seeds. Supported
grid sizes are odd integers from 1 to 9. Five points use sixths; nine use tenths.
The default remains one point because the wider probe found no gain on the tested
routes. Unequal splits require the CUDA joint backend and prepared device layouts;
unsupported libraries/settings produce an explicit error, without a CPU fallback.

The additive `spacepdhcg_gtoc12_joint_prepared_insertion_grid_host` API takes the
grid size and emits `4 * points^2` rows per layout, in deployment-fraction,
collection-fraction, borrowing-seed order. Original epoch shifts and borrowing
admission rules are unchanged. The midpoint rows preserve the legacy arithmetic.
Only the new visits change within the already allocated split intervals; ordinary
flight-time, thrust, mining-stay and mass checks still decide feasibility.

The Python integration scales its default layout batch inversely with the grid
area, retaining approximately 16,384 candidate rows per call rather than expanding
memory by the grid area. Explicit `layouts_per_batch` still controls the number of
layouts. Shared edge inputs upload once per prepared source. Source snapshots,
workspace reuse and invalid-input behavior retain their previous contracts.

## What the wider search found

The v787 probe starts from the same v779 certified route summaries and exclusions
as the previous checkpoint. Both GPUs give identical evaluation and rejection
counts. All ten probes find zero feasible insertions.

| Route | Neighbour request / radius | Grid points | Evaluated schedules |
|---|---:|---:|---:|
| 1786 | 60 / 2.5 | 1 | 12,992 |
| 1786 | 60 / 2.5 | 3 | 116,928 |
| 1786 | 600 / 4 | 3 | 780,192 |
| 1786 | 3,000 / 8 | 3 | 3,725,568 |
| 1786 | 3,000 / 8 | 5 | 10,348,800 |
| 2297 | 60 / 2.5 | 1 | 1,890 |
| 2297 | 60 / 2.5 | 3 | 17,010 |
| 2297 | 600 / 4 | 3 | 117,369 |
| 2297 | 3,000 / 8 | 3 | 768,852 |
| 2297 | 3,000 / 8 | 5 | 2,135,700 |

The five-point pair evaluates **12,484,500 schedules**; all ten probes total
18,025,301 with overlapping neighbourhoods. Counts exclude disabled borrowing
seeds. They include preflight rejections and are not counts of certified trajectory
solutions. The catalogue neighbourhood's existing coarse element prefilter remains.

On RTX 5090, the widest five-point searches take 19.969 and 8.792 seconds. These
are single probe observations including setup, transfers and result handling,
not repeated timing benchmarks or a speedup claim. Raw H100 timings and all
telemetry are retained alongside them.

A separate v790 diagnostic retains all final mass-gate failures: 77 for route
1786 and two for route 2297. The best remaining weighted cargo is approximately
311.159 and 84.178 kg respectively, with negative mass margins of 32.862 and
60.463 kg. Those already reduced cargo amounts are below the incumbents, so none
is a promising score-gain candidate to send to refinement. They remain rejected;
no mass, thrust or verification threshold was relaxed. Denser splitting alone
does not fix the two routes' insertion limitations. The next search work should
generate different routes and explore the rest of the fleet.

## Validation and complete replay

Both GPUs pass **142 regression tests**, with seven skips for absent historical
fixtures. All **36 insertion/layout/grid tests** pass memcheck, racecheck and
synccheck with no errors or hazards. The grid tests cover both archived route
structures, three/five/nine points, cold and cached geometry, exact legacy
midpoint rows, independently reconstructed split fractions, unchanged original
epochs, every enabled result, valid mass details, stable survivor ranking through
the ordinary `JointSettings` entry point, and invalid input/output preservation.

The v789 full mission replay explicitly selects a three-point grid. Each GPU
performs **36 native solves: 34 converge and two return infeasible**. All 33
accepted route legs converge and certify on CUDA. Both full-fleet independent and
official physics checks pass. The result remains **23 ships, 195 asteroids,
14,044.353 raw kg and 12,843.556 weighted kg**. Differences below 1e-8 kg are not
score gains. The raw campaign's `improved` field compares with its older v733 input;
the evidence audit separately compares the score against v780.

The replay performs 136,694 joint evaluations in 193 batches. Insertion uses
5,138 layouts in 13 batches, with 51,030 disabled seed rows. The geometry stage
computes 1,185,023 hops and reuses 154 cached hops. Shared source upload remains
575,640 bytes, but full result and epoch downloads grow to 122,328,872 and
57,318,912 bytes. GPU survivor compaction is still needed to remove those wasted
transfers; metadata compilation, orchestration and survivor sorting still involve
Python. This is not a fully GPU-controlled fleet optimiser yet.

Source is frozen from `59a62be3` with the recorded overlays. Initial native builds
and tests are v786; v788 adds the production settings integration and its tests
without changing the CUDA core. Both complete numerical replays use v788 and the
new v786 core, including the previously published archived-control initializer.

## Retrieved evidence

[Evidence, checksums and reproduction scripts](../results/lambda/2026-09-09/gpu-insertion-grid-v791/)
contain both native libraries, frozen source, test logs, every probe, rejected
mass candidates, native solve records and full numerical mission replays. Each
archive has 918 hashed payload members, checked before and after transfer.

H100 replay:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-insertion-grid-v791\h100\v789\fleet\Result.txt`

There is no improved fleet to promote. The existing visualiser continues to use
the verified v780 fleet; [startup and loading instructions](GPU_INSERTION_LAYOUTS.md#retrieved-h100-result)
are unchanged.
