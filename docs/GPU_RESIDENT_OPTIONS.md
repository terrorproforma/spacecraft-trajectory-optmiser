# Keep return and collection options on the GPU

CUDA-generated compact options now stay in owned device tables through pricing,
selection and return pruning. The native selection arithmetic and stable ordering
are unchanged. A full-catalogue one-ship comparison measured **8.99% less complete
process time on RTX 5090** and **2.28% less on H100**. Each mode has two observations;
the H100 ranges overlap, so its timing advantage is not firmly established.

Validation also exposed an independent replay error: unconstrained DOP853 steps
could cross changes in the cubic thrust polynomial. That verifier now integrates
each polynomial span separately, preserving its original error tolerances. Solver
and physics acceptance limits were not relaxed. The best retained fleet still
passes and scores **12,805.194102 weighted kg**.

## Resident option ownership and execution

`spacepdhcg_orbitweaver_hop_options_resident` uses the same Lambert evaluation,
finite filtering, stable sort and packing as the existing host bridge. It copies
packed rows device-to-device into an immutable table. A table owns its allocation
independently of Lambert scratch, which can be resized or overwritten immediately.
Only the selected-row count is downloaded during creation.

`spacepdhcg_gtoc12_collection_resident` evaluates those device rows directly with
the existing collection kernels, then downloads the result and selected tuple.
The non-associative 1e-12 tie fold retains input order. Return pruning also uses
the existing first-feasible device mode, avoiding iteration over rows on the CPU.

Tables enforce creating-thread/device ownership and are released with their
Python object or backend context. The native API provides an explicit host read
for consumers that actually need all rows; these reads are counted. Neither
candidate campaign performs one. Query construction, count/result inspection and
route/fleet orchestration remain host work. This is not a fully GPU-controlled
application, and retained tables still require device allocations and a D2D copy.

The CUDA search bridge enables residency by default. Set
`SPACEPDHCG_TEST_GTOC12_RESIDENT_OPTIONS=0` to use the host option bridge. Direct
`GpuLambert.paired_options` calls retain their existing list-returning default;
the resident form is explicit there. CPU screening and public host APIs remain
available for their existing uses.

## Complete-process comparisons

Runs alternate baseline/candidate/candidate/baseline with one binary, toggling
only option residency. Each searches one ship with beam width 16, 48 neighbours,
three refinement candidates, four retiming attempts, two-day nodes and up to 40
SCvx iterations. Timings include process exit and cleanup.

| Hardware | Baseline runs | Resident runs | Median change |
|---|---|---|---|
| RTX 5090 | 28.6577, 31.2957 s | 26.8906, 27.6739 s | 29.9767 to 27.2822 s; 8.99% less |
| H100 | See archived stage times | See archived stage times | 29.7313 to 29.0538 s; 2.28% less |

Initial plans and logical search counts match exactly: 45,188,558 branches,
2,782,091 collection options, 5,930 collection queries, and 1,131,935 compact
options generated in 2,267 calls. All eight runs pass both final mission checkers
and retain 548.254620 weighted kg. Logical counts include reused work; they are
not counts of unique, fully optimized trajectories. Refinement variability can
affect total time, so the measured improvement is not solely a PCIe bandwidth
claim or a universal speedup.

| Data movement in this fixture | Before | Resident path |
|---|---:|---:|
| Compact-generation result downloads | 27,175,508 bytes | 9,068 bytes of counts |
| Selection option-row uploads | 66,770,184 bytes | 0 |
| Explicit resident-table host reads | — | 0 |
| Resident selection results and winning tuples | — | 242,480 bytes |

The last row includes 132 return-pruning queries in addition to the original
5,930 selections. Early v496/v497 telemetry omitted those 5,280 output bytes and
reported 237,200; the final counter includes them. Full-array transfer counts
were unaffected. Query metadata is still uploaded, so these rows are not an
exhaustive CPU/GPU transfer ledger.

## Accuracy, lifetime and mission checks

The captured real fixture has 5,930 selection queries over 992 distinct tables
(10,856,640 bytes of unique rows). A native replay on both GPUs compares the host
and resident bridges **bitwise**, then checks original captured winners and costs.
All queries pass. A separate native probe covers producer destruction, scratch
independence, exact ties, invalid queries, invalid rows, empty tables, ownership
and read capacity. Memcheck, synccheck and racecheck pass on both GPUs.

Python tests cover grow/shrink/chunked Lambert generation, sorted and unsorted
options, both Earth directions and asteroid transfers, repeated selections,
explicit reads, context closure, foreign-thread rejection, and GPU return pruning.
All **103 final tests pass on both GPUs**. The suite also covers SCvx, workspace reuse, CLI gates, the
official example and three published reference fleets. Final source and library
hashes are in the v506/v507 reports.

The wider four-ship run takes 213.3378 seconds and retains 29 mined asteroids and
2,088.668592 weighted kg. It passes both mission checkers and a subsequent replay
with the corrected verifier. Compared with the preceding pool-only 210.8269 s
observation, this does not establish a four-ship speedup.

## Independent replay diagnosis and correction

The first default-enabled suites, v498/v500, each had 75 passes and two failures.
One compared a resident object directly with a Python list; the test now reads
values explicitly for comparison. The other exceeded the unchanged 1e-5 kg
fresh/reused Lagrange mass-comparison bound by reporting differences of 11–26 mg.
Both corresponding trajectories passed the existing physics checks.

Repeated old/new solver comparisons reproduced variation, and stricter solver
tolerances did not eliminate it. GPU mass states agreed much more closely than
the DOP853 certificate masses. No stricter experimental solver setting was
promoted, and the workspace mass-comparison assertion remains unchanged.

For identical saved thrust histories, replay split at interpolation knots agreed
with independent 16- and 32-point quadrature to 3.2e-12 kg across 16 cases. The
old unsegmented integrator differed by up to 3.7654e-6 kg in that capture. The
regression test fails on the old implementation at the original 1e-12 DOP853
relative tolerance; it requires mass agreement within 1e-8 kg, including when
viewer output samples do not line up with thrust knots.

`t_eval` specifies dense-output samples, not adaptive integration boundaries.
The corrected verifier restarts DOP853 whenever its cubic interpolation stencil
changes. Constant-thrust arcs and single-polynomial arcs retain one integration
call. Requested history epochs remain ordered and unduplicated; boundary states
are carried forward even when no output sample is requested there. This fixes
the verifier instead of relaxing the optimiser's tests or acceptance thresholds.

The corrected verifier rechecked the archived 23-ship incumbent, 15-ship GPU fleet
and current four-ship result. All pass, with unchanged weighted scores. The
incumbent's raw returned mass is 14,047.802875 kg. No new official submission or
leaderboard position is claimed.

## Reproducible evidence

- [Local implementation, captures, failures and final validation](../results/lambda/2026-09-08/gpu-resident-options-local-v506/summary.json)
- [H100 matched campaign analysis](../results/lambda/2026-09-08/gpu-resident-options-v497/analysis.json)
- [H100 failed default-on suite](../results/lambda/2026-09-08/gpu-resident-options-v500/report.json)
- [H100 final validation](../results/lambda/2026-09-08/gpu-resident-options-v507/report.json)

Archives retain original reports, commands, source snapshots, input fixtures and
binary hashes. Each published folder has a SHA-256 manifest, and retrieved raw
archives have member manifests checked after download. Recipes retain original
machine paths; they require the pinned QOCO SOC-step build, CUDA/cuDSS and GTOC12
catalogue. The existing [larger-fleet visualiser](GPU_FLEET_RECOVERY.md#display-and-reproduce)
remains available; these changes do not replace the fleet incumbent.
