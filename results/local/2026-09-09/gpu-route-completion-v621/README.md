# Batched CUDA route completion: v621

The normal CUDA library now has an opt-in retained workspace for batched route-completion costing. It preserves the corrected scalar search's mining, actual-mass fuel calculation, authority checks, model inflation and final dry-mass-plus-cargo gate. This package records local RTX 5090 correctness tests and a bounded component benchmark. It contains no new Lambert solve, low-thrust certification, H100 completion run or fleet promotion.

| Completed experiment | GPU calls | Candidate evaluations | Result |
| --- | ---: | ---: | --- |
| Native test and saved controls | 3 | 304 | Passed |
| Production Python adapter, four models | 8 | 1,048 | Passed |
| CPU/GPU costing comparison | 60 | 14,200 per backend | Passed |

These are 71 completion kernel calls and 15,552 GPU proxy evaluations in total. The benchmark also performs 60 timed CPU batches containing 14,200 scalar completions. It repeats 20 distinct historical controls; repeated evaluations are not new solutions. Native and adapter correctness timings include instrumentation and are not performance measurements.

## Measured scope

For fitted models, warm complete-method medians improve from 4.850 to 1.617 ms at 24 candidates (3.00x), and from 9.544 to 2.812 ms at 48 (3.39x). Small flat batches regress: 0.44x at four and 0.96x at 24. The option remains off by default. The [complete benchmark results](runs/benchmark/RESULTS.md) retain all twelve groups, first-use costs, four warm samples per backend, packing, native/event, reconstruction, setup and close scopes.

At 1,024 fitted candidates the median complete method takes 41.207 ms; the median paired packing fraction is 91.39%. The separate mean kernel/event time is 0.101 ms. This supports moving retained metadata and route construction next; it does not establish whole-search speed or certified solutions per second. [Benchmark summary](runs/benchmark/summary.json) and [raw report](runs/benchmark/output/benchmark-report.json) preserve the exact measurements.

All failure classifications, indices and pickup decisions match. The saved adapter audit independently checks 42,240 fields, with maximum ordinary finite difference 4.55e-13 and maximum Decimal65 fuel discrepancy 9.73e-14 kg across 1,812 costed legs. Cargo and gains match exactly. All four owners close; observed Lambert calls are zero. The native controls include invalid mining, strict boundaries, model/authority precedence, repeated pickups and CPython 3.12 compensated cargo summation. Tolerances compare numeric outputs; they do not change acceptance gates.

## Source and evidence

The [normal full-core manifest](build/fullcore/manifest.json) binds committed base `7eb8828f61bd35ec0abaf98c7651c5e82384ae27` plus exactly four completion C++/CMake files. It is an uncommitted frozen source tree, not a synthetic Git commit. Its compiled source-tree SHA256 is `e12c5a94b940eb003d6b513b3418ba681fbf294996f78315b706dcd7655384ff`; the tested full library SHA256 is `cb977ff09b206de807996a8a21d7ccec41bb6a2a4c35a37c10ebe4883b83d2fb`. The binary is identified but not distributed here.

Host Python is pinned separately by the [CPU b report](python/cpu-attempt-b/report.json), tree `decb2a37cdfe2119d7d371d1d1ac75dcb39590be420a90b17485d0cea483fa27`. Later unrelated branch changes are excluded. All native/adapter/benchmark GPU tests use this full core; the earlier standalone build received CPU checks only. Its smaller source archive and original logs are retained. The full-core completion kernel uses 90 registers and 160 stack bytes on SM120. Existing persistent default/common resource footprints match their recorded baseline; resource equality is not a runtime-parity claim.

- [Controls and historical measured-mass decomposition](controls/README.md), including 20 historical and 22 synthetic C API fixtures.
- [Native GPU report](controls/gpu-output/report.json) and [adapter GPU report](runs/adapter/gpu-output/report.json), including raw outputs, lifecycle counters and exact input/source pins.
- [Independent orchestration and readback reviews](reviews/orchestration/REPORT.md), including the unchanged scalar AST and 272 scheduling patterns.
- [Exact mass-elimination design](reviews/mass-design/DESIGN.md) and [diagonal-metric design](reviews/mass-metric/DESIGN.md), with recoverable coefficient inputs and the required source/map dependencies.
- [Documentation snapshot](documentation/GPU_ROUTE_COMPLETION.md), whose links retain their original repository-relative locations. Documentation is outside both compiled and host-runtime fingerprints.

The mass notes are CPU mathematics for a possible later experiment. Their exact scan/absolute-sum analysis yields a represented-coefficient step-product bound at most 0.9025, with SOC-tied steps. They contain no implemented solver, convergence measurement or speed claim. The old uneliminated scaling cannot simply be reused after introducing cumulative mass coupling. Original KKT gates, virtual controls and affine constants remain requirements of any future implementation.

The historical return proxy overprices fuel on 20 of 23 retained routes. CUDA reproducing a proxy's decisions does not fix such false rejections or improve the fleet score. The [execution-plan snapshot](documentation/SOTA_EXECUTION_PLAN_2026-09-09.md) separates model admission, solver convergence and verified mission progress.

## Lossless reproduction

`sha256.json` indexes every final package file except itself. C++ source archives are retained once per build scope. Python snapshots use one content-addressed [source archive](python/source-objects.tar.gz) plus [path/size/hash maps](python/source-snapshots.json); the benchmark shares the exact CPU b bytes. Large native adapter and benchmark readbacks are stored losslessly in member-indexed `readbacks.tar.gz` files. Original scalar fields, NPZ array data, rejected cases, preparation failures and failed validation attempts remain present. There are no executable binaries, key files or caches in the package.

From this package directory, using Python 3.12:

```text
python -B reproduce/portable.py
python -B reproduce/portable.py --materialize /new/path/v621-evidence
python -B reproduce/recheck.py --out /another/new/path/v621-cpu-recheck
```

The first command verifies all file, archive-member and source-object hashes. Materialization restores the original evidence layout and exact host snapshots to a new directory. The last command replays saved native/adapter vectors and both mass coefficient audits using Python only, without loading CUDA. It writes fresh reports under the new output directory; it never overwrites this package. The mass source materialization is explicitly only the two prior C++ members consumed by that audit, with the unchanged original manifest; it is not presented as the full old archive.

The original runners are preserved for provenance. They contain machine-specific paths, device/library fingerprints and one-use launch guards; they are not automatic rerun instructions. Fresh GPU experiments require a separately reviewed environment and output location.
