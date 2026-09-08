# Parallel Lambert directions: opt-in small-batch execution

The CUDA hop solver can now allocate one full warp to each of the short- and
long-way zero-revolution directions. Both warps independently use the existing
scan and root algorithm, then share their candidates in block memory. Strict
cost comparison preserves the short-way tie rule. Padding threads participate
in the final barrier, including odd-sized batches. No physics tolerance changes.

The final implementation uses this layout only through 1,024 transfers per batch.
It leaves the larger-batch dispatch unchanged. Enable it before launching with:

```bash
export SPACEPDHCG_TEST_GTOC12_PARALLEL_DIRECTIONS=1
```

The default is off; `0` explicitly selects the production dispatcher. The switch
is resolved during launch or graph capture; already captured graphs retain their
selected method. It adds no CPU numerical processing or new device/host transfers.

## Measurements

Full-catalogue one-ship runs use A/B/B/A order, two complete processes per mode on
the same frozen binary per GPU. They retain the previous 16-wide beam, 48
neighbours, three refinements, four retiming attempts, two-day nodes, 40 SCvx
iterations, CUDA numerical backends, QOCO graphs and zero Ruiz.

| GPU | Production dispatcher, seconds | Parallel small batches, seconds | Median difference |
| --- | --- | --- | --- |
| RTX 5090 | 24.8487, 24.3926 | 25.1990, 23.2708 | 24.6207 → 24.2349 s; 1.57% less |
| H100 80 GB | 28.1554, 32.9089 | 28.6526, 27.7579 | 30.5321 → 28.2053 s; 7.62% less |

Ranges overlap on both GPUs. One slow H100 baseline materially affects its
median. These samples do not establish a reliable overall speedup, so this is
an opt-in execution strategy, not a default performance claim.

All eight final-policy campaigns evaluate 45,188,558 logical transfer branches
and 2,782,091 collection options. They pass both mission verifiers, retaining
548.254620 weighted kg with differences below 1e-6 kg. Those counters measure
screening and search work, not certified low-thrust solutions.

Final-policy microbenchmarks reuse identical preconstructed inputs, with five
warmups and 30 samples per A/B/B/A block. Times include the native host API,
transfers and synchronization. At 31 transfers per batch, the measured ratios
are 1.40× locally and 1.17× on H100. At 1,024, they are 1.057× and 1.074×.
At 65,536, both modes select the same production kernel; measured variation is
not a different algorithm's speedup or regression.

## Rejected layouts and validation

The first prototype split one warp into two half-warps. It preserved every
checked result but did not consistently improve small batches and regressed
at 65,536 transfers. Allocating two full warps for *all* batch sizes improved
several small-batch timings but also regressed at 65,536 on both GPUs. Both
prototypes, all measurements and their exact source/binaries are archived.

The final small-batch policy passes **106 tests on each GPU**, including exact
comparisons of every hop-result field at batch sizes 31, 257, 1,024, 1,025 and
16,385; scan sizes 16, 31, 32, 33, 256 and 1,024; and bisection/interpolated roots.
Coverage includes invalid inputs, odd batches, dispatch boundaries, CPU reference
agreement, independent orbit closure, ephemeris/fused tables and resident tables.
Native graph replay and Lambert memcheck, synccheck and racecheck pass on both GPUs.

Two isolated build attempts initially lacked Git metadata and then a pinned
third-party patch. Their failed configuration logs are retained. The first local
test run passed 78 cases before encountering an omitted archived table fixture;
restoring that exact pinned fixture completed the remaining tests. Neither issue
required a numerical change. The final policy passes the entire 106-test set.

[Reproducible evidence](../results/lambda/2026-09-09/gpu-parallel-directions-v622/)
contains per-process reports, raw microbenchmark samples, trajectories and
certificates, source snapshots, runtime libraries, commands and SHA-256 manifests.
Sources are frozen from `8a5dffee` plus the recorded kernel/test overlays, isolating
the comparisons from concurrent fleet-scoring and joint-optimisation work.

No new fleet record or competition submission results from this experiment. The
incumbent remains 12,805.194 weighted kg. CPU orchestration and symbolic work remain;
Lambert sanitizer passes do not resolve the documented full-QOCO/cuDSS failures.
