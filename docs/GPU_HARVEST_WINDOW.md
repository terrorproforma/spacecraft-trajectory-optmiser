# Resident GPU harvest-window pricing

The route substitution search prices harvesting windows repeatedly. Previously,
`CollectPairTable.harvest_window_cost` downloaded resident float32 Lambert tables
and calculated the minimum on the CPU. Profiling the standard one-ship campaign
found 692 table reads transferring 11,143,968 bytes through this path.

The CUDA implementation retains the time-of-flight grid and reduction scratch
with each device table. Parallel kernels apply the original thrust-authority
gate, inflation model and rocket equation, then reduce the minimum. Each query
returns one double. This preserves pricing of the same float32 table values;
it does not replace the low-thrust trajectory solver or mission verification.

The Python interface still assembles policy parameters and orchestrates queries.
Fitted inflation also uploads a per-row geometry vector. Consequently this change
does not make the whole application GPU native. Flat and ratio models need no
geometry upload. Missing native support raises a rebuild error in CUDA mode.

## Measured outcome

H100 comparison, ordered baseline/candidate/candidate/baseline:

| Run | Complete CLI time (s) | Verified weighted kg |
|---|---:|---:|
| Baseline 0 | 52.149356 | 548.254620 |
| GPU window 0 | 53.385340 | 548.254620 |
| GPU window 1 | 54.438435 | 548.254620 |
| Baseline 1 | 53.644853 | 548.254620 |

Every mission passed the locally executed official checker and the independent
checker. All four runs evaluated 45,188,558 transfer branches and 2,782,091
collection options. The candidate replaced 11,143,968 bytes of full-table reads
with 5,536 scalar-result bytes over 692 queries, with no fitted-geometry upload.
Median time **increased 1.92%** (52.897105 to 53.911887 seconds). Two samples per
mode do not establish a general performance difference or an overall speedup.

The instrumented RTX 5090 runs recorded 79.799544 and 75.131914 seconds before
and after, with the same search counts and checked mission. They are single
samples, include profiling, and are not a speedup claim. Full table-read
instrumentation recorded zero such reads after the change. The measured time
inside the original reads was only 0.069411 seconds, so download volume alone
does not identify the dominant remaining runtime cost.

## Accuracy and reproducibility

Both RTX 5090 and H100 passed 92 relevant tests. H100 Compute Sanitizer ran all
12 harvest tests with zero errors. The tests compare GPU minima with host
pricing of the same float32 tables, across flat, ratio and fitted models,
time-of-flight boundaries, empty windows and repeated changes of mass, with an
absolute 1e-9 kg tolerance. Full table downloads are forbidden during the tested
resident queries. This sanitizer scope does not cover the entire vendor solver.

[Evidence](../results/lambda/2026-09-08/gpu-harvest-window-v378/summary.json)
includes all four complete H100 campaign reports, logs, exact source overlays,
runtime hashes, local profiles and execution recipes. The retrieval verified
97 remote files. The initial configure attempt failed because the copied source
had no Git metadata; its log and the explicit isolated-source recovery recipe
are retained alongside the successful run. No GPU test ran on that failed build.

Run the tests against a rebuilt native library with
`SPACEPDHCG_GTOC12_GPU_TESTS=1` and the normal CUDA library/data environment:

```sh
python -m pytest tests/test_gtoc12_gpu_harvest_window.py \
  tests/test_gtoc12_gpu_resident_collect_tables.py \
  tests/test_gtoc12_gpu_collect_tables.py tests/test_gtoc12_gpu_collect_dp.py \
  tests/test_gtoc12_gpu_scvx.py tests/test_gtoc12_gpu_cli.py -q
```

The subsequent recovery implementation shares this native translation unit;
its combined build is validated separately from these frozen v378 measurements.
