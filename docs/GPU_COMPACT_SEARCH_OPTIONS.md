# CUDA filtering and ordering of search options

This records the return/collection option migration. The subsequent
[GPU initial Earth beam](GPU_EARTH_BEAM.md) also migrates initial-beam ephemerides,
scoring and ranking; the remaining-work discussion below describes this earlier
measurement stage.

The return-window and collection-hop search paths can now compute total delta-v,
filter invalid transfers and order return candidates in C++/CUDA. Previously
Python downloaded every detailed screening result and performed these operations
on the host. The new bridge returns only `(delta_v, departure, flight duration)`
triples for valid transfers.

Return ordering remains ascending delta-v, descending departure epoch, then
original row order for exact ties. Collection options retain input order. CUB
merge sort and stable selection operate on retained workspace buffers; each
transfer still uses the existing orbital propagation, Lambert branch selection,
root solve and Earth velocity allowances. No physics or acceptance tolerance
changes.

This is enabled by default inside CUDA screening scope.
`SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS=0` selects the corrected host option
builder for controlled comparisons. CPU screening keeps its existing path.

The host bridge accepts batches larger than the workspace screening capacity,
screens them in chunks and orders the combined result. Scratch allocation grows
when necessary and is reused for subsequent calls. Workspace telemetry includes
the retained scratch. CUDA computes the valid count before the host downloads
only the compact rows. This remains a blocking host-output interface: Python
constructs schedule axes and metadata tuples, orchestrates the beam and fleet,
and runs independent mission checks. It is not complete GPU control of search.

The fallback option builder also now computes the full delta-v vector once.
Previously it accessed a property that allocates that entire vector twice per
valid row, causing quadratic host work. Controlled compact-path comparisons use
this corrected fallback as their baseline.

Native tests cover synthetic exact ties, invalid and nonfinite costs, all-invalid
batches, chunk boundaries, scratch growth and reuse, invalid API arguments and
workspace destruction. Python integration tests compare compact options exactly
with detailed CUDA screening, including Earth allowances and changed epochs.
Search tests forbid fallback to CPU ephemerides, detailed screening and host
delta-v calculation. Complete campaigns additionally require both the official
checker and independent mission replay to pass.

All **27 selected Python tests** pass on RTX 5090 and H100. The native probe's
20 synthetic cases and public API parity checks pass on both, with zero memory,
synchronization or race errors. Final default-on Python integration passes under
memory checking on both GPUs and synchronization checking on H100. Separate final
default campaigns pass both mission checkers in 30.191 seconds locally and 32.766
seconds on H100; those are individual confirmations, not paired comparisons.

## Complete-run measurements

Both comparisons use baseline/candidate/candidate/baseline process order, two
runs per mode, the same native binary and the same search settings. The baseline
already includes the correction to repeated host delta-v vector allocation.

| GPU | Host option median | CUDA option median | Less complete-run time |
|---|---:|---:|---:|
| RTX 5090 | 30.658 s | 30.613 s | 0.14% |
| H100 | 37.548 s | 34.286 s | 8.69% |

The local comparison is effectively flat, with overlapping timings. H100 shows
a measured improvement on this fixture; two samples per mode do not establish a
universal speedup. Every run retains 45,188,558 logical transfer branches,
2,782,091 collection options and 548.254620 weighted kg. All existing screening
counters match; two new counters record compact rows and their download bytes.

The migrated paths process 1,132,379 transfers per one-ship run and return
1,131,935 valid options. Output falls from 81,531,288 bytes of detailed results
to 27,175,508 bytes, including 2,267 downloaded counts: approximately 66.7% less
output for these paths. This is not the total application transfer volume.

A separate local comparison of the vector-allocation correction alone had
overlapping timings and no clear complete-run improvement; that evidence is
retained too. A direct profile of the compact path records 2,267 calls consuming
1.54 seconds including synchronization and Python tuple creation. Native SCvx
remains the largest measured component, followed by collection-table construction
and detailed screening elsewhere. Initial-beam scoring and orchestration still
provide concrete CPU migration targets.

A wider local confirmation completed in **231.976 seconds**, retaining four
ships, 29 mined asteroids and **2,088.668592 weighted kg**, with both mission
checkers passing. It processed 169,753,864 logical branches and 18,250,121
collection options. This is a single wider validation, not a paired speed test.

The best retained fleet is still **12,805.194 weighted kg**. These changes improve
the execution path without establishing a new fleet score.

- [Local comparisons, wider fleet, profile and final validation](../results/lambda/2026-09-08/gpu-compact-options-local-v429/summary.json)
- [H100 paired comparison and native validation](../results/lambda/2026-09-08/gpu-compact-options-v426/summary.json)
- [H100 final default configuration](../results/lambda/2026-09-08/gpu-compact-options-v430/summary.json)

The archives retain commands, exact source overlays, runtime hashes, raw campaign
reports, verifier outputs and per-file SHA-256 manifests. The local core is the
frozen v423 build; H100 compiles the native overlay in v426 and reuses that binary
for final default validation. The QOCO numerical runtime is unchanged.
