# Retained catalogue fingerprints for CUDA completion

The completion model no longer rescans the same immutable catalogue on every
request batch. The loaded pinned catalogue owns immutable byte-backed NumPy
arrays. The host retains up to four SHA-256 prefix states for the pricing
policies using that snapshot; the existing CUDA model retains the corresponding
geometry. Across the two measured routes, catalogue hashes fall from **1,302 to
4**, with **1,298 reuses**. This removes 99.7% of those catalogue scans.

This is a reduction in CPU work feeding the retained CUDA model. Search
orchestration, request packing and mutable return-grid fingerprinting still run
on the host. It does not make the complete mission pipeline GPU controlled.

## Freshness and compatibility

Read-only NumPy views of writable memory are deliberately excluded from reuse:
another alias could change their values. A cached catalogue must have immutable
`bytes` backing every relevant array. The cache key includes backing-object
identity, data address, dtype, shape and strides; strong references prevent
address or identity recycling. Switching catalogue snapshots clears all cached
prefixes. Pricing policies have a four-entry least-recently-used limit.

Mutable custom catalogues remain supported and are hashed on every batch, as
before. Return grids, masks and time-of-flight arrays are also checked every
time. The full model signatures retain the same hash-stream order and values;
this is neither a weaker checksum nor an identity-only cache for writable data.
Geometry, pricing, dynamics, FP64 arithmetic and physics tolerances are unchanged.

`load_catalogue()` now returns shared immutable arrays. Code that intentionally
perturbs elements should make array copies and construct a replacement catalogue.
`parse_catalogue_text()` and custom constructors keep their existing mutable
array behavior; `immutable_copy()` explicitly owns an immutable snapshot.

## Measurement and validation

The final comparison is v820. Each GPU runs both modes once as warmup, then
three measured repeats per mode in alternating order. Both modes use the same
immutable catalogue, retained collection-DP buffers, native core and route
inputs. Setting `SPACEPDHCG_TEST_GTOC12_CACHE_CATALOGUE_DIGEST=0` disables only
the catalogue-prefix reuse. Timings measure complete `RouteSearch.run`, excluding
process startup and initial catalogue loading. The two routes produce 517 and
472 surrogate candidates respectively; these are not certified solutions.

| GPU | Original ship | Fresh median | Cached median | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 10 | 21.817 s | 21.030 s | +3.74% |
| RTX 5090 | 21 | 19.792 s | 19.614 s | +0.91% |
| Lambda H100 | 10 | 19.439 s | 18.086 s | +7.48% |
| Lambda H100 | 21 | 18.653 s | 17.379 s | +7.33% |

Every candidate file is byte-identical across modes/repeats on each GPU and
matches the preceding saved benchmark output. The final two-route H100 medians
correspond to about 27.9 surrogate candidates per second before refinement.
Packing time falls from 1.596 to 0.295 seconds for ship 10 and from 1.681 to
0.376 seconds for ship 21 on H100. The local throughput effect is smaller;
three repeats do not establish a statistical confidence bound for the 0.91%
observation. These measurements do not establish a universal solver speedup.

The initial single-prefix experiment, v816, is preserved separately. Alternating
table/non-table policies still caused 356 hashes across the two routes. The
final bounded policy cache removes that repeated work. Its source changes are
limited to the host fingerprint cache and its regression test; the CUDA core
is byte-identical to the integrated v808 build.

These runs freeze the numerical tree at `5f698731` plus the four catalogue/cache
source and test files. The later route-ephemeris and fleet-admission changes in
`106e0cdc` are separate work and are not included in these timing measurements.

The final runtime passes 161 tests on each GPU. Thirty-two tests pass under each
of memcheck, racecheck and synccheck, and the native endpoint-controller tests
also pass separately under all three tools. Tests cover immutable ownership,
mutable aliases, replaced arrays, policy switching and eviction, live return
grids, native metadata/forward-gate agreement, collection reuse, exact seeded
trajectories and controller acceptance. The early runner failures are retained:
an incorrect executable directory and an omitted `.gitignore` test fixture.
Neither failure started a trajectory-refinement run. The final test bundle
includes the exact missing fixture from the pinned base revision.

The integrated endpoint/collection core replays 14 prescribed route proposals
on both GPUs, making 147 native QOCO/SCvx leg attempts. Five complete routes
certify; this replay produces no meaningful new fleet score. The latest
published 13,023.704901 kg frontier becomes the warm incumbent for CUDA selection,
so the newer ship-8 and ship-19 gains are preserved. The resulting fleet passes
both full-fleet physics checkers locally and on Lambda at **13,023.704901 weighted
kg / 14,291.006160 raw kg**, with 23 ships and 199 asteroids. Differences of order
1e-10 kg are roundoff, not score improvements.

The frontier's two newly added routes retain their local GPU-refinement
provenance. This work does not claim those two routes were reoptimized on H100.
The downloaded H100 result does contain fresh complete-fleet checking and dense
verifier replay, including the coast intervals absent from the earlier display.

[Exact runtime, paired measurements and downloaded result](../results/lambda/2026-09-09/gpu-catalogue-cache-v822/README.md).

The next useful work is broader route construction and removal of measured host
search control. The tested finite route pool is exhausted; repeating its solve
cannot establish global mission optimality or produce new trajectory choices.
