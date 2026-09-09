# Shared resident catalogues for CUDA completion

Pricing changes can now reuse the asteroid catalogue already on the GPU. The
additive `spacepdhcg_gtoc12_completion_model_with_catalogue` C API creates an
independent immutable pricing model whose orbit allocation is shared with its
source. It skips repeated catalogue validation, host packing, allocation and
upload. Atomic reference counting releases the device catalogue when its last
model is destroyed. Either model can be destroyed first.

The original create/destroy/evaluate API is unchanged. New pricing policies and
return grids still receive the same structural validation and their own copied
buffers. Failed creation leaves the source usable. Request partitions, original
forward gates, return-row overflow checks, FP64 arithmetic and physics tolerances
are unchanged. Evaluation still performs no allocation.

The Python caller permits sharing only for the immutable byte-backed catalogue
snapshot already validated by the current model. Strong references prevent
object/address recycling. A replacement catalogue triggers a fresh upload;
mutable custom catalogues continue through content checks and fresh construction
when their values change. Older native libraries without the additive symbol
retain the existing full-construction behavior. The counters now distinguish
model rebuilds, catalogue uploads, shared reuses and uploaded catalogue bytes.

The experimental compact-completion mode is still required. Setting
`SPACEPDHCG_TEST_GTOC12_RETAIN_COMPLETION_CATALOGUE=0` disables this optimization
for paired measurements. Catalogue hash-prefix caching and collection-DP reuse
remain enabled in both measured modes.

This is another reduction in host setup and data transfer. Pricing/grid buffers,
host request packing, search decisions and repeated dispatches remain. It does
not make the whole fleet search GPU controlled. No new certified fleet is
produced by this comparison; the last independently verified result remains
13,023.704901 weighted kg / 14,291.006160 raw kg, with 23 ships and 199 asteroids.

## Verification and reproducibility

The frozen base is `6c686cce48763a5ab34687a74d810f5ce693589b`, with only the
completion-model C++ implementation/header, Python owner and its tests overlaid.
Concurrent seed-generation work is outside this measurement. Both RTX 5090 and
Lambda H100 compile the same C++ sources. The final validation runtime is v826;
its native library is byte-identical to v823, with exact C++ source equality
checked before reuse.

The initial tests found a stale scalar-reference geometry cache in the new
catalogue-replacement fixture: both upload modes correctly changed geometry,
while the reference kept its old value. The fixture now clears that independent
cache. The lifetime test also explicitly exercises FIT5 orbital reads after
either model is destroyed. Original failing logs are retained as v823 evidence.

Final validation passes 165 tests per GPU, with 36 tests under each of CUDA
memcheck, racecheck and synccheck. Native endpoint-controller tests pass normally
and separately under all three tools. Coverage includes both ownership release
orders, failed replacement, changed immutable catalogue content, changed pricing
and grids, metadata agreement and unchanged forward acceptance/rejection gates.

The paired benchmark uses the same two saved route searches, one warmup per mode
and three measured repeats per mode in alternating order. Timings cover
`RouteSearch.run`, excluding process startup and initial catalogue loading.
These are surrogate route candidates, not independently certified trajectories.

| GPU | Original ship | Fresh catalogue median | Resident catalogue median | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 10 | 21.229 s | 20.737 s | +2.37% |
| RTX 5090 | 21 | 19.249 s | 18.810 s | +2.33% |
| Lambda H100 | 10 | 17.862 s | 17.846 s | +0.09% |
| Lambda H100 | 21 | 17.125 s | 16.938 s | +1.11% |

The small H100 effects should not be interpreted as an established general
solver speedup: three repeats provide no confidence bound, and the 0.09%
observation is effectively flat. The two-route retained medians correspond to
about 25.0 surrogate candidates/s locally and 28.4 on H100. These rates cover
517 and 472 completed surrogate candidates, respectively. Each route also
screens many more transfer branches; a branch evaluation is not a solution.

All eight runs on each GPU produce byte-identical candidate files and match
the previously saved search outputs. Catalogue uploads fall from 147 to 1 for
ship 10 and from 209 to 1 for ship 21. Across both routes, upload volume falls
from **854.4 MB to 4.8 MB**, a **99.44% reduction**. Model rebuilds remain 356:
pricing and grid buffers are still independent and reconstructed. These are
incremental measurements with the preceding hash-prefix optimization enabled,
not a comparison against the original uncached implementation.

Four focused ownership tests also pass `memcheck --leak-check full` on both
GPUs, reporting zero bytes leaked in zero allocations. This covers pricing
replacement, immutable catalogue replacement and both model release orders.

[Downloaded runtime, raw timings, audit and reproduction instructions](../results/lambda/2026-09-09/gpu-resident-catalogue-v827/README.md).
