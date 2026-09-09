# CUDA beam admission

The deploy beam can now apply reserve, Earth-return feasibility and diversity
limits before downloading candidates. This extends the retained CUDA expansion
workspace. It borrows immutable Earth-return option tables from their owning
thread/device, uploads only table descriptors and evaluates reserve and return
eligibility across multiple GPU blocks. Return rows stay on the device.

Admission must preserve the ranked greedy rule: a rejected child cannot consume
a diversity slot. A cooperative 256-thread kernel processes ranked tiles,
checks conflicts across lanes and elects the earliest eligible child. Selected
rows consume the same deployed-set and first-asteroid limits as the Python
reference. Set comparison uses exact asteroid multisets, with no hash collisions.
Only accepted rows are downloaded and materialized for subsequent collection
tour scoring. The ordered admission stage remains a single cooperative block;
its cost and scalability must be measured separately from parallel eligibility.

The operator preserves the original FP64 operation order for reserve and cargo
estimates and applies the same Earth-return authority threshold. It does not
relax trajectory dynamics or verification tolerances. Surrogate admission is
not independent trajectory certification.

`SPACEPDHCG_TEST_GTOC12_GPU_ADMISSION=0` retains host admission for matched
comparisons. Custom admission callbacks and nonresident return-option tables
retain their established host behavior. The ordinary CPU backend is unchanged.
Collection-tour control, host screening dispatch and first-level admission
remain work toward complete GPU execution.

## Paired measurements

Both GPUs pass 190 tests. Fifty-two tests per GPU pass each of CUDA memcheck,
racecheck and synccheck, with separate native controller checks. Fourteen GPU
tests per device also pass full leak checking with zero leaks or errors.

The two fixed route searches produce byte-identical candidate files across all
eight runs per GPU and match the preceding CUDA-expansion checkpoint. There are
517 and 472 surrogate candidates, respectively. Both comparison modes use the
same frozen build; only the admission environment flag changes. Each mode has
one warm-up and three alternating measured samples. Whole-search timing excludes
process startup and catalogue loading.

| GPU | Route | Host admission median | CUDA admission median | Throughput change |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 12.793 s | 13.074 s | -2.15% |
| RTX 5090 | Ship 21 | 11.221 s | 11.042 s | +1.62% |
| H100 | Ship 10 | 6.998 s | 6.958 s | +0.59% |
| H100 | Ship 21 | 6.317 s | 6.083 s | +3.85% |

The RTX sample ranges overlap; this checkpoint does not establish a general
whole-search speedup there. H100's first-route change is also small. The clear
architectural gain is removing host admission work: Python materialization
falls from 105,755 to 1,092 children across both routes, and ranked-result
downloads fall from 9,408,696 to 96,096 bytes, about 99% less. CUDA processes
the same 1,251,916 valid children. The individual admitted totals are 576 and
516. This is search throughput, not certified solutions per second.

The frozen source base is `a8cd81e520a51390ea8f818d421fc56dcdd669d3`, with eight
owned source/test overlays. v839 contains the build and safety checks, v840 the
paired searches, v842 leak checks and v843 the subsequent profile. The immutable
v841 evidence includes exact source and binaries, every paired candidate file,
raw profiles, reports and a portable byte/content audit.

[Downloaded evidence and copy-paste audit instructions](../results/lambda/2026-09-09/gpu-beam-admission-v841/README.md).

## Where the remaining time goes

The new RTX ship-10 profile records 7.77 million calls. Route completion takes
12.93 cumulative profiled seconds out of a 14.80-second profiled search. Its
collection-DP and forward-scheduling paths account for most remaining work:
1,152 collection-tour preparations, 1,731 DP solves, 6,443 return-option builds
and 8,552 collection selections. These are overlapping cumulative profiling
measurements, not additional benchmark time or separate speedup claims.
Batching those operations and retaining their device state across route
completions is the next substantial step toward GPU-controlled search.

This admission change adds no newly certified trajectory and promotes no fleet
score. The separately verified 24-ship fleet reaches 13,526.961241 weighted kg /
14,915.044490 raw kg; it was advanced by recovering an existing route, outside
this benchmark. [Fleet result and provenance](GTOC12_FLEET_ADDITION.md).
