# Compact completion metadata: measured v622 result

The compact path improved this completion component at larger batch sizes, but
did not win at every practical size. With the unchanged fitted model, a warm
48-request batch took **2.196 ms versus 2.689 ms** for ordinary metadata
(1.225× throughput, 18.3% less time). At 24 requests it was slightly slower.
These measurements support keeping the mode opt-in rather than changing the
default for every workload.

Both arms called the same frozen production `GpuCompletion.run` through the
compact-g standalone component library on the local RTX 5090. The method timer
includes Python packing, source/model hashing and validation, native execution
and transfers, and plan reconstruction. Both used the official catalogue; no
saved geometry was injected. Capture and evidence checks were outside the timer.

| Model | Batch | First ordinary ms | First compact ms | Warm ordinary ms | Warm compact ms | Warm ordinary / compact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 24 | 10.474 | 4.457 | 1.225 | 1.860 | 0.658× |
| Flat | 48 | 1.845 | 3.636 | 1.973 | 2.203 | 0.896× |
| Flat | 256 | 6.635 | 7.412 | 7.081 | 5.822 | 1.216× |
| Flat | 1024 | 27.563 | 19.585 | 26.693 | 18.776 | 1.422× |
| Existing fit | 24 | 2.500 | 4.040 | 1.773 | 1.831 | 0.968× |
| Existing fit | 48 | 3.620 | 3.824 | 2.689 | 2.196 | 1.225× |
| Existing fit | 256 | 11.468 | 7.372 | 10.839 | 5.775 | 1.877× |
| Existing fit | 1024 | 43.860 | 20.000 | 40.945 | 18.966 | 2.159× |

Each row has exactly one first call and four warm calls per arm; warm entries
are medians, with all individual samples and their range retained in
`summary.json`. A ratio above one favors compact. First order alternated between
groups, followed by ABBA/BAAB warm order. These are eight fresh-workspace groups
in one process, not independent cold processes. The first ordinary workspace
creation took **170.274 ms** and was excluded and recorded separately. The first
ordinary method also includes lazy Python imports; its apparent first-call
advantage for compact must not be generalized. Catalogue loading took 38.441 ms.

Packing remains the largest measured cost. In the fitted 1024-request case,
median packing fell from **37.574 to 16.024 ms**. Compact packing/source checks
still consumed **84.6%** of the matching warm method samples, while the median
CUDA event interval was **95.776 microseconds**. Compact throughput there was
**53,991 repeated proxy evaluations/s** including the full method. At fitted
48-request size it was 21,861/s, with 70.5% spent in packing/source checks.
Source-array hashing ran on every call; the native model was configured once
per compact workspace. Ordinary fit geometry cached 42 pairs; compact had no
host pair-geometry cache. The flat model did not need pair geometry.

The finite run completed with **80 evaluation calls**, **13,520 repeated
candidates per arm**, and **20 unique historical controls** (five archived
routes × two prefix-mass choices × two fixed models). It returned 1,370
proxy-accepted evaluations per arm; these are repeated controls, not newly found
missions. The frozen source implies 160 kernel launches: 40 ordinary finishing
kernels plus 40 compact calls containing three kernels each. This count is not
a profiler trace. The worker took 7.536 seconds including construction,
validation, compression and evidence writing; dividing the case count by that
time would not measure the production method.

All 80 saved per-call comparisons passed: exact classifications, cargo, pickup
order and request preservation; the reported maximum absolute finite difference
against the fixed CPU oracles was **4.5475e-13**, over 3,585,000 field comparisons.
The five repetitions in each arm had identical numerical payloads. Cross-arm
flat payloads matched byte-for-byte; fit payloads differed within the prescribed
FP64 bounds while exact gates and cargo still matched. All 80 raw NPZs, call
status files and comparison reports are retained. This summary checks recorded
results and timing arithmetic; the independent raw/formula review is maintained
separately in `../completion-pack-review-v622`.

This is a historical proxy-completion benchmark using saved delta-v values, not
end-to-end trajectory optimization or current-fleet performance. No Lambert
solves, SCvx refinements, refitting, new route certification or score change
occurred. The component library scope does not itself establish new full-core
integration coverage. These ratios must not be multiplied by the earlier v621
CPU/GPU ratios from a different benchmark setup.

The remaining measured opportunity is reducing Python request packing and
source/model validation work while retaining correct invalidation, then supplying
these batches directly from native search. This run does not establish that
removing checks, enlarging batches or changing the default improves a complete
mission search.

`analyse.py` uses saved files only. It rechecks all 406 preparation hashes,
terminal status, exact work/order counts, saved comparison status, medians,
paired packing shares and every raw-output file hash. `summary.json` contains
the complete numerical table; `raw-output-sha256.json` indexes all 255 output
files. The original frozen preparation remains unchanged.

Key identities:

- Preparation: `64d36a429381edeae90f3300520516f5310337f8d154ed292d5c89642143ba9d`.
- Component library: `9c3e6a250002893363d2f4dcf68babe3b7158508b5b97149f83c8261437f1fa7`.
- Terminal report: `ccd74d20bec17954b72e731bc05341ce93168e80c4876cefd336d5ee8591483a`.
- Terminal launch: `97a29a527989f7ea30a5cb4fd690722cf8b63cf76e57801c31fd3b4a047606c3`.
