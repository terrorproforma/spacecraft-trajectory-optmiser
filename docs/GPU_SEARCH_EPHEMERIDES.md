# GPU ephemerides for return and collection search

This records the ephemeris migration measurements. Subsequent
[compact CUDA options](GPU_COMPACT_SEARCH_OPTIONS.md) also migrate cost summation,
filtering and return ordering; the host-option discussion below describes the
implementation measured in this report.

Return-window and collection-hop candidate generation now sends orbital elements
and paired departure epochs / flight durations to the existing C++/CUDA screening
bridge. CUDA propagates both bodies, builds Cartesian Lambert requests, solves
the transfers and applies departure/arrival velocity allowances. These paths no
longer calculate body positions and velocities on the CPU before screening.

This extends the device element-based path already used for retiming to
`RouteSearch._return_options` and `RouteSearch._collect_hop_options`. The native
library is unchanged. Request data is 16 bytes per transfer plus a 136-byte
orbital description per batch, replacing the prior 160-byte Cartesian requests.
Full screening results still return to the host. Option filtering, ordering,
beam orchestration and independent mission checks remain CPU work.

The path is enabled inside the CUDA screening scope by default.
`SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES=0` selects the previous CPU-ephemeris
path for controlled comparisons. CPU screening retains its existing behavior.

## Measurements

Nested direct wall timers identified CPU ephemeris and option preparation in
return and collection search. An additional trace found 1,122 collection-table
requests with 1,122 distinct immutable inputs, so another table cache would not
help that one-ship fixture. Both profiles and the trace are archived.

| GPU | Previous complete CLI median | GPU ephemeris median |
|---|---:|---:|
| RTX 5090 | 32.406 s | 31.929 s |
| H100 | 39.962 s | 38.696 s |

Each comparison uses baseline/candidate/candidate/baseline process order, with
two runs per mode. Timing ranges overlap, so these measurements do **not**
establish a significant overall speedup. The demonstrated improvement is GPU
residency: another **1,132,379 transfer requests** use device-built ephemerides
in each one-ship candidate run. All runs preserve 45,188,558 logical branch
requests, 2,782,091 collection options and **548.254620 weighted kg**, with both
final mission checkers passing. The `completed_element_hops` counter increases
as expected; all other screening counters match. Logical requests include cached
tables elsewhere in the pipeline and are not counts of unique solved missions.

A wider local run completed in **230.390 seconds**, retaining four ships,
29 mined asteroids and **2,088.668592 weighted kg** with both checkers passing.
It preserves 169,753,864 logical branches and 18,250,121 collection options;
another **7,726,204 transfer requests** use device-built ephemerides. This is a
single wider confirmation, not a paired timing experiment.

The final local profile still finds substantial CPU work in option preparation
and the initial beam, while SCvx remains the largest measured component. In
particular, option list construction repeatedly accesses `total_delta_v`, whose
property currently allocates a complete vector on each access. Removing that
repeated work and retaining compact options on the device are subsequent targets.

## Accuracy and evidence

Twenty-three selected tests pass locally and on H100. Tests compare costs,
feasibility and option ordering with CPU-generated ephemerides, reject any CPU
ephemeris call in the migrated search paths, and exercise workspace reuse,
partial batches, empty/invalid inputs, Earth allowances and vector validation.
Final tests include fractional epochs and durations, circular, inclined,
high-eccentricity and retrograde orbits. Independent Kepler propagation closes
the tested transfers within **0.01 km** in position and **1e-8 km/s** in velocity.
Memory and synchronization checking pass on both GPUs. Separate final default-on
campaigns pass both mission verifiers.

No thrust, integration, conic qualification or final mission acceptance tolerance
was relaxed. The best retained fleet remains **12,805.194 weighted kg**. This is
another part of GPU migration, not completion of the whole application.

- [Local profiles, paired runs, wider fleet and default validation](../results/lambda/2026-09-08/gpu-paired-ephemerides-local-v419/summary.json)
- [H100 paired runs and tests](../results/lambda/2026-09-08/gpu-paired-ephemerides-v417/summary.json)
- [H100 final default configuration](../results/lambda/2026-09-08/gpu-paired-ephemerides-v420/summary.json)

Archives include exact Python overlays, commands, runtime hashes, raw reports,
profiles and SHA-256 manifests. Local runs use the unchanged native v411 build;
H100 uses the unchanged native v412 build. H100 Python source is frozen v412
plus each recorded overlay. The local baseline profile uses commit `2bea8738`;
later local runs add their recorded overlays. All recipes run from the repository
root with their recorded data/runtime paths restored.
