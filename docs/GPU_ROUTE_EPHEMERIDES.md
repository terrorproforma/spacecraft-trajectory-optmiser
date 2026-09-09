# CUDA route-boundary orbital states

The native route pipeline now calculates its Earth and asteroid boundary states
in a CUDA batch. `ScvxSettings.ephemeris_backend="auto"` selects CUDA when the
outer SCvx loop is CUDA; the existing CPU route path keeps CPU states. An explicit
`"cpu"` setting is available for comparisons. A CUDA error is reported and does
not silently switch the route to CPU.

The implementation reuses the existing double-precision Kepler state calculation
in `orbitweaver_gpu.cu`. It copies a route's immutable orbital elements once,
deduplicates exact body/epoch requests, and caches the returned states across the
route's mass passes. Event schedules, cargo and physics tolerances are unchanged.
Both complete-fleet checkers remain independent.

This removes CPU orbital-state arithmetic from route-boundary preparation. The
Python bridge still downloads boundary vectors for `LegBoundary`, and the
existing native solve interface uploads them. It therefore does not establish
a fully device-controlled mission pipeline. The C interface also provides a
caller-stream device launch with no allocation, transfer or synchronization;
connecting that interface to a retained native route workspace is subsequent
integration work. Event/export callbacks are counted separately.

## Local verification and complete preparation timing

The fresh CUDA 12.8 / SM120 build uses source tree
`737137c4e7e85c762d413e094d69b9df423c409c6ab7581b82ce1f6eee754063`
assembled from `f9e05c2` plus the ten recorded route-ephemeris files. Unrelated
working-tree changes are excluded. No old compiled objects are reused.

- 56 CPU tests pass, followed by the actual CUDA/CPU state-equation test.
- Native API and CUDA Graph execution tests pass, including invalid rows,
  ownership, capacity and output-preservation cases.
- Compute Sanitizer reports zero memory errors on the native test.
- Four actual archived route prescriptions pass five alternating CPU/CUDA
  comparisons each: 410 CUDA states and 720 uncached CPU calls in total.
- The largest state difference is **1.765 mm** in position and
  **7.3183e-14 km/s** in velocity. The unchanged thresholds are 1e-3 km and
  1e-9 km/s. Every archived launch-boundary vector also passes its existing,
  stricter `64 * epsilon * max(1, max_abs(vector))` representation check.

The comparison includes deduplication, Python/native setup, element upload,
batch execution, state download and workspace destruction. Both methods include
their small measurement counters. Catalogue parsing and module imports are
shared setup and recorded separately. These are five observations per route,
not a broad throughput benchmark.

| Archived route | CPU median | CUDA median | CPU / CUDA |
| --- | ---: | ---: | ---: |
| Ship 8, rank 0 | 1.390 ms | 1.682 ms | 0.826 |
| Ship 8, rank 1 | 1.575 ms | 1.727 ms | 0.912 |
| Ship 19, rank 6 | 1.662 ms | 1.739 ms | 0.956 |
| Ship 19, rank 7 | 1.422 ms | 1.658 ms | 0.858 |

**There is no complete-method speedup at these small batch sizes.** Fresh
workspace creation and destruction dominate the small CUDA calculation. The
first CUDA sample also costs 221.074 ms with context/library/kernel setup; it
is retained in the raw observations. Later samples reuse the process context
but still create and destroy an ephemeris workspace for every route. This result
supports removing that lifecycle and host bridge in the native route integration,
not claiming a faster complete solver from a small kernel measurement.

The local build and GPU-test records are under
`build/performance/gpu-route-ephemeris-v630`; the frozen prescriptions, every
state readback and complete comparison are under
`build/performance/route-ephemeris-audit-v630/execution-a`. The comparison report
SHA256 is
`e4fecfe0a333a8ee52a313657a032fe6860d606bf41977eb05d03bc7f9d62d28`.

The [portable evidence package](../results/local/2026-09-09/gpu-route-ephemeris-v630/README.md)
retains the source/build records, local tests, complete state outputs and
unexecuted H100 build/test delivery. Lambda is reachable, but automatic approval
review rejected the source upload pending human confirmation of the exact
destination. There is no H100 build or new GPU measurement for this change yet.

These orbital-state tests perform no trajectory optimization and cannot change
the fleet score. Fixed-cargo refinements and both complete-fleet checks are
separate acceptance gates.
