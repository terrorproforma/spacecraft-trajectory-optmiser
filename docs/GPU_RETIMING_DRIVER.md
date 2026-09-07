# Device-controlled retiming — 8 September 2026

A conditional CUDA graph now owns the full price/mass search for a fixed visit
order. Each body execution runs the existing multi-block scheduling kernels and
forward accounting, then a device controller corrects the mass profile, updates
the price bracket, performs bisection and retains the best weighted candidate.
There is one graph submission and one packed result download for the complete
search, with no CPU decision or synchronization between its evaluations.

The loop preserves the reference driver's behavior: mass corrections carry into
the next price; failed authority checks include the refused leg's mass; two
bisections follow bracketing; and strictly greater weighted haul plus orphan
credit replaces the incumbent candidate. Equal objectives retain the first
candidate. The payload sum and orphan-credit sum preserve their separate CPU
iteration orders. Nonfinite price/profile updates are rejected before another
evaluation, matching the native validation that the host driver previously ran.

CUDA builds now require toolkit **12.4 or later** for conditional graph nodes.
Measurements here use CUDA 12.8 on H100 and RTX 5090. NVIDIA documents the
[conditional WHILE node and capture API](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/).
The normal CUDA backend enables this driver; an older native library raises a
rebuild error. `gpu.retime_cuda_driver = False` selects the host driver for
diagnostics. Disabling forward accounting/graphs or using a custom driver method
also retains the reference path.

Initial route-mass construction, visit/stage packing, Python route objects,
search orchestration and fleet management still involve the CPU. This change
completes the device price/mass loop, not the entire GPU-native application.

## Matched performance

The same binary and inputs are used for both modes. Twelve alternating pairs
are measured per scope, discarding the first pair and reporting medians. The
host reference already uses GPU forward accounting. Both modes retain exactly
the same selected schedule and objective; device mode must record a driver call.

| Scope | GPU | Host driver | Device driver | Time reduction |
|---|---|---:|---:|---:|
| Complete retained-table retiming | H100 | 2.67144 ms | **1.60237 ms** | **40.0%** |
| Complete retained-table retiming | RTX 5090 | 2.77837 ms | **1.88028 ms** | **32.3%** |
| Fresh-table retiming | H100 | 59.78141 ms | 58.63832 ms | 1.9% |
| Fresh-table retiming | RTX 5090 | 215.36192 ms | 215.36785 ms | approximately zero |

The cached fixture performs six prices over existing transfer tables. Fresh
runs build 412,116 Lambert branches first, which still dominates their timing.
These are warm-runtime measurements of one fixture, not complete fleet-search
or mission-certification throughput claims.

## H100 trace

Separate Nsight captures cover 20 warm complete retimings per mode:

| Captured operation | Host driver | Device driver |
|---|---:|---:|
| Scheduling / forward evaluations | 120 | 120 |
| Graph submissions | 120 | **20** |
| Stream synchronizations | 120 | **20** |
| Result downloads | 120 | **20** |
| Host-to-device copies | 480 | **120** |

The device trace shows 120 controller executions and 20 initial/final driver
kernels, confirming that the conditional body repeats on the GPU. Instrumented
timings are not the benchmark medians above. The new controller adds work while
removing CPU round trips: camp, path completion and leg kernels account for
30.0%, 29.8% and 28.7% of device kernel time; the controller adds 10.5%.
Its best-output copy and the serial final-path scan are concrete next targets.
The long host `cudaMemcpyAsync` duration includes waiting for preceding GPU work;
it must not be interpreted as pure PCIe transfer time.

## Accuracy and mission replay

- **131 tests pass on each GPU**, including 20 driver cases. They compare prices,
  round limits, final corrected profiles, schedules, weighted objectives,
  cooperative/orphan handling, changed weights/pins, first-candidate ties, stale
  libraries and overflowing prices against the reference driver.
- H100 memcheck, initcheck and synccheck each pass **68 cases with zero errors**.
- Existing physics, objective and acceptance gates remain unchanged. The prior
  1e-10 kg tolerance for CPU/CUDA forward mass arithmetic remains in place;
  this port does not change it.
- A fresh 13-leg H100 refinement takes **9.258 s** and passes both verifiers.
  Independent replay returns **526.4887063658 kg**, six mined asteroids and no
  violations; the official checker returns **526.489 kg**. Maximum discrepancies
  are 0.472343 km position, 8.039e-8 km/s velocity and 9.527e-11 kg mass.
  This is a validation replay, not a refinement-speedup measurement.

The mission records one device driver call containing six scheduling/forward
evaluations and one resident table build. The incumbent fleet remains
**12,805.194 weighted kg**. The visualiser retains its earlier v235 dataset and
provenance; the new v257 numerical replay is downloaded separately.

## Evidence and reproduction

Code: `b2fdc059`, based on `06b4cf84`.
[Downloaded evidence](../results/lambda/2026-09-08/gpu-driver-retime-v256/)
contains build logs, source/runtime hashes, local/H100 tests, sanitizer logs,
paired benchmarks, both raw Nsight captures/CSV exports and the complete mission.

With the rebuilt CUDA library, pinned catalogue and `PYTHONPATH=src` configured:

```sh
python results/lambda/2026-09-08/gpu-driver-retime-v256/benchmark_driver_v253.py \
  build/performance/driver-comparison \
  results/lambda/2026-09-08/gpu-driver-retime-v256/input-refinements.json \
  results/lambda/2026-09-08/gpu-driver-retime-v256/input-return.json
```

Archived launchers contain exact validation, tracing and mission commands;
absolute paths describe the measured environments and need adapting elsewhere.
