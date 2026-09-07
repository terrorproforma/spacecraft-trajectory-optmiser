# GPU forward mass accounting — 8 September 2026

Retiming now computes selected-path mining yield, departure masses, thrust
authority, propellant consumption and the final dry-mass/payload check on CUDA.
The arithmetic runs inside the existing schedule-completion kernel and captured
graph. It adds no kernel launch or result-download synchronization. The retained
packed output now includes the forward result and per-leg mass/inflation arrays.

The host still constructs Python route objects, corrects mass profiles, chooses
propellant prices and compares weighted/orphan objectives. This is a prerequisite
for a device-controlled pricing loop, not completion of the GPU-native application.
The original CPU forward implementation remains the independent arithmetic
reference and supports custom retimer implementations. Standard CUDA retiming
requires the new native entry point; an old library raises an explicit rebuild
error. `gpu.retime_cuda_forward = False` selects CPU bookkeeping for diagnostics.

## Matched measurements

The same native binary is used for both modes. Only forward accounting is toggled.
Each scope alternates 12 CPU/GPU pairs, discards the first pair and reports the
remaining median. Fresh runs build transfer tables; cached runs retain them and
execute the complete six-price retiming loop. Every pair selects the same epochs
and objective; GPU mode must report completed forward calls.

| Scope | GPU | CPU forward | CUDA forward | Time reduction |
|---|---|---:|---:|---:|
| Retained-table retiming | H100 | 3.47394 ms | 2.64513 ms | **23.9%** |
| Retained-table retiming | RTX 5090 | 3.02032 ms | 2.77037 ms | **8.3%** |
| Fresh-table retiming | H100 | 60.62491 ms | 59.72756 ms | 1.5% |
| Fresh-table retiming | RTX 5090 | 218.95969 ms | 219.48529 ms | -0.2% |

Fresh-table timing has no material overall improvement. These warm-runtime
measurements are not a fleet-search or mission-certification throughput claim.
A separate paired comparison against the published warp-selection library is
archived under `measurement` / `release-comparison`; it explicitly disables the
new feature for the old library. Its cached metric measures ordinary DP without
forward accounting and must not be confused with the complete cached loop above.

## Accuracy and failure handling

- **111 tests pass on each GPU.** Forward parity covers different initial/profile
  masses and prices, cooperative collections, measured return overrides, matching
  mass-round decisions, missing deployers, short stays and cache invalidation.
- H100 memcheck, initcheck and synccheck each pass **48 cases with zero errors**.
- Schedule epochs, payloads and selected delta-V values match exactly in the
  parity fixtures. CUDA and NumPy exponential evaluation can differ in accumulated
  mass by a few 1e-13 kg. Comparisons of propellant/final mass use an absolute
  1e-10 kg tolerance; the resident-table test formerly used exact dictionary
  equality for these two scalars. All other fields keep their prior exact checks.
  No trajectory acceptance, dynamics, thrust or objective gate was relaxed.
- Forward failure ordering and the refused leg's mass prefix follow the CPU
  implementation, so the host mass-correction loop receives the same information.
  Policy, visit and epoch changes invalidate the retained forward result.

The H100 mission replay uses all six native forward evaluations, 412,116 Lambert
branches and one resident table build. Its 13-leg low-thrust refinement finishes
in **8.909 s**; this single replay is validation, not evidence of a refinement
speedup. Independent propagation accepts **526.4887063654 kg**, and the official
checker accepts **526.489 kg**, with six mined asteroids and no violations.
Maximum independent discrepancies are 0.474406 km position, 8.085e-8 km/s velocity
and 7.959e-11 kg mass under unchanged gates.

The incumbent fleet remains **12,805.194 weighted kg**. The existing visualiser
retains its v235 mission and provenance; this newer numerical replay is archived
as v252 and does not replace or improve the incumbent fleet.

## Evidence and reproduction

Code commit: `ac95a2cd`, based on `e57d6b7d`.
[Downloaded evidence](../results/lambda/2026-09-08/gpu-forward-retime-v251/)
contains both benchmarks, native build logs, source/runtime hashes, tests,
sanitizers and the complete independently checked mission.

The native library was built in H100 run v249. Final run v251 uses the exact same
binary and verifies unchanged C++/CUDA hashes, adding the Python stale-library
guard and its test. Both source manifests are retained. Local evidence records
the frozen binary path and SHA-256. Mission v252 records both CUDA and QOCO hashes.

With `PYTHONPATH=src`, the pinned catalogue and rebuilt CUDA library configured:

```sh
python results/lambda/2026-09-08/gpu-forward-retime-v251/benchmark_forward_v248.py \
  build/performance/forward-comparison \
  results/lambda/2026-09-08/gpu-forward-retime-v251/input-refinements.json \
  results/lambda/2026-09-08/gpu-forward-retime-v251/input-return.json
```

The archived launchers record the exact build, validation and replay commands.
Their absolute paths describe the measured environments and need adapting on a
different machine. The next architectural step is to retain the mass/price state,
failure decisions and best weighted candidate on the GPU between evaluations.
