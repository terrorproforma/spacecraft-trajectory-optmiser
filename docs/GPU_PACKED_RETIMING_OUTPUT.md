# Single-download CUDA retiming output — 8 September 2026

The native retimer now writes its status, objective, arrival/departure indices,
selected delta-v values and sweep data into one retained device allocation. It
downloads that allocation once, then copies the fields into the caller's buffers
after stream completion. The 13-leg fixture returns only 349 bytes.

Previously these fields occupied six separate device allocations and required
six device-to-host calls per evaluation. The new internal layout preserves the
existing C APIs, including optional path fields. Double arrays remain aligned;
all bytes are initialized even when no feasible schedule exists. Caller arrays
are independent of workspace storage and are not modified on failed validation.

## Hill-climbing result

The first implementation used a retained pinned host buffer. It improved cached
pricing, but the fresh-workspace benchmark exposed an allocation cost. A direct
pinned/pageable comparison found practically identical cached performance while
the ordinary host buffer avoided about 0.7 ms in the H100 fresh-workspace sample.
The selected implementation therefore uses a retained ordinary host byte buffer.
Both variants and their test results are archived.

The final comparison alternates the published graph runtime (`17c5b77d`) and the
new runtime, using the same current Python wrapper and exact input mission. Full
retiming uses eight pairs, with the first pair omitted from the median. Cached
pricing uses 42 alternating pairs over six prices, omitting the first pair.
Every pair produces exactly the same schedule and objective.

| Workload | GPU | Published runtime | Single download | Time reduction |
|---|---|---:|---:|---:|
| Cached price evaluation | H100 | 0.75849 ms | 0.72476 ms | 4.4% |
| Cached price evaluation | RTX 5090 | 1.24933 ms | 1.06227 ms | 15.0% |
| Complete return-sweep retiming | H100 | 63.714 ms | 63.447 ms | 0.4% |
| Complete return-sweep retiming | RTX 5090 | 225.158 ms | 223.880 ms | 0.6% |

The full workload still evaluates 412,116 Lambert branches, builds one resident
table set and runs six price evaluations. Differences below 1% do not establish
a material complete-retiming speedup. The verified improvement is in repeated
cached pricing, with fewer result-transfer calls and allocations.

## CUDA trace changes the next priority

A separate Nsight Systems trace captures 100 warm, unswept, cached H100 price
evaluations, after table construction and graph warm-up. It records 200 input
copies, **100 output copies**, and 100 graph launches. Kernel totals are:

| Kernel | Calls | Mean duration | Share of kernel time |
|---|---:|---:|---:|
| Leg selection | 1,300 | 36.740 µs | **83.7%** |
| Camp selection | 1,300 | 4.280 µs | 9.8% |
| Final selection/backtracking | 100 | 37.259 µs | 6.5% |

All device-to-host transfers together take only 146.7 µs across those 100 calls.
`cudaMemcpyAsync` consumes 59.46 ms of host API time, but this includes waiting
for prior GPU work. It is not evidence of 59 ms of physical data transfer.
Kernel durations sum to 57.05 ms. This makes the per-arrival serial TOF loop in
the leg kernel the next primary target, with final selection another reduction
opportunity. A GPU reduction must preserve the earliest-TOF tie winner exactly.

The local trace captures API calls but lacks device-kernel/memory activities;
the archive retains the empty-report notices. It cannot establish local kernel
percentages. Trace timings are diagnostic observations, not replacements for
the uninstrumented paired benchmark.

## Validation

- **93 tests pass locally and on H100.** Two new cases cover odd/even path
  layouts, graph and ordinary execution, every legacy output API, buffer guards,
  detached outputs and untouched caller arrays on invalid input.
- The existing CPU parity, changed-policy, return-refusal, infeasibility and
  workspace replacement cases also pass with the consolidated output.
- H100 memcheck and initcheck each pass **30 cases with zero errors**.
- A fresh 13-leg H100 mission replay takes **7.713 s** and passes both checkers:
  official **526.489 kg**, independent **526.4887063655 kg**, six mined asteroids,
  no violations. Maximum replay discrepancies are 0.448668 km position,
  7.662e-8 km/s velocity and 7.24e-11 kg mass. This single refinement observation
  does not establish a refinement speedup. Physics and objective gates are unchanged.

The fleet remains **12,805.194 weighted kg**. The existing visualiser continues
to show the earlier v235 replay, retaining its own samples and compute metadata;
the v244 repeat is archived separately.

## Evidence and reproduction

Code: `41293ed5`, based on `17c5b77d`.
[Downloaded evidence](../results/lambda/2026-09-08/gpu-packed-retime-v243/) includes
the rejected pinned variant, final measurements, source/runtime hashes, test
logs, native Nsight reports and the full independently verified solution.

The benchmark takes an output directory, original refinements, measured return,
the previous Python wrapper, and previous/current CUDA library paths:

```sh
python results/lambda/2026-09-08/gpu-packed-retime-v243/benchmark_packed_retime_v242.py \
  build/performance/packed-replay \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-return.json \
  src/spacepdhcg/gtoc12/gpu_retime.py OLD_LIBRARY NEW_LIBRARY
```

Set `PYTHONPATH=src` and the pinned `SPACEPDHCG_GTOC12_DATA`. The wrapper is
unchanged by this tranche. `old` and `candidate` identify the published/final
runtimes in final reports; the older pinned experiment's helper calls its
candidate `graph`, a historical label, although both sides enable graph replay.

The complete application still needs device-side forward mass bookkeeping and
price/mass control, along with fleet search improvements. Those requirements
remain open; this change removes a measured overhead without changing the solver.
