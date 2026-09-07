# Batched independent GTOC12 propagation

The opt-in CUDA verification backend propagates all event-to-event trajectories
in a solution file as one GPU batch. It uses an independent FP64 DOP853
implementation, rather than the optimiser's RK4 discretisation. The existing
mission checker still enforces rendezvous, mass, thrust, mining, fleet-size and
score rules. Its default backend remains the CPU reference.

## Execution and accuracy

`cpp/cuda/src/gtoc12_verification.cu` assigns one warp to each independent leg,
with four warps per block. The state components cooperate through shared memory;
each warp advances its own adaptive integration without host decisions. The
device C API accepts resident buffers and a CUDA stream, supports graph capture,
and neither allocates nor synchronizes with the host. The retained host bridge
uploads batch descriptors and samples and downloads final records once per call.

The physical model uses km, km/s, seconds, kg and newtons. It includes central
solar gravity, cubic Lagrange thrust interpolation and mass flow from the norm
of the actual interpolated thrust. Burn interpolation follows the same stencil
selection and duplicate-epoch convention as the CPU verifier. Integration steps
stop at stencil knots; daily burn and five-day coast sample grids, including
endpoints, retain the CPU certificate's minimum-radius sampling rule. This is
not a proof of continuous minimum radius between those samples.

The fixed tolerances are `rtol=1e-12`, position `atol=1e-7 km`, velocity
`atol=1e-10 km/s`, and mass `atol=1e-9 kg`. No fast-math option is used. Invalid
descriptors, nonfinite/nonphysical states, step exhaustion and step underflow
produce explicit failure statuses and NaN final records; the Python bridge
raises rather than returning a qualified result or using a CPU fallback.

Tests compare circular and eccentric coasts, including a 15-year coast, against
analytic Kepler propagation. Constant thrust has an analytic mass check. Variable
thrust is compared with a substantially tighter CPU DOP853 integration limited
to steps of 1/16 day. This extra oracle matters: unrestricted CPU adaptive steps
can cross multiple interpolation stencils despite a small local error estimate.
The varying-thrust fixture agrees with the tightly stepped oracle within the
test's one-metre position, 1e-9 km/s velocity and 1e-7 kg mass bounds. Ordinary CPU
verifier comparisons are retained separately.

DOP853 coefficients come from SciPy 1.18.1. The generated header records the
source SHA-256, and `scripts/gpu/generate_dop853_coefficients.py` reproduces the
constants. SciPy's BSD license is retained in the header and
`third_party/licenses/scipy-dop853.txt`, which is included in native installation.
See [SciPy's DOP853 documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html)
for the algorithm and original Hairer implementation references.

## Measured fleet workload

The existing v11 fleet contains 413 legs, 19,261 burn arcs and 57,634 samples.
Both GPUs completed every leg. Every leg was also propagated by the existing
CPU verifier on the same machine; the raw comparisons are retained in
[`gpu-verification-v176`](../results/lambda/2026-09-07/gpu-verification-v176).

| Measurement | Local RTX 5090 | Lambda H100 |
| --- | ---: | ---: |
| Warm batch, median of five | 29.95 ms | 12.41 ms |
| Serial Python CPU propagation of the same 413 legs | 18.40 s | 37.07 s |
| Largest CPU/GPU position difference | 0.182 m | 0.176 m |
| Largest CPU/GPU velocity difference | 8.38e-12 km/s | 8.31e-12 km/s |
| Largest CPU/GPU mass difference | 6.42e-11 kg | 6.28e-11 kg |

GPU timings include host uploads and final downloads, but exclude file parsing,
packing, workspace creation and mission-rule evaluation. CPU times include
serial propagation only. These comparisons use the existing Python/SciPy CPU
implementation, not a tuned batched native CPU comparator. They do not establish
the speedup of the optimiser or the complete application. H100 parsing/packing
took about 1.09 seconds, substantially longer than the propagation itself.

The initial H100 check passed 17 tests; five archived-data tests were explicitly
skipped because that staged runtime has no pinned catalogue archive. Its
14-test CUDA memory check reported zero errors. A separate native device-buffer
graph replay and ownership test passed; racecheck reported zero hazards.
The integrated local library passed **296 GTOC12 tests**, including GPU
qualification and exact score reproduction for the three archived 39-, 37- and
36-ship reference missions. The native graph/ownership test also passed.

For our current 23-ship fleet, complete verification **from already parsed
inputs** took 19.278 seconds on the CPU backend and 0.532 seconds with CUDA
propagation: **36.25× in this one end-to-end comparison**. That includes packing
and all mission checks but excludes catalogue loading and solution-file parsing
(recorded separately). Both backends passed, with the same 194 collected
asteroids and exactly the same fixed-bonus score, 12805.194102488575 kg.
This is verification speed, not search or trajectory-optimisation speed.

## Run it

Build the ordinary `spacepdhcg_cuda` CMake target; the new propagator is included
and does not require QOCO to perform verification. In a configured Linux/WSL
environment:

```bash
export PYTHONPATH=src
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/absolute/path/to/libspacepdhcg_cuda.so
export SPACEPDHCG_GTOC12_DATA=/absolute/path/to/pinned/gtoc12/data
python -m spacepdhcg gtoc12 verify /absolute/path/to/Result.txt --propagation-backend cuda
```

`--official` additionally invokes the original independent competition binary
when installed. CUDA verification currently requires the fixed tolerance above
and does not export a dense history; asking its Python API for another tolerance
or a history raises an explicit error. Existing CPU viewer-history generation
is still available.

To reproduce the propagation comparison (choose a new output filename):

```bash
python scripts/gpu/benchmark_gtoc12_verifier.py \
  results/lambda/2026-09-06/fleet_master_v11/fleet/Result.txt \
  build/verifier-comparison.json --cpu-legs 413 --repeats 5
```

The C++ device API, host bridge and optional `certify_legs_cuda` interface enable
GPU propagation; file parsing, packing, rule evaluation, score bookkeeping,
viewer history generation and fleet search still contain CPU work. Neither a
new fleet score nor completion of the full GPU-native application is claimed.
