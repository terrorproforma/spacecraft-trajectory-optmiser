Native route-completion preparation
===================================

This bundle contains a frozen standalone build of the new `RouteSearch._finish`
CUDA costing stage. It contains no GPU executions, trajectory solves, fleet
promotions or throughput measurements. The shared library was compiled for
SM90 and SM120. The normal project CMake file includes the new source with
`--fmad=false`; its existing `tests/*_test.cu` discovery registers the native
test. This preparation did not configure or rebuild the complete production core.

`manifest.json` binds the four frozen source files, source archive, exact build
commands, compiler version, standalone library and linked test. The source
identity is a SHA256 tree, not a Git commit. `build_completion_native_v621.py`
contains the complete no-GPU build procedure. `cpu-fixture-check.log` covers
fixture construction, ABI sizes and the independent compensated-sum boundary;
it does not claim that native costing passed on those fixtures.

Root-controlled next checks are the native test's two kernel calls (257 ordinary
ragged candidates followed by five retained-workspace edge cases), then the
separately frozen 20 historical and 22 synthetic CPU-oracle cases. The executable
without arguments runs those two native test calls. `--cpu-only` never queries a
CUDA device and is the only executed test mode in this preparation.

The interface retains candidate/deployment/flight and output buffers. One GPU
thread performs each candidate's short sequential mass chain; candidates run in
parallel. Logical payload allocation is `72*max_candidates + 36*max_deploys +
200*max_legs` bytes, excluding CUDA stream/event/runtime overhead. Every evaluation
reuses those buffers. There is no trajectory propagation, Lambert solve, payload
shrinking, candidate ranking or interpolation-grid search in this stage.

The wrapper resolves existing model/certified-cell metadata and pair geometry.
CUDA computes stays, pickup matching, mining, cargo, actual departure masses,
authority, inflation, rocket-equation fuel and the final dry-plus-cargo gate.
First failure follows the original deployment dictionary and forward-leg order.
Duplicate pickups increase running mass again but update one dictionary entry;
zero pickups have explicit markers. Cargo uses CPython 3.12 compensated float
summation in first-pickup insertion order. Invalid mining stay has its own result
code because the Python oracle raises `ValueError` there.

The five-term fit uses sequential FP64 products/additions without FMA; NumPy's
matrix multiplication may use a different reduction order. Comparisons therefore
need bounded FP64 numeric parity plus exact discrete gate agreement. This does
not assert universal bitwise equivalence at every possible rounding boundary.

Resource output reports SM90: 80 registers, zero stack/shared/local; SM120: 84
registers, 160 stack bytes, zero shared/local. Existing solver code is untouched
by this standalone build. Neither resource numbers nor compilation demonstrate
production integration, speedup or independently certified physics.
