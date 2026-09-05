# Local GPU IPM optimization checkpoint

The local RTX 5090 experiments use a separate copy of QOCO commit
`09f049597deef2a7ead15b3da19a9456ff7d4e53`, CUDA 12.8, SM 120, FP64 and 32-bit
indices. The shared pinned checkout and existing campaign libraries are unchanged.
This is an optional backend experiment, not an automatic production backend switch.

## Implemented changes

- A scoped cuBLAS reduction handle replaces creating/destroying a handle for every
  dot product, infinity norm and minimum-absolute-value reduction. The native C++
  adapter discovers the optional paired extension and closes the scope after each
  solve, including cold retries. Unmodified libraries remain supported. Handles
  belong to the calling thread/device and do not survive the enclosing solve.
- Sparse forward and symmetric products use retained GPU-built row maps. Stable
  radix sorting preserves the CSC entry order within each row; one thread writes
  each result. The symmetric product gathers the mirrored off-diagonal entries
  from the original CSC column. Transpose products use 256 threads per block
  instead of one thread per block. Numeric values remain in their original buffer.
  Matrix synchronization refreshes the map using retained scratch allocations.
- A stopping-test correction computes the objective-coefficient norm from `c`,
  replacing an erroneous use of `x`. The regression constructs a large primal
  value with stationarity residual 0.5: the old relative test accepts it, while the
  corrected test rejects it. Neither the requested tolerances nor the independent
  trajectory acceptance criteria were relaxed.
- Optional deterministic cuDSS factorization makes the tested landing solve
  reproducible together with the ordered sparse products. This is specific to the
  same hardware, inputs and software configuration; it is not a cross-platform
  reproducibility claim. See [NVIDIA's reproducibility documentation](https://docs.nvidia.com/cuda/cudss/general.html).
- An optional checked cuDSS interface derives function-pointer types from the
  installed headers, handles the changed CSR signature in 0.8, and rejects a
  runtime with a different major/minor version before calling that interface.
  This enables controlled upgrade experiments; it does not qualify cuDSS 0.8.

## Matched measured result

Two warmups and seven measured samples per variant, alternating fresh processes:

| Median | Original reduction handles | Scoped reduction handles | Speedup |
|---|---:|---:|---:|
| SCvx | 1.474888 s | 0.623454 s | 2.37x |
| GPU IPM solve calls | 1.436294 s | 0.583023 s | 2.46x |
| Complete process | 1.929623 s | 1.045398 s | 1.85x |

Both builds include the same ordered products, corrected stopping scale and
deterministic factorization. Thus this comparison isolates handle reuse. Every
sample performed 54 inner iterations and two accepted outer steps on the displaced
20-interval P1-C 3DOF landing fixture. The final objective was
0.49448537334898213, canonical residual 8.5196031142981985e-12 and terminal residual
5.7354944404952589e-12, with qualification at 1e-8. The sample's
`trajectory_difference` is displacement from the reference, not CPU/GPU error.

Raw samples, qualification checks, stdout/stderr and binary hashes are in
`artifacts/performance/qoco-deterministic-scoped-handles-pd3.json`. Earlier aborted
comparisons (`qoco-scoped-handles-pd3.json` and
`qoco-gather-scopes-correct-stopping-pd3.json`) retain the failing samples. They
must not be reported as successful performance comparisons. Sparse products alone
did not eliminate cuDSS convergence variation; deterministic factorization was
needed for the matched measurement.
The checked-loader follow-up was built separately and passed seven further
identical 54-iteration landing repeats; the timing table refers to the frozen
pre-loader-check binaries identified in the benchmark artifact.

An API trace of the same deterministic work shows allocations falling from
5,752 to 1,135, frees from 7,059 to 903, and device synchronizations from 11,965
to 5,809. Kernel launches remain 12,263 in both traces. These are CUDA API
counts, not a GPU kernel timeline: the installed Nsight Systems 2024.6 does not
capture the RTX 5090 kernel timeline in this environment.

## Validation and limits

The standalone operator test checks independent long-double arithmetic,
rectangular and empty matrices, duplicate entries, symmetric products, topology
refresh, numeric updates, bitwise repeatability, nested reduction scopes and
unscoped calls after cleanup. CUDA memcheck, initcheck, synccheck and racecheck
passed; memcheck reported no device leaks. The stopping regression fails before
the correction and passes afterward, including memcheck.

The complete landing solve passed memcheck, initcheck and synccheck. **Full
racecheck reports hazards inside cuDSS 0.7.1.6's deterministic factorization kernel,
with both CUDA 12.8 and the isolated CUDA 13.2 sanitizer.** A check restricted to
the first factorization-kernel launch passes, so it does not clear the full-solve
finding. Keep this dependency issue open; do not suppress it or treat the whole
backend as sanitizer-qualified. Logs are under `build/performance/native-checks/`
with prefixes `qoco-v4`, `qoco-stopping` and `qoco-gather`.

The isolated cuDSS 0.8.0.10 upgrade was rejected: after adapting its breaking API,
memcheck found an out-of-bounds shared-memory read in its deterministic forward
solve kernel (`cudss::fwd_dtmn_ker`), and the native solve failed. Enabling
superpanels also failed. The checked loader successfully rejects 0.8 headers
paired with a 0.7 runtime. Logs: `qoco-v7-memcheck.log`, `qoco-v8-repeat7.err`,
and `qoco-abi-mismatch.log`. See the [cuDSS 0.8 migration guide](https://docs.nvidia.com/cuda/cudss/migration_guide.html)
for the changed function signatures; a library-path substitution alone is unsafe.
The final checked build also passes the inverse mismatch test (0.7 headers with
0.8 runtime), the native unavailable-backend contract, and the refreshed standalone
operator/stopping checks. The existing 6DOF handback contract passes; its candidate
is rejected by the outer loop, so this is not a 6DOF convergence result.

The pipeline still has CPU numerical work: QOCO conversion, scaling, KKT assembly,
plus host decisions and upstream scalar reductions. The adapter's residual audit
and dual mapping now run on CUDA, as described below.
GPU sparse maps and handle reuse do not make the complete planner GPU resident.
The maps also consume additional retained device memory (indices and sort scratch).
Large trajectories, other dynamics families, batching, stream-safe concurrency and
the full GPU GTOC12 path remain separate qualification and implementation work.

## Reproduce the optional build

Run from the repository in WSL. Use an empty destination outside the shared
upstream checkout; QOCO's CMake writes a generated header into its source tree.
Set the paths below to the reviewed upstream checkout, isolated build directory,
CUDA installation and the chosen cuDSS headers/library:

```bash
python3 scripts/gpu/prepare_qoco_gpu.py \
  --source "$QOCO_SOURCE" --destination "$EXPERIMENT/source" \
  --gather --correct-stopping --deterministic --checked-cudss-abi
cmake -S "$EXPERIMENT/source" -B "$EXPERIMENT/build" -G Ninja \
  -DQOCO_ALGEBRA_BACKEND=cuda -DQOCO_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_FLAGS="-I$CUDSS_INCLUDE" -DCUDSS_LIB="$CUDSS_LIBRARY" \
  -DBUILD_QOCO_DEMO=OFF
ninja -C "$EXPERIMENT/build" -j 3 qoco
"$CUDA_ROOT/bin/nvcc" -std=c++17 -O3 -arch=sm_120 \
  -I"$EXPERIMENT/source/include" -I"$EXPERIMENT/source/lib/qdldl/include" \
  cpp/cuda/tests/qoco_gpu_operators_test.cu \
  -L"$EXPERIMENT/build" -lqoco -ldl -Xlinker -rpath="$EXPERIMENT/build" \
  -o "$EXPERIMENT/qoco_gpu_operators_test"
```

Include the cuDSS library directory and CUDA runtime directory in
`LD_LIBRARY_PATH`; the cuDSS directory must contain `libcudss.so` because the
backend loads it dynamically. Use matching cuDSS 0.7.1 headers and runtime for
the measured candidate; the 0.8 probe is rejected. Run `qoco_gpu_operators_test` and its
`--stopping-only` mode, then the native integration executable's
`--p1c-qoco-repeatability 7` mode with
`SPACEPDHCG_QOCO_LIBRARY="$EXPERIMENT/build/libqoco.so"`.

For the matched control, prepare a second source copy with the same options plus
`--original-handles`. `--unmodified` prepares a completely unmodified upstream
control and cannot be combined with other changes. Run
`scripts/gpu/benchmark_qoco.py --help` for the alternating benchmark; it acquires
the shared local GPU lock and aborts while saving evidence if qualification fails.

## Device residual audit and dual mapping

The native adapter now evaluates the unscaled KKT certificate and maps QOCO duals
on CUDA. Fixed CSC topology is converted to ordered row gathers on the GPU once;
numeric updates reuse retained buffers. Parallel row and variable tasks evaluate
the equality, conic, stationarity and complementarity terms, followed by
deterministic reductions. Each SOC is an independent task. Non-finite inputs and
intermediates fail the certificate instead of disappearing through max operations.
The mathematical normalization and acceptance tolerances are unchanged.

Prepared backends expose `qoco_gpu_get_solution`, a borrowed view of completed,
unscaled device vectors. The adapter copies the primal device-to-device, maps
duals directly into the driver's buffer, and downloads only six audit scalars.
Older libraries without the optional export upload their host solution into
retained audit buffers; the audit itself still runs on CUDA. This compatibility
path is measured separately. Upstream QOCO still downloads its solution, and the
adapter still retains a host primal for accepted warm starts. Those transfers
and the CPU conversion/scaling/KKT assembly have **not** been eliminated.

`qoco_gpu_audit_test` checks an independent dense long-double reference, empty
constraint blocks, duplicate sparse entries, multiple SOC sizes, numeric updates,
dual transforms, reproducibility, a nonblocking stream, and NaN/infinity rejection.
Repeated updates and audits allocate no further buffers. CUDA memcheck (including
leaks), initcheck, synccheck and racecheck all pass for this test. Full landing
memcheck and synccheck pass with `SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE=1`, which
compares the old CPU certificate and mapping to the new device outputs. Seven
repeated landing solves and the older-library compatibility path also pass this
oracle. Production runs leave that flag unset. The existing 6DOF handback
contract passes, but it rejects its candidate; it does not establish 6DOF solver
convergence or exercise the new audit on a converged 6DOF problem.

The paired fresh-process benchmark uses the same prepared backend for both core
libraries, two warmups and seven measured samples per variant:

| Measurement | Previous CPU audit | GPU audit |
| --- | ---: | ---: |
| Median SCvx | 660.848 ms | 668.247 ms |
| Median complete process | 1068.991 ms | 1112.926 ms |
| Inner iterations / accepted steps | 54 / 2 | 54 / 2 |

Every sample qualifies at 1e-8 with objective 0.49448537334898213 and terminal
residual 5.735494440495259e-12. The GPU certificate differs only at roundoff
(8.519603114298186e-12 versus 8.519603114298199e-12). This small fixture does
**not** demonstrate an additional speedup: SCvx is 1.1% slower and process time
4.1% slower by these medians. The earlier scoped-handle speedup is a separate
measurement. Large-problem audit scaling remains to be measured.

Evidence: [paired samples](../artifacts/performance/qoco-gpu-audit-pd3.json) and
[validation checkpoint](../artifacts/performance/qoco-gpu-audit-checkpoint.json).
The benchmark predates the final memory-export correction: its `peak_device_bytes`
field covers only the persistent CQP workspace. Current exports add driver and
audit-owned peak allocations and label the result
`native_owned_peak_upper_bound_excludes_qoco_cudss`; the validated landing reports
506880 bytes instead of the old partial 187660-byte count. The sum is a conservative
bound on known allocations, **not** total device memory: opaque QOCO/cuDSS
allocations remain outside this counter. Adapter transfer counters likewise do
not instrument upstream library-internal copies.

Use `benchmark_qoco.py --baseline-core OLD/libspacepdhcg_cuda.so
--optimized-core NEW/libspacepdhcg_cuda.so` in addition to its existing backend
arguments to reproduce a core-adapter comparison. The tool hashes both core
libraries and selects them through the process library search path. The full
CUDA build now excludes the standalone QOCO operator test from its automatic
test glob, since that test requires the isolated upstream headers and library.

## Queued operators, device cone reductions and SOC step safety

Two additional opt-in preparation flags enable the next measured candidate:
`--queued-operators --device-cone-reductions`, alongside `--gather
--correct-stopping --deterministic --checked-cudss-abi`. The queued-operator flag
requires gathers and scoped handles. No remote campaign or pinned checkout was
modified, and the experimental backend remains opt-in.

Within a solve, vector and sparse-product kernels remain ordered on the default
CUDA stream without waiting for the entire device after each operation. Host
scalar reads retain their synchronization, as do matrix setup/refresh and cuDSS
operations. The outermost scope checks stream completion before destroying its
resources. Unscoped calls retain synchronous behavior. Independent chained
operations cover in-place updates, both sparse directions, nested scopes and
completion before returning to the caller.

Cone line-search block reductions now finish on CUDA and download one scalar,
including the step safeguard and scaling. The LP step stays on the device while
SOC limits are calculated. Cone residual and line-search scratch buffers are
retained for the solve and reused between calls; scope teardown frees them.
There are no host block-reduction loops in this path.

The tests also exposed two correctness defects in the pinned backend:

- The final cone residual reduction inspected at most 1024 partial blocks. A
  violated LP constraint at index 262144 was ignored. The replacement uses a
  grid-stride reduction covering every partial. The old binary fails this
  regression and the new binary reports the expected residual of 7.
- The SOC line-search branches for a linear quadratic expression or an initial
  boundary point could return an infeasible step, or unnecessarily return zero.
  For example, `x=(1,0,0), dx=(-1,1,0)` permits a maximum step of 0.5, whereas the
  old branch allowed 1. The corrected code enforces the first boundary of
  `c + b*t + a*t*t`, handles `a=0` explicitly, and retains the stable quadratic
  root calculation for nonzero `a`. Six exact/near-linear and boundary fixtures
  agree with independent long-double feasibility bisection. The old binary fails
  the boundary test.

These corrections change the convergence path; iteration counts must not be
assumed equal. The combined candidate is measured against the previous GPU-audit
backend, with two warmups and seven fresh-process measured samples per variant:

| Case | Previous median SCvx | New median SCvx | Speedup | Inner iterations |
| --- | ---: | ---: | ---: | ---: |
| 20-interval displaced landing, 1e-8 gate | 617.517 ms | 242.203 ms | 2.55x | 54 → 28 |
| Planner 3DOF example, 40 intervals, 1e-6 gate | 638.399 ms | 426.664 ms | 1.50x | 44 → 41 |
| Planner 6DOF example, 20 intervals, 1e-6 gate | 2045.458 ms | 1684.074 ms | 1.21x | 162 → 179 |

Complete-process speedups are respectively 1.55x, 1.27x and 1.19x. Every sample
passes its unchanged qualification gates, with two accepted steps and objectives
within an absolute 1e-8 of its control. Both planner examples also pass independent
replay, coefficient parity and the continuous-time checks. The 6DOF candidate
requires six outer attempts instead of five, so it is faster despite additional
solver work. These are scoped results, not a universal speedup multiplier.

For the 1e-8 landing, the final objective is 0.49448537365291756 (control
0.49448537334898213), canonical residual 2.618510687647072e-11 and terminal
residual 7.034629823099436e-13. All seven additional CPU-audit-oracle solves also
qualify. The prior native-audit migration remains a separate measurement.

Intermediate comparisons isolate queued operators (623.111 → 495.660 ms, same
54 iterations) and GPU cone reductions/scratch reuse before the SOC boundary
correction (536.114 → 486.666 ms, same 54 iterations). The full-solve API profile
records device-wide waits 5809 → 65 and allocations 1183 → 381. That profile also
changes 54 → 28 iterations, so its call-count reductions include both improved
convergence and implementation changes. Nsight captures API calls only here,
not an RTX 5090 kernel/memory timeline.

Validation includes all four CUDA sanitizers for the new cone kernels, queued
operator chains, independent feasibility checks, scope cleanup and large-grid
coverage. Full landing memcheck (zero leaks), initcheck and synccheck pass.
**Full landing racecheck still reports 30 hazards inside cuDSS 0.7.1.6's
deterministic factorization kernel.** This run returns a numerically qualified
point but is not racecheck-clean. A separate candidate using cuDSS's
nondeterministic mode fails the unchanged landing gate (one accepted step,
terminal residual 5.4844079940608025e-8 versus 1e-8). It is rejected; disabling
deterministic mode is not a qualified workaround.

The final prepared candidate is `/home/angus/build-qoco-gpu-cones-v14`; the
nondeterministic rejected probe is `build-qoco-gpu-cones-v15`. Evidence is in
[the checkpoint](../artifacts/performance/qoco-queued-checkpoint.json),
[landing samples](../artifacts/performance/qoco-queued-device-cones-pd3.json),
[3DOF planner samples](../artifacts/performance/qoco-queued-planner-pd3.json),
[6DOF planner samples](../artifacts/performance/qoco-queued-planner-pd6.json), and
[API counts](../artifacts/performance/qoco-queued-api-profile.json). Full
representative planner results are linked by the checkpoint.

Use `qoco_gpu_operators_test --cone-reductions-only` for the new numerical and
large-grid checks, and `--cone-boundaries-only` for the SOC regression alone.
`scripts/gpu/benchmark_planner.py` benchmarks a canonical problem document,
locks the local GPU, hashes its inputs and executables, checks every physics
gate plus objective agreement, and retains the full result for each sample.
Normalize an example first with `spacepdhcg validate`; the native executable
expects radians/canonical units. Unit normalization is outside these timings;
native independent replay is included.

CPU conversion, equilibration, KKT setup, upstream scalar decisions, accepted
warm-state storage and outer decisions remain. The dependency race issue, large
trajectory scaling, batching and the GPU GTOC12 path remain open work under the
full GPU-native goal.

## Retained topology and values-only updates

The native adapter now downloads the six sparse index arrays once. Subsequent
updates compare the device arrays exactly against retained device copies and
download a four-byte mismatch flag. This detects in-place changes even when
the pointer and supplied fingerprint are unchanged. Numeric downloads are
queued as a batch with one stream completion; destination lifetimes remain
valid on errors. Coefficient conversion, bound classification, cone mapping
and CSC assembly still execute on the CPU. This is not a GPU conversion yet.

The additional opt-in preparation flag `--values-only-updates` makes QOCO's
numeric update entry point upload values without uploading indices or rebuilding
GPU sparse gather maps. Explicit full-matrix synchronization still refreshes
topology. The measured source/binary is `/home/angus/build-qoco-gpu-values-v16`,
using all v14 flags plus this flag. It is not promoted to the default dependency.

The device topology tests cover empty arrays, nonblocking stream ordering,
in-place mutations, grid-stride tails, equal contents at a changed address,
dimension rejection and no allocations during validation. They pass all four
CUDA sanitizers. QOCO arithmetic tests cover values-only updates followed by
full topology updates against independent dense arithmetic; all four sanitizers
pass. Seven landing solves agree with the independent CPU audit at the unchanged
1e-8 gate. Full landing memcheck (zero leaks), initcheck and synccheck pass. The
previous cuDSS race finding remains open; it was not remeasured in this checkpoint.

Two warmups and seven alternating fresh-process samples per variant show no
clear additional speedup over v14:

| Case | Before SCvx | Cached SCvx | Before process | Cached process |
| --- | ---: | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 260.056 ms | 255.066 ms | 644.094 ms | 652.264 ms |
| 20-interval 6DOF planner, 1e-6 | 1662.863 ms | 1657.908 ms | 2027.960 ms | 2046.950 ms |

Iteration counts are unchanged (28 and 179 respectively), as are accepted steps
and physics gates. The retained topology saves repeated data movement, but its
creation and exact validation have costs. In the two-step landing, native D2H
traffic falls 137504 → 118172 bytes, initial H2D rises 116552 → 135888 bytes,
and native memory rises 506880 → 526220 bytes. These counters exclude opaque
QOCO/cuDSS traffic and memory; they are not whole-process transfer measurements.
The initial upload is amortized across updates. Do not claim a runtime win here.

A separate nondeterministic cuDSS probe tightened accurate inner tolerances by
10x. Ten solves qualified, but the eleventh failed the unchanged gate: first
inner natural residual 2.338862291e-7 and final terminal residual
5.484484238e-8 versus 1e-8. The planned 15 repeats stopped at that failure.
The candidate is rejected and its experimental stopping-margin flag removed;
the source wrapper, provenance and failed sample are retained in the evidence.
Neither factorization-mode change nor extra accuracy has resolved the vendor issue.

See [checkpoint and validation](../artifacts/performance/qoco-cached-topology-checkpoint.json),
[landing samples](../artifacts/performance/qoco-cached-topology-pd3.json), and
[6DOF planner samples](../artifacts/performance/qoco-cached-topology-planner-pd6.json).
The planner benchmark now accepts `--baseline-core` and `--optimized-core` to
compare native adapter changes with separately hashed libraries. The next
architectural work is compiled numerical conversion on CUDA, then device
equilibration and KKT updates; topology caching alone does not remove these CPU paths.

## Compiled CUDA numerical conversion

The native adapter now compiles the numerical conversion once. Each output
coefficient has an ordered list of canonical input entries and constant factors.
Parallel CUDA gathers evaluate quadratic, equality and conic matrix values,
objectives and right-hand sides. The maps preserve duplicate-entry accumulation
and ordinary/rotated SOC transforms. Separate multiplication/addition rounding
preserves the CPU expression order rather than introducing fused operations.

CUDA also validates every numerical input, bound classification and quadratic
symmetry before the converted values can be used. Changed sparse arrays, cone
descriptors, or finite/equality bound patterns reject the update. Rejected
conversion leaves the accepted formulation intact. The output stays in retained
device storage and updates the independent GPU KKT audit with device-to-device
copies. No CPU row construction, sorting, symmetry scan or numerical conversion
remains in repeated updates.

Initial structure discovery still uses the CPU converter. It compiles and uploads
the maps, then runs the GPU conversion before setup. The current QOCO host API
still needs converted numerical values on the CPU, so each update downloads a
packed output buffer and one validation integer. CPU equilibration, KKT updates,
solver scalar decisions and warm/outer state remain. This is a migration of
repeated conversion arithmetic, not the completed end-to-end GPU pipeline.

`SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE=1` enables the old CPU conversion as
an independent oracle; production runs omit it. A synthetic CQP covers mixed
scalar/box constraints, ordinary and rotated affine/variable cones, duplicate
entries, nonzero buffer offsets, changed coefficients and mutation rejection.
It also checks a successful solve after rejected conversions. Independent CUDA
tests cover zero outputs, grid-stride tails, fixed summation order, symmetry,
all bound classes, nonfinite inputs/results and allocation reuse. All four
sanitizers pass for the new kernels. The synthetic adapter test and full landing
pass memcheck (zero leaks), initcheck and synccheck. Seven repeated landing solves
and the actual 6DOF planner pass both CPU conversion and KKT audit oracles.
The existing cuDSS race finding remains open and was not remeasured.

Matched local measurements use the same v16 QOCO backend and frozen d35fe7e core
as the control, two warmups and seven alternating measured samples per variant:

| Case | Previous SCvx | CUDA conversion SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 258.174 ms | 249.038 ms | 657.813 → 637.775 ms |
| 20-interval 6DOF planner, 1e-6 | 1657.412 ms | 1635.301 ms | 2024.929 → 2002.448 ms |

The median SCvx changes are 3.7% and 1.4%, with unchanged 28/179 inner iterations,
two accepted steps, objectives and independent physics gates. These modest
measurements are not a universal speedup. In the landing, initial conversion
including compilation rises 0.771 → 1.644 ms; the complete numeric-update phase
falls 1.479 → 1.097 ms. Retained maps cost memory: native peak rises
526220 → 758744 bytes. Initial uploads also increase the small case's total
native H2D from 135888 to 286728 bytes and D2H from 118172 to 150840 bytes.
These counters exclude opaque QOCO/cuDSS allocations and transfers. Direct device
updates into QOCO are still needed to remove the remaining output round trip.

[Checkpoint and validation](../artifacts/performance/qoco-device-conversion-checkpoint.json),
[landing samples](../artifacts/performance/qoco-device-conversion-pd3.json), and
[full planner comparison](../artifacts/performance/qoco-device-conversion-planner-pd6.json)
retain the measured binaries' hashes and representative complete planner results.
