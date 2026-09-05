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

## Device coefficient updates and Ruiz equilibration

The optional `--device-numeric-updates` preparation flag now exposes a retained
device update context. The native adapter supplies the resident converted
P/A/G/c/b/h buffer directly. CUDA inserts regularization-only diagonal entries,
updates transpose values, scales matrices and vectors, and reduces diagnostic
ranges. Requested Ruiz passes run on CUDA during both initial native setup and
later updates. Norms and scaling span multiple blocks; each cone uses a block
reduction to preserve its common scaling factor. FP64 arithmetic and independent
physics qualification remain unchanged.

Correction to the earlier CPU-path inventory: QOCO's numerical KKT coefficient
updates already ran on CUDA through retained CSR maps. This extension reuses
that path. Initial KKT assembly and symbolic setup still use CPU structure.
The new work removes CPU numerical update/equilibration and the converted-array
round trip; it does not complete the entire GPU-native pipeline.

Preparation also fixes three pinned-backend update bugs exposed by the new
tests: the missing host `scale_arrayf` branch silently omitted objective
scaling, transpose updates used the permutation in the wrong direction, and
quadratic updates mishandled inserted diagonal entries. External callers that
use CPU Ruiz setup receive the initially scaled device c/b/h vectors when they
create the device context. The native adapter avoids that CPU Ruiz setup.

The extension owns its solver's numerical updates: do not mix it with legacy
host update calls. Host matrix/vector shadows become stale. The adapter still
supports older libraries without the extension; a partial extension or a device
update failure is rejected. A failed update requires fresh solver setup before
reuse. The context is destroyed before the solver, and a producer-stream event
orders the input before QOCO's default-stream work. Each update downloads nine
scalar doubles (72 bytes), with no numerical-array download or new allocation.
Initial structure discovery, sparse assembly, scalar/outer decisions and warm
state remain host work. Verbose mode and explicit CPU audit/conversion oracles
still request host converted values.

The standalone test compares three successive updates against the corrected
CPU reference and checks scaling equations independently with long-double
expressions. Seven cases cover Ruiz 0/1/4, 17/1031 variables, mixed linear/SOC
constraints, absent constraints, missing quadratic diagonal entries, and a zero
quadratic objective. It uses an offset device pointer and a nonblocking producer
stream. All four CUDA sanitizers pass; memcheck reports zero leaks. Native
mixed-cone conversion tests pass with Ruiz 0 and 4 and an independent 1e-8 KKT
gate. The Ruiz-4 native case and real landing pass memory, initialization and
synchronization checks. Seven landing solves and the actual 6DOF planner also
pass both CPU conversion and KKT audit oracles.

The prior cuDSS deterministic-factorization race finding remains open and was
not remeasured here. The clean standalone update racecheck excludes
factorization. This backend remains optional and is not promoted to the default.
Intermediate v18/v19 comparisons failed and led to the fixes above; their local
prepared sources are retained, and neither is a qualified candidate.

Final paired RTX 5090 results compare frozen 4556941/v16 against the new core/v20
with two warmups and seven alternating measured samples per variant:

| Case | Previous SCvx | Device update SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 259.198 ms | 259.853 ms | 653.673 → 644.673 ms |
| 20-interval 6DOF planner, 1e-6 | 1675.632 ms | 1667.908 ms | 2064.490 → 2047.173 ms |

Overall runtime is essentially flat. Both cases retain 28/179 inner iterations,
two accepted steps and the same objectives and independent physics gates. These
mission benchmark policies request zero Ruiz iterations; nonzero Ruiz is tested
in the synthetic solver cases, not benchmarked on these missions.

The landing's median complete numerical-update phase improves 1.137 → 0.488 ms
(2.33x for that phase), but it is a small fraction of total time. Initial
conversion/setup remain about 1.68/32.02 ms. Native D2H counters fall
150840 → 110000 bytes (27.1%); native H2D remains 286728 bytes and native peak
758744 bytes. Those counters exclude QOCO/cuDSS, including this extension's
retained buffers and scalar downloads, and must not be presented as whole-process
memory or traffic. The earlier pre-initial-Ruiz implementation also measured
flat landing performance and is retained separately.

[Checkpoint, source provenance and sanitizer outputs](../artifacts/performance/qoco-device-updates-checkpoint.json),
[final landing samples](../artifacts/performance/qoco-device-updates-final-pd3.json),
and [final planner comparison](../artifacts/performance/qoco-device-updates-final-planner-pd6.json)
record hashes and representative complete planner results. Prepared-source hashes
were reproduced in a fresh directory. The full GPU-native goal remains active.

To reproduce this candidate, use the existing isolated CUDA build recipe with
all these preparation flags (the destination must be a new directory):

```bash
python3 scripts/gpu/prepare_qoco_gpu.py --source "$PINNED_QOCO" \
  --destination "$CANDIDATE/source" --gather --correct-stopping --deterministic \
  --checked-cudss-abi --queued-operators --device-cone-reductions \
  --values-only-updates --device-numeric-updates
```

The standalone update test intentionally stays outside the native test glob;
compile it against that prepared source and library, then run under the same
cuDSS runtime and GPU lock as the operator tests:

```bash
nvcc -std=c++17 -O3 -arch=sm_120 \
  -I"$CANDIDATE/source/include" -I"$CANDIDATE/source/lib/qdldl/include" \
  cpp/cuda/tests/qoco_gpu_numeric_update_test.cu \
  -L"$CANDIDATE/build" -lqoco -ldl -o "$CANDIDATE/qoco_gpu_numeric_update_test"
```

With this backend selected, `SPACEPDHCG_TEST_QOCO_DEVICE_UPDATE_REQUIRED=1`
makes `native_qoco_conversion_test` assert that the device path was used.
Pass positional argument `4` to exercise initial and repeated Ruiz-4; the default
is Ruiz-0. `SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE=1` and
`SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE=1` enable the independent host oracles.
Omit test-oracle variables from performance runs. The final local build directory
was named `build-fixed`; adjust the link/runtime path when using that saved build.

## GPU scalar extrema and retained NaN checks

The next optional backend adds `--device-scalar-reductions` to the preceding
preparation flags. Infinity norms and minimum-absolute-value reductions now
return the value directly from CUDA. The previous implementation downloaded a
cuBLAS extremum index, then downloaded the selected element in a second operation.
NaN checks now use retained scalar scratch instead of allocating, initializing
from the host and freeing a flag for every call. These operations use the same
default stream and shared scope lifecycle as the existing cone reductions.

Inputs through 4096 entries use one block; larger inputs use bounded grid-stride
partial reductions followed by a device reduction. All threads participate in
the block barriers. Finite extrema remain exact FP64 values; NaNs explicitly
propagate through the extrema reduction. No floating-point atomics or changed
stopping tolerances are involved. Host scalar decisions and one scalar result
transfer per call remain; batching those decisions is still required for the
fully resident iteration loop.

The standalone `qoco_gpu_scalar_test.cu` checks exact extrema, empty inputs,
subnormals, infinities, signed zeros, NaNs, offset pointers, block/grid tails up
to 1048579 entries, and nested/unscoped calls before and after cleanup. Build it
using the preceding standalone command with the scalar test filename. All four
CUDA sanitizers pass, with zero leaked allocations. The native Ruiz-4 case,
seven repeated landing solves and actual 6DOF planner pass the independent CPU
conversion/KKT oracles. Full landing memory, initialization and synchronization
checks also pass. The prior cuDSS factorization race remains unresolved and was
not rerun; this optional backend is not promoted to the default dependency.

Matched RTX 5090 measurements use the same frozen 038695b core with v20 versus
v21 backends, two warmups and seven alternating measured samples per variant:

| Case | Previous SCvx | Scalar reduction SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 245.661 ms | 227.428 ms | 660.829 → 605.240 ms |
| 20-interval 6DOF planner, 1e-6 | 1662.882 ms | 1536.696 ms | 2027.660 → 1905.294 ms |

Both SCvx medians improve about 1.08x (7.4%/7.6% less time). Iterations remain
28/179, with two accepted steps and the same objectives and independent physics
results. These are local matched measurements, not universal speedup claims;
the raw samples retain timing variability, including a slow baseline warmup.

A separate qualified landing API trace supports the reduction in round trips:
`cudaStreamSynchronize` calls fall 847 → 401, `cudaMemcpyAsync` 1232 → 786,
`cudaMemcpy` 953 → 925 and allocations/frees 405/325 → 379/299. Kernel launches
remain 6065. This is an Nsight Systems 2024.6 CUDA API trace, not an RTX 5090 GPU
kernel timeline; API durations include initialization and queued work and should
not be treated as exclusive GPU phase times.

[Checkpoint, sanitizer output, source provenance and API counts](../artifacts/performance/qoco-device-scalars-checkpoint.json),
[landing samples](../artifacts/performance/qoco-device-scalars-pd3.json), and
[planner samples](../artifacts/performance/qoco-device-scalars-planner-pd6.json)
retain the measured binary hashes and complete representative planner results.
Prepared-source hashes reproduce in a fresh directory. Remaining CPU decisions,
initial structure/KKT assembly, dependency correctness, large trajectory scaling
and GPU GTOC12 remain part of the active goal.

## Batched GPU stopping metrics

The optional `--batched-stopping` flag keeps twelve norm results and five dot
products on the GPU, combines them there, and returns one six-double packet:
primal residual, dual residual, gap and their three relative stopping scales.
The prior implementation returned each norm/dot separately. Sparse operators and
cuBLAS `Ddot` remain in use; dot results use device pointer mode and the prior
pointer mode is restored before returning. This uses the asynchronous scalar
result behavior documented by [NVIDIA](https://docs.nvidia.com/cuda/cublas/index.html#scalar-parameters).
Retained scratch and default-stream ordering protect intermediate values when
the next operation reuses a work vector. The six-scalar combination preserves
the original expression order with explicit double multiplication/addition.

The flag requires `--device-scalar-reductions`, `--queued-operators` and
`--correct-stopping`. Add it to the v21 preparation command. The loader checks
both cuBLAS pointer-mode symbols before using them. The standalone
`qoco_gpu_stopping_test.cu` builds against the isolated headers/library using the
same standalone recipe as the scalar test.

`SPACEPDHCG_TEST_QOCO_BATCHED_STOPPING_COMPARE=1` compares all six metrics with
the previous implementation at every stopping check. It is omitted in measured
runs. The standalone test independently calculates the metrics on the CPU with
long-double arithmetic, covering nontrivial scaling vectors, symmetric sparse
quadratics, absent constraints, zero quadratic input, 17/4103 variables,
successive iterate changes, nested/unscoped execution and restoration of cuBLAS
host scalar mode. All four CUDA sanitizers pass, including zero leaks. Native
Ruiz-4, seven repeated landing solves and the actual 6DOF planner pass metric
comparison plus CPU conversion/KKT oracles. Full landing memory, initialization
and synchronization checks also pass.

Matched local RTX 5090 measurements use the same frozen 038695b core, v21 versus
v22 backends, two warmups and seven alternating measured samples per variant:

| Case | Previous SCvx | Batched stopping SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 237.979 ms | 184.374 ms | 724.607 → 661.227 ms |
| 20-interval 6DOF planner, 1e-6 | 1540.333 ms | 1269.197 ms | 1918.977 → 1619.480 ms |

These are 1.29x/1.21x SCvx speedups at unchanged 28/179 inner iterations, two
accepted steps, objectives and independent physics gates. Complete process
speedups are 1.10x/1.18x. Raw timing variability is retained; these measurements
do not establish a universal multiplier.

A separate qualified landing API trace shows synchronous copies falling
925 → 595, asynchronous copies 786 → 636, and stream synchronizations
401 → 251. There are thirty additional metric-combination launches
(6065 → 6095 total launches), with unchanged 379 allocations/299 frees.
The control trace is from the identical saved v21 binary. As above, these are
API counts, not exclusive GPU kernel timings.

The stop/continue policy, best-iterate bookkeeping and scalar work in objective,
centering, line search and iterative refinement still involve the CPU. Initial
structure/KKT assembly, outer control, larger trajectory scaling and GPU GTOC12
also remain. The known cuDSS factorization race was not remeasured and stays open;
this candidate remains optional. The full goal is active.

[Checkpoint, source hashes, sanitizer output and API counts](../artifacts/performance/qoco-batched-stopping-checkpoint.json),
[landing samples](../artifacts/performance/qoco-batched-stopping-pd3.json), and
[planner comparison](../artifacts/performance/qoco-batched-stopping-planner-pd6.json)
retain the evidence. Prepared-source hashes were reproduced in a fresh directory.

## Shared objective and complementarity calculation

`--batched-iteration-scalars` extends the stopping packet with the objective and
mu (the mean slack/dual dot product). It requires `--batched-stopping` and uses
the same FP64 cuBLAS dot products. The quadratic product is consumed before its
regularization/scaling correction changes the work vector, and `c' * x` is shared
with the stopping calculation. CUDA preserves the separate quadratic
regularization correction and guarded division by the objective scale. An absent
inequality block produces mu zero. The production loop now gets eight doubles
in one packet instead of separately computing/downloading objective and mu.
The six-metric entry point remains available.

The standalone stopping test accepts `--iteration` to require and exercise the
extended entry point. It checks both interfaces, independent long-double
objective/mu formulas with nontrivial scales, the zero-scale division contract,
regularization removal, empty constraints and zero quadratic input. The native
comparison mode now checks all eight values against the previous calculations
at every iteration. Native Ruiz-4, seven landing solves and the actual 6DOF
planner pass this comparison plus independent conversion/KKT/physics checks.
All four standalone kernel sanitizers pass; full landing memory, initialization
and synchronization checks pass with zero leaks.

The initial v23 integration comparison failed because the new C oracle calls
lacked their function declarations. The corrected v24 includes `kkt.h` and was
built with `-Werror=implicit-function-declaration`. The failed prototype and
its output are retained. A subsequent comment cleanup rebuild changed binary
hashes, including after the original source was restored. The initial timing
artifacts remain as superseded evidence; their original optimized binary is no
longer retained. The final dependency was copied to an immutable `final/`
directory, then fully revalidated and remeasured. Use those final artifacts and
hashes for subsequent comparisons.

Final paired RTX 5090 measurements compare frozen v22 with the final v24 library,
using the same 038695b core, two warmups and seven alternating measured samples:

| Case | Previous SCvx | Shared iteration SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 186.770 ms | 175.702 ms | 575.536 → 575.912 ms |
| 20-interval 6DOF planner, 1e-6 | 1240.639 ms | 1163.785 ms | 1613.479 → 1530.128 ms |

SCvx medians improve 1.063x/1.066x. Landing complete-process time is flat;
6DOF improves 1.054x including startup/replay. The landing QOCO solve subphase
alone is slightly slower (132.608 → 136.713 ms); these independently computed
medians and raw variability must not be presented as a uniform gain in every
phase. Iterations remain 28/179, with two accepted steps, unchanged objectives
and the same independent physics gates.

A qualified API trace of the final library shows 251 → 131 stream waits,
636 → 516 asynchronous copies and 6095 → 6005 launches. Synchronous copies
remain 595 and allocations/frees remain 379/299. These counts support fewer
round trips and duplicated operations; they are not exclusive GPU timings.

[Final checkpoint and validation](../artifacts/performance/qoco-batched-iteration-checkpoint.json),
[final landing samples](../artifacts/performance/qoco-batched-iteration-final-pd3.json),
and [final planner samples](../artifacts/performance/qoco-batched-iteration-final-planner-pd6.json)
record the immutable binaries, source provenance and representative complete
planner outputs. The final backend is
`/home/angus/build-qoco-gpu-iteration-v24/final/libqoco.so`.

Stop/best-iterate policy, centering, line search, iterative-refinement control,
initial structure/KKT assembly, outer control and GPU GTOC12 remain unfinished.
The known cuDSS factorization race remains open and was not remeasured; this
backend stays optional. The full GPU-native goal is active.

## GPU centering and fused iterate updates

The `--device-step-control` preparation option requires `--batched-stopping`
and its prerequisites. Both cone line searches now write into retained device
scratch, using the existing LP/SOC boundary kernels. CUDA computes the affine
centering vectors, cuBLAS writes the two dot products to device memory, and a
small kernel computes the clipped centering factor. Final x/y/s/z updates share
one multi-block kernel that reads the device step length. This also avoids the
reference minimum macro's repeated line-search evaluation.

FP64 arithmetic and boundary rules are unchanged, including the near-zero step
cutoff, clipping order and empty-centering NaN behavior. The early exit for a
NaN search direction remains in place. One sigma and one alpha scalar still
return to the host: combined-RHS construction and stop/best-iterate policy have
not yet moved fully onto the device.

The standalone `qoco_gpu_step_control_test.cu` compares centering and all four
updates against independent long-double arithmetic. Feasibility bisection
covers LP, SOC, mixed and empty cones, boundary and nearly linear directions,
tiny steps, 262145 LP entries and 1027 SOCs. Offset pointers and canaries check
vector bounds; nested and unscoped calls check scratch lifetime. The optional
`SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE=1` also checks each sigma/alpha against
the prior calculation. Native Ruiz-4, seven landing solves and the actual 6DOF
planner pass that comparison, the eight-metric comparison, and independent
conversion/KKT/physics gates. All four standalone sanitizers pass. Full landing
memory, initialization and synchronization checks pass both with and without
test oracles; memory checks report zero leaks.

Frozen v24 and v25 libraries were compared on the RTX 5090 using the unchanged
038695b core, two warmups and seven alternating measured samples per variant:

| Case | Previous SCvx | GPU step SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 173.834 ms | 156.348 ms | 581.578 → 588.903 ms |
| 20-interval 6DOF planner, 1e-6 | 1166.371 ms | 1032.315 ms | 1522.864 → 1390.632 ms |

SCvx medians improve 1.112x/1.130x; QOCO solve medians improve
128.149 → 110.148 ms and 1050.970 → 934.503 ms. Startup-inclusive landing is
1.3% slower, while startup-inclusive 6DOF improves 1.095x. The optimized landing
repeat 2 took 1.300420 s and remains in the measured samples. Its cause is not
established. No outliers were discarded. Iterations remain 28/179, with two
accepted steps, unchanged objectives and the same physics gates.

A qualified landing API trace shows synchronous copies 595 → 483, asynchronous
copies 516 → 460, stream waits 131 → 75 and launches 6005 → 5697. Allocations
and frees remain 379/299. Nsight Systems 2024.6 supplies API counts here, not an
RTX 5090 GPU timeline or exclusive stage timings.

[Checkpoint and validation](../artifacts/performance/qoco-device-steps-checkpoint.json),
[landing samples](../artifacts/performance/qoco-device-steps-pd3.json) and
[planner samples](../artifacts/performance/qoco-device-steps-planner-pd6.json)
record hashes, reproduced source, sanitizer output and complete representative
planner results. The retained binary is
`/home/angus/build-qoco-gpu-step-control-v25/final/libqoco.so`, SHA256
`a48278eaaab05add8b3098e9817fdad52bed2328dc88b2317783203e62fa443a`.
It was frozen before validation and measurement.

The existing cuDSS factorization race remains open and was not remeasured.
This optional backend has not replaced the production default. Host control,
initial structure/KKT assembly, iterative refinement, larger trajectory
scaling, batching and GPU GTOC12 remain part of the active full goal.

## Combined RHS experiments: no general promotion

Two opt-in experiments move the combined search-direction RHS further onto
CUDA. `--device-combined-rhs` requires `--device-step-control`. It consumes the
device centering factor and the existing s-dot-z result, computes mu on CUDA,
and replaces two Jordan products, the identity shift, negation and correction
with one cone kernel. The identity shift uses cached SOC offsets, removing its
quadratic prefix scan. The original cone products and subtraction order remain
the numerical reference.

`--queued-centering-metadata` additionally pins the existing workspace allocation
without changing its layout. Sigma's metadata copy is queued, so the host can
dispatch the next solve without waiting for that copy. No production host code
reads sigma before a later synchronous result or scope completion. Workspace
destruction explicitly completes the stream before freeing the pinned storage.
The existing standalone centering API still returns its scalar synchronously.

The extended standalone test checks the full combined RHS against independent
long-double Jordan products/inversion with identity NT scaling, as well as the
old implementation. Real landing and 6DOF tests exercise nonidentity scalings.
Tests cover empty/mixed cones, large cone sets, direction/iterate preservation,
canaries, pinned allocation, metadata completion at a stream event, scope
lifetime, and execution with and without test oracles. Native Ruiz-4, seven
landing solves and the 20-interval planner pass the step/RHS/eight-metric
comparisons and independent conversion/KKT/physics checks. All four standalone
sanitizers and full landing memory/initialization/synchronization checks pass;
the latter also pass with all test oracles disabled. Memory checks report zero
leaks. The existing cuDSS factorization race remains open and was not rerun.

The first, synchronous-metadata v26 trial regressed landing SCvx from
151.187 to 165.627 ms. Its 6DOF SCvx median improved only 1.009x, although its
solver subphase improved 1.041x. The 463.112 ms optimized landing sample remains
in the evidence. This trial did not replace v25.

The queued-metadata v27 trial compares against frozen v25 with two warmups and
seven alternating measured samples per variant, unchanged core and tolerances:

| Case | v25 SCvx | v27 SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 148.964 ms | 157.818 ms | 537.325 → 582.600 ms |
| 20-interval 6DOF, 1e-6 | 1042.609 ms | 1031.575 ms | 1400.294 → 1421.922 ms |
| 500-interval 6DOF, 1e-6 | 721.923 ms | 713.041 ms | 1227.712 → 1220.916 ms |

The small solver subphase medians are effectively flat. At 500 intervals the
solver subphase regresses 252.728 → 258.415 ms despite a 1.012x SCvx improvement.
These results do **not** establish a reliable general speedup. Frozen v25 remains
the performance baseline; the new options remain experiments, disabled unless
explicitly selected. Separate v26/v27 timing batches do not establish the cause
of either regression.

All paired objectives and physics gates agree, with 28/179/34 inner iterations
for the three respective cases. Single 100/500-interval qualification probes
also pass, but are not speedup benchmarks. The full 500-interval planner passes
all enabled conversion/KKT/RHS/metric comparisons. Its lower iteration count
than the 20-interval case prevents treating these as monotonic scaling timings.

The v27 landing API trace confirms 483 → 455 synchronous copies,
460 → 488 asynchronous copies and 5697 → 5585 launches. Stream waits increase
75 → 76 for pinned workspace destruction; pinned allocations/frees increase
4 → 5. Device allocations/frees remain 379/299. This is API evidence, not an
RTX 5090 GPU timeline or proof that fewer calls improve total time.

The 500-interval baseline now spends a median 343.772 ms in QOCO setup versus
252.728 ms solving. Setup includes both host and GPU work; split that phase
before attributing its cost to a particular routine. This is the next profiling
target, rather than continuing to assume iteration scalar calls dominate.

[v26 evidence](../artifacts/performance/qoco-device-combined-rhs-checkpoint.json),
[v27 evidence](../artifacts/performance/qoco-queued-combined-rhs-checkpoint.json),
[500-interval matched samples](../artifacts/performance/qoco-queued-combined-rhs-planner-pd6-500.json)
and [larger qualification probes](../artifacts/performance/qoco-rhs-scaling-probes.json)
retain regressions, hashes, sanitizer logs and complete representative results.
Frozen candidates are `/home/angus/build-qoco-gpu-combined-rhs-v26/final/libqoco.so`
(SHA256 `811e7bd2b198cda88e78caa97f468388651bb875adf9612f5f0c2527e726c7f1`)
and `/home/angus/build-qoco-gpu-combined-rhs-v27/final/libqoco.so`
(SHA256 `e99226f73b2f1c1d9f9fb988cc2a0fd24b6e7a4b1e245cb05d4b99c6ccb89e6c`).
Both were frozen before validation and measurement. The full goal remains active.

## Setup profiling, rejected GPU ordering, and ownership fixes

The next RTX 5090 checkpoint splits setup instead of attributing all of it to
GPU work. `--profile-setup` adds optional stage diagnostics, enabled only with
`SPACEPDHCG_TEST_QOCO_SETUP_PROFILE=1`. These are completion-fenced wall
intervals and perturb overlap; they are not kernel timings or speedup samples.
Diagnostic mode separates vendor reordering and symbolic factorization; normal
execution preserves the combined analysis call.

Three fresh-process probes per size in frozen v30 pass the independent planner
certificate. At 500 intervals, median reordering is **231.925 ms**, symbolic
factorization 43.553 ms, vendor handle creation 47.201 ms, host KKT assembly
2.087 ms and CSC-to-CSR conversion 3.640 ms. One probe has a 428.090 ms symbolic
outlier and 781.974 ms workspace-vector stage; those samples remain recorded,
with no established cause. Reordering is consistently 231.520–236.089 ms.
The [NVIDIA documentation](https://docs.nvidia.com/cuda/cudss/doc_output/types.html)
identifies reordering as CPU work even when numerical execution is on the GPU.
The measured cost makes this a material obstacle to the fully GPU-native goal.

[Split-stage evidence](../artifacts/performance/qoco-setup-split-profile.json)
includes hashes, source provenance and full diagnostic output. The
[earlier v28 profile](../artifacts/performance/qoco-setup-stage-profile.json)
labels the combined vendor analysis `symbolic_analysis`; do not compare that
field directly with v30's symbolic-only field.

The opt-in `--gpu-degree-ordering` experiment computes undirected degrees and
a stable permutation using CUDA/CUB, then supplies the device array through
the installed cuDSS 0.7.1 user-permutation API. This is static degree ordering,
not approximate minimum degree. The standalone test checks independent CPU
ordering, stable ties, bijection, duplicate edges, disconnected graphs and
output guards at sizes 0, 1, 13, 4099 and 65539. All four standalone sanitizers
pass. Native conversion and landing/20/500-interval planner probes also pass.

**Reject v29 as a performance option.** Its diagnostic 500-interval probe takes
2.416 s solving and 2.894 s in SCvx, compared with roughly 0.25 s solving in
the existing default-ordering measurements. This is not a paired speedup
estimate. Landing iterations increase from 28 to 36. Numerical gates pass,
but the experiment provides no reason to promote it. Factorization fill was
not measured, so it is only a possible explanation for the regression.
The v29 `create_csr` stage includes GPU permutation creation/submission, and
its `symbolic_analysis` still covers combined analysis.

[Ordering checks and historical prepared source](../artifacts/performance/qoco-gpu-ordering-checkpoint.json)
and [qualification probes](../artifacts/performance/qoco-degree-ordering-probes.json)
retain the rejected result. The standalone CUDA test needs the isolated QOCO
headers/library, as with the existing step-control test; compile
`cpp/cuda/tests/qoco_gpu_ordering_test.cu` with NVCC C++17, `-arch=sm_120`,
the prepared `include` and `lib/qdldl/include` paths, and `-lqoco -ldl`.

The retained opt-in `--setup-lifetimes` correction (v31, based on v25) fixes
three setup ownership issues without changing numerical algorithms:

- Free the temporary host KKT matrix and its three arrays after analysis.
- Retain CSR row/column indices until the vendor matrix is destroyed.
- Create RHS/solution dense wrappers before passing them to analysis.

The Linux diagnostic interposer `cpp/cuda/tests/qoco_host_kkt_tracker.cpp`
tracks only the four allocations returned by `construct_kkt`, not the entire
host heap. Seven landing workspaces leaked 559,188 bytes in v25; one 500-interval
workspace leaked 5,315,104 bytes. The same v31 runs free all tracked bytes.
Build the tracker with `g++ -std=c++17 -O2 -fPIC -shared`, the isolated QOCO
`include` and `lib/qdldl/include` paths, and `-ldl`. Set
`LD_PRELOAD=<tracker.so>:<candidate-libqoco.so>` for this diagnostic only.
[Host allocation evidence](../artifacts/performance/qoco-host-lifetime-comparison.json)
records both libraries, commands and full outputs. Retaining CSR indices does
increase live device storage during solving; existing native-owned peak
telemetry excludes these QOCO allocations.

Matched v25/v31 measurements use unchanged frozen core, two warmups and seven
alternating measured samples per variant, with all test diagnostics disabled:

| Case | v25 SCvx | v31 SCvx | Complete process |
| --- | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 148.338 ms | 150.244 ms | 557.965 → 574.286 ms |
| 20-interval 6DOF, 1e-6 | 1064.851 ms | 1061.243 ms | 1438.595 → 1446.565 ms |
| 500-interval 6DOF, 1e-6 | 731.741 ms | 730.562 ms | 1231.769 → 1228.340 ms |

These establish no general speedup. Retain v31 for the ownership correction;
v25 remains the historical performance comparison. All paired physics gates
and objectives agree, with unchanged 28/179/34 inner iterations. The 4.005 s
baseline 500-interval warmup is retained in the raw measurements.

Native Ruiz-4, seven repeated landing solves, 20/500-interval conversion/KKT,
step and eight-metric comparisons pass. All four standalone step sanitizers
pass. Full landing memory, initialization and synchronization checks pass
with and without test oracles. Full production racecheck was rerun because
vendor input lifetimes changed: it still exits 99 with **30 cuDSS factorization
race errors**. This remains unresolved; standalone racecheck does not qualify
vendor factorization. No report was suppressed.

[Lifetime checkpoint](../artifacts/performance/qoco-setup-lifetimes-checkpoint.json),
[source reproduction](../artifacts/performance/qoco-setup-lifetimes-source-reproduction.json)
and the [500-interval matched benchmark](../artifacts/performance/qoco-setup-lifetimes-planner-pd6-500.json)
retain all evidence. Frozen v31 is
`/home/angus/build-qoco-gpu-lifetimes-v31/final/libqoco.so`, SHA256
`4a8a0d15e83bcdd68fc0910718a30602417f61ec4cf2e315a7f06a0cb2b358a7`.
All prepared source hashes reproduce. The profiling/lifetime flag combination
also prepares successfully; the measured v31 binary contains no profiling or
degree-ordering option. The next ordering design should use explicit trajectory
stage metadata and separators, with factorization cost measured as well as
ordering time. CPU assembly, generic reordering, host solver and outer control,
the vendor race and the full GPU-native goal remain open.
