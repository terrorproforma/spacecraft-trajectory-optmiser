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

## GPU trajectory ordering and separator trees

The next checkpoint moves trajectory graph labeling, ordering and separator
size generation to CUDA. The optional native adapter extension supplies actual
device state/control/virtual index maps; it does not infer the problem family
from matrix dimensions. The adapter owns one device copy of those maps, records
the allocation and three device copies in telemetry, and retains them across
solver reconstruction. A test-only proxy forces a numerical failure after the
caller frees its maps and destroys its original stream. Reconstruction and
independent KKT comparisons pass on a new stream.

The GPU algorithm labels constraint supports from the sparse graph, propagates
labels through cone blocks before assigning epigraphs, and constructs interval
separators. Local auxiliary vertices are assigned to descendant leaf groups.
A GPU edge check moves endpoints of unexpected cross-partition edges to the
root. Sorting creates contiguous groups, and GPU integer reductions count their
sizes. Both the permutation and the separator-size array are supplied directly
from device memory to cuDSS. This still leaves generic host KKT assembly and
host solver/SCvx control; it is not a fully GPU-native pipeline yet.

Passing only a permutation was insufficient. The local-elimination prototype
reduced the 500-interval factor count from 1,089,492 to 976,632 entries, yet its
single probe spent 1.837 s solving. NVIDIA documents that omitting separator-tree
information can lose factorization/solve parallelism, and supports externally
generated trees with contiguous node groups in bottom-up level order.
[cuDSS tree format and reuse](https://docs.nvidia.com/cuda/archive/13.1.1/cudss/advanced_features.html#user-provided-elimination-tree-data-saving-reordering)
describes that contract; the installed 0.7.1 runtime accepts the generated tree.
The final ordering also changes grouping within the permutation, so the trials
do not isolate a speedup caused solely by adding metadata.

The current option chooses at most eight levels, with a minimum of two: a direct
runtime probe showed that cuDSS 0.7.1 rejects `ND_NLEVELS=1`. It does not silently
substitute a CPU ordering when supplied maps contain duplicate/out-of-range
indices. Synthetic native tests without trajectory metadata continue to use the
ordinary backend; trajectory production probes confirm the GPU extension runs.

Enable `--trajectory-ordering --trajectory-tree` in addition to the v31 flags
(`--setup-lifetimes` plus all v25 options). The permutation-only option remains
available for comparisons. The reference backend and routing defaults are
unchanged. **Retain the tree option as an experimental large-case candidate;
do not promote it generally or claim full sanitizer qualification.**

Matched RTX 5090 measurements use frozen v31/v42 QOCO libraries and the same
frozen native core containing the metadata ownership fix. Each variant has two
warmups and seven alternating measured samples; test/profiling environments are
disabled. All paired objectives and independent physics gates agree:

| Case | Reference SCvx | GPU tree SCvx | Complete process | Inner iterations |
| --- | ---: | ---: | ---: | ---: |
| 20-interval landing, 1e-8 | 154.919 ms | 180.697 ms | 564.663 → 581.701 ms | 28 → 36 |
| 20-interval 6DOF, 1e-6 | 1038.735 ms | 1159.789 ms | 1396.007 → 1503.814 ms | 179 → 176 |
| 500-interval 6DOF, 1e-6 | 745.262 ms | 553.157 ms | 1223.355 → 1048.186 ms | 34 → 34 |

The large case improves **1.347x in SCvx and 1.167x including process startup**.
Its median QOCO setup falls 356.258 → 135.697 ms, while the solver subphase
regresses 261.967 → 303.429 ms. Smaller cases regress. These are case-specific
results at unchanged tolerances, not a universal multiplier or a scaling curve.

A separate completion-fenced 500-interval diagnostic measures 3.046 ms for the
stage containing GPU ordering/tree creation and submission, 10.951 ms for the
vendor reordering phase, and 49.514 ms for symbolic factorization. The final
factor count is 976,603. The installed cuDSS version has no factor-FLOP query;
recorded `-1`, status 4, and zero output bytes mean unsupported. Diagnostic
fences and statistics queries perturb overlap and are excluded from benchmarks.

The standalone test checks permutation bijection, repeatability, shuffled device
maps, offset views in native integration, duplicate and out-of-range metadata,
output guards and non-default streams. Independent CPU elimination bounds the
front size for chain graphs. For trees, a CPU check reconstructs contiguous
groups from the returned counts and verifies every graph edge joins comparable
tree nodes. Cases include long-range couplings and a 70,001-vertex graph.
All four standalone sanitizers pass. The owned-metadata reconstruction test
passes memory, initialization and synchronization checks, with no leaked device
allocations. Native Ruiz-4, seven repeated landing solves, and 20/500-interval
conversion/KKT/step/eight-metric comparisons also pass.

Full production landing memory/initialization/synchronization checks pass.
Full racecheck **still exits 99**, now reporting **38 errors**, all in
`cudss::factorize_dtmn_ker`, versus 30 reports with the reference ordering.
These are retained unsuppressed. Standalone kernel checks do not clear this
vendor failure, and report counts alone do not measure its severity.

Frozen artifacts:

- QOCO: `/home/angus/build-qoco-gpu-trajectory-v42/final/libqoco.so`, SHA256
  `8c96980e396412aaeda225d71578feaa7bf906ebff34c004e369d6b8f794e203`.
- Native core: `/home/angus/build-spacepdhcg-trajectory-v41/final/libspacepdhcg_cuda.so`,
  SHA256 `ba045295670b421e049fa1fc21d399279fbfa523d71b2062d3f07b9620823d14`.
- The [checkpoint](../artifacts/performance/qoco-gpu-tree-checkpoint.json) contains
  complete sanitizer output, commands, additional binary hashes and measurement
  summaries. [Source reproduction](../artifacts/performance/qoco-gpu-tree-reproduction.json)
  confirms every prepared QOCO source hash.
- [Matched large-case samples](../artifacts/performance/qoco-gpu-tree-pd6-500.json),
  [factor/tree profile](../artifacts/performance/qoco-gpu-tree-factor-profile.json),
  [permutation-only factor profile](../artifacts/performance/qoco-trajectory-factor-profile.json)
  and [rejected/intermediate trials](../artifacts/performance/qoco-trajectory-ordering-history.json)
  retain the evidence and historical ordering source.

Build `qoco_trajectory_ordering_test.cu` against the isolated library using NVCC
C++17/SM120 as for the previous ordering test. For the ownership test, build
`qoco_recreate_probe.cpp` as a shared test proxy with isolated QOCO includes,
`-Wl,--no-as-needed -lqoco -ldl`; point `SPACEPDHCG_QOCO_LIBRARY` at the proxy
and run `native_qoco_conversion_test 4 trajectory`. Other native test invocations
use the actual backend. The proxy and diagnostic environments are never used
for timing. Tree-depth/within-group ordering, the remaining factorization
regression and the vendor race are the next local targets. The whole goal stays active.
## Factorization and cuDSS runtime variants

The v48–v55 hill climb changes only private local builds. No shared cuDSS
installation, pinned upstream checkout, production default or H100 campaign
changes. All variants use FP64, int32 indices and sm_120 without fast-math;
the native core stays frozen at v41. Physics tolerances and independent
qualification remain unchanged.

| Build | Runtime and change from trajectory-tree v42 | Result |
| --- | --- | --- |
| v48 | 0.7.1.6, force multiblock factors | Physics probes pass; full landing initcheck fails in vendor `preprocess_block_mapping` with an uninitialized host-copy source. |
| v49 | v48 with deterministic mode disabled | Probes pass with more iterations; same initialization failure. |
| v50 | 0.7.1.6, standard kernels, default factor algorithm | 20-interval 6DOF fails its terminal certificate and exhausts the trust region. |
| v51 | 0.8.0.10, deterministic, default factor algorithm | Native/landing probes pass; 6DOF crashes. Diagnostic memcheck first reports an invalid shared read in `cudss::fwd_dtmn_ker`. |
| v52 | v51 with forced multiblock factors | 6DOF still reports an illegal memory access. |
| v53 | 0.8.0.10, deterministic, vendor ordering without our permutation/tree | Landing crashes; memcheck reports the same forward-solve kernel's invalid shared read. |
| v54 | 0.8.0.10, standard kernels, trajectory tree | Initial probes and oracles pass; later 6DOF repeat fails qualification. Rejected. |
| v55 | v51 with superpanels enabled | 6DOF still reports an illegal memory access. |

No variant is promoted. The standard-kernel failures return API success but
fail the independent nonlinear terminal gate. In v54's failed warmup, canonical
residual is 3.297e-10 while terminal residual is 4.517e-6 against a 1e-6 limit:
1988 inner iterations, 18 outer attempts, 3 accepted and 15 rejected steps.
The final trust radius reaches 0.0001. This is a convergence/qualification
failure, not evidence that the smaller canonical residual permits a faster
result. It follows earlier successful v54 20/500-interval probes and independent
conversion/KKT/stopping/step-control comparisons, making repeatability essential.

v54 passes all four **full landing** sanitizers, including racecheck with zero
hazards. That narrows the older race problem but does not qualify this candidate
for production or establish sanitizer coverage of 6DOF. The prepared 6DOF
sanitizer helper was not run after its repeatability failure. v48/v49 stop at
initcheck; v51's diagnostic memcheck permits kernel-level error recovery and
reports 800 errors, whose later entries can be cascades. v52/v55 have no further
sanitizer claims. No sanitizer reports are suppressed.

The completed v54 landing benchmark uses two warmups and seven alternating
measured samples per variant, the same core and explicit per-variant runtimes:

| Median | v42 / cuDSS 0.7.1.6 deterministic | v54 / cuDSS 0.8.0.10 standard | Ratio |
| --- | ---: | ---: | ---: |
| SCvx | 194.813 ms | 151.352 ms | 1.287x |
| QOCO solve | 143.096 ms | 114.191 ms | 1.253x |
| Complete process | 600.362 ms | 571.919 ms | 1.050x |
| Inner iterations | 36 | 36 | — |

This is a landing-only result for a rejected candidate. The 6DOF campaign
stops at its first optimized warmup; there is no 6DOF median or completed
500-interval matched campaign for v54. Earlier single probes are diagnostic,
not speedup evidence. The prior v42 experimental large-case result remains
the last retained trajectory-tree checkpoint, with its previously documented
small-case regressions and vendor race limitations.

Reproduction and evidence:

- [Frozen variants, source snapshots, CMake caches, hashes, complete check logs and compact results](../artifacts/performance/qoco-factor-runtime-checkpoint.json).
- [Completed matched landing campaign](../artifacts/performance/qoco-cudss08-v54-pd3.json).
- [Interrupted 6DOF campaign, including failed warmup](../artifacts/performance/qoco-cudss08-v54-pd6.json).
- [Failed native 6DOF result with trajectory and iterations](../artifacts/performance/qoco-cudss08-v54-rejected-pd6.json).

`prepare_qoco_gpu.py --multiblock-factorization` and `--superpanels` are opt-in
experiments requiring `--checked-cudss-abi`. The trajectory-tree submission
selects the renamed enum for 0.8. The optional benchmark arguments
`--baseline-cudss /path/to/libcudss.so` and
`--optimized-cudss /path/to/libcudss.so` select separate runtime search directories
and record each library's SHA-256. Their directories must expose `libcudss.so`
pointing to the selected runtime. This allows a private pip `--target`
installation without changing the shared environment; the ordinary default
benchmark behavior is preserved when these arguments are omitted.

NVIDIA documents the [0.8 enum/CSR API migration](https://docs.nvidia.com/cuda/cudss/migration_guide.html)
and [symmetric-indefinite and user-tree fixes](https://docs.nvidia.com/cuda/cudss/release_notes.html).
Those release notes do not establish that our failures are fixed.
[Deterministic mode uses different kernels](https://docs.nvidia.com/cuda/cudss/types.html);
the next investigation should address conditioning and SCvx progress sensitivity
while retaining the same terminal certificate, then revisit faster kernels.
CPU KKT assembly and host control still prevent claiming the whole pipeline
is GPU-native. The full goal remains active.

## Per-solve recovery-state isolation

The v58 patch fixes recovery state that outlives the convex problem it describes.
The pinned `qoco_solve` calls `initialize_ipm` but does not invalidate
`work->best_valid`, `best_iter` or `best_metric`. SCvx updates coefficients and
scaling between calls. If the new solve stalls before beating the old metric,
`restore_best_iterate` copies the old scaled vectors into the new solve and may
label them solved-inaccurate using an obsolete residual. This explains why the
failed v54 trace repeatedly returns the preceding trajectory. The independent
canonical and nonlinear gates correctly prevent its certification.

`prepare_qoco_gpu.py` now resets best-iterate validity and per-solve iteration
counters before initialization in every patched build. The explicit
`--reset-solve-state` flag remains accepted; `--unmodified` preserves the original
benchmark control. Existing device allocations
and explicit primal warm starts are retained. The prepared v58 QOCO tree differs
from v54 in **only `src/qoco_api.c`**; source hashes reproduce from a fresh copy.
The native core additionally corrects `forcing_satisfied`: pure QOCO requires
the requested tolerance itself, whereas its previous telemetry used the
PDHCG re-solve trigger multiplier (5). No acceptance gate or tolerance changes.

The [standalone regression](../cpp/cuda/tests/qoco_solve_state_test.cu) solves a
scalar equality-constrained QP, updates its right-hand side from 1 to 4, and
forces iteration-limit recovery after installing an unbeatable old progress
metric. The original library restores old best-iteration metadata; the patched
one returns a point satisfying the new equality and clears old counters.
The test then verifies that an explicit warm start and the same allocated
workspace remain usable. Both zero and four Ruiz iterations are covered.
The test initially violated QOCO's ownership contract by allocating the solver
on the stack; that harness error was corrected before the final passing runs.
Its failure output is retained separately.

Before finding this bug, tighter linear refinement (v56: 10 iterations, 1e-10)
and smaller equality regularization (v57: 1e-13) both failed repeated 6DOF
qualification. Cold-start-only runs failed on repetition four. These changes
were rejected and the original numerical settings restored. The old v54
6DOF memory/initialization/synchronization checks passed; its racecheck timed
out after 240 seconds and is not a pass. These experiments separate a
demonstrated state-lifetime bug from unproven conditioning explanations.

The candidate uses frozen QOCO v58 with standard cuDSS 0.8.0.10 kernels and
native core v58, against v42 with deterministic cuDSS 0.7.1.6 and core v41.
GPU tree ordering and physics settings are otherwise unchanged. Initial
qualification includes nine consecutive 20-interval 6DOF runs, one 500-interval
run and seven landing repetitions. Independent numerical oracles pass at both
6DOF sizes and another seven landing repetitions. Each matched campaign uses
two warmups and seven alternating measured samples per variant; every sample
passes its unchanged certificate and objective comparison.

| Median SCvx | v42 reference | v58 candidate | Ratio |
| --- | ---: | ---: | ---: |
| Landing, 20 intervals | 203.147 ms | 164.095 ms | 1.238x |
| 6DOF, 20 intervals | 1169.896 ms | 806.350 ms | 1.451x |
| 6DOF, 500 intervals | 549.632 ms | 462.553 ms | 1.188x |

Complete-process median ratios are 1.101x, 1.309x and 1.058x. These compare
the **combined runtime/kernel choice and state fix**; they do not measure a
reset-only speedup. The 20-interval candidate's SCvx mean is **1.252 s versus
1.173 s**, and its maximum is **2.887 s versus 1.204 s**. At 500 intervals,
means are 0.526 versus 0.555 s, but maxima are 0.689 versus 0.593 s.
All outliers remain in the reports. The runtime configuration is therefore
experimental; no blanket/default promotion or uniform-speedup claim is made.

Full landing memory, initialization, synchronization and race checks pass.
Full 20-interval 6DOF memory, initialization, synchronization and race checks pass.
At 500 intervals, memory, initialization and synchronization checks pass.
The standalone state-isolation regression passes all four sanitizers with both
zero and four Ruiz iterations, including leak checking.
The [checkpoint](../artifacts/performance/qoco-reset-v58-checkpoint.json) records
each additional sanitizer's exact fixture and terminal result; missing scopes
and timeouts must not be interpreted as passes. In particular, no 500-interval
racecheck is claimed. The numerical regression's original/patched results,
frozen hashes, rejected experiments and helper sources are included there.

The deterministic cuDSS 0.8 retry (v59, now including the state reset) still
fails 20-interval 6DOF with an illegal memory access after passing native and
landing probes. It is rejected without a timing claim. The state-lifetime fix
does not resolve that separate vendor-kernel failure.

- [Prepared-source reproduction](../artifacts/performance/qoco-reset-v58-reproduction.json).
- [Landing measurements](../artifacts/performance/qoco-reset-v58-pd3.json).
- [20-interval 6DOF measurements](../artifacts/performance/qoco-reset-v58-pd6.json).
- [500-interval 6DOF measurements](../artifacts/performance/qoco-reset-v58-pd6-500.json).

The full objective remains active. CPU KKT assembly, host solver/SCvx control,
remaining runtime variance and broader physics-family qualification still need
work. Existing shared runtimes, pinned upstream checkouts and Lambda campaigns
remain untouched.

## Device solution ownership and host Ruiz consistency

The next checkpoint retains two changes with different purposes. The new
`--device-io` preparation option offers device solution output and a per-solver
accepted-primal cache in QOCO's existing GPU `x0` storage. The native adapter
enables it only when the complete extension is present. Separately, all patched
preparations now synchronize the objective and right-hand-side vectors after
legacy host Ruiz equilibration. Neither change selects a different production
backend, changes precision, or relaxes qualification.

### Completed solution and accepted-start contracts

In device mode, `qoco_solve` completes unscaling on its CUDA stream without
exporting `x/s/y/z` to host solution arrays. The GPU audit consumes borrowed
device pointers. Accepting a candidate copies its unscaled primal to GPU `x0`;
a cold or rejected solve retains that accepted cache. Enabling the next warm
start uses the saved device vector with the new solve's scaling. Each workspace
owns its mode and saved-state flag; no new GPU allocations or global cache are
introduced. The native adapter skips its host primal allocations in this mode.

Legacy output remains the default for direct QOCO callers. Explicit
`qoco_gpu_download_solution` supports host consumers and the opt-in independent
CPU audit. A host `qoco_set_x0` overwrite invalidates the saved GPU acceptance
flag. Solver reconstruction negotiates device mode again and starts with an
empty accepted cache. Copy failures propagate through the adapter; the driver
includes the last accepted-start transfer in its report immediately.

### Scaling regression discovered by the ownership test

The first non-unit test fails before enabling device IO. With P=3, A=2, G=-4,
c=1, b=2, h=4 and four Ruiz passes, v58 retains device c/b/h as 1/2/4 while the
host scaled values are approximately 0.666667/1.915207/2. It reports solved at
**x=1.0442737824**, although the equality requires x=1. After changing A to 4,
the stale-vector path returns x=0.6484197773 instead of 0.5.

The patch publishes c/b/h at the end of host equilibration, including the
zero-pass path used when re-equilibrating previously scaled data. The regression
now gives x=1.000000000000086 and x=0.500000000000010, with exact host/device
coefficient parity. A transition from four to zero Ruiz passes also passes.
The native device-update path performs its actual Ruiz calculation on CUDA;
these host copies repair the legacy setup/update boundary.

The first failed assertion exited before cleanup and produced leak reports.
That output is retained as a failed test, not classified as a new allocator
leak. The dedicated regression always cleans up, reproduces the bug on frozen
v58, and passes on v62 under all four sanitizers.

### Matched results and the failed objective comparison

Both variants use the same isolated cuDSS 0.8.0.10 standard-kernel runtime on
the local RTX 5090. The control is QOCO/core v58; the final candidate is QOCO
v62/core v63. Each completed campaign uses two warmups and seven alternating
measured samples per variant, with unchanged physics gates and an absolute
1e-8 objective-comparison limit.

| Final batch | Control SCvx median | Candidate SCvx median | SCvx ratio | Complete-process ratio |
|---|---:|---:|---:|---:|
| Landing, N20 | 188.758 ms | 186.763 ms | 1.011x | 0.974x |
| 6DOF, N20 | 1.171099 s | 1.027652 s | 1.140x | 1.033x |
| 6DOF, N500 | Campaign stopped | Campaign stopped | No claim | No claim |

N20 6DOF still has substantial convergence variability: measured inner-iteration
medians are 225 versus 203; candidate SCvx spans 0.476–1.896 s. Its earlier v61
batch regresses the SCvx median from 0.796724 to 0.864510 s (0.922x), while
landing is effectively flat (0.997x). The complete distributions and earlier
batch remain in the evidence; the later favorable median does not establish
a general improvement.

The final N500 campaign stops at candidate repeat 5: objective
0.512975718214113 differs from the initial control by **2.3926362e-8**. The
earlier v61 campaign stops at **5.5537667e-8**. Both runs pass every independent
physics certificate gate, but fail the stricter comparison required by this
benchmark. Neither campaign supplies a complete timing comparison.

A separate, predeclared ten-repeat comparison also finds an objective spread
of **8.6953563e-8 on the unchanged v58 control**. The ten v61 repeats in that
diagnostic span 9.7051622e-10, but that does not erase its earlier failure or
establish reliability. The decision is to retain the GPU IO capability as an
opt-in experiment and the scaling synchronization as a correctness fix, with
no whole-pipeline speedup or backend-promotion claim.

A separately qualified final landing Nsight Systems 2024.6 trace shows:

| CUDA API | Control calls | Candidate calls |
|---|---:|---:|
| Synchronous copy | 564 | 558 |
| Asynchronous copy | 422 | 424 |
| Stream synchronization | 83 | 87 |
| Kernel launch | 7046 | 7046 |
| Allocation | 398 | 398 |

Device IO removes eight solution exports and one warm-start upload; the Ruiz
fix adds three setup vector uploads. Two accepted-primal D2D copies and four
explicit completion waits preserve ownership/completion contracts. This is
CUDA API evidence, not an RTX5090 occupancy profile. A discarded LD_PRELOAD
probe could not intercept the statically linked CUDA runtime and is not used
as transfer evidence.

### Validation, frozen artifacts and scope

QOCO v62 passes the non-unit device-output/accepted-start test and the host
Ruiz regression at zero and four passes under memcheck with leak checking,
initcheck, synccheck and racecheck: 16 standalone checks. QOCO v62/core v61
also pass all four full landing and N20 checks, plus N500 memory,
initialization and synchronization checks. **N500 racecheck was not run.**

The final core v63 only adds immediate accepted-copy reporting and its direct
test. It passes the native conversion/ownership test at four Ruiz passes under
all four sanitizers, seven landing repetitions, N20/N500 qualification and all
independent CPU/GPU comparisons. Legacy-library compatibility passes. The
earlier full sanitizer results retain their exact core-v61 scope rather than
being relabeled as final-core runs. Nine plain N20 repetitions and the full
oracle results are also recorded.

Frozen final libraries:

- `/home/angus/build-qoco-gpu-device-io-v62/final/libqoco.so`:
  `2445292ccef52ede26d7a5a370cb3f5f99f9b95e6dca644b24eb1cd2a2f4c58a`.
- `/home/angus/build-spacepdhcg-device-io-v63/final/libspacepdhcg_cuda.so`:
  `b4bfba99813663f26e10d244381a2a7491d724a0cc7287b129fce8d79c3393c6`.

Add `--device-io` to the v58 preparation flags to reproduce the capability.
State-history reset and host Ruiz vector synchronization apply automatically
to patched preparation; `--unmodified` remains an unchanged control. Preparation
now fingerprints the extended workspace header, device-IO header, API and
equilibration sources, and hashes the actual final CUDA algebra source.
Prepared-file reproduction matches the frozen v62 source.

- [Complete checkpoint, raw failures, helper sources and test scopes](../artifacts/performance/qoco-device-io-v62-checkpoint.json).
- [Prepared-source reproduction](../artifacts/performance/qoco-device-io-v62-reproduction.json).
- [Final landing campaign](../artifacts/performance/qoco-device-io-v62-pd3.json).
- [Final N20 campaign](../artifacts/performance/qoco-device-io-v62-pd6.json).
- [Stopped N500 campaign](../artifacts/performance/qoco-device-io-v62-pd6-500.json).

The full GPU-native goal remains active. CPU KKT setup, host solver/SCvx
control, production independent replay and numerical variability remain.
Shared runtimes, pinned upstream sources and remote campaigns are unchanged.

## GPU KKT CSR assembly

The optional `--gpu-kkt` preparation mode replaces the backend's host
`construct_kkt`, host CSC-to-CSR conversion and the serial construction/upload
of P, A, G, NT and NT-diagonal update maps. It consumes the existing device CSC
matrices. One CUDA thread handles each sparse entry; CUDA prefix scans locate
cone blocks, and integer searches locate each upper-triangular cone entry.
Stable sorting of 64-bit row/column keys produces ordered CSR, followed by
parallel value/map scatter and row-offset generation. Matrix values are copied
without numerical atomic additions. The exact original NT entry order and
regularization values are preserved.

The final implementation packs nine temporary arrays into one GPU allocation,
with 256-byte alignment for each view. CUB sort/scan scratch uses one further
allocation: ten temporary allocations become two. Final CSR arrays and maps
remain separately owned by the solver until vendor teardown. Legacy host map
slots remain null and the unused CPU CSR conversion routine is omitted from
this build. There is no global assembly cache or CPU fallback in this path.

The native input conversion, initial QOCO matrix creation, host transposes and
regularization still precede this boundary. Moving KKT CSR construction does
not make the entire setup or solver control loop GPU-native. The existing
device coefficient/scaling updates consume the new maps without changing the
numerical solver settings.

### Construction and ownership validation

`SPACEPDHCG_TEST_QOCO_KKT_COMPARE=1` runs the independent CPU constructor during
testing and checks every CSR offset, column, value and all five map arrays.
Production runs omit it. The direct assembly test calls an explicit test-only
entry point, so it cannot silently pass against a library without GPU KKT
support. Its seven fixtures include empty A/G blocks, empty equality rows,
duplicate entries, pure SOCs, mixed orthant/SOCs, a 257-dimensional cone and
513 separate cones. The latter cases exercise multiple GPU blocks and scans.

The initial fixture coupled duplicate-entry assembly to cuDSS analysis, which
rejects that raw duplicate input on both the old v62 and new v65 libraries.
The final direct test isolates assembly from vendor analysis and confirms
exact duplicate/map parity. Production integration uses the existing native
canonicalization. The compile-placement error in v64 and the rejected fixture
are preserved in the checkpoint rather than counted as passes.

Final QOCO v67 with core v63 passes:

- The seven direct assembly cases under memcheck with leak checking, initcheck,
  synccheck and racecheck.
- Native conversion/CPU audit at four Ruiz passes, seven landing repetitions,
  N20/N500 qualification, and all independent numerical oracles.
- Full landing and N20 runs under all four sanitizers.
- Full N500 runs under memory, initialization and synchronization checking.
  **N500 racecheck is not included.**
- Forced numerical failure and solver reconstruction after caller index arrays
  and the original stream are released, normally and under all four sanitizers.
  The proxy exists only in the test directory; production code contains no
  fault injection.

### Timing results and limits

The CPU-KKT control is QOCO v62. Both candidates use the same core v63,
isolated cuDSS 0.8.0.10 standard kernels and RTX 5090. Full campaigns alternate
two warmups and seven measured samples per variant. All samples in both
campaigns pass the unchanged physics gates and absolute 1e-8 objective
comparison. This does not invalidate earlier objective-repeatability failures.

| Final pooled candidate v67 | Control SCvx median | GPU SCvx median | SCvx ratio | Complete-process ratio |
|---|---:|---:|---:|---:|
| Landing N20 | 151.384 ms | 162.474 ms | 0.932x | 0.969x |
| 6DOF N20 | 956.487 ms | 629.467 ms | 1.520x | 1.351x |
| 6DOF N500 | 444.173 ms | 448.727 ms | 0.990x | 0.987x |

The favorable N20 result also changes the inner-iteration median from 189 to
122. It cannot be attributed to faster assembly. In the earlier v66 batch,
N20 regresses from 499.038 to 576.248 ms (0.866x); landing and N500 ratios are
0.978x and 0.953x. All raw distributions remain available.

An additional direct comparison of separate temporaries (v66) with pooled
temporaries (v67) uses two warmups and **15 measured samples per variant** on
landing, with 36 inner iterations throughout. SCvx medians are 163.307 →
146.647 ms (1.114x), and whole-process medians improve 1.057x. However, QOCO
setup medians are **30.028 → 30.038 ms**, effectively flat. The data confirms
fewer allocation calls; it does not establish an 11% KKT-assembly speedup.
Even the final CPU-versus-GPU setup medians are nearly flat at N20 6DOF
(71.381 → 71.702 ms) and N500 (105.959 → 105.843 ms).

Separately qualified landing Nsight Systems 2024.6 traces confirm the intended
operation changes:

| CUDA API | CPU KKT v62 | GPU v66 | Pooled GPU v67 |
|---|---:|---:|---:|
| Synchronous copy | 558 | 550 | 550 |
| Allocation | 398 | 408 | 400 |
| Free | 306 | 316 | 308 |
| Kernel launch | 7046 | 7068 | 7068 |
| Stream synchronization | 87 | 88 | 88 |

Eight uploads disappear when CSR and update maps are generated on CUDA.
Pooling removes eight allocations and frees from the first GPU version. The
extra kernels perform the newly parallel assembly, scans and sorting. These
are CUDA API counts, not an RTX5090 occupancy profile. The implementation is
retained as an opt-in GPU-native setup capability; mixed timings do not earn
a general speedup claim or a default backend change.

### Reproduction and remaining work

Add `--gpu-kkt` to the v62 preparation flags. It requires `--setup-lifetimes`
and currently rejects combination with legacy `--profile-setup`, whose CPU
phase markers no longer apply. Nsight/API evidence and full setup timing are
recorded separately. Normal builds without `--gpu-kkt` retain the previous
constructor. The new header is fingerprinted in preparation provenance, and
the final prepared files reproduce exactly.

Frozen final QOCO:
`/home/angus/build-qoco-gpu-gpu-kkt-v67/final/libqoco.so`, SHA256
`67c8981047fdbe756a5b361f112ed56a6dc43b215416245f8b06115a29c860dd`.
It uses the unchanged frozen core
`/home/angus/build-spacepdhcg-device-io-v63/final/libspacepdhcg_cuda.so`.
The checkpoint fingerprints both, the runtime, direct test and recovery proxy,
and preserves the earlier v66 source overrides.

- [Checkpoint, failures, test scopes and helper sources](../artifacts/performance/qoco-gpu-kkt-v67-checkpoint.json).
- [Prepared-source reproduction](../artifacts/performance/qoco-gpu-kkt-v67-reproduction.json).
- [Final landing campaign](../artifacts/performance/qoco-gpu-kkt-v67-pd3.json).
- [Final N20 campaign](../artifacts/performance/qoco-gpu-kkt-v67-pd6.json).
- [Final N500 campaign](../artifacts/performance/qoco-gpu-kkt-v67-pd6-500.json).
- [Direct pooled-temporary comparison](../artifacts/performance/qoco-gpu-kkt-v67-packing.json).

The complete goal remains active. Initial matrix setup/transposes/
regularization, CPU solver and SCvx control, independent production replay,
conditioning and broader physics-family qualification still require work.
Shared runtimes, pinned upstreams and remote campaigns remain untouched.

## Device numerical-update setup maps

The device numerical updater now reuses the original matrix's GPU gather entry
order for A/G transpose updates. Stable row sorting produces the same inverse
mapping as the CPU transpose constructor, including duplicate-entry order.
Two duplicate GPU arrays, their host inverse-permutation loops and their uploads
are removed. These are borrowed solver-owned pointers: the native adapter
destroys the update context before solver teardown or reconstruction, and
numerical updates preserve matrix topology.

CUDA kernels also construct the original-P source map, locate regularized P's
diagonal entries, and copy cone boundaries from the existing device workspace.
The source map binary-searches the sorted inserted-diagonal positions. Their
initial list still comes from host regularization and is uploaded when needed;
initial matrix setup, host transposes and regularization remain separate work.
A four-byte validation result and a setup stream synchronization are retained.
This replaces duplicate numerical-update setup work, not the entire setup path.

`SPACEPDHCG_TEST_QOCO_UPDATE_MAPS_COMPARE=1` independently compares every source,
diagonal, transpose and cone-boundary map with its legacy host construction.
The flag is omitted from timing runs. The extended numeric-update test includes
1031-variable cases with inserted diagonal entries and with off-diagonal P,
as well as Ruiz 0/1/4, unconstrained and zero-quadratic cases. It checks three
updates against both the CPU updater and independent scaling equations.

The frozen candidate is QOCO v68, SHA256
`828f6b9249909134c10475dba3111b110bf03872b398c3ab98efc8db5a0c3ece`.
The control is frozen QOCO v67; both use unchanged core v63 and isolated cuDSS
0.8.0.10 standard kernels on RTX 5090. Two warmups and seven measured samples
per variant pass all unchanged physics and absolute 1e-8 objective comparisons.

| Case | Control SCvx median | Candidate SCvx median | SCvx ratio | Process ratio |
|---|---:|---:|---:|---:|
| Landing N20 | 186.214 ms | 189.909 ms | 0.981x | 1.024x |
| 6DOF N20 | 1078.986 ms | 665.487 ms | 1.621x | 1.292x |
| 6DOF N500 | 478.516 ms | 478.223 ms | 1.001x | 0.999x |

N20's inner-iteration median changes from 204 to 112; its improvement cannot
be attributed to faster map setup. Landing has 36 inner iterations throughout,
and N500's median changes from 34 to 35. The architectural change is retained
within the optional device updater, with no general speedup or backend-default
claim. Numerical variability remains unresolved.

Complete setup medians are also mixed: landing 37.426 → 41.241 ms,
N20 80.001 → 78.492 ms, and N500 114.961 → 132.524 ms. Removing these CPU
map loops is not evidence of faster complete setup in this batch.

Qualified landing Nsight traces show the expected operation changes:

| CUDA API calls | v67 | v68 |
|---|---:|---:|
| Synchronous copy | 550 | 545 |
| Asynchronous copy | 424 | 425 |
| Allocation | 400 | 398 |
| Free | 308 | 307 |
| Kernel launch | 7068 | 7071 |
| Stream synchronization | 88 | 89 |

This input needs no inserted-P upload. Five map uploads disappear, with one
small asynchronous validation readback added. The new temporary buffer's
destructor still calls `cudaFree(nullptr)` when empty, so the API free count
falls by one despite removing two real allocations. The traces measure API
activity; they do not establish occupancy or a map-phase speedup.

Full landing, the original eight update cases and the extended nine-case test
pass all four sanitizer tools. Full N500 passes memory, initialization and synchronization checks;
N500 racecheck and full N20 sanitizer coverage are not included in this
checkpoint. Forced reconstruction passes normally and under all four tools
after the caller's original index arrays and stream are released. Independent
KKT, conversion, audit, device-update, device-I/O and stopping/step oracles also
pass on landing, N20 and N500.

- [Frozen binaries, helper sources, checks and API traces](../artifacts/performance/qoco-update-maps-v68-checkpoint.json).
- [Prepared-source reproduction](../artifacts/performance/qoco-gpu-kkt-v68-reproduction.json).
- [Landing timings](../artifacts/performance/qoco-gpu-kkt-v68-pd3.json).
- [N20 timings](../artifacts/performance/qoco-gpu-kkt-v68-pd6.json).
- [N500 timings](../artifacts/performance/qoco-gpu-kkt-v68-pd6-500.json).

## Experimental GPU scratch-vector arena

`--vector-arena` replaces the 26 post-analysis scratch-vector device allocations
with one solver-owned allocation. Every view has 256-byte alignment; one GPU
memset initializes all views and padding. Host mirrors use `calloc` for existing
inspection interfaces. Input/scaling vectors and matrix allocations are outside
this arena. A setup completion synchronization preserves the caller's existing
initialization boundary.

Each float-vector metadata record identifies arena ownership so ordinary vector
destruction frees its host mirror without freeing a GPU slice. The solver frees
the complete arena after its vectors and vendor linear-system resources are
destroyed. There is no global arena or cross-solver cache. Preparation verifies
the exact 26-call replacement, and runtime checks verify both the view count and
final arena capacity. Builds without `--vector-arena` retain separate allocations.

This candidate is **experimental and not qualified for promotion**. In the
matched N20 campaign, optimized measured sample 2 returns a physics-certified
objective of 0.51297605935915824 versus reference 0.51297569119164033. Its
3.681675179e-7 difference exceeds the unchanged absolute 1e-8 objective gate.
The campaign stops at that failure, preserving its result. No matched N500
campaign follows it. Passing memory tests or later repeats cannot erase this
failed qualification.

An additional alternating diagnostic runs each frozen library 20 times against
the original reference, recording every result without relaxing the gate. The
unpooled v68 control qualifies 20/20, with objective spread 1.0121e-9. Arena v69
qualifies 19/20, with spread 3.6750e-7; its failing run misses the objective gate
by the same scale as the initial failure. All 40 runs pass the physics gates.
This reproduces the candidate's qualification failure; the clean ownership
checks do not establish its numerical cause or justify promotion.

The complete landing batch (two warmups and seven measured samples per variant)
has 36 inner iterations throughout. Its SCvx median is 169.559 → 161.358 ms
(1.051x), and complete-process ratio is 1.026x. Setup medians are 32.268 →
32.441 ms, effectively flat. This landing result is not a general speedup claim
for the failed candidate.

Qualified landing Nsight API traces confirm the intended resource changes:

| CUDA API calls | Separate vectors v68 | Arena v69 |
|---|---:|---:|
| Synchronous copy | 545 | 519 |
| Allocation | 398 | 373 |
| Free | 307 | 282 |
| Asynchronous memset | 565 | 566 |
| Stream synchronization | 89 | 90 |
| Kernel launch | 7071 | 7071 |

The independent ownership test checks four solver shapes, up to 1031 variables
and a 257-dimensional SOC. Three solvers coexist. Every nonempty view is aligned,
in bounds and disjoint, starts at exact device/host zero and retains its unique
tag after every vector is written. Destroying the middle solver and allocating
another preserves the surviving solvers' data. An explicit arena capability
symbol prevents the test from silently passing against the separate-vector build.

The ownership test, nine numerical-update cases and full landing each pass all
four sanitizer tools. Full N500 passes memory, initialization and synchronization
checking; N500 racecheck and full N20 sanitizer coverage are not included.
Forced reconstruction passes normally and under all four tools. Initial
landing/N20/N500 numerical-oracle runs also pass, but have the narrower scope
described above and do not override the matched objective failure.

Frozen QOCO v69 is
`/home/angus/build-qoco-gpu-gpu-kkt-v69/final/libqoco.so`, SHA256
`353b4cc498f5e385dfc4364e98284e53d13fa614d5720c8f2cd7ec9d5254cb81`.
Control v68, core v63, isolated cuDSS 0.8.0.10 standard kernels and RTX 5090
remain unchanged. Prepared-source reproduction is exact.

- [Checkpoint, ownership scopes and helper sources](../artifacts/performance/qoco-vector-arena-v69-checkpoint.json).
- [Landing timing distribution](../artifacts/performance/qoco-gpu-kkt-v69-pd3.json).
- [Failed N20 objective comparison](../artifacts/performance/qoco-gpu-kkt-v69-pd6.json).
- [Paired repeatability diagnostic retaining every failure](../artifacts/performance/qoco-vector-arena-v69-repeatability.json).
- [Prepared-source reproduction](../artifacts/performance/qoco-gpu-kkt-v69-reproduction.json).

## Inaccurate-exit best-iterate experiment

`--restore-inaccurate-best` returns the saved best qualified QOCO iterate on an
inaccurate exit, using the existing GPU buffers and restoration operation before
unscaling. Accurate exits and all tolerances are unchanged. v70 adds this policy
to unpooled v68; v71 adds it to pooled v69. The option remains experimental.

A diagnostic proxy observes matching first-solve fingerprints for device matrix
topology/values, objective/right-hand-side vectors, Ruiz scales and initialization
settings across 24 v68/v69 runs, but differing iteration counts and results. These
are fingerprints, not a byte-for-byte proof or an identified cause. The added
downloads/synchronization make these traces unsuitable for performance claims.
Source inspection finds that numerical-error exits already restore the saved
best point, while inaccurate exits can return a worse latest point.

An exploratory eight-run-per-variant proxy sweep compares unchanged behavior,
best-point return and two tighter refinement policies. All 64 samples pass the
physics and objective gates. Best-point return has the lowest median SCvx time
in both tested builds, but the subsequent direct implementation test contradicts
any claim that this alone fixes the outliers:

| Direct implementation | Objective-qualified / 20 | Physics-qualified / 20 | Maximum absolute objective difference |
|---|---:|---:|---:|
| v68 control | 20 | 20 | 6.377e-10 |
| v70 best return | 19 | 20 | 3.682e-7 |
| v71 arena plus best return | 20 | 20 | 1.341e-9 |

Every sample uses the original 0.51297569119164033 reference, absolute 1e-8
objective gate, and 1e-6 physics gate. v70 sample 18 fails; it remains recorded.
No matched timing campaign follows this failure. v71's passing batch cannot
establish that its allocation layout resolves the numerical sensitivity.

The failed run rejects six candidates on the inner forcing requirement, shrinking
the trust radius from 1 to 0.015625. Two steps then pass. The last uses
0.9999992144 of the available radius, yet satisfies the outer loop's absolute
0.02 step tolerance and ends the solve. Its final reported step is zero because
the final replay compares the retained trajectory with itself. This identifies
an outer convergence/reporting weakness; the original inner-solve sensitivity
remains a separate question.

The direct audit verifies that each exercised inaccurate exit returns all four
saved vectors with the proper physical scaling and the corresponding saved
objective/residuals. Changed-RHS reuse, warm-start retention, numerical-update
tests and landing/N20/N500 numerical oracles pass for both builds. Exact prepared
source reproduction passes. Complete sanitizer scopes and frozen fingerprints
are in the checkpoint; these checks do not override the objective failure.

- [Checkpoint and failed outer-iteration trace](../artifacts/performance/qoco-best-return-v71-checkpoint.json).
- [Input/solve fingerprint traces](../artifacts/performance/qoco-arena-v69-solve-trace.json).
- [Exploratory policy sweep](../artifacts/performance/qoco-accuracy-v69-probe.json).
- [Direct implementation repeatability](../artifacts/performance/qoco-best-return-v71-repeatability.json).

## SCvx convergence at a small trust radius

The outer loop now requires an accepted step to satisfy the existing absolute
step tolerance **and** lie inside the existing near-boundary threshold of the
radius used for that solve. Shrinking the trust region below the step tolerance
can no longer make a boundary-limited step establish convergence. The guard is
evaluated before radius expansion and retained with the accepted trajectory.
No physics, objective, inner forcing or configured tolerance is relaxed.

Final replay now preserves the step used by the outer convergence decision.
It does not promote an iteration-limit or cancellation result to converged by
comparing the retained point with itself. This also repairs a separately
reproduced cancellation bug: cancelling a feasible zero HCW trajectory before
the first outer iteration previously returned certified/converged with zero
iterations. The new core returns cancelled and uncertified.

The native regression deliberately sets the N20 initial radius to 0.015625,
below the unchanged 0.02 step tolerance. The old core certifies a two-iteration
boundary result with an objective about 3.68e-7 from the reference and reports
zero step. A separate two-iteration case with step tolerance 1e-12 also exposes
the final-replay promotion. Both limited-budget cases correctly remain
unconverged in the new core. With sufficient budget it takes an interior step,
meets the original objective gate and reports the accepted displacement.

The 60-run diagnostic retains every sample, including baseline failures:

| Configuration | Objective-qualified | Physics-qualified | Maximum objective difference |
|---|---:|---:|---:|
| Small-radius input, core v63 / QOCO v68 | 8/20 | 20/20 | 3.7012e-7 |
| Same input, guarded core v72 / QOCO v68 | 20/20 | 20/20 | 7.7509e-10 |
| Original input, guarded core v72 / QOCO v70 | 20/20 | 20/20 | 8.5263e-10 |

All comparisons retain the 0.51297569119164033 reference and absolute 1e-8
objective gate. These samples support the stopping correction, not a claim that
all numerical variability is eliminated. v70's earlier failure remains part of
its separate record; its policy is not promoted independently.

The complete original-input benchmark uses QOCO v68 and the same cuDSS runtime
on both sides, with two warmups and seven measured samples per workload/variant.
All 54 samples pass their unchanged quality gates:

| Workload | Old SCvx median | Guarded SCvx median | Inner-iteration medians |
|---|---:|---:|---:|
| Landing | 176.179 ms | 171.186 ms | 36 → 36 |
| N20 6DOF | 570.046 ms | 648.000 ms | 102 → 112 |
| N500 6DOF | 458.284 ms | 465.257 ms | 34 → 36 |

This is retained as a correctness fix, with no general speedup claim. The
small-radius diagnostic likewise takes longer with the guard (730.661 →
874.078 ms median), while the old core fails matched-quality qualification.
Inner iteration sensitivity, conditioning and CPU setup/control remain open.

Four native regression tests pass. The normal native conversion test, seven
landing repetitions, N20 and N500 pass the existing independent numerical
oracles. One complete small-radius trajectory passes memory/leak checking with
zero errors and leaks. No new GPU kernel code is introduced. These are the exact
new-core test scopes; the earlier best-return N20 racecheck timeout is not a pass.

Frozen core v72 is
`/home/angus/build-spacepdhcg-boundary-v72/final/libspacepdhcg_cuda.so`, SHA256
`993fd2c0b0765b3696507aea086952177d0c686367d0b0c02f1f9707c9e2e241`.
No backend default or frozen runtime is replaced.

- [Checkpoint, commands, fingerprints and test output](../artifacts/performance/scvx-boundary-v72-checkpoint.json).
- [Boundary and iteration-limit regressions](../artifacts/performance/scvx-boundary-v72-checks.json).
- [Cancellation reproduction](../artifacts/performance/scvx-boundary-v72-cancellation.json).
- [Repeatability including baseline failures](../artifacts/performance/scvx-boundary-v72-repeatability.json).
- [Landing](../artifacts/performance/scvx-boundary-v72-pd3.json), [N20](../artifacts/performance/scvx-boundary-v72-pd6.json), [N500](../artifacts/performance/scvx-boundary-v72-pd6-500.json) timing distributions.
