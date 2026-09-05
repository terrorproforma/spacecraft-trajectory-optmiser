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
