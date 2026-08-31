# Local GPU execution roadmap

**Programme:** SpacePDHCG Paper 1, followed by OrbitWeaver scale experiments  
**Repository:** `terrorproforma/spacecraft-trajectory-optmiser`  
**Pinned upstream PDHCG:** `167c8b72b4b96d2f94d405b8763e485514192b81`  
**Minimum upstream CUDA requirement:** 12.4  
**Primary implementation language:** C++20/CUDA; Python is orchestration, reference, and analysis only

This is the execution manual for completing the remaining GPU-dependent programme on a local NVIDIA
machine. It is deliberately more detailed than a normal roadmap. Follow the gates in order. Do not
start multi-GPU work before the single-GPU correctness and persistent-ownership gates pass.

---

# 1. What remains

The pre-GPU programme has already supplied:

- fixed-pattern CQP structures and topology fingerprints;
- mutable numerical CQP values;
- native C++ dynamics, variational RK4 sensitivities, transcriptions, and SCvx drivers;
- deterministic and robust CPU truth models;
- adaptive forcing/trust-region policies;
- accelerator pointer, stream, lifetime, and optional DLPack contracts;
- native wheels carrying the stable C ABI;
- benchmark matrices, run manifests, Paper 1 result schema, figure/table definitions, and H1–H6
  decision rules.

The remaining critical path is:

```text
G0 host/native freeze
    ↓
G1 real pinned upstream one-shot CUDA correctness
    ↓
G2 persistent single-GPU PDHCG workspace
    ↓
G3 device-resident deterministic SCvx
    ↓
G4 adaptive/inexact and hybrid GPU experiments
    ↓
G5 scenario-aware NCCL multi-GPU execution
    ↓
Paper 1 result freeze and manuscript
    ↓
OrbitWeaver route × scenario scaling
```

No performance hypothesis is evaluated before the corresponding gate's correctness conditions pass.

---

# 2. Hardware profiles

## 2.1 Minimum development profile

Use this only for correctness and early kernel development:

- x86-64 Linux;
- one NVIDIA GPU supported by CUDA 12.4 or newer;
- at least 16 GB GPU memory;
- at least 32 GB host RAM;
- modern 8-core CPU;
- local NVMe storage with at least 100 GB free;
- no display workload on the benchmark GPU if avoidable.

A smaller GPU may run G1 and small G2 cases but cannot establish memory or crossover claims.

## 2.2 Recommended single-GPU measurement profile

- one A100 80 GB, H100 80 GB, H200, or comparable datacentre GPU;
- 128 GB or more ECC host RAM;
- PCIe 4/5 x16 or SXM attachment;
- dedicated machine access;
- stable cooling and configurable power/clock policy;
- Ubuntu 24.04 LTS;
- local NVMe scratch.

Consumer GPUs are acceptable for development but power, thermals, display scheduling, and limited ECC
must be recorded. Results are hardware-specific; do not generalise across GPU classes.

## 2.3 Recommended multi-GPU profile

- 2, 4, or 8 identical GPUs;
- NVLink/NVSwitch preferred; PCIe-only runs remain useful but are a separate topology class;
- one MPI rank per GPU initially;
- NCCL matching the installed CUDA/driver stack;
- sufficient host memory for duplicated input/reference data and logging;
- CPU cores pinned per rank;
- known NUMA/GPU/PCIe topology.

Do not mix GPU models in primary scaling results. Mixed hardware may be an exploratory appendix only.

## 2.4 Machine exclusivity

During measured runs:

- no other CUDA processes;
- no automatic OS updates;
- no scheduled backups or indexing;
- no display compositor on the measured GPU where possible;
- no dynamic cloud instance migration;
- no thermal throttling;
- no MIG reconfiguration between runs;
- no power-limit change within a comparison campaign.

Record any unavoidable shared-machine condition in every affected run.

---

# 3. Operating-system and driver preparation

## 3.1 Recommended software stack

Use one locked stack for a complete primary campaign:

- Ubuntu 24.04 LTS;
- a production NVIDIA driver compatible with the chosen toolkit;
- CUDA toolkit 12.4 or newer;
- CMake 3.24 or newer;
- Ninja;
- GCC/G++ supported by the CUDA toolkit;
- Python 3.11 or 3.12;
- OpenMPI for distributed runs;
- NCCL runtime and development headers for distributed runs;
- Nsight Systems and Nsight Compute from the same CUDA/toolkit family;
- `compute-sanitizer`;
- `numactl`, `hwloc`, and standard Linux performance tools.

PDHCG's pinned upstream documentation requires NVIDIA CUDA 12.4+, CMake/GCC/NVCC, and MPI/NCCL for
its optional distributed build. Its Python interface is single-GPU; multi-GPU operation uses the
C++ executable under MPI.

## 3.2 Do not let the repository install the driver

Driver/toolkit installation is machine-administration work. Use NVIDIA's documented installation
method for the exact machine. The committed bootstrap script deliberately refuses to install or
replace the NVIDIA driver or CUDA toolkit.

After installation, reboot and verify:

```bash
nvidia-smi
nvcc --version
which nvcc
readlink -f "$(which nvcc)"
```

If several CUDA toolkits exist, select one explicitly:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Never rely on an old `/usr/bin/nvcc` when `CUDA_HOME` points elsewhere.

## 3.3 Record GPU mode before changing it

Run as an administrator if permitted:

```bash
nvidia-smi -L
nvidia-smi -q > machine-before.txt
nvidia-smi topo -m > topology-before.txt
```

Record:

- persistence mode;
- ECC mode;
- MIG mode;
- compute mode;
- power limit;
- application clocks, if supported;
- current/max graphics and memory clocks;
- temperatures;
- PCIe generation/width;
- BAR1 size;
- GPU UUIDs.

For a dedicated datacentre GPU, it is usually useful to enable persistence mode:

```bash
sudo nvidia-smi -pm 1
```

Do not change ECC or MIG merely to improve a benchmark. If either changes, reboot as required and
treat the new setting as a different hardware configuration.

## 3.4 Clock and power policy

Primary options, in descending preference:

1. lock supported application clocks for repeatability;
2. otherwise keep the default power limit and verify no throttling;
3. never compare solvers under different clock/power settings.

Inspect supported clocks:

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
```

If administrative policy permits, use model-appropriate supported values rather than copying a
clock from another GPU:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl <supported_power_limit_watts>
sudo nvidia-smi -lgc <minimum_graphics_clock>,<maximum_graphics_clock>
```

Save the commands and final `nvidia-smi -q` output. If clocks cannot be locked, record temperature,
power, and clock traces during every measured run.

## 3.5 NUMA and topology

Capture:

```bash
lscpu
numactl --hardware
nvidia-smi topo -m
lspci -tv
```

For multi-GPU runs, establish which CPU NUMA node is nearest each GPU and network interface. Begin
with one MPI rank pinned to the nearest CPU cores:

```bash
mpirun --map-by ppr:1:numa --bind-to core ...
```

Verify the actual binding with `--report-bindings`. Do not assume rank order equals GPU order; set
`CUDA_VISIBLE_DEVICES` or map ranks explicitly.

---

# 4. Repository checkout and immutable campaign commit

## 4.1 Clone and select a commit

```bash
git clone https://github.com/terrorproforma/spacecraft-trajectory-optmiser.git
cd spacecraft-trajectory-optmiser
git fetch --all --prune
git checkout main
git pull --ff-only
git status --short
git rev-parse HEAD
```

The status must be empty. Record the 40-character commit. Do not benchmark a moving branch.

For implementation work, create a narrow branch from current `main`:

```bash
git checkout -b feat/persistent-cuda-workspace
```

For locked evaluation, tag or record the exact evaluation commit after the implementation PR is
merged:

```bash
git checkout main
git pull --ff-only
git tag -a paper1-gpu-campaign-v1 -m "Paper 1 GPU campaign v1"
git push origin paper1-gpu-campaign-v1
```

Do not retag an existing campaign identifier.

## 4.2 Directory policy

Use:

```text
_upstream/                 ignored exact upstream checkouts
build/                     ignored build trees
results/gpu/<campaign>/    raw local evidence
artifacts/                 optional copied/archived evidence
```

Never commit large Nsight traces or raw benchmark outputs directly to Git unless they are tiny.
Commit manifests, compact summaries, schemas, and checksums. Store large artifacts in GitHub Actions,
a release, or a durable object store referenced by hash.

---

# 5. Bootstrap the machine

## 5.1 Conservative automated bootstrap

From a clean repository:

```bash
bash scripts/gpu/bootstrap_ubuntu.sh \
  --install-system \
  --cuda /usr/local/cuda-12.6
```

For later distributed work, after the NVIDIA repository providing NCCL is configured:

```bash
bash scripts/gpu/bootstrap_ubuntu.sh \
  --install-system \
  --install-distributed \
  --cuda /usr/local/cuda-12.6
```

The script:

- installs ordinary Ubuntu development packages only when asked;
- does not install the driver/toolkit;
- creates `.venv-gpu`;
- installs the native SpacePDHCG editable wheel;
- validates the packaged C ABI;
- configures/builds/tests the native host tree;
- checks out the exact PDHCG commit and tree;
- writes `results/gpu/environment.json`.

## 5.2 Manual equivalent

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip build
python -m pip install -e '.[dev]'

cmake -S cpp -B build/native-host -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSPACEPDHCG_BUILD_C_API=ON \
  -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
  -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON
cmake --build build/native-host --parallel
ctest --test-dir build/native-host --output-on-failure

bash scripts/gpu/checkout_pinned_pdhcg.sh
python scripts/gpu/verify_environment.py \
  --repository . \
  --output results/gpu/environment.json
```

## 5.3 Stop conditions

Do not proceed if:

- repository is dirty;
- native C++ tests fail;
- packaged C ABI fails to load;
- environment verifier reports CUDA below 12.4;
- no GPU is visible;
- compiler/toolkit combination is unsupported;
- pinned PDHCG commit/tree does not match the lock file;
- free disk space is insufficient.

---

# 6. Gate G0 — freeze host/native truth

Run the complete host gate before every new GPU campaign commit:

```bash
source .venv-gpu/bin/activate
python -m ruff check .
python -m pytest -q

cmake -S cpp -B build/native-gate -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSPACEPDHCG_BUILD_C_API=ON \
  -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
  -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON
cmake --build build/native-gate --parallel
ctest --test-dir build/native-gate --output-on-failure

cmake -S cpp -B build/native-sanitized -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSPACEPDHCG_BUILD_C_API=ON \
  -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
  -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
  -DSPACEPDHCG_NATIVE_ENABLE_SANITIZERS=ON
cmake --build build/native-sanitized --parallel
ctest --test-dir build/native-sanitized --output-on-failure
```

Build and inspect the native wheel:

```bash
rm -rf dist
python -m build --wheel
python -m pip install --force-reinstall dist/*.whl
python - <<'PY'
import spacepdhcg
from spacepdhcg.native import c_api_version, native_available, native_version
assert native_available()
assert c_api_version() == 1
assert native_version() == spacepdhcg.__version__
print(spacepdhcg.__version__)
PY
```

Archive:

- test logs;
- compiler and CMake versions;
- wheel filename and SHA-256;
- environment record;
- commit.

G0 passes only if all results are green.

---

# 7. Gate G1 — pinned upstream one-shot CUDA correctness

This closes GitHub issue #1. Do this before changing upstream internals.

## 7.1 Automated gate

```bash
source .venv-gpu/bin/activate
bash scripts/gpu/run_first_gate.sh \
  --cuda /usr/local/cuda-12.6 \
  --gpu 0 \
  --intervals "8 32 128" \
  --tolerance 1e-6
```

The script:

1. rejects a dirty repository;
2. records the machine and exact commit;
3. checks out the pinned upstream commit/tree;
4. builds its C++ executable;
5. installs its Python binding;
6. runs exact-optimum box-QP and SOC trajectory fixtures against CPU references;
7. seals the results with SHA-256 hashes and a reproducible archive.

## 7.2 Manual upstream build

```bash
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
bash scripts/gpu/checkout_pinned_pdhcg.sh

cmake -S _upstream/pdhcg -B build/pdhcg-one-shot -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPDHCG_COMPILE_DISTRIBUTED=OFF
cmake --build build/pdhcg-one-shot --clean-first --parallel
build/pdhcg-one-shot/pdhcg --help

export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$CUDACXX"
python -m pip install --no-deps --force-reinstall ./_upstream/pdhcg
python -c 'import pdhcg; print(pdhcg.__version__)'
```

The pinned public C API creates a `qp_problem_t`, accepts native Q/A/F matrices, bounds, variable and
affine cones, optional primal/dual starts, and returns duals ordered `[dual_A, dual_F]`. It is
one-shot; persistence is not inferred from allocator caching.

## 7.3 Correctness expansion

After the automated gate passes, add:

- HCW box QP, then HCW SOC;
- tolerances `1e-3`, `1e-4`, `1e-6`, `1e-8` where convergence is practical;
- cold, primal, and primal-dual starts;
- interval sizes `20`, `50`, `100`, `500`, then larger until the development GPU limit;
- repeated identical and small-update cases;
- objective, independent primal/dual/cone/gap residuals;
- terminal and dynamics checks.

Do not time the first CUDA context creation as steady-state solve time. Record it separately.

## 7.4 G1 acceptance

G1 passes when:

- every declared small reference case solves;
- PDHCG objective agrees with CPU reference within the locked quality rule;
- independently computed canonical and nonlinear residuals pass;
- both QP and SOC paths run on the selected GPU;
- upstream commit/tree and toolchain are archived;
- no CPU fallback is possible or silently used;
- run evidence is sealed.

If any case fails, stop. Diagnose formulation/API mapping before persistence work.

---

# 8. Implement the persistent CUDA workspace — contribution B

Work on a dedicated branch and PR:

```bash
git checkout main
git pull --ff-only
git checkout -b feat/persistent-cuda-workspace
```

## 8.1 Integration strategy

The pinned public PDHCG API is one-shot. Do not repeatedly call `create_qp_problem()` and
`solve_qp_problem()` and label the result persistent. The persistent implementation must retain
processed problem data, CUDA descriptors, scaling/preconditioner state, solver iterates, and device
buffers.

Use one of these explicit strategies:

### Preferred: minimal pinned upstream patch

1. keep `_upstream/pdhcg` as an ignored exact checkout;
2. add deterministic patch files under `third_party/patches/pdhcg/`;
3. apply patches only after verifying the locked commit/tree;
4. expose lower-level lifecycle functions without changing numerical kernels unnecessarily;
5. record patch SHA-256 in the build/run manifest.

### Acceptable: linked internal adapter

Compile selected pinned upstream source files into a SpacePDHCG CUDA target and expose internal
lifecycle functions from a namespaced adapter. This is acceptable only if every imported source path
and commit remains pinned and licence notices are preserved.

Do not copy an untracked snapshot of upstream source into the repository.

## 8.2 Upstream code map to inspect

At the pinned revision, inspect at least:

```text
include/pdhcg.h                 public create/solve/start/free API
include/pdhcg_types.h           problem, result, parameter, cone types
src/pdhcg.c                     top-level one-shot orchestration
src/pdhg_core_op.cu             device iteration/state operations
src/preconditioner.c            scaling/preconditioner creation and transforms
src/cone_dispatch.cu            cone projections
src/permute.cu                  row/column permutation
src/partition_utils.c           partition helpers
distributed/distributed_solver.cu
distributed/distributed_utils.cu
distributed/distributed_conic.cu
```

Build a call graph from `solve_qp_problem()` downward and mark:

- host allocations;
- device allocations;
- H2D/D2H copies;
- descriptor creation/destruction;
- preprocessing/scaling;
- solver-state initialization;
- iteration loop;
- residual calculation;
- result extraction;
- cleanup.

Commit this map as `docs/PDHCG_PERSISTENCE_CODE_MAP.md` before modifying the implementation.

## 8.3 Native directory layout

Recommended:

```text
cpp/cuda/
├── CMakeLists.txt
├── include/spacepdhcg/cuda/
│   ├── persistent_pdhcg.hpp
│   ├── persistent_pdhcg_c_api.h
│   ├── device_buffers.hpp
│   ├── stream_event.hpp
│   ├── allocation_ledger.hpp
│   └── upstream_adapter.hpp
├── src/
│   ├── persistent_pdhcg.cu
│   ├── upstream_adapter.cu
│   ├── coefficient_kernels.cu
│   ├── residual_kernels.cu
│   └── c_api.cpp
└── tests/
    ├── persistent_cw_test.cu
    ├── persistent_soc_test.cu
    ├── pointer_contract_test.cu
    ├── allocation_lifecycle_test.cu
    └── stream_lifetime_test.cu
```

Keep generic native C++ headers in `cpp/include`. CUDA implementation details stay in `cpp/cuda`.

## 8.4 Workspace ownership

The concrete workspace should own or retain:

### Immutable topology

- dimensions and cone inventory;
- Q/A/F CSC offsets and row indices;
- topology fingerprint;
- permutations and inverse permutations;
- cuSPARSE sparse-matrix descriptors;
- any fixed cone metadata and projection workspaces;
- logical scenario/time partition metadata when applicable.

### Mutable numerical data

- Q/A/F values;
- objective;
- scalar and variable bounds;
- affine-cone offsets;
- exact-penalty/trust/scenario coefficients.

### Solver state

- current and averaged primal/dual iterates;
- previous iterates used for restart or residuals;
- step sizes and restart counters;
- scaling vectors and refresh metadata;
- proximal/Krylov state needed for quadratic steps;
- residual scratch;
- reduction scratch;
- result/status storage.

### CUDA resources

- device id;
- consumer stream or owned stream;
- completion/error events;
- cuBLAS/cuSPARSE/cuSOLVER handles bound to the stream;
- CUDA memory pool or allocator policy;
- allocation ledger;
- optional captured CUDA graph and invalidation epoch.

### External lifetime

- retained DLPack managed-tensor borrows when zero-copy views are used;
- checkpoint metadata;
- cancellation and in-flight state.

## 8.5 Lifecycle state machine

Implement explicit states:

```text
uninitialised
created
values_updated
warm_started
solving
solved
failed
cancelled
destroyed
```

Rules:

- topology may be supplied only at creation;
- values cannot update while a solve is in flight unless double-buffering is explicitly implemented;
- a successful value update invalidates the previous solution but retains eligible warm state;
- a topology fingerprint change is a hard error;
- destroy waits for or safely cancels in-flight operations;
- every failure records CUDA/upstream error and moves to a defined state;
- reset distinguishes iterate reset, scaling reset, and full workspace reset.

## 8.6 C/C++ API

The implementation must follow `docs/ACCELERATOR_INTEROP.md`. Minimum operations:

```text
create from fixed topology + accelerator exchange
update numerical values asynchronously
apply/replace primal-dual warm start asynchronously
solve asynchronously with requested tolerance and iteration limit
query/wait completion event
retrieve compact diagnostics
checkpoint retained state
restore compatible checkpoint
reset iterates
refresh scaling
cancel where supported
destroy
```

Each call returns an explicit status; no exception may cross the C ABI.

## 8.7 Creation sequence

On workspace creation:

1. set CUDA device;
2. validate exchange ABI and topology fingerprint;
3. validate every DLPack/buffer view, pointer attribute, dtype, length, stride, access, and storage
   device;
4. establish producer-to-consumer stream dependencies;
5. allocate owned topology only if borrowing is not selected;
6. upload/capture Q/A/F indices exactly once;
7. copy/store cone descriptors;
8. build cuSPARSE descriptors;
9. perform fixed permutations/partition preprocessing;
10. allocate all numerical, iterate, residual, and cone scratch buffers;
11. compute initial scaling/preconditioner;
12. initialise solver state;
13. record creation completion event;
14. record allocation counts and bytes by category;
15. return an opaque workspace handle.

After this point, the topology allocation count must remain unchanged.

## 8.8 Numerical update sequence

A values-only update:

1. verifies the topology fingerprint;
2. waits on producer readiness in the consumer stream;
3. validates replaceable views;
4. copies values with `cudaMemcpyAsync` or accepts validated same-device zero-copy views;
5. updates only descriptors whose value pointer changes;
6. computes coefficient-change metrics;
7. decides scaling reuse/refresh using the committed policy;
8. preserves or invalidates graph capture according to pointer/shape stability;
9. records completion event and update epoch;
10. records bytes, time, and allocation delta.

There must be no post-create topology allocation or index copy.

## 8.9 Warm starts

Support:

- no warm start;
- primal only;
- primal and dual;
- full retained internal state when the prior solve is from the same workspace.

Validate dimensions and finiteness. Project or repair a warm start only when the operation is
mathematically defined and recorded. Report whether the solver accepted, modified, or discarded it.

## 8.10 Scaling reuse and refresh

Implement the three locked modes:

- `always_refresh`;
- `reuse` with a bounded reuse budget;
- `refresh_if_needed` using coefficient/bound change metrics and residual behaviour.

Record:

- coefficient-change norm/maximum;
- scaling min/max;
- refresh reason;
- refresh time;
- reuse count;
- whether retained iterates were rescaled consistently.

A scaling refresh must not silently destroy warm starts.

## 8.11 Asynchronous solve

The solve call:

1. waits on value/warm-start events in the consumer stream;
2. updates requested tolerance/iteration limit without reconstructing topology;
3. runs the PDHCG outer/inner iteration loop;
4. periodically evaluates stopping/restart quantities on device;
5. avoids host synchronization on every iteration;
6. records compact final diagnostics to pinned host memory only after a completion event;
7. retains final iterates and reusable internal state;
8. records work counters and timing components;
9. returns a future/event handle.

The host SCvx driver may wait once per CQP acceptance decision. It must not cause hidden
`cudaDeviceSynchronize()` calls inside the inner loop.

## 8.12 Independent residuals

Implement canonical residual calculation independently of upstream termination status:

- scalar-row primal violation;
- box violation;
- affine cone distance;
- stationarity/dual residual;
- complementarity/natural residual;
- objective/gap where defined;
- scaled and unscaled norms.

Use the residual in `docs/INEXACT_SCVX_THEORY.md`. Compare against CPU residuals on every small gate.

## 8.13 Allocation ledger

Wrap every SpacePDHCG-owned CUDA allocation/free and record:

- category;
- bytes;
- pointer;
- creation epoch;
- free epoch;
- peak active bytes;
- peak reserved bytes if a pool is used.

During debug runs, intercept or instrument upstream allocations sufficiently to determine whether
one-shot internals still allocate per solve. H1 cannot pass if hidden topology allocations remain.

## 8.14 G2 correctness tests

For every small QP/SOCP reference:

1. create once;
2. solve cold;
3. update values ten times;
4. solve with no/primal/primal-dual warm starts;
5. compare every result to one-shot PDHCG and CPU references;
6. assert fingerprint unchanged;
7. assert post-create topology allocation/copy count zero;
8. assert device pointers remain stable where promised;
9. checkpoint, destroy, restore, and reproduce a solve;
10. run with default and non-default CUDA streams;
11. run with copied CUDA storage and managed storage;
12. run DLPack inputs from CuPy/PyTorch/JAX-compatible producers;
13. test premature producer release and reject/use safe retained ownership;
14. test cancellation/destruction ordering.

## 8.15 Compute Sanitizer

Run debug-size tests:

```bash
compute-sanitizer --tool memcheck --leak-check full \
  ./build/cuda-tests/persistent_cw_test

compute-sanitizer --tool racecheck \
  ./build/cuda-tests/stream_lifetime_test

compute-sanitizer --tool initcheck --track-unused-memory \
  ./build/cuda-tests/persistent_soc_test

compute-sanitizer --tool synccheck \
  ./build/cuda-tests/persistent_cw_test
```

Archive complete logs. Zero errors are required before performance measurement.

## 8.16 G2 acceptance

G2 passes when:

- persistent results equal one-shot/CPU references at matched quality;
- topology allocations and index copies are zero after creation;
- update uses the declared stream;
- warm starts are retained correctly;
- independent residuals pass;
- sanitizer tests are clean;
- no hidden CPU fallback;
- every lifecycle failure path is tested;
- issue #2's implementation criteria are met, although H1 performance remains for G3.

---

# 9. Device-side coefficient generation and variational RK4

The production CPU code now has analytic variational RK4 for 3-DoF, 6-DoF, and low-thrust models.
Port those equations, not the finite-difference reference, to CUDA.

## 9.1 Kernel decomposition

Recommended first implementation:

- one CUDA block or cooperative group per trajectory interval;
- one thread group computes nonlinear flow and RK4 stage states;
- threads cooperatively compute `f_x`, `f_u`, `Phi`, and `Gamma` entries;
- fixed CSC value positions are compiled or precomputed lookup arrays;
- direct writes into Q/A/F value buffers;
- separate small kernels for bounds/objective/cone offsets;
- one reduction for coefficient-change metrics;
- no dynamic allocation inside kernels.

For scenario bundles, launch over `(scenario, interval)`.

## 9.2 6-DoF quaternion rule

After the raw RK4 augmented step, apply

```text
J_N(q) = (I - q_hat q_hat^T) / ||q||
```

to quaternion output rows of `Phi` and `Gamma`, exactly matching the native CPU implementation.
Check radial tangent components against zero.

## 9.3 Correctness tests

For random admissible state/control samples:

- CPU variational vs GPU variational matrices;
- GPU variational vs independent finite-difference RK4 reference;
- affine reference reproduction;
- quaternion radial tangency;
- physical-boundary cases such as zero low-thrust sigma;
- repeated deterministic output under the locked deterministic mode.

Use double precision first. Mixed/float precision is a later exploratory campaign.

## 9.4 Performance evidence

Record separately:

- state propagation;
- Jacobian/variational integration;
- sparse coefficient writes;
- bounds/cone update;
- change metric;
- total coefficient generation.

Compare against CPU coefficient generation and host-to-device copy, but do not treat this secondary
result as H1/H2 by itself.

---

# 10. Gate G3 — device-resident deterministic SCvx

Connect the existing native outer drivers to the persistent CUDA backend.

## 10.1 Initial order

1. HCW fixed QP/SOCP update loop;
2. 3-DoF powered descent;
3. 6-DoF powered descent;
4. long-horizon low thrust.

Do not begin with 6-DoF.

## 10.2 Outer iteration dataflow

The steady-state outer loop should be:

```text
reference trajectory already on GPU
→ variational coefficient kernels
→ values-only workspace update
→ retained primal-dual/internal warm start
→ PDHCG solve
→ canonical residual kernel
→ nonlinear replay and dense path check
→ compact merit/acceptance data copied to host
→ host or device trust/forcing decision
→ next outer iteration
```

A first implementation may keep the trust decision on the host because it transfers only compact
scalars. Full trajectory/matrix transfers are not allowed in the steady-state region.

## 10.3 Acceptance parity

For small cases, run CPU and GPU outer drivers from the same initial reference. Compare:

- accepted/rejected iteration sequence;
- requested/achieved inner residuals;
- trust radius;
- predicted/actual reduction;
- virtual control;
- objective;
- final state/control trajectory;
- nonlinear dynamics/path/terminal residuals.

Exact iteration parity is desirable but not mandatory if both satisfy the locked result gate.
Unexpected divergence must be explained before performance runs.

## 10.4 H1 measurement

Measure three regimes:

- cold one-shot;
- first solve after persistent creation;
- steady-state same-topology updates.

Instrument:

```text
T_topo
T_coeff
T_create
T_update
T_scaling
T_h2d
T_solve
T_residual
T_replay
T_acceptance
T_d2h
T_CQP
T_SCvx
```

Record allocation/copy counts and bytes. Evaluate H1 exactly according to
`papers/paper1/CLAIMS_AND_DECISION_RULES.md`.

## 10.5 Nsight Systems

Profile correctness-sized and medium cases, not every benchmark repeat:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cusparse \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output results/gpu/profiles/persistent-pd3 \
  <command>
```

Add NVTX ranges for:

- topology creation;
- coefficient generation;
- numerical update;
- scaling refresh;
- inner solve;
- residual;
- nonlinear replay;
- collective;
- D2H diagnostics.

Inspect for:

- device-wide synchronization;
- repeated allocation;
- repeated descriptor creation;
- large H2D trajectory/matrix copies;
- launch gaps;
- serial kernels that should overlap;
- context initialization included in steady-state timing.

## 10.6 Nsight Compute

Profile representative kernels:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --export results/gpu/profiles/pdhcg-kernels \
  <command>
```

Use focused metric sets for final runs to reduce perturbation. Check:

- achieved occupancy;
- memory bandwidth;
- cache hit rates;
- sparse operation efficiency;
- branch divergence in cone projections;
- register pressure;
- launch dimensions;
- precision throughput.

Never use profiled timings as primary benchmark timings.

## 10.7 G3 acceptance

G3 passes when:

- all deterministic families meet quality gates;
- steady-state data residency is demonstrated by traces/counters;
- H1 is supported, mixed with a scale boundary, or honestly rejected;
- result schema records validate;
- no sanitizer or lifetime errors;
- no post-create topology allocation;
- no full trajectory/CQP host roundtrip in steady state.

---

# 11. Gate G4 — contribution D: adaptive and hybrid experiments

## 11.1 Freeze policies before evaluation

The primary policies are:

- `fixed-tight`;
- `fixed-loose`;
- `adaptive`;
- `adaptive+polish`;
- pure GPU IPM;
- `hybrid-pdhcg-ipm`.

Tuning data and evaluation data must be separate. After choosing global policy parameters, commit them
and rerun evaluation from a clean commit.

## 11.2 Required policy logic

For each outer iteration record:

- phase;
- requested tolerance;
- achieved canonical residual;
- backend-native residuals;
- inner work;
- re-solve flag;
- trust action;
- predicted and actual reduction;
- scaling mode/refresh;
- warm-start mode;
- final polish handoff.

If a rejected step is under-solved, re-solve the identical CQP before trust shrink. Verify that the
CQP values/fingerprint are identical across the re-solve.

## 11.3 GPU IPM baselines

Build each available comparison from a pinned source revision. Create lock files analogous to
`third_party/pdhcg.lock.json` before primary evaluation. Record:

- source commit;
- CUDA/toolchain;
- compile flags;
- cone support;
- quadratic-objective representation;
- precision;
- linear solver/factorisation options;
- warm-start capability;
- failure/OOM behaviour.

Do not force an unsupported problem through a mathematically different relaxation without labelling
it. If an objective or cone must be epigraphed, report both the original and transformed dimensions.

## 11.4 Hybrid handoff

The hybrid sequence is:

1. persistent PDHCG builds a qualified candidate at adaptive tolerance;
2. convert primal and dual to the IPM convention;
3. preserve variable/row/cone ordering or apply an audited permutation;
4. qualify warm start before solve;
5. run the final IPM on the same final convex model;
6. independently recompute residuals;
7. nonlinear replay the polished trajectory;
8. include conversion/setup/polish in total time.

A hybrid run fails if the warm-start conversion is inconsistent even when the IPM eventually solves
from scratch.

## 11.5 H5 and H6 matrix

Use nonlinear families P1-C, P1-D, and P1-E. Sweep:

- intervals from the locked benchmark matrix;
- conditioning bins;
- cold/primal/primal-dual starts;
- requested final quality tiers;
- accepted initial dispersion classes.

Use at least five measured repeats and the randomised instance count in the benchmark matrix. Apply
H5/H6 decision thresholds without alteration.

## 11.6 Energy measurement

Preferred: NVML sampling at 10–20 Hz or higher, integrating power over the measured region. At
minimum:

```bash
nvidia-smi \
  --query-gpu=timestamp,index,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate \
  --format=csv,noheader,nounits \
  --loop-ms=50 \
  > power.csv &
POWER_PID=$!
<benchmark command>
kill "$POWER_PID"
```

Also record CPU/system energy where available; otherwise label GPU-only energy. Subtracting idle
power is exploratory unless the methodology is locked in advance. Report sampling gaps.

## 11.7 G4 acceptance

G4 passes when:

- every policy result validates against the compact schema;
- H5 and H6 are supported/rejected/mixed under preregistered rules;
- matched-quality comparison is enforced;
- final theorem diagnostics are present;
- floor-dominated or CT-error-dominated cases are flagged;
- negative results remain in the dataset.

---

# 12. Gate G5 — contribution C: scenario-aware multi-GPU

Only start after G2/G3 single-GPU correctness.

## 12.1 Build prerequisites

Verify:

```bash
nvidia-smi -L
nvidia-smi topo -m
mpirun --version
ldconfig -p | grep -i nccl
```

Build pinned upstream distributed target:

```bash
cmake -S _upstream/pdhcg -B build/pdhcg-distributed -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPDHCG_COMPILE_DISTRIBUTED=ON
cmake --build build/pdhcg-distributed --clean-first --parallel
```

Start with one node. Multi-node work is a separate campaign with network topology and NCCL transport
configuration recorded.

## 12.2 Process/device model

Initial model:

- one MPI rank per GPU;
- rank-local CUDA context and stream;
- scenario ownership fixed for one solve;
- local CQP data and iterates reside on the owning GPU;
- shared control/risk arrowhead represented explicitly;
- NCCL communicator created once;
- MPI used for process/bootstrap/control and NCCL for GPU reductions.

Avoid multi-threaded multi-GPU ownership in the first implementation. It complicates context and
lifetime diagnosis.

## 12.3 Scenario-aware partition

Use the committed deterministic whole-scenario partition as the default. Weight scenarios by a
measured or modelled cost including:

- local Q/A/F nonzeros;
- cone slots/type cost;
- time nodes;
- nonlinear replay work;
- risk augmentation;
- expected coefficient-update work.

Record predicted and measured load. Compare against generic nonzero-balanced upstream partitioning on
the identical global CQP.

## 12.4 Distributed algebra

For the block-arrow CQP, each GPU computes:

- local Q product;
- local A/F products;
- local cone projections;
- local transpose contributions;
- local residual components;
- local scenario objective/risk contributions.

Collectives combine only shared-arrowhead quantities and global norms/statistics. Implement the
minimum sufficient reductions rather than all-reducing full local vectors.

Document each collective:

| Collective | Payload | Frequency | Mathematical purpose |
|---|---:|---:|---|
| shared primal/gradient reduction | arrowhead dimension | per relevant iteration | non-anticipativity coupling |
| residual norm reduction | few scalars | check interval | stopping/restart |
| risk aggregate | scenarios or compact partials | outer/CQP iteration as needed | expected/worst/CVaR |
| global work/status | few scalars | solve end | diagnostics |

Do not use an implicit host gather for a claimed GPU-resident reduction.

## 12.5 NCCL stream ordering

Create dedicated compute and collective streams only after the one-stream implementation is correct.
When overlapping:

1. local kernels record an event when shared contributions are ready;
2. collective stream waits on that event;
3. NCCL collective runs;
4. collective stream records completion;
5. dependent compute waits on completion;
6. scenario-local independent work overlaps where mathematically valid.

Validate with `compute-sanitizer --tool racecheck` and Nsight Systems.

## 12.6 CVaR/worst-case semantics

For robust risk:

- expected risk reduces weighted partial sums;
- worst-case reduces maxima with deterministic tie handling;
- CVaR maintains threshold/excess variables in the declared CQP; do not substitute a host-only
  postprocessing calculation for the optimization epigraph;
- risk duals and epigraph residuals participate in the canonical stopping test.

## 12.7 Determinism

Primary correctness mode:

- fixed scenario ordering and partition;
- fixed rank/GPU mapping;
- deterministic seeds;
- deterministic reduction option where practical;
- float64;
- no asynchronous nondeterministic host update.

Performance mode may use faster NCCL reductions. Report solution variability across repeats and
compare to deterministic mode.

## 12.8 Strong scaling

Fix global `(N,S)` and run `G=1,2,4,8` where hardware permits. Record:

- total time;
- local compute;
- exposed collective time;
- overlap estimate;
- bytes/count;
- load imbalance;
- peak memory per GPU;
- speedup;
- efficiency;
- nonlinear quality.

Do not report efficiency without the one-GPU result on the same multi-GPU machine/configuration.

## 12.9 Weak scaling

Keep scenario/nonzero workload per GPU approximately constant while increasing `G`. Record the same
metrics plus throughput. State exactly how global `S` and problem data change.

## 12.10 H2/H3/H4

- H2: compare factorisation-free compute crossover to qualified GPU IPM baselines.
- H3: grow until memory crossover or all solvers fail; preserve OOM evidence.
- H4: compare scenario-aware against generic partitioning under matched global CQP and topology.

Apply the locked thresholds. Do not change them after viewing results.

## 12.11 Multi-GPU launch template

Example single-node launch; adjust CPU binding to actual topology:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mpirun -np 4 \
  --map-by ppr:1:numa \
  --bind-to core \
  --report-bindings \
  -x CUDA_DEVICE_ORDER \
  -x NCCL_DEBUG \
  -x NCCL_DEBUG_SUBSYS \
  <spacepdhcg distributed command>
```

If rank-to-GPU mapping is not one-to-one under this command, implement an explicit mapping and archive
it. For PCIe-only systems, consider `NCCL_P2P_DISABLE` only as an exploratory diagnostic; never alter
it between primary compared methods.

## 12.12 G5 acceptance

G5 passes when:

- 1-GPU distributed path matches monolithic single-GPU/CPU truth;
- every multi-GPU result passes canonical/nonlinear quality;
- non-anticipativity and risk residuals pass;
- communicator and device ownership are persistent;
- sanitizer and race checks are clean;
- strong/weak scaling data are complete, including failures;
- H2/H3/H4 are resolved or honestly left censored;
- issue #4 acceptance criteria are met.

---

# 13. Benchmark execution discipline

## 13.1 Warm-up and repeats

Default locked values:

- two warm-up runs;
- seven measured deterministic repeats;
- twenty random instances per randomised coordinate;
- same execution order policy across solvers.

Randomise or rotate solver order to reduce temporal bias, while retaining paired instances. Record
order.

## 13.2 Synchronisation boundaries

For measured asynchronous CUDA regions:

- record a CUDA event before and after the region on the relevant stream;
- synchronize only the ending event outside the region;
- use host monotonic time for complete end-to-end duration;
- report both event and host time;
- do not rely on unsynchronised host timers around kernel launches.

## 13.3 Cold versus warm

Define:

- cold: new process/context/workspace;
- first persistent: existing process, newly created workspace;
- warm: same workspace, new numerical values;
- repeated-identical: same values, retained state;
- primal warm start;
- primal-dual warm start;
- full internal-state continuation.

Never label CUDA context reuse as solver warm start.

## 13.4 Timeout and OOM

Set declared timeouts by family/scale. On timeout:

- terminate cleanly where possible;
- capture last progress/residual;
- preserve logs and partial traces;
- label `timeout`, not `failed`.

On OOM:

- record requested size;
- capture CUDA error and `nvidia-smi` memory state;
- record active/reserved memory before failure;
- label `oom`;
- do not silently reduce size or switch precision.

## 13.5 Quality first

A fast result that fails canonical or nonlinear gates is unqualified. Preserve its timing but exclude
it from winner/Pareto claims.

---

# 14. Evidence directory and sealing

Use one directory per run:

```text
results/gpu/<campaign>/<run-id>/
├── environment.json
├── run-manifest.json
├── paper1-result.json
├── command.txt
├── stdout.log
├── stderr.log
├── raw.json
├── nvidia-smi-before.txt
├── nvidia-smi-after.txt
├── power.csv
├── profile.nsys-rep
├── profile.ncu-rep
├── sanitizer.log
└── evidence-index.json
```

Not every run needs profiles, but the schema uses explicit null/missing optional artifacts rather
than invented paths.

Seal a run:

```bash
python scripts/gpu/archive_run.py \
  results/gpu/<campaign>/<run-id> \
  --repository . \
  --require-clean-repository \
  --archive results/gpu/<campaign>/<run-id>.tar.gz
```

Copy the archive and printed SHA-256 to durable storage. Never edit a sealed run. If metadata is
wrong, create a new run ID.

---

# 15. Paper 1 aggregation

## 15.1 Validate compact results

Every primary result must pass:

```python
from spacepdhcg.experiments import read_paper1_result
read_paper1_result("paper1-result.json")
```

The validator rejects:

- unknown family/solver IDs;
- incomplete qualified residuals;
- fewer than five deterministic repeats;
- inconsistent GPU solver metadata;
- invalid artifact hashes;
- missing persistent allocation evidence;
- malformed quantiles;
- non-finite values.

## 15.2 Generate figures only from validated summaries

Implement figure scripts under `papers/paper1/figures/` after data exist. Each script:

1. loads validated compact results;
2. records the exact filter query and run IDs;
3. retains failure/censoring records;
4. applies the aggregation in `FIGURE_SCHEMA.md`;
5. writes source JSON plus PDF/PNG;
6. never contains manually typed numerical coordinates.

## 15.3 Apply preregistered decisions

Resolve H1–H6 using `CLAIMS_AND_DECISION_RULES.md`. Store one machine-readable decision record per
hypothesis containing:

- input run IDs;
- comparison coordinates;
- practical threshold;
- bootstrap method/seed;
- point estimate and confidence interval;
- supported/rejected/mixed/unresolved;
- censored coordinates;
- notes.

Negative/mixed results belong in Table T08 and the main discussion.

---

# 16. Pull-request and issue workflow

Use one PR per major gate:

1. `feat/persistent-cuda-workspace` — G2 implementation;
2. `feat/device-scvx` — coefficient kernels and G3 integration;
3. `exp/adaptive-hybrid-gpu` — G4 harness and locked configs;
4. `feat/scenario-aware-multigpu` — G5 implementation;
5. `results/paper1-gpu-campaign-v1` — compact summaries/figures/manuscript only.

Each implementation PR must include:

- CPU/native gates;
- CUDA correctness tests on the local machine or self-hosted runner;
- sanitizer evidence hashes;
- no unqualified speed claim;
- updated status/issue body;
- exact upstream/patch lock.

Issue closure:

- #1 closes after G1 evidence is archived;
- #2 closes after G2 correctness/lifecycle and H1 instrumentation are implemented; if project
  convention requires H1 result in the same issue, close after G3 H1 evaluation;
- #3 closes only after theorem assumptions are checked and H5/H6 experiments resolve;
- #4 closes after G5 correctness and H2/H3/H4 scaling evidence.

Do not close an issue merely because code compiles.

---

# 17. First-day command sequence

On a prepared Ubuntu GPU machine:

```bash
# 1. Clean canonical checkout
git clone https://github.com/terrorproforma/spacecraft-trajectory-optmiser.git
cd spacecraft-trajectory-optmiser
git checkout main
git pull --ff-only
git status --short

# 2. Select CUDA explicitly
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 3. Bootstrap and record
bash scripts/gpu/bootstrap_ubuntu.sh \
  --install-system \
  --cuda "$CUDA_HOME"
source .venv-gpu/bin/activate

# 4. Run G0 again from the chosen campaign commit
python -m ruff check .
python -m pytest -q
ctest --test-dir build/native-host --output-on-failure

# 5. Run G1
bash scripts/gpu/run_first_gate.sh \
  --cuda "$CUDA_HOME" \
  --gpu 0 \
  --intervals "8 32 128" \
  --tolerance 1e-6

# 6. Inspect and copy the sealed archive
find results/gpu/first-gate -maxdepth 2 -type f -print
```

Do not start persistent implementation until the G1 JSON and independent checks are clean.

---

# 18. Troubleshooting decision tree

## 18.1 `nvcc`/driver mismatch

Symptoms: unsupported PTX, runtime driver error, build selecting wrong CUDA.

Actions:

1. compare `nvidia-smi` driver and `nvcc --version`;
2. inspect `which nvcc` and `CUDACXX`;
3. delete the affected build tree;
4. configure with explicit `CMAKE_CUDA_COMPILER`;
5. record the corrected stack as a new environment artifact.

## 18.2 Upstream Python build finds wrong CUDA

Set both:

```bash
export CUDACXX="$CUDA_HOME/bin/nvcc"
export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$CUDACXX"
```

Reinstall with a clean pip/build cache if necessary. Verify actual compile commands.

## 18.3 One-shot objective passes but independent residual fails

Likely causes:

- row/cone ordering mismatch;
- affine offset sign;
- lower/upper bound mapping;
- SOC slot ordering;
- dual convention;
- scaling/unscaling mismatch;
- result copied before completion.

Compare smallest exact fixture and dump canonical Q/A/F/bounds/cones from both paths. Do not loosen
tolerance to hide the mismatch.

## 18.4 Persistent result differs from one-shot

Disable in order:

1. warm start;
2. scaling reuse;
3. zero-copy views;
4. graph capture;
5. asynchronous overlap;
6. device-side coefficient fill.

At each step compare buffer hashes and residuals. The first feature whose removal restores parity
localises the defect.

## 18.5 Hidden allocations remain

Use allocation ledger plus Nsight Systems. Common sources:

- upstream solve-level scratch creation;
- cuSPARSE descriptor recreation;
- cone workspace size query followed by allocation;
- CUB temporary storage;
- scaling/permutation rebuild;
- framework DLPack materialisation;
- graph recapture;
- result conversion.

Move storage-size queries to creation and retain maximum required scratch.

## 18.6 Multi-GPU hang

Check:

- every rank enters collectives in identical order;
- rank/GPU mapping;
- NCCL communicator creation errors;
- stream event cycles;
- MPI progress and thread level;
- peer access;
- network interface selection;
- one rank failed before a later collective.

Run two GPUs with `NCCL_DEBUG=INFO`, reduce to one collective, and impose per-rank logs. Never debug an
8-GPU full SCvx run first.

## 18.7 Thermal or clock variability

Inspect power trace, throttling reasons, temperature, and pstate. If the machine reaches a different
steady thermal state across solvers, increase warm-up and rotate order. Do not discard slow repeats
without a preregistered hardware-fault rule.

## 18.8 IPM OOM but PDHCG succeeds

Verify:

- same mathematical problem and precision;
- transformed objective/cones recorded;
- IPM memory options at documented best feasible settings;
- no arbitrary memory cap;
- complete process/device peak memory;
- valid IPM solution before failure boundary.

Then retain OOM as H3 evidence.

---

# 19. Final Paper 1 freeze

Paper 1 is ready to freeze only when:

- all G0–G5 applicable gates are complete;
- H1–H6 have decision records;
- all primary compact results validate;
- all figures/tables follow the frozen schemas;
- failure/censored data are included;
- the theorem assumptions are discussed against observed diagnostics;
- no claim exceeds the tested hardware/regime;
- exact commits, patches, toolchains, and artifact hashes are recorded;
- manuscript text distinguishes implemented architecture, measured result, and inference.

Create a results PR from a clean branch and require all ordinary CI plus compact-result validation.
After merge, tag the paper state and archive the tag, repository bundle, and evidence indices.

---

# 20. OrbitWeaver after Paper 1

Once the persistent/multi-GPU trajectory oracle is stable:

1. replace the host OrbitWeaver continuous arc adapters with the persistent GPU backend;
2. batch arcs by compatible topology/fidelity;
3. allocate route candidates across a candidate axis and scenarios across a scenario axis;
4. reuse warm-start tokens between coarse, refined, robust, and certified stages;
5. integrate arc lower bounds with column generation and dynamic discretisation;
6. run route × scenario throughput experiments from `benchmarks/paper2_matrix.json`;
7. retain high-fidelity certification independently of the optimization backend;
8. report how GPU arc throughput changes the tractable routing frontier, not merely arc-kernel speed.

This begins Paper 2. It does not alter Paper 1's preregistered solver claims.
