# Gate G5 Physical Multi-GPU Execution Runbook

## Scope and claim boundary

This runbook prepares and operates the physical 2/4/8-GPU validation campaign. The tooling was
built and logically tested on a one-GPU WSL host. No 2/4/8-GPU result, scaling efficiency, energy
measurement, or Gate G5 acceptance exists until the physical checklist below is completed on the
target systems and the sealed evidence is reviewed.

`g5_physical_validation_harness` validates launch, rank/device ownership, MPI/NCCL communication,
telemetry, checkpoint guards, cancellation, and explicit failure paths. It is intentionally marked
`launch-and-collective-harness-only`; it does not make trajectory-quality or scaling claims. Full
P1-F campaign execution must use the integrated robust-scenario solver executable implementing the
same command and rank-telemetry contracts.

## External prerequisites

### Hardware and scheduler

- One scheduler-exclusive Linux node with exactly the requested minimum of 2, 4, or 8 visible GPUs.
- All primary-campaign GPUs have identical model, memory capacity, driver, ECC state, compute mode,
  and clock/power policy. MIG is disabled.
- No GPU compute processes exist before preflight. At least 90% of every GPU's memory is free.
- NVLink, NVSwitch, or PCIe connectivity is exposed by `nvidia-smi topo -m`; the actual topology is
  evidence, not inferred from the model name.
- GPU-local NUMA CPU affinities are available. The node has enough non-overlapping physical cores
  to bind one CPU set per rank.
- A stable non-loopback NIC is up. For multi-node extensions, the fabric and selected NCCL network
  plugin must be homogeneous and separately qualified; this runbook's primary matrix is one node.
- Wall power or BMC/DCGM access is required before energy can be a campaign result. `nvidia-smi`
  board-power sampling is retained but is not a substitute for preregistered external measurement.

### Administrator policy

- The scheduler allocation must be exclusive and prevent GPU sharing for the complete run.
- Persistence mode and an approved fixed application-clock/power policy must be applied before
  preflight, then left unchanged. Record the administrative command log.
- ECC counters are clean or any pre-existing counters are recorded and approved.
- NTP/PTP time synchronisation is healthy.
- `ulimit -l`, open-file limits, shared memory, and process limits accommodate NCCL/OpenMPI.
- No MPS, MIG, container GPU oversubscription, fake ranks, or remapped virtual GPUs are permitted.
- A watchdog or scheduler timeout must preserve the run directory after timeout or rank loss.

### Software and storage

- NVIDIA driver compatible with CUDA 12.8; do not install a Linux driver through this repository.
- CUDA toolkit 12.8, OpenMPI 4.1.x, and matching CUDA-12 NCCL runtime/development packages.
- CMake 3.24 or newer, Ninja, GCC/G++, Python 3.11 or newer, `numactl`, `iproute2`, and Git.
- Pinned PDHCG commit `167c8b72b4b96d2f94d405b8763e485514192b81`, tree
  `62b05e6c1bedd385f6c267af3645ae4aae0421b4`.
- A local POSIX filesystem with atomic rename and at least 100 GiB free for logs/samples, plus
  independent durable archive storage. Do not execute directly into object-store/FUSE paths.
- SHA-256 verification must be supported at both local and archive destinations.

## Build and preflight

Start from a clean, reviewed commit. Build before preflight so the exact CMake cache and harness
binary are included in the topology fingerprint.

```bash
bash scripts/gpu/checkout_pinned_pdhcg.sh
BUILD_TYPE=Release BUILD_DIR=build/g5-physical RUN_ONE_RANK_GPU=0 \
  bash scripts/gpu/run_g5_build_matrix.sh

python3 scripts/gpu/g5_campaign.py capture-preflight \
  --expected-gpus 8 \
  --build-directory build/g5-physical \
  --output /evidence/preflight/g5-node-preflight.json
```

Preflight returns nonzero for insufficient/heterogeneous GPUs, low free memory, active compute
processes, MIG, unsupported compute mode, missing affinity/NIC/toolchain data, dirty source, wrong
PDHCG pins, or missing build hashes. Never bypass a primary-campaign failure. Preserve
`preflight-raw/`, including full `nvidia-smi -q`, NVLink status, topology matrix, CPU/NUMA/network
data, tool versions, and command stderr.

## Logical command generation on any host

These manifests are permanently non-executable and are only command/schema evidence:

```bash
python3 scripts/gpu/g5_campaign.py generate-plan \
  --config benchmarks/g5_launch_probe.json \
  --logical-dry-run \
  --repository-commit "$(git rev-parse HEAD)" \
  --executable /opt/spacepdhcg/bin/g5_physical_validation_harness \
  --run-root /evidence/g5-runs \
  --output build/g5-logical-plan

python3 scripts/gpu/g5_campaign.py verify-plan \
  build/g5-logical-plan/launch-plan.json
```

The verifier checks all manifest/rankfile invariants and the installed OpenMPI options, CUDA
compiler, NCCL runtime/development packages, and linker-visible NCCL library without starting MPI
ranks.

## Physical launch probe

Generate a plan from the passing preflight. Select only rank counts supported by the allocation:

```bash
python3 scripts/gpu/g5_campaign.py generate-plan \
  --config benchmarks/g5_launch_probe.json \
  --preflight /evidence/preflight/g5-node-preflight.json \
  --gpu-count 8 \
  --repository-commit "$(git rev-parse HEAD)" \
  --executable "$PWD/build/g5-physical/distributed-tools/g5_physical_validation_harness" \
  --run-root /evidence/g5-runs \
  --output /evidence/g5-plans/8gpu-probe

python3 scripts/gpu/g5_campaign.py verify-plan \
  /evidence/g5-plans/8gpu-probe/launch-plan.json
```

Review `command.json`, its rankfile, `CUDA_VISIBLE_DEVICES`, GPU UUID/PCI ordering, disjoint CPU
sets, NUMA nodes, NIC, timeout, and evidence path. Execution requires an exact confirmation:

```bash
python3 scripts/gpu/g5_campaign.py execute \
  --manifest /evidence/g5-plans/8gpu-probe/COMMAND_ID/command.json \
  --preflight /evidence/preflight/g5-node-preflight.json \
  --confirm EXECUTE-PHYSICAL-G5
```

The launcher records per-rank files, OpenMPI/NCCL logs, 200 ms GPU memory/power/clock/temperature
samples, return status, missing ranks, and partial logs. `qualification_claim` and
`multi_gpu_scaling_verified` remain false by construction.

## Full strong/weak campaign plans

Use `benchmarks/g5_physical_campaign.json`. It freezes:

- ranks/GPUs 1, 2, 4, and 8;
- strong scenario counts 1,000 and 10,000;
- weak scenario count 1,000 per GPU;
- 100/500-node problems;
- scenario-aware and generic nonzero-balanced whole-scenario partitions;
- expected, worst, CVaR-0.9, and CVaR-0.99 risk semantics;
- two warmups, seven measured repeats, and twenty frozen seeds;
- matched one-GPU monolithic references.

Point `--executable` at the reviewed P1-F integrated solver for scientific runs. The executable must
emit one `rank-N.json` per rank with collective bytes/count/frequency/purpose, exposed/overlapped
communication, local compute and predicted/measured load, peak/free memory, timing, energy,
canonical residuals, non-anticipativity, risk epigraph/duals, and nonlinear replay quality.

Run order is fixed:

1. G=1 correctness and monolithic reference.
2. G=2 correctness, checkpoint/restart, and failure campaign.
3. G=4 correctness and failure campaign.
4. G=8 correctness and failure campaign.
5. Strong scaling, then weak scaling, scenario-aware before generic at each coordinate.
6. Repeat any censored coordinate only under the preregistered retry policy; retain failed attempts.

## Explicit failure campaign

Failure generation is separate from primary plans, requires one repeat, adds `--test-mode`, and
sets `SPACEPDHCG_G5_FAILURE_TEST=1`. It can never be enabled implicitly.

```bash
python3 scripts/gpu/g5_campaign.py generate-plan \
  --config benchmarks/g5_launch_probe.json \
  --preflight /evidence/preflight/g5-node-preflight.json \
  --gpu-count 8 \
  --failure-mode rank_failure \
  --repository-commit "$(git rev-parse HEAD)" \
  --executable "$PWD/build/g5-physical/distributed-tools/g5_physical_validation_harness" \
  --run-root /evidence/g5-failures \
  --output /evidence/g5-plans/8gpu-rank-failure
```

Execute and review each mode: `rank_failure`, `communicator_error`, `collective_order`,
`cancellation`, `checkpoint_restart`, `topology_mismatch`, `device_mismatch`, and `timeout`.
Timeout/rank-loss plans are expected to produce partial evidence. Ordinary OpenMPI is not ULFM;
rank loss is expected to terminate the job, not recover the communicator.

## Seal and transfer

Seal successful runs, and separately seal expected partial failure evidence:

```bash
python3 scripts/gpu/seal_g5_evidence.py seal /evidence/g5-runs/RUN_ID \
  --archive /archive/g5/RUN_ID.tar.gz

python3 scripts/gpu/seal_g5_evidence.py seal /evidence/g5-failures/RUN_ID \
  --archive /archive/g5/RUN_ID.tar.gz \
  --allow-partial-failure-evidence

python3 scripts/gpu/seal_g5_evidence.py verify \
  /archive/g5/RUN_ID.tar.gz.seal.json
```

Sealing requires a clean repository, refuses overwrite, creates a reproducible archive and
write-once SHA-256 descriptor, and makes the archive/index read-only. Copy archive and seal to
independent storage, verify there, then apply storage retention/immutability controls.

## Physical evidence checklist

- [ ] Clean source commit, branch, CMake cache hash, harness/solver hash, and PDHCG pins match.
- [ ] Exclusive allocation and GPU UUID/model/count/memory/driver homogeneity pass.
- [ ] Topology, NVLink/NVSwitch/PCIe, CPU/NUMA/NIC binding, clocks, power, ECC, MIG, and compute mode
      are captured.
- [ ] One MPI rank owns one unique PCI-ordered GPU and a disjoint CPU set.
- [ ] Launch probe passes at 2, 4, and 8 ranks with expected NCCL reduction values.
- [ ] All eight failure modes produce the expected status and preserve partial logs.
- [ ] Checkpoint/restart preserves rank, device, topology, partition, primal/dual/full-state ownership.
- [ ] Every primary coordinate has all warmups/repeats/seeds and a matched monolithic reference.
- [ ] Collective, overlap, compute/load, memory, timing/energy, residual/risk, and nonlinear fields are
      complete; no implicit host-gather metric is represented as a device reduction.
- [ ] Archives and seal descriptors verify after independent transfer.
- [ ] Review explicitly decides G5; tooling output alone never marks the gate PASS.
