# Deferred do/test-later backlog: physical multi-GPU

Scope: `full-multi-gpu-v1` historical continuation or a future explicitly versioned successor.
Status: **deferred; not a blocker for `single-gpu-v1`**

This backlog moves execution priority only. It does not delete, weaken, or rewrite the G5
implementation, physical campaign tooling, acceptance rules, preregistration, or one-rank
historical evidence.

## Preserved G5 execution entry points

- Runbook: `docs/G5_PHYSICAL_EXECUTION_RUNBOOK.md`
- Implementation/status contract: `docs/G5_IMPLEMENTATION_STATUS.md`
- Frozen campaign matrix: `benchmarks/g5_physical_campaign.json`
- Launch probe: `benchmarks/g5_launch_probe.json`
- Preflight/planner/runner: `scripts/gpu/g5_campaign.py`
- Core planner and evidence logic: `src/spacepdhcg/experiments/g5_campaign.py`
- Physical MPI/NCCL harness: `g5_physical_validation_harness`
- Build matrix: `scripts/gpu/run_g5_build_matrix.sh`
- Evidence seal: `scripts/gpu/seal_g5_evidence.py`
- Schemas: `experiments/schema/g5_campaign.schema.json` and
  `experiments/schema/g5_implementation.schema.json`

The exact commands, immutable-plan hashes, failure injection guards, one-rank baseline,
topology/device checks, and write-once sealing rules in the runbook remain authoritative.

## Exact deferred requirements

1. G5 physical 2/4/8-GPU correctness against monolithic CPU/single-GPU truth.
2. G5 physical one-stream and overlap sanitizer/profiler validation.
3. G5 physical cancellation, rank-failure, communicator, checkpoint, and restart validation.
4. G5 scenario-aware versus generic physical partition comparison.
5. G5 same-machine strong scaling at 1/2/4/8 GPUs.
6. G5 weak scaling at fixed per-GPU scenario/nonzero work.
7. G5 physical scaling, communication, memory, energy, and H4 decision evidence.
8. OrbitWeaver distributed route-by-scenario correctness on 2/4/8 physical GPUs.
9. OrbitWeaver physical throughput, scaling, energy, memory-crossover, and tractability claims.

## Preserved G5 acceptance

Before any G5 PASS or scaling statement, execute the seven acceptance steps already recorded in
`docs/G5_IMPLEMENTATION_STATUS.md`:

1. Verify one-rank-per-GPU UUID, PCIe/NUMA, and `CUDA_VISIBLE_DEVICES` mapping on one node with 2,
   4, and 8 identical GPUs.
2. Compare monolithic CPU/single-GPU truth with distributed expected, worst, and CVaR CQPs,
   including epigraph primal/duals, canonical residuals, nonlinear replay, and
   non-anticipativity.
3. Exercise one-stream and overlap paths under memcheck, racecheck, initcheck, synccheck, and
   Nsight Systems.
4. Validate cancellation, injected rank/communicator failure, incompatible checkpoints,
   topology mutation, identical-rank-set restart, and explicit unrecoverable rank loss.
5. Compare scenario-aware and nonzero-balanced partitions on the identical global CQP/topology.
6. Run strong and weak scaling with same-machine one-GPU baselines and complete
   compute/communication/load/memory/failure/nonlinear-quality telemetry.
7. Resolve or honestly censor H2/H3/H4 under the frozen decision rules.

No logical rank, MPS substitute, one-rank execution, compile result, or synthetic fixture may
satisfy a physical requirement.

## Deferred OrbitWeaver acceptance

After G5 physical acceptance, run distributed OrbitWeaver with identical route/scenario inputs at
1/2/4/8 GPUs, independently certify retained trajectories, retain all failures, and only then
evaluate throughput, scaling, energy, crossover, or tractability-frontier claims. These products
belong to a separate scope and must never be merged into a `single-gpu-v1` freeze.
