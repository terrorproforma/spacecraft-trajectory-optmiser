# GPU-controlled integration and SCvx reference refresh

Experimental native core **v134**, using the unchanged QOCO **v128** runtime,
lets captured integration work read its step count and enable flag directly
from GPU memory. The SCvx reference refresh now consumes the GPU's decision to
switch into higher-resolution polishing without a host branch or an additional
command download. This is a step toward enclosing the full SCvx loop on device;
the outer attempt loop, time limit, conic dispatch and final reports still use
the CPU.

Set `SPACEPDHCG_TEST_GTOC12_DEVICE_REFRESH=1` to select the new refresh path.
The existing device qualification, assembly validation, canonical validation,
numeric replay, native replay and IPM graph options remain separately selected.
The default refresh path is retained for comparison. Neither the equations nor
the physics, objective or solver acceptance tolerances change.

## Device API and ordering

`spacepdhcg_gtoc12_discretisation_launch_controlled_device` takes retained device
scalars for the step count and an optional enable flag. It allocates nothing,
downloads nothing and performs no host synchronization. Its launches are
capture-compatible for both propagation and linearisation, and both ZOH and
four-node Lagrange controls.

- A null enable pointer means always execute; any nonzero value enables work.
- Disabled work preserves all numerical outputs and the existing invalid flag.
- An enabled step count below one sets the invalid flag and preserves numerical
  outputs. Enqueue success does not imply valid dynamics; consumers must check
  the flag before accepting those outputs.
- Input pointers must belong to the workspace device, remain alive through
  completion, and follow the existing stream ordering and non-aliasing contract.

SCvx queues the controlled propagation, gated parallel metric reductions and
gated reference update after candidate acceptance. The final update clears the
refresh flag only after all consumers have used it. Disabled refresh preserves
the candidate metrics, reference merit and error state. The regular per-attempt
command download then observes the completed refresh, including any error.

## Local verification

The frozen v134 runtime and tests are in
`/home/angus/build-spacepdhcg-gtoc12-v134/final` in WSL.

- Both holds pass 24 changing-input graph replays, independent analytic mass
  evolution and affine closure checks. An additional 28 controlled replays
  change the step count, enable state, invalid values and recovery. All retained
  outputs match the fixed-step API bit for bit when enabled and valid.
- Twelve captured reference-refresh cases cover enable/skip, both signs of
  nonzero enable values, invalid steps and complete reference-state parity with
  the former fixed-step chain.
- Memcheck, initcheck and synccheck report zero errors on both tests. These are
  isolated dynamics/control checks; the previously recorded full conditional
  IPM memcheck failure remains unresolved.
- The existing controller and parallel reductions pass, as do 82 Python GPU
  integration tests including independent CPU dynamics comparisons.
- The broader regression passes all 325 tests in 363.61 seconds. Its planner
  checks still use the separately recorded v77 executable; this does not imply
  that fleet search has moved to the GPU.

Six balanced triples retain all **36 complete transfer runs**, including
warmups. Every run passes independent trajectory certification and the unchanged
1e-5 kg final-mass gate against 2445.3111007852112 kg.

| Execution, all with QOCO v128 | Median complete transfer, excluding warmup |
|---|---:|
| Published core v133 | 350.887 ms |
| Core v134, host refresh | 323.730 ms |
| Core v134, GPU refresh | 364.085 ms |

These variable local timings do **not** establish an additional speedup. The
control-transfer evidence is narrower and concrete: all 12 GPU-refresh runs
download exactly `16 * (attempts + 1)` outer command bytes. The host-refresh
v134 runs make 12 additional command downloads in total, and the v133 runs make
nine, reflecting their different iteration histories. Each avoided refresh
download also avoids its stream wait. There is still one command read per
attempt and one initial read; these counters exclude solver/bridge reports.

The source, runtime hashes, helpers, checks and complete raw results are retained
in [the v134 checkpoint](../artifacts/performance/native-refresh-v134-checkpoint.json).
The first build's test-only initializer-list compilation failure and corrected
build log are retained too.

## Remaining work

The strict coast-constraint variability is still open. The deterministic vendor
mode and extra Ruiz scaling experiments did not produce a qualified improvement;
see [their retained results](QOCO_DETERMINISM_DIAGNOSTICS.md). No experimental
cuDSS setting is used by v134.

Next, propagate the device step-count/enable contract through conic assembly and
solver replay, replace the host outer-loop dispatch, and complete setup,
workspace reuse and fleet search on the GPU. No new fleet is claimed: the web
visualiser still displays v11 at 12,805.194102488575 weighted kg. Lambda's H100
remains occupied by the existing campaign, which has not been interrupted.
