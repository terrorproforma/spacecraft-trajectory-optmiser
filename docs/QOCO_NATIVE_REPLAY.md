# Native solver replay and independent device audit

For the subsequent objective qualification and SCvx consumer integration, see
[GPU qualification and SCvx consumers](GTOC12_DEVICE_QUALIFICATION.md). The
measurements and limitations below describe the v128 checkpoint.

Experimental native core v128 connects the v126 prepared solver replay API to
the independent CUDA residual audit. After a synchronous priming solve, native
cold solves can queue solver execution, independent residuals, dual mapping and
primal copying on the same stream, then collect reports with one final wait.
The adapter no longer waits on the CPU between solver completion and audit.

Select `SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1` with the v126 prepared QOCO runtime
and `SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1`. Default native execution remains unchanged.
The optional `SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE` switch prints actual replay
calls. The first solve and a newly rebuilt/stale graph use synchronous GPU
priming. No CPU numerical solver is introduced.

The existing warm-inaccurate cold-retry path still consumes host-mutated solver
settings, so native warm solves retain their existing synchronous path. GTOC12
currently requests independent cold subproblem solves and therefore exercises
the new path. Moving that retry policy and full SCvx dispatch onto the device
remains unfinished.

## Independent audit and report budget

`qoco_gpu_audit_run_device` exposes the existing independent audit kernels and
retained result without a host download or wait. It is graph-capturable. The
synchronous audit API now wraps this operation and explicitly downloads the
same six residual scalars. Arithmetic, cone rejection and dual mapping remain
unchanged.

The first integration, v127, downloaded the entire 64-byte completion packet in
addition to the audit. It passed physics checks but failed eight existing
transfer-budget assertions: successful updates downloaded 120 bytes instead of
at most 64. Those failures and the frozen v127 runtime are retained.

v128 validates the completion ABI on the GPU and retains only status and IPM
iteration count for transitional host dispatch. The adapter downloads those
eight bytes plus the 48-byte independent audit and existing validation flags.
The original **64-byte per-update gate remains unchanged**. Remaining completion
metrics and all trajectory vectors stay on the GPU.

Retained CUDA events measure replay and audit separately. Replay-mode solve and
residual timings use GPU event intervals; the synchronous path uses its previous
host timing. Compare external complete-transfer wall time across modes, rather
than treating the internal timing methods as identical.

## Current boundaries

This removes a CPU dependency between the solver and its independent audit. It
does not remove the final status/audit download, GTOC12 objective-gap download,
assembly-validation downloads, host numerical-update orchestration or the outer
SCvx command loop. Initial topology/setup/analysis and vendor warm-up remain
host work. Full GPU-controlled trajectory execution and a new fleet search are
still unfinished; the visualiser continues to show the existing v11 fleet.

## Verification and retained failures

The enabled path passes all **51 integration tests**, including the unchanged
64-byte transfer limit, and the broader **324-test regression suite**. Native
controller checks pass. PD6 N20/N500 independently certify under unchanged 1e-8
reference-objective limits, with errors 5.399e-10 and 9.705e-10.

The independent audit test now captures and replays its device operation, checks
against dense reference arithmetic and verifies that it performs no internal
download. Its memcheck and initcheck runs report zero errors; memcheck reports
zero leaks. A separate 64-case graph probe verifies completion ABI validation,
status and iteration extraction, retained allocations and absence of internal
downloads. Normal, memcheck, initcheck and synccheck runs pass without errors.

The first v128 synchronous-ablation integration run failed one coast
qualification case on its first solve, before replay; the other 50 tests passed.
All 27 targeted follow-up coast solves across the old core, new synchronous
core and replay core qualified. Those follow-ups and the 324-test pass do not
erase the original failure. Qualification variability remains unresolved.

Full native-pipeline memcheck still aborts with CUDA 999 during initial
conditional-refinement warm-up in QOCO v126, before native replay. The abort
reports 208 errors and 206 outstanding allocations. The full pipeline remains
**not sanitizer-qualified** despite the isolated audit/status successes.

## Complete-transfer measurements

Six balanced triples run warm-up and measured transfers with the same QOCO v126
runtime. All **36 transfers** pass independent physics and the unchanged 1e-5 kg
gate against 2445.3111007852112 kg final mass.

| Native core / mode | Median measured complete transfer |
|---|---:|
| Previous v107 core | 337.209 ms |
| v128 synchronous ablation | 570.374 ms |
| v128 native replay | 324.569 ms |

The display RTX 5090 has unlocked clocks and the runs vary widely. No reliable
additional overall gain is established. All warm-ups and outliers are retained.
Trace logging confirms 148 actual native replay calls across the six replay
processes; its cost is included in these measurements.

The [v128 checkpoint](../artifacts/performance/native-replay-v128-checkpoint.json)
embeds five current sources, 11 runtime hashes, 41 helpers and 12 evidence reports,
including frozen v127 source and failed tests. SHA-256:
`27c8f7fbae4817f9bd86839c08d251155cac5199a916fe719b3a66335b1efdac`.
Lambda's H100 remains occupied at 100% utilisation by its existing campaign;
the check was read-only and no work was offloaded.
