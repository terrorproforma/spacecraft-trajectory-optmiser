# Native GPU SCvx diagnostic checkpoint, 2026-09-07

**Experimental, not ready for promotion.** The native adapter now connects
discretisation, conic assembly, IPM, independent conic auditing, SCvx decisions
and reference refresh inside a CUDA conditional graph. Repeated trajectory
qualification still fails in both ordinary and graph dispatch. The fleet score
has not changed. No overall speedup is claimed.

The local suite passed **337 tests**, and Lambda H100 passed **94 integration
tests**. A separate eight-process H100 comparison then qualified only **4/8**
trajectories against both independent physics and the known final-mass target.
All four unshifted cases qualified; all four shifted-state cases missed at least
one target. Local repeated tests also fail. A passing single suite is therefore
insufficient evidence of solver reliability.

## Concrete corrections

`scripts/gpu/prepare_qoco_stopping_accuracy.py`, applied after the QOCO133
outer-graph preparation, produces the QOCO135 diagnostic runtime:

- For the Ruiz transformation `x_original = D*x`, `y_original = E*y/k`,
  `z_original = F*z/k`, `s_original = inverse(F)*s`, the slack norm must use
  `inverse(F)`. The old metric used `F`.
- Complementarity is `abs(s'*z/k)`. The old metric multiplied both vectors
  by `F`, introducing an erroneous `F^2` factor.
- The quadratic objective product must be formed before applying `inverse(D)`
  for the stationarity norm. Primal and dual objective scales must then be
  divided by `k`.
- Termination now checks the larger of complementarity and the actual absolute
  primal/dual objective gap. Complementarity alone cannot certify objective
  accuracy away from stationarity. Explicit host arithmetic audits use the same
  corrected definition.
- Zero CPU Ruiz column/cost norms use identity scaling, matching the GPU update
  rule. Previously these could produce extreme scales and NaNs during priming.
- Graph completion restores the saved best iterate on an inaccurate exit as
  well as an iteration-limit or numerical-error exit.

The new original-equation test checks eight quantities on 64 QP iterates,
including nonuniform scaling, zero complementarity with nonzero objective gap,
absent constraints and zero quadratic terms. It passes on RTX 5090 and H100.
The legacy metric/graph-lifetime test, 128 queued exact-parity solves, 72 nested
results, 24 invalid-input cases and 32 changing analytic QPs also pass on both.
The replay fixture now requests a tighter relative solver tolerance to retain
its existing absolute residual requirement after correcting objective scaling;
its residual, primal and objective acceptance limits were not relaxed.

These corrections do **not** yet fix every complete-trajectory failure. Tighter
linear refinement, smaller regularization and absolute-only stopping were
measured as diagnostics and reverted. Their failed attempts are retained.

## Native outer graph

Enable explicitly with `SPACEPDHCG_TEST_GTOC12_OUTER_GRAPH=1`, together with
`SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION=1`,
`SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1`,
`SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1` and
`SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1`. Unsupported combinations return an explicit
error. This experimental path is limited to the tested SM90/SM120 targets.

Setup and two or three priming solves remain host-dispatched. The following
SCvx iterations use one enclosing graph launch. Graphs borrow solver buffers
under an exclusive lease; destruction drains the stream and releases graphs
before deleting solver storage. Changing parameters, invalid-input guards and
resource ownership have direct native tests.

An on-device ledger counts executed attempts and solver iterations. A device
deadline is checked between attempts; it does not preempt an IPM solve. Five
deadline/iteration-limit/error cases pass locally. Four recursive graph DOT
dumps contain eight conditional nodes each, 48 or 49 device-to-device copies,
and no host nodes or host/device memcpy nodes. Final reports and trajectories
are downloaded after completion. Per-iteration graph phase timings are absent
rather than misleading zeros. This does not claim the complete application,
setup, verification or fleet search is GPU native.

## Reproduction and evidence

The candidate core is frozen locally at
`/home/angus/build-spacepdhcg-gtoc12-v158/final`; QOCO135 is at
`/home/angus/build-qoco-gpu-device-ir-v135/final`.
The H100 source overlay, builds and results are under
`/home/ubuntu/spacepdhcg-diagnose-v162`.

Apply the stopping preparation to a fresh prepared QOCO133 source tree, then
build the shared library using the recorded CUDA 12.8/cuDSS 0.8 commands. Do not
apply it twice or overwrite frozen runtimes. Run
`qoco_equilibrated_stopping_test`, `qoco_gpu_stopping_test` (without the whole-IPM
flag, to exercise its own metric-graph lifetime), and the replay/nested probes.
Use the retained comparison scripts for complete trajectories; their success
criterion includes convergence, independent certification and final mass
`2445.3111007852112 +/- 1e-5 kg`.

[Summary and archive checksums](../results/lambda/2026-09-07/gpu-outer-v162/summary.json)
and the adjacent local/H100 archives retain successful and failed runs,
commands, source hashes, graph dumps and certificates. The v160 experiment
started close to build completion and is excluded from timing claims. No new
sanitizer result is claimed for this combined candidate.

The next diagnostic should explain the shifted-coordinate stopping/audit
disagreement and preserve the current physics/objective gates. Repeated
qualification, archived recovery cases, native lifetime/guard tests and
sanitizers must pass before promoting this candidate or restarting a large
optimisation campaign.
