# Native GPU initialization from an archived trajectory — v624

[Frozen source, raw results and independent audit package](../results/local/2026-09-09/native-trajectory-seed-v624/README.md).

The v623 control failed to regenerate a known feasible GTOC12 departure leg.
Its last conic subproblem qualified, but independent propagation missed the
target by 10,193 km. The archived trajectory for that same physical leg missed
by approximately 46 m. This motivates an initialization experiment, not a change
to the physical constraints or proof that initialization was the sole cause.

## Additive native path

`ZohTrajectorySeed` supplies the archived physical initial state, node epochs,
thrust vectors and source digest. It requires the exact generated ZOH mesh,
matching initial position/mass and matching fixed departure velocity. A free
departure velocity is preserved as recorded. A seed is an iterate, not a
certificate; the digest identifies its source but does not prove feasibility.

The native bridge uploads the physical initial state and thrust schedule once.
On the GPU, DOP853 advances each original constant-thrust interval, constructs
the state scaling and Gamma, and writes directly into the retained SCvx buffers.
There is no trajectory interpolation, clipping, target snapping, CPU propagation
or seed-state download in this solve path. The existing conic solver, SCvx
acceptance rules, budgets and independent physics certification remain in force.
The ordinary Lambert initialization and explicit NumPy ablation remain available.

One forward rollout has a causal dependency between intervals; its seven state
components cooperate in a warp. Subsequent SCvx interval linearization remains
parallel. The initializer has a global integration-step budget and invalidates
all output nodes on failure. A separate inspection bridge downloads the seed
only for comparison with the independently propagated CPU reference.

The native result now initializes its departure/arrival excess velocities from
the actual reference state. Previously these outputs remained zero until a
candidate was accepted, which could misrepresent a return before any accepted
step. Tests exercise both fixed/free endpoints and a zero-accepted timeout.

## Fixed mission experiment

The selected leg is ship 23, Earth to asteroid 30805, MJD 64403 to 64973,
initial mass 3,000 kg. All 159 archived constant-thrust burns and 126 coast
intervals fit the existing two-day grid exactly: 285 intervals, 286 nodes.
No control resampling or mission schedule change is needed.

The archived departure excess-speed norm is 6.000000002323666 km/s. The initial
iterate preserves that recorded value; the native optimization bound and the
existing independent verification tolerance are unchanged. Archive boundary
serialization differs from regenerated ephemeris in a few components by about
1e-11 km or 1e-14 km/s. The experiment uses archived boundary values and records
these differences. It is a diagnostic comparison with v623, not a byte-identical
paired timing benchmark.

The experiment is bounded to one supplied-seed native leg solve, with the same
40 main plus four polishing iterations, and one independent leg certificate.
No complete fleet replacement or leaderboard claim follows from this control.
The current native mission backend is GPU QOCO; this experiment does not measure
PDHCG convergence or establish a PDHCG speedup.

## Validation and results

The RTX 5090 control passes. The inspection bridge reproduces the independent
CPU trajectory to a maximum node-position difference of **5.148 mm**, velocity
difference 1.816e-13 km/s and mass difference 2.865e-11 kg. All scaled controls,
including Gamma, match exactly. The inspection takes 0.2609 s including its
initialization and download; this is not a steady-state kernel timing.

| Measurement | Supplied-seed native control |
| --- | ---: |
| SCvx iterations / accepted steps | 1 / 1 |
| Qualified QOCO subproblems / IPM iterations | 1 / 20 |
| Native adapter call wall time | 0.4163 s |
| Complete solution-reported wall time | 0.4178 s |
| Worker including inspection, refinement, certificate and evidence writes | 0.8476 s |
| Independent arrival position error | 0.057110396 km |
| Independent arrival velocity error | 5.36985e-9 km/s |
| Independently propagated arrival mass | 2580.675614107 kg |
| Final maximum normalized dynamics defect | 2.12136e-13 |
| Final virtual-control infinity norm | 2.33346e-15 |

The one conic subproblem qualifies with relative primal residual 1.9573e-12,
relative dual residual 9.1188e-12 and absolute objective gap 7.7586e-12. The outer
controller accepts its candidate and confirms it with finer propagation. It
finishes during priming, before entering the outer WHILE graph. Selecting graph
execution therefore does not imply graph iterations occurred in this run.

A separate CPU verifier replay of the final postprocessed controls also passes
the unchanged local gates. It gives a 0.057137424 km position error,
5.37161e-9 km/s velocity error and 2580.675614107172 kg final mass. Its scalar
position-error metric differs from the GPU certificate by 2.7 cm; the GPU's full
final-state vector was not retained, so this is not a vector-distance comparison.
The CPU replay retains its complete final state and all 160 segment endpoints.
It performs 159 DOP853 burn integrations and one analytical coast, without
optimization or GPU calls. Its 0.756 s, including observation overhead, is
separate verification work and is excluded from the GPU solve timing.

The trajectory input payload is 6,920 bytes: seven initial-state values and
286 thrust vectors. A further 2,288 bytes of node times are uploaded separately;
topology and other setup transfers also remain outside this trajectory counter.
The completed trajectory download is 25,168 bytes. Raw native arrays are saved
before the existing pipeline thrust postprocessing, and final certificate arrays
are retained separately. That existing clamp/coast cleanup still runs in Python;
this initializer does not make all surrounding pipeline arithmetic GPU-native.
Its CUDA port must preserve the same controls and certificate before replacing
that path. The certificate's legacy `rk4_vs_dop853_km` field is
not a new CPU DOP853 comparison under the CUDA certifier.

All 57 Python API/regression tests pass, with two existing catalogue-dependent
tests skipped. The 15 experiment-harness tests and the 10 saved-input mapping
tests pass. The SCvx controller, existing verifier and new ZOH CUDA tests pass;
Compute Sanitizer memory, race and synchronization checks report no errors.
The initial build's compiler naming collision is fixed and retained in the
evidence, along with an intermediate build; all runtime results use the final
fresh c build. Its core SHA-256 is
`cbe6207592e3bea5c8cefad624df9d6275f556858803e723a3e29e603e528550`.

This is a successful regeneration from a known feasible control schedule. The
propellant remains essentially the archived 419.324 kg; there is no mission-score
gain. The observed call time is not a paired cold/warm speedup or fleet throughput
measurement. The independently verified incumbent stays at **12,843.555696
fixed-bonus weighted kg**, 23 ships and 195 asteroids. This extension has been
tested on RTX 5090; an H100 run and a complete seeded-route regression remain.

The next mission gate is to regenerate the complete archived route with the
newly propagated mass carried between legs, then use compatible verified
initializations for changed-route refinement. PDHCG's own original-coordinate
convergence gates remain a separate requirement before native mission integration.

A parse-only preflight finds that all 19 flight control schedules fit their
actual production meshes: 2,494 nodes, 2,475 intervals and 834 burn intervals.
One transfer also has a 118-day departure wait and a 270-day arrival wait, which
need separate certification. Its flown endpoint states are not printed in the
archive, and the Earth-return spacecraft velocity must not be used as Earth
body velocity. These boundaries require the existing ephemeris model. The next
mass is the independently propagated mass plus fixed event changes; it must not
be reset to an archived mass or replaced by an optimizer state-node value.
