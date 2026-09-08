# Fixed-route initialization and mass handoffs — v625

[Frozen sources, raw results and independent audits](../results/local/2026-09-09/native-route-and-dual-polish-v625/README.md).

The [v624 native initializer](GPU_VERIFIED_TRAJECTORY_INITIALIZATION.md) regenerated
one archived departure leg successfully. Extending that test to a complete route
requires controls to be initialized with the mass from the preceding independent
certificate, including deployment and collection events. Resetting each leg to
its archived mass would conceal accumulated fuel errors.

## Additive route API

`refine_fixed` now accepts two optional callbacks, in this order:

```python
boundary_factory(leg, generated_boundary, index) -> LegBoundary
seed_factory(leg, effective_boundary, index) -> ZohTrajectorySeed | None
```

The generated boundary uses the existing body ephemeris, fixed epochs, carried
cargo and independently certified arrival mass. The boundary callback may restore
an archived decimal representation of its position and velocity vectors. Every
component must be within `64 * float64_epsilon * max(1, max(abs(reference_vector)))`
of the generated vector. This is a representation guard, not a relaxation of the
trajectory verifier. Epochs, initial and minimum final mass, and free-velocity
flags must remain exactly equal. All four vectors must be finite three-vectors.
In particular, an archived Earth-return spacecraft velocity cannot replace the
Earth body velocity.

Callbacks receive immutable FP64 coordinate snapshots backed by bytes. A callback
cannot change the comparison reference through a NumPy array alias or re-enable
writing with `setflags`. Returned coordinates are independently snapshotted before
the seed callback and solver use them. Existing callers without hooks retain the
original unseeded solve signatures.

The seed provider receives the effective boundary after all preceding mass
events. The native initializer then propagates the supplied physical controls
using that actual initial mass. The hook does not interpolate controls, rescale
thrust, import archived trajectory nodes or change the prescribed cargo/schedule.
The existing seed validation, conic qualification, SCvx acceptance, independent
leg certification and route master remain in force.

Initialization errors stop before the affected native solve. The deadline is
checked again after initialization. A post-leg callback can reject an independent
wait certificate; its exception retains the completed leg and diagnostic record
and prevents route-master certification, including after the final flown leg.
Partial failures never trigger a retry, retiming or cargo reduction.

## Scope and verification

The focused CPU-only seed and route contract suite passes **70 tests**, with no
skips or native library calls. It checks actual certified mass ordering through
deployment/collection, coordinate immutability, rejected physical boundary changes,
initialization failure and deadlines, and failure retention after the last leg.
Ruff passes on both changed files.

These Python orchestration hooks connect the existing native GPU solve path.
Body ephemeris generation and the existing thrust postprocessing still include
CPU arithmetic. Complete fleet emission and both independent and official fleet
checks are required before treating a regenerated route as a valid replacement.
Reproducing an incumbent at unchanged cargo does not increase the mission score.

## Complete route result

The single RTX 5090 regression passes all **19 flown legs and both waits**.
Every leg takes one SCvx iteration; no cold solve, standalone seed inspection,
retiming or cargo reduction is performed. All 834 archived burn intervals fit
the exact meshes, with 2,494 total nodes. The final dry mass is
502.861383200 kg while carrying the prescribed 654.099931554 kg home.

| Measured scope | Result |
| --- | ---: |
| Native solves / SCvx iterations | 19 / 19 |
| QOCO subproblems / inner iterations | 19 / 468 |
| Independent GPU flight / wait certificates | 19 / 2 |
| Route refinement, certification and callbacks | 3.875972 s |
| Complete worker, including full-fleet checks and evidence | 25.192757 s |
| Independent CPU full-fleet check within that worker | 20.211418 s |
| Fixed-bonus weighted fleet score | 12,843.555695585 kg |

The route time corresponds to about 4.90 certified seeded legs per second for
this one known-feasible route. It is neither fresh mission-search throughput nor
a paired cold/warm or solver-baseline comparison. The 38 runtime CPU body-state
evaluations and Python thrust cleanup are still outside the fully GPU-native
destination. CPU full-fleet verification is separately visible above.

The deployment-to-collection interval around flown leg 9 includes a 118-day
departure wait and a 270-day arrival wait. MJD 67193 is a canonical solver
boundary only: it is not emitted as an event. The departure-wait propagation
differs from that canonical state by 0.797 mm; that residual is retained.
The arrival wait starts from the actual GPU flight-certificate final state,
with no body-state reset at MJD 67418. Its final position error is 33.16 m.
Both waits preserve mass exactly in the returned FP64 state.

The emitted solution retains all 20 real ship-23 events and leaves the other
22 ships byte-identical. Both the original independent CPU verifier and official
verifier pass on the same complete Result bytes, SHA-256
`6fa61648cbc958aec0f93b941d0bafc650fa6f0bcb2497b60743481a88e53ba8`.
These checks propagate the unbroken real-event interval across both waits and
the flight. Separate leg/wait certificates alone do not qualify the fleet.
The weighted-score difference from the retained incumbent is 8.37e-11 kg of
serialization roundoff. This zero-gain control does not promote the incumbent.

The saved route's legacy scheduler fields report its last one-leg submission;
the separately instrumented actual-call log records all 19. After the run,
`FrozenLegRunner` was changed to accumulate each new scheduler snapshot, including
partial failures, without recounting a previous snapshot on early validation
failure. Its adapter's existing workspace counters remain cumulative. Three
additional CPU cases cover this fix; the final focused suite passes **73 tests**
and Ruff. The measured frozen source and raw report remain unchanged. The
reporting-only revision does not require another GPU solve.

## H100 portability control

A fresh SM90 build from the exact v624 c source passes all three CUDA test
binaries and the memory, race and synchronization sanitizer runs. Its one
archived first-leg control also passes: one accepted SCvx step, 20 QOCO
iterations, a 0.287122 s native call and a 57.826 m independently propagated
arrival error. The worker takes 0.588052 s including its 0.259825 s inspection.
Its physical seed payload remains 6,920 bytes, plus 2,288 bytes of node times.
All owned processes exit and the GPU is idle afterward.

This H100 result validates the initializer and first-leg path on a second GPU.
It is not a full-route H100 measurement. QOCO is the native mission backend;
PDHCG's separate [dual-correction diagnostic](PDHCG_CONSTRAINED_DUAL_POLISH.md)
does not change that attribution.
