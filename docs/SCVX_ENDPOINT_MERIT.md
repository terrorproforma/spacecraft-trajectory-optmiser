# Correct endpoint merit for GPU trajectory refinement

The native outer controller omitted endpoint errors from its merit function.
An exactly propagated starting trajectory could therefore look inexpensive even
when it missed the prescribed arrival. The corrected GPU controller now prices
the original endpoint constraints consistently in the reference, candidate and
predicted model costs. The conic problem and independent physics tolerances are
unchanged.

This was isolated using a fixed ship-8 route from the historical v767 fleet.
Only its 16048-to-33441 transfer changes: departure moves from MJD 66503 to 66467
and arrival from 67028 to 67012. The initial thrust schedule shifts by 36 days,
retains every burn duration/vector, and ends with a 20-day coast. An explicit
mesh seam at MJD 66992 preserves its final one-day burn. All other generated
nodes remain present; no thrust interpolation or CPU trajectory initialization
is introduced.

The supplied trajectory misses the changed arrival by 9,358,851 km and
0.357576 km/s, although its normalized local dynamics defect is 1.33e-15.
Previously its merit was just 0.0838692 fuel units. The first two qualified
conic corrections cost more than that, so the controller rejected both and
collapsed its trust region. The final infinite virtual-control field was the
never-accepted sentinel; those first two candidates had virtual norms near 1e-17.

## The correction

One GPU warp evaluates the original normalized endpoint constraints: initial
position, velocity and mass equal to one, plus final position and velocity.
Free Earth velocities retain their explicit excess-velocity auxiliaries and
their norm bounds. Legal nonzero excess velocity is not penalized, and no final
mass equality is introduced.

The L1 boundary penalty uses the existing conic deadband and virtual-control
weight. It appears in both nonlinear merits and the predicted model merit.
The raw boundary violation must also pass the existing defect threshold before
stationary convergence or finer-propagation confirmation. The implementation
adds no host readback and leaves the C ABI unchanged.

## Matched local result

The two runs use identical route prescriptions, seed controls, mesh, settings,
runtime libraries and independent certificate rules. The new native build
differs from the tested control in the outer controller and its test file only.
The changed leg's inherited mass is recomputed from the preceding certificates;
it differs by 8.97e-10 kg between runs. Its epochs, boundary position/velocity and
supplied thrust arrays are bit-identical. This is a matched campaign comparison,
not a claim that every runtime input bit is identical.

| Changed transfer | Before | After |
| --- | ---: | ---: |
| Outer iterations | 18 | 4 |
| Accepted steps | 0 | 4 |
| Reported solve time | 8.868 s | 0.290 s |
| Independently certified | no | yes |
| Arrival position error | no certificate | 37.7 m |

The complete corrected route passes all 17 flight certificates and three waiting
coasts. Both original complete-fleet checkers accept the same emitted Result,
SHA-256 `931defc362c9299d28d089aa46130c05f92bf9ee38ea03d870b4dc2a83915946`.
The other 22 ship sections remain byte-identical to the historical input.

That historical fleet improves from 12,843.555696 to **12,843.927775 weighted kg**,
a gain of 0.372079 kg, with 14,044.791239 raw kg. The worker takes 26.917 s,
including 21.035 s for independent CPU full-fleet propagation. It makes 17 native
calls and 34 outer iterations. These are bounded campaign measurements, including
an initial failed run, rather than a repeated throughput benchmark.

The v627 display retains this historical comparison. A separate v628 composition
applies only the newly certified ship 8 to the higher v799 fleet: its previous
ship 8 is byte-identical to the historical control, and the replacement conflicts
with no other ship. Fresh independent and official full-fleet checks accept
**12,992.407741 weighted kg / 14,271.485284 raw kg**, a 0.372079 weighted-kg gain
over v799, with 23 ships and 200 asteroids. The other 22 current ship sections
remain byte-identical. Composition takes 23.890 s and invokes no new GPU solves;
the new controller was measured locally, not rerun on H100.
[Current fleet, exact Result and loading instructions](../results/local/2026-09-09/current-fleet-composition-v628/README.md).

The mesh API passes 82 CPU regression tests; the matched replay passes 22 harness
tests. The native controller, verifier and ZOH initializer tests pass, including
64 endpoint/auxiliary cases and boundary convergence guards. Memory, race and
synchronization sanitizer checks report no errors. The new native measurements
are on RTX 5090; they do not establish H100 parity for this correction.

[Frozen inputs, source, failures, certificates and loading instructions](../results/local/2026-09-09/endpoint-merit-and-gpu-dual-v627/README.md).

## Next decision

The improvement removes a specific obstruction to refining changed routes.
Apply it to the current fleet's candidate-generation/refinement path and measure
qualified score gain per campaign. QOCO is still the mission subproblem backend;
this result does not measure PDHCG convergence. Keep that core's accuracy and
complete-time comparison as a separate acceptance gate. The
[updated execution priorities](SOTA_EXECUTION_PLAN_2026-09-09.md#current-checkpoint-and-next-work)
connect those experiments to the overall score and performance goals.
