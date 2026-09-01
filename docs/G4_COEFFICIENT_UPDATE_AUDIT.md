# G4 fixed-pattern coefficient update audit

This audit records which CQP numbers may change without changing sparse topology. `Q` is the
quadratic objective, `A` the scalar-constraint matrix, `F` the affine-cone matrix, `c` the linear
objective, `l/u` scalar and variable bounds, and `bK` the affine-cone offset.

## HCW

- Boundary state changes update `c` at every state node using the initial-to-target interpolation.
- Initial and terminal boundary rows update scalar `l/u`.
- `Q`, dynamics `A`, control-limit rows, `F`, variable bounds, and SOC acceleration offsets are
  fixed for a fixed horizon, timestep, mean motion, weights, and acceleration limit.
- HCW has no trust region, virtual control, exact penalty, nonlinear path linearisation, or
  scenario weight in this transcription.

## 3-DoF powered descent

- A new state/control reference updates state/control tracking entries in `c`.
- Every dynamics Jacobian and affine defect updates its fixed `A` slot and dynamics `l/u`.
- Initial and terminal targets update boundary `l/u`.
- The virtual-control L1 weight updates epigraph entries in `c`; virtual quadratic and epigraph
  regularisation weights update their diagonal `Q` slots when configured.
- State/control trust scales remain fixed in `F`; each reference updates the corresponding `bK`
  centre, and each trust-radius change updates every stage and terminal SOC radius in `bK`.
- Thrust, glide-slope, tilt, altitude, mass, sigma, and epigraph-bound coefficients are fixed for a
  fixed physical model. Fuel weight updates sigma entries in `c`.

## Low thrust

- All 3-DoF objective, dynamics, boundary, virtual-control, trust, fuel, and bound updates apply.
- Each position reference additionally updates the three `A` coefficients of its minimum-radius
  supporting halfspace. Its lower bound remains the physical minimum radius.
- Thrust SOC coefficients and mass/throttle bounds remain fixed for a fixed model.

## 6-DoF powered descent

- All 3-DoF objective, dynamics, boundary, virtual-control, trust, fuel, and bound updates apply
  with 14-state and 7-control dimensions.
- Each quaternion reference updates four `A` coefficients (`2 q_ref`) and both equality bounds
  (`1 + ||q_ref||^2`) for the unit-quaternion linearisation.
- Thrust, torque, glide, angular-rate, tilt, altitude, mass, sigma, and epigraph coefficients stay
  fixed for a fixed physical model.

## Scaling, forcing, and scenarios

- Numerical scaling is derived workspace state, not a CQP coefficient. The frozen
  `refresh_if_needed` policy reevaluates it after numerical updates when the registered range-ratio
  condition is met; a scaling refresh must not change unscaled `Q/A/F/c/l/u/bK`.
- Forcing policy changes solve tolerances and iteration limits only. It must not mutate the CQP.
  A rejected refined re-solve therefore requires an identical full numerical fingerprint.
- These four production transcriptions contain no scenario-weighted objective or constraint
  terms. Distributed scenario/risk CQPs are outside this driver and must not be silently mapped
  onto these layouts.

The CUDA update descriptor stores all sparse position maps at create time. Iterations launch
values-only kernels against those maps; they do not allocate, copy topology indices, resize a
matrix, or stage the full CQP through host memory.
