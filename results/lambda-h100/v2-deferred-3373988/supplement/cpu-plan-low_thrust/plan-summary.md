# Plan summary: Low-thrust transfer: 40-minute tangential radius raise from a 7000 km circular orbit (500 kg, 1 N)

- Family: `low_thrust` (low-thrust two-body transfer; central-body inertial Cartesian)
- Status: **certified** — converged and independently certified (CPU reference execution)
- Certified: **yes**
- Execution: `cpu_reference` via backend `cpu_reference` (policy `clarabel_scvx_reference`)

## Problem

- Intervals: 40, final time 2400 s, step 60 s
- Initial state: [7000, 0, 0, 0, 7.546053290107541, 0, 500]
- Terminal target (free components shown as `free`): [-5952.149848617802, 3686.049123066764, 0, -3.973625734575774, -6.4145609503740335, 0, "free"]
- Terminal fixed flags: [true, true, true, true, true, true, false]
- Canonical units: {"position": "km", "velocity": "km/s", "mass": "kg", "thrust": "N", "gravitational_parameter": "km^3/s^2", "time": "s"}

## Result

- Objective: 0.149921716 (mean_k(sigma_k) / maximum_thrust (normalised fuel))
- Outer iterations: 2 (accepted 2, rejected 0), inner iterations 55
- Final trust radius: 1
- Propellant used: 0.0122336, final mass 499.988
- Terminal position error (independent replay): 7.822e-11 km; velocity error 7.372e-14 km/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 6.119e-13 | — |
| dynamics defect (scaled) | 0 | 0 |
| path violation (normalised) | 3.21e-13 | 3.21e-13 |
| terminal residual (scaled) | 7.822e-14 | 7.822e-14 |
| virtual control | 1.412e-15 | — |
| replay parity | — | 0 |
| continuous-time violation (dense replay) | — | 3.21e-13 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 6.119e-13 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 3.21e-13 | 1e-06 |
| device_terminal_residual | yes | 7.822e-14 | 1e-06 |
| virtual_control | yes | 1.412e-15 | 1e-06 |
| independent_replay_parity | yes | 0 | 1e-09 |
| independent_dynamics_defect | yes | 0 | 1e-06 |
| independent_path_violation | yes | 3.21e-13 | 1e-06 |
| independent_terminal_residual | yes | 7.822e-14 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 0 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0 |
| topology_seconds | 0.00012136 |
| coefficient_seconds | 0 |
| workspace_create_seconds | 0.31373 |
| solve_seconds | 0.040929 |
| recovery_seconds | 0 |
| replay_seconds | 0 |
| acceptance_seconds | 0 |
| scvx_total_seconds | 1.2563 |
| independent_replay_seconds | 0.00013983 |
| plan_wall_seconds | 1.2565 |
